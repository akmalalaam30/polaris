"""Graph-based navigation route optimisation.

A navigation graph is built over the navigable cells of the analysis grid
(8-connected), with every edge carrying three measured quantities:

``distance_km``   great-circle length of the edge
``risk``          mean total risk of the two endpoint cells
``hours``/``fuel_tonnes``
                  derived from an explicit speed model: service speed reduced by
                  ice concentration (scaled by the vessel's ice class) and by
                  wave-induced added resistance.

Edge cost is expressed in **equivalent kilometres** so the three terms can be
traded off transparently::

    cost = w_distance * distance
         + w_risk     * risk * distance * RISK_KM_EQUIVALENT
         + w_fuel     * fuel_tonnes     * FUEL_KM_EQUIVALENT

Three profiles are produced from the same graph, differing only in weights:

``shortest``  minimise distance
``safest``    minimise accumulated risk exposure
``polaris``   balance distance, risk and fuel, shaded by the vessel's stated
              risk tolerance

Routing uses A* (Dijkstra is also exposed) via NetworkX.  The heuristic is
``w_distance * great_circle_to_goal``, which is admissible because the risk and
fuel terms are non-negative - so A* returns a true optimum for the given cost.

Reported fuel and duration are **model estimates** from the documented speed
model, not measurements, and POLARIS makes no claim about savings it has not
computed from these same numbers.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from app.config import STATIONS, Settings, get_settings
from app.services import landmask
from app.services.risk_engine import RiskGridResult, VesselProfile, assess_route, smoothstep
from app.utils.geo import GridSpec, haversine_km, initial_bearing_deg, path_length_km
from app.utils.logging import get_logger
from app.utils.validation import ValidationError, validate_coordinate

log = get_logger("services.route_optimizer")

#: Exchange rate between safety and distance: sustaining a risk of 1.0 over one
#: kilometre costs the same as this many kilometres of extra steaming.
RISK_KM_EQUIVALENT = 4.0
#: Exchange rate between fuel and distance, in equivalent km per tonne burned.
FUEL_KM_EQUIVALENT = 12.0

PROFILES = ("shortest", "safest", "polaris")

#: 8-connected neighbourhood.
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# Cost configuration
# ---------------------------------------------------------------------------
@dataclass
class CostWeights:
    """Weights of the three cost terms, in equivalent-kilometre space."""

    distance: float = 1.0
    risk: float = 1.0
    fuel: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {"distance": self.distance, "risk": self.risk, "fuel": self.fuel}


def profile_weights(profile: str, vessel: VesselProfile, settings: Settings) -> CostWeights:
    """Cost weights for a named optimisation profile."""
    profile = profile.lower()
    if profile == "shortest":
        # Pure distance. Risk still blocks impassable cells via the graph mask.
        return CostWeights(distance=1.0, risk=0.0, fuel=0.0)
    if profile == "safest":
        # Distance is kept at a small non-zero weight so the search still
        # terminates promptly and does not wander through equal-risk water.
        return CostWeights(distance=0.05, risk=1.0, fuel=0.0)
    if profile == "polaris":
        # A risk-tolerant vessel discounts the risk term and cares more about
        # fuel; a risk-averse one does the opposite.
        tolerance = vessel.risk_tolerance
        return CostWeights(
            distance=settings.route_distance_weight,
            risk=settings.route_risk_weight * (1.4 - tolerance),
            fuel=settings.route_fuel_weight * (0.4 + 0.8 * tolerance),
        )
    raise ValidationError(f"Unknown route profile {profile!r}; expected one of {PROFILES}")


# ---------------------------------------------------------------------------
# Speed / fuel model
# ---------------------------------------------------------------------------
def speed_factor(concentration: float, wave_height_m: float, vessel: VesselProfile) -> float:
    """Fraction of service speed achievable in the given conditions.

    * **Ice** - speed is unaffected in open water, and falls towards 12% of
      service speed as concentration approaches the vessel's ice capability.
      Ramming/backing progress below that is not modelled.
    * **Waves** - added resistance and voluntary speed reduction remove up to
      25% of speed by 8 m significant wave height.
    """
    cap = max(0.1, vessel.ice_capability)
    ice_penalty = smoothstep(float(concentration or 0.0), 0.10, min(0.99, cap + 0.10))
    ice_factor = 1.0 - 0.88 * float(ice_penalty)
    wave_factor = 1.0 - 0.25 * float(smoothstep(float(wave_height_m or 0.0), 3.0, 8.0))
    return float(np.clip(ice_factor * wave_factor, 0.08, 1.0))


def segment_cost(
    distance_km: float,
    risk: float,
    concentration: float,
    wave_height_m: float,
    vessel: VesselProfile,
) -> tuple[float, float]:
    """``(hours, fuel_tonnes)`` for one edge under the documented speed model."""
    speed = max(0.5, vessel.speed_km_h * speed_factor(concentration, wave_height_m, vessel))
    hours = distance_km / speed
    fuel = hours * (vessel.fuel_consumption_tpd / 24.0)
    return hours, fuel


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
@dataclass
class NavigationGraph:
    """The navigation graph plus the fields used to build it."""

    graph: nx.Graph
    grid: GridSpec
    vessel: VesselProfile
    navigable: np.ndarray
    risk: np.ndarray
    concentration: np.ndarray
    wave_height: np.ndarray
    degraded_nodes: set[tuple[int, int]] = field(default_factory=set)

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def node_position(self, node: tuple[int, int]) -> tuple[float, float]:
        return self.grid.cell_center(*node)

    def nearest_node(self, lat: float, lon: float) -> tuple[tuple[int, int], float]:
        """Closest graph node to a coordinate, with the snap distance in km."""
        if not self.graph.number_of_nodes():
            raise ValidationError("Navigation graph is empty: no navigable water in the domain")
        nodes = np.array(list(self.graph.nodes))
        lats = self.grid.lats[nodes[:, 0]]
        lons = self.grid.lons[nodes[:, 1]]
        d = haversine_km(lat, lon, lats, lons)
        k = int(np.argmin(d))
        return (int(nodes[k, 0]), int(nodes[k, 1])), float(d[k])

    def summary(self) -> dict:
        return {
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "grid": self.grid.to_dict(),
            "degraded_nodes": len(self.degraded_nodes),
        }


def build_navigation_graph(
    risk_result: RiskGridResult,
    fields: Mapping[str, np.ndarray] | None = None,
    vessel: VesselProfile | None = None,
    allow_high_risk: bool = False,
) -> NavigationGraph:
    """Build the 8-connected navigation graph over navigable water.

    ``allow_high_risk`` keeps cells that exceed the risk ceiling (but are still
    open water) in the graph, tagged as degraded.  It is used only as a fallback
    when no fully compliant route exists, and the result is flagged as such.
    """
    grid = risk_result.grid
    vessel = vessel or risk_result.vessel
    fields = fields or {}
    risk = np.nan_to_num(risk_result.total, nan=1.0)
    conc = np.nan_to_num(np.asarray(fields.get("sea_ice_concentration", np.zeros(grid.shape)), float), nan=0.0)
    waves = np.nan_to_num(np.asarray(fields.get("wave_height", np.zeros(grid.shape)), float), nan=0.0)

    water = landmask.navigable_mask(grid)
    allowed = water if allow_high_risk else risk_result.navigable
    degraded = set()
    if allow_high_risk:
        blocked = water & ~risk_result.navigable
        degraded = {(int(i), int(j)) for i, j in zip(*np.where(blocked))}

    g = nx.Graph()
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            if allowed[i, j]:
                g.add_node((i, j), latitude=float(grid.lats[i]), longitude=float(grid.lons[j]))

    for (i, j) in list(g.nodes):
        lat1, lon1 = grid.lats[i], grid.lons[j]
        for di, dj in _NEIGHBOURS:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= rows or nj < 0 or nj >= cols or not allowed[ni, nj]:
                continue
            if (ni, nj) < (i, j):
                continue  # undirected: add each pair once
            lat2, lon2 = grid.lats[ni], grid.lons[nj]
            d_km = float(haversine_km(lat1, lon1, lat2, lon2))
            if d_km <= 0:
                continue
            edge_risk = float(0.5 * (risk[i, j] + risk[ni, nj]))
            edge_conc = float(0.5 * (conc[i, j] + conc[ni, nj]))
            edge_wave = float(0.5 * (waves[i, j] + waves[ni, nj]))
            hours, fuel = segment_cost(d_km, edge_risk, edge_conc, edge_wave, vessel)
            g.add_edge(
                (i, j), (ni, nj),
                distance_km=d_km,
                risk=edge_risk,
                concentration=edge_conc,
                wave_height=edge_wave,
                hours=hours,
                fuel_tonnes=fuel,
                degraded=bool((i, j) in degraded or (ni, nj) in degraded),
            )

    log.info("Navigation graph: %d nodes, %d edges (high-risk allowed=%s)", g.number_of_nodes(), g.number_of_edges(), allow_high_risk)
    return NavigationGraph(
        graph=g, grid=grid, vessel=vessel, navigable=allowed,
        risk=risk, concentration=conc, wave_height=waves, degraded_nodes=degraded,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
@dataclass
class RouteSegment:
    """One graph edge as traversed by the route."""

    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    distance_km: float
    risk: float
    concentration: float
    hours: float
    fuel_tonnes: float


@dataclass
class Route:
    """A computed route with its metrics and provenance."""

    profile: str
    waypoints: list[tuple[float, float]]
    segments: list[RouteSegment]
    weights: CostWeights
    vessel: VesselProfile
    status: str = "ok"
    notes: list[str] = field(default_factory=list)
    algorithm: str = "astar"
    start_snap_km: float = 0.0
    end_snap_km: float = 0.0
    risk_assessment: dict = field(default_factory=dict)

    @property
    def distance_km(self) -> float:
        return float(sum(s.distance_km for s in self.segments))

    @property
    def duration_hours(self) -> float:
        return float(sum(s.hours for s in self.segments))

    @property
    def fuel_tonnes(self) -> float:
        return float(sum(s.fuel_tonnes for s in self.segments))

    @property
    def mean_risk(self) -> float:
        total_d = self.distance_km
        if total_d <= 0:
            return 0.0
        return float(sum(s.risk * s.distance_km for s in self.segments) / total_d)

    @property
    def max_risk(self) -> float:
        return float(max((s.risk for s in self.segments), default=0.0))

    @property
    def max_concentration(self) -> float:
        return float(max((s.concentration for s in self.segments), default=0.0))

    def geojson(self) -> dict:
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(lon, 5), round(lat, 5)] for lat, lon in self.waypoints],
            },
            "properties": {
                "profile": self.profile,
                "distance_km": round(self.distance_km, 2),
                "duration_hours": round(self.duration_hours, 2),
                "fuel_tonnes": round(self.fuel_tonnes, 3),
                "mean_risk": round(self.mean_risk, 4),
                "max_risk": round(self.max_risk, 4),
                "status": self.status,
            },
        }

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "status": self.status,
            "algorithm": self.algorithm,
            "distance_km": round(self.distance_km, 2),
            "duration_hours": round(self.duration_hours, 2),
            "estimated_fuel_tonnes": round(self.fuel_tonnes, 3),
            "mean_risk": round(self.mean_risk, 4),
            "max_risk": round(self.max_risk, 4),
            "max_sea_ice_concentration": round(self.max_concentration, 3),
            "n_waypoints": len(self.waypoints),
            "waypoints": [
                {"latitude": round(lat, 5), "longitude": round(lon, 5)} for lat, lon in self.waypoints
            ],
            "geometry": self.geojson(),
            "cost_weights": self.weights.as_dict(),
            "start_snap_km": round(self.start_snap_km, 2),
            "end_snap_km": round(self.end_snap_km, 2),
            "risk_assessment": self.risk_assessment,
            "notes": self.notes,
        }


def _edge_weight_fn(weights: CostWeights):
    def weight(_u, _v, data) -> float:
        return (
            weights.distance * data["distance_km"]
            + weights.risk * data["risk"] * data["distance_km"] * RISK_KM_EQUIVALENT
            + weights.fuel * data["fuel_tonnes"] * FUEL_KM_EQUIVALENT
        )

    return weight


def _heuristic_fn(nav: NavigationGraph, goal: tuple[int, int], weights: CostWeights):
    goal_lat, goal_lon = nav.node_position(goal)
    scale = max(weights.distance, 1e-6)

    def heuristic(node: tuple[int, int], _goal) -> float:
        lat, lon = nav.node_position(node)
        return scale * float(haversine_km(lat, lon, goal_lat, goal_lon))

    return heuristic


def _crosses_restricted_water(
    points: Sequence[tuple[float, float]], nav: "NavigationGraph"
) -> bool:
    """True if the straight polyline leaves the graph's navigable cells.

    Simplification must never turn a legal path into one that cuts a headland,
    so every simplified leg is re-sampled and checked against the same mask the
    graph was built from.
    """
    from app.utils.geo import densify_great_circle

    grid = nav.grid
    step_km = min(grid.dlat, grid.dlon) * 55.0
    for k in range(len(points) - 1):
        lat1, lon1 = points[k]
        lat2, lon2 = points[k + 1]
        for lat, lon in densify_great_circle(lat1, lon1, lat2, lon2, step_km=max(step_km, 10.0)):
            if not grid.contains(lat, lon):
                return True
            i, j = grid.index_of(lat, lon)
            if not nav.navigable[i, j]:
                return True
    return False


def _safe_simplify(
    points: Sequence[tuple[float, float]], nav: "NavigationGraph"
) -> list[tuple[float, float]]:
    """Simplify only as far as the result stays inside navigable water."""
    for tolerance in (0.05, 0.02, 0.01):
        candidate = _simplify(points, tolerance)
        if not _crosses_restricted_water(candidate, nav):
            return candidate
    return list(points)


def _simplify(points: Sequence[tuple[float, float]], tolerance_deg: float = 0.02) -> list[tuple[float, float]]:
    """Drop near-collinear waypoints (Douglas-Peucker in lat/lon space)."""
    pts = list(points)
    if len(pts) < 3:
        return pts

    def rdp(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(seq) < 3:
            return seq
        (x1, y1), (x2, y2) = seq[0], seq[-1]
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy) or 1e-9
        worst_i, worst_d = 0, 0.0
        for k in range(1, len(seq) - 1):
            x0, y0 = seq[k]
            d = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / norm
            if d > worst_d:
                worst_i, worst_d = k, d
        if worst_d <= tolerance_deg:
            return [seq[0], seq[-1]]
        return rdp(seq[: worst_i + 1])[:-1] + rdp(seq[worst_i:])

    return rdp(pts)


def _build_route(
    nav: NavigationGraph,
    node_path: Sequence[tuple[int, int]],
    profile: str,
    weights: CostWeights,
    algorithm: str,
    start_snap_km: float,
    end_snap_km: float,
    simplify: bool,
) -> Route:
    segments: list[RouteSegment] = []
    waypoints: list[tuple[float, float]] = []
    for k, node in enumerate(node_path):
        lat, lon = nav.node_position(node)
        waypoints.append((lat, lon))
        if k == 0:
            continue
        data = nav.graph.get_edge_data(node_path[k - 1], node)
        prev_lat, prev_lon = nav.node_position(node_path[k - 1])
        segments.append(
            RouteSegment(
                from_lat=prev_lat, from_lon=prev_lon, to_lat=lat, to_lon=lon,
                distance_km=data["distance_km"], risk=data["risk"],
                concentration=data["concentration"], hours=data["hours"],
                fuel_tonnes=data["fuel_tonnes"],
            )
        )

    notes: list[str] = []
    status = "ok"
    restricted = [
        k for k in range(1, len(node_path))
        if nav.graph.get_edge_data(node_path[k - 1], node_path[k]).get("degraded")
    ]
    if restricted:
        status = "degraded"
        max_conc = max(segments[k - 1].concentration for k in restricted)
        max_risk = max(segments[k - 1].risk for k in restricted)
        reasons = []
        if max_conc >= min(0.99, nav.vessel.ice_capability + 0.12):
            reasons.append(
                f"sea-ice concentration reaches {max_conc:.2f}, above the {nav.vessel.ice_class} "
                f"hull's working limit"
            )
        if max_risk > 0:
            reasons.append(f"peak cell risk on the restricted section is {max_risk:.2f}")
        notes.append(
            "No route exists that satisfies every navigability constraint. This path crosses "
            + str(len(restricted))
            + " restricted segment(s) ("
            + "; ".join(reasons)
            + ") and requires manual review or icebreaker support."
        )

    display = _safe_simplify(waypoints, nav) if simplify else waypoints
    route = Route(
        profile=profile, waypoints=display, segments=segments, weights=weights,
        vessel=nav.vessel, status=status, notes=notes, algorithm=algorithm,
        start_snap_km=start_snap_km, end_snap_km=end_snap_km,
    )
    if start_snap_km > 1.0:
        notes.append(
            f"Start position is on land or outside navigable water; snapped {start_snap_km:.1f} km "
            f"to the nearest navigable point."
        )
    if end_snap_km > 1.0:
        notes.append(
            f"Destination is on land or outside navigable water; snapped {end_snap_km:.1f} km "
            f"to the nearest navigable point."
        )
    return route


def find_route(
    nav: NavigationGraph,
    start: tuple[float, float],
    end: tuple[float, float],
    profile: str = "polaris",
    settings: Settings | None = None,
    algorithm: str = "astar",
    simplify: bool = True,
) -> Route:
    """Compute one route between two coordinates for a given profile."""
    settings = settings or get_settings()
    weights = profile_weights(profile, nav.vessel, settings)
    start_node, start_snap = nav.nearest_node(*start)
    end_node, end_snap = nav.nearest_node(*end)
    if start_node == end_node:
        raise ValidationError(
            "Start and destination snap to the same grid cell; choose points further apart "
            "or increase the grid resolution."
        )

    weight_fn = _edge_weight_fn(weights)
    try:
        if algorithm == "dijkstra":
            node_path = nx.dijkstra_path(nav.graph, start_node, end_node, weight=weight_fn)
        else:
            node_path = nx.astar_path(
                nav.graph, start_node, end_node,
                heuristic=_heuristic_fn(nav, end_node, weights), weight=weight_fn,
            )
    except nx.NetworkXNoPath as exc:
        raise RouteNotFoundError(
            f"No {profile} route exists between the requested points within the current "
            f"navigability constraints."
        ) from exc
    except nx.NodeNotFound as exc:  # pragma: no cover - guarded by nearest_node
        raise RouteNotFoundError(str(exc)) from exc

    return _build_route(nav, node_path, profile, weights, algorithm, start_snap, end_snap, simplify)


class RouteNotFoundError(RuntimeError):
    """Raised when the graph contains no path between the endpoints."""


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------
def resolve_location(value: Mapping | str, label: str) -> tuple[float, float, str | None]:
    """Accept ``{"latitude":..,"longitude":..}``, ``{"station": "maitri"}`` or a station name."""
    if isinstance(value, str):
        value = {"station": value}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object with coordinates or a station name")

    station = value.get("station") or value.get("name")
    if station and (value.get("latitude") is None or value.get("longitude") is None):
        key = str(station).strip().lower().replace(" ", "_").replace("_station", "")
        if key not in STATIONS:
            raise ValidationError(
                f"Unknown station {station!r}; known stations: {sorted(STATIONS)}"
            )
        entry = STATIONS[key]
        return entry["latitude"], entry["longitude"], entry["name"]

    lat, lon = validate_coordinate(value.get("latitude"), value.get("longitude"), label)
    return lat, lon, value.get("name")


def optimize_routes(
    risk_result: RiskGridResult,
    start: Mapping | str,
    end: Mapping | str,
    fields: Mapping[str, np.ndarray] | None = None,
    vessel: VesselProfile | None = None,
    profiles: Sequence[str] = PROFILES,
    settings: Settings | None = None,
    icebergs: Sequence[Mapping] | None = None,
    algorithm: str = "astar",
) -> dict:
    """Compute every requested profile between two points.

    Falls back to a degraded graph (open water above the risk ceiling) only if
    no compliant route exists, and marks the result accordingly.
    """
    settings = settings or get_settings()
    vessel = vessel or risk_result.vessel
    start_lat, start_lon, start_name = resolve_location(start, "start")
    end_lat, end_lon, end_name = resolve_location(end, "destination")

    for lat, lon, label in ((start_lat, start_lon, "start"), (end_lat, end_lon, "destination")):
        if not risk_result.grid.contains(lat, lon):
            raise ValidationError(
                f"{label} ({lat:.3f}, {lon:.3f}) is outside the POLARIS domain "
                f"(lat {risk_result.grid.lat_min}..{risk_result.grid.lat_max}, "
                f"lon {risk_result.grid.lon_min}..{risk_result.grid.lon_max})"
            )

    nav = build_navigation_graph(risk_result, fields, vessel, allow_high_risk=False)
    routes: dict[str, Route] = {}
    errors: dict[str, str] = {}
    degraded_nav: NavigationGraph | None = None

    for profile in profiles:
        try:
            routes[profile] = find_route(
                nav, (start_lat, start_lon), (end_lat, end_lon), profile, settings, algorithm
            )
        except RouteNotFoundError as exc:
            log.warning("%s route unavailable on the compliant graph: %s", profile, exc)
            if degraded_nav is None:
                degraded_nav = build_navigation_graph(risk_result, fields, vessel, allow_high_risk=True)
            try:
                routes[profile] = find_route(
                    degraded_nav, (start_lat, start_lon), (end_lat, end_lon), profile, settings, algorithm
                )
            except (RouteNotFoundError, ValidationError) as exc2:
                errors[profile] = str(exc2)
        except ValidationError as exc:
            errors[profile] = str(exc)

    for profile, route in routes.items():
        route.risk_assessment = assess_route(risk_result, route.waypoints, icebergs)

    request_id = uuid.uuid4().hex[:16]
    return {
        "request_id": request_id,
        "computed_at": datetime.now(timezone.utc),
        "start": {"latitude": start_lat, "longitude": start_lon, "name": start_name},
        "destination": {"latitude": end_lat, "longitude": end_lon, "name": end_name},
        "vessel": vessel.to_dict(),
        "graph": nav.summary(),
        "fallback_graph": degraded_nav.summary() if degraded_nav is not None else None,
        "risk_grid": risk_result.summary(),
        "routes": {p: r.to_dict() for p, r in routes.items()},
        "recommended_profile": "polaris" if "polaris" in routes else (next(iter(routes), None)),
        "comparison": compare_routes(routes),
        "errors": errors,
        "data_mode": risk_result.data_mode,
    }


def compare_routes(routes: Mapping[str, Route]) -> dict:
    """Side-by-side comparison of the computed profiles.

    Differences are stated relative to the shortest route when one exists; they
    are computed from this run's own numbers, not asserted.
    """
    if not routes:
        return {}
    reference = routes.get("shortest") or next(iter(routes.values()))
    out: dict[str, dict] = {}
    for name, route in routes.items():
        out[name] = {
            "distance_km": round(route.distance_km, 2),
            "duration_hours": round(route.duration_hours, 2),
            "estimated_fuel_tonnes": round(route.fuel_tonnes, 3),
            "mean_risk": round(route.mean_risk, 4),
            "max_risk": round(route.max_risk, 4),
            "status": route.status,
        }
        if reference is not route and reference.distance_km > 0:
            out[name]["distance_vs_shortest_pct"] = round(
                100.0 * (route.distance_km - reference.distance_km) / reference.distance_km, 2
            )
            if reference.fuel_tonnes > 0:
                out[name]["fuel_vs_shortest_pct"] = round(
                    100.0 * (route.fuel_tonnes - reference.fuel_tonnes) / reference.fuel_tonnes, 2
                )
            if reference.mean_risk > 0:
                out[name]["mean_risk_vs_shortest_pct"] = round(
                    100.0 * (route.mean_risk - reference.mean_risk) / reference.mean_risk, 2
                )
    return out


def route_rows(result: Mapping, data_mode: str) -> list[dict]:
    """Convert an :func:`optimize_routes` result into ``route_results`` rows."""
    rows = []
    for profile, route in result["routes"].items():
        rows.append(
            {
                "request_id": result["request_id"],
                "computed_at": result["computed_at"],
                "profile": profile,
                "start_name": result["start"].get("name"),
                "end_name": result["destination"].get("name"),
                "start_latitude": result["start"]["latitude"],
                "start_longitude": result["start"]["longitude"],
                "end_latitude": result["destination"]["latitude"],
                "end_longitude": result["destination"]["longitude"],
                "distance_km": route["distance_km"],
                "duration_hours": route["duration_hours"],
                "fuel_tonnes": route["estimated_fuel_tonnes"],
                "mean_risk": route["mean_risk"],
                "max_risk": route["max_risk"],
                "n_waypoints": route["n_waypoints"],
                "status": route["status"],
                "geometry": route["geometry"],
                "parameters": {
                    "vessel": result["vessel"],
                    "cost_weights": route["cost_weights"],
                    "risk_weights": result["risk_grid"]["weights"],
                },
                "notes": "; ".join(route["notes"]) or None,
                "data_mode": data_mode,
            }
        )
    return rows


def persist_routes(session, result: Mapping, data_mode: str) -> list[int]:
    from app.database import repositories as repo

    return repo.save_routes(session, route_rows(result, data_mode))
