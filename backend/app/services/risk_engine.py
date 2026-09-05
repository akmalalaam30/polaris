"""Navigation risk engine.

The engine converts environmental state into a **decision-support** risk score
in ``[0, 1]`` (0 = negligible, 1 = extreme).  It is not a safety guarantee: it
ranks options under uncertainty using the information that is actually
available, and it reports which layers were missing.

Total risk is a weighted sum of five normalised components::

    total = (w_ice*R_ice + w_berg*R_berg + w_wx*R_wx + w_ocn*R_ocn + w_con*R_con)
            / sum(weights of the components that had data)

Re-normalising over the *available* weights means a missing layer degrades the
confidence of the score rather than silently pulling it towards zero; the
missing layers are listed in the result.

Every component mapping below states its thresholds explicitly so they can be
reviewed, tuned per vessel, or replaced with operator-specific tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.services import landmask
from app.utils.geo import GridSpec, haversine_km, point_segment_distance_km
from app.utils.logging import get_logger
from app.utils.validation import clamp01

log = get_logger("services.risk_engine")

RISK_GRID_VERSION = "1.0.0"

#: Ice capability by class.  The value is the sea-ice concentration a vessel can
#: work in before risk saturates; it scales the whole sea-ice risk curve.
ICE_CLASS_CAPABILITY: dict[str, float] = {
    "none": 0.30,        # open-water hull, avoid all but the loosest ice
    "1c": 0.45,
    "1b": 0.55,
    "1a": 0.70,
    "1a_super": 0.82,    # PC6/PC7 equivalent, typical polar research vessel
    "pc5": 0.88,
    "icebreaker": 0.95,
}
DEFAULT_ICE_CLASS = "1a_super"


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------
@dataclass
class VesselProfile:
    """Vessel parameters used by the risk engine and the route optimiser.

    These are *model parameters*, not a description of any particular ship.
    Defaults approximate a polar research/resupply vessel of the class operated
    on Indian Antarctic expeditions.
    """

    name: str = "generic polar research vessel"
    ice_class: str = DEFAULT_ICE_CLASS
    speed_knots: float = 12.0
    #: Fuel burn at service speed, tonnes per day.
    fuel_consumption_tpd: float = 25.0
    draft_m: float = 7.0
    #: 0 = risk averse (prefers long, safe routes), 1 = risk tolerant.
    risk_tolerance: float = 0.3

    def __post_init__(self) -> None:
        self.ice_class = str(self.ice_class).strip().lower().replace(" ", "_").replace("-", "_")
        if self.ice_class not in ICE_CLASS_CAPABILITY:
            log.warning("Unknown ice class %r; falling back to %r", self.ice_class, DEFAULT_ICE_CLASS)
            self.ice_class = DEFAULT_ICE_CLASS
        self.speed_knots = float(np.clip(self.speed_knots, 1.0, 30.0))
        self.fuel_consumption_tpd = float(np.clip(self.fuel_consumption_tpd, 0.1, 500.0))
        self.draft_m = float(np.clip(self.draft_m, 0.5, 25.0))
        self.risk_tolerance = float(np.clip(self.risk_tolerance, 0.0, 1.0))

    @property
    def ice_capability(self) -> float:
        return ICE_CLASS_CAPABILITY[self.ice_class]

    @property
    def speed_km_h(self) -> float:
        return self.speed_knots * 1.852

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ice_class": self.ice_class,
            "ice_capability": self.ice_capability,
            "speed_knots": self.speed_knots,
            "fuel_consumption_tpd": self.fuel_consumption_tpd,
            "draft_m": self.draft_m,
            "risk_tolerance": self.risk_tolerance,
        }


@dataclass
class RiskWeights:
    """Relative weight of each risk component (need not sum to 1)."""

    sea_ice: float = 0.40
    iceberg: float = 0.25
    weather: float = 0.20
    ocean: float = 0.10
    constraint: float = 0.05

    @classmethod
    def from_settings(cls, settings: Settings) -> "RiskWeights":
        w = settings.risk_weights
        return cls(
            sea_ice=w["sea_ice"], iceberg=w["iceberg"], weather=w["weather"],
            ocean=w["ocean"], constraint=w["constraint"],
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "sea_ice": self.sea_ice, "iceberg": self.iceberg, "weather": self.weather,
            "ocean": self.ocean, "constraint": self.constraint,
        }


@dataclass
class RiskGridResult:
    """A generated risk grid with its components and provenance."""

    grid: GridSpec
    generated_at: datetime
    valid_at: datetime
    horizon_hours: int
    components: dict[str, np.ndarray]
    total: np.ndarray
    navigable: np.ndarray
    weights: RiskWeights
    vessel: VesselProfile
    available_layers: list[str]
    missing_layers: list[str]
    data_mode: str
    n_icebergs: int = 0

    @property
    def mean_risk(self) -> float:
        vals = self.total[self.navigable]
        return float(np.nanmean(vals)) if vals.size else float("nan")

    def summary(self) -> dict:
        ocean_cells = int(np.sum(landmask.navigable_mask(self.grid)))
        return {
            "generated_at": self.generated_at,
            "valid_at": self.valid_at,
            "horizon_hours": self.horizon_hours,
            "grid": self.grid.to_dict(),
            "cells_total": int(self.total.size),
            "cells_ocean": ocean_cells,
            "cells_navigable": int(np.sum(self.navigable)),
            "cells_blocked": int(ocean_cells - np.sum(self.navigable)),
            "mean_risk": self.mean_risk,
            "max_risk": float(np.nanmax(self.total)) if np.isfinite(self.total).any() else None,
            "weights": self.weights.as_dict(),
            "vessel": self.vessel.to_dict(),
            "available_layers": self.available_layers,
            "missing_layers": self.missing_layers,
            "n_icebergs_considered": self.n_icebergs,
            "data_mode": self.data_mode,
            "grid_version": RISK_GRID_VERSION,
        }


# ---------------------------------------------------------------------------
# Helper curves
# ---------------------------------------------------------------------------
def smoothstep(x, lo: float, hi: float):
    """Hermite ramp from 0 at ``lo`` to 1 at ``hi`` (array-safe)."""
    if hi <= lo:
        return np.where(np.asarray(x, float) >= hi, 1.0, 0.0)
    t = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _finite(arr: np.ndarray | None) -> bool:
    return arr is not None and np.isfinite(np.asarray(arr, float)).any()


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------
def sea_ice_risk(
    concentration: np.ndarray,
    vessel: VesselProfile,
    forecast_concentration: np.ndarray | None = None,
    ice_edge_distance_km: np.ndarray | None = None,
) -> np.ndarray:
    """Risk from sea ice.

    Three contributions:

    * **Concentration** - negligible below the 15% detection threshold, rising
      to 1 as concentration approaches the vessel's ice capability and beyond.
    * **Trend** - a forecast increase in concentration adds risk, because the
      vessel may be beset by the time it arrives.
    * **Marginal ice zone** - within 40 km of the ice edge conditions change
      quickly and the analysis is least reliable, so a small premium applies.
    """
    conc = np.nan_to_num(np.asarray(concentration, dtype=float), nan=0.0)
    cap = vessel.ice_capability
    base = smoothstep(conc, 0.15, min(0.99, cap + 0.15))

    trend = np.zeros_like(base)
    if _finite(forecast_concentration):
        delta = np.nan_to_num(np.asarray(forecast_concentration, float), nan=0.0) - conc
        trend = 0.35 * smoothstep(delta, 0.05, 0.35)

    marginal = np.zeros_like(base)
    if _finite(ice_edge_distance_km):
        d = np.nan_to_num(np.asarray(ice_edge_distance_km, float), nan=1e4)
        marginal = 0.15 * (1.0 - smoothstep(d, 0.0, 40.0))

    return np.clip(base + trend * (1.0 - base) + marginal * (1.0 - base), 0.0, 1.0)


def _berg_hazard_scale(area_km2: float | None) -> float:
    """Size weighting: larger bergs are both harder to avoid and more damaging.

    Ranges from 0.45 for a small berg (<10 km2) to 1.0 for a giant tabular berg
    (>1000 km2), on a logarithmic scale.
    """
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
        return 0.6
    return float(np.clip(0.45 + 0.55 * (math.log10(max(area_km2, 1.0)) / 3.0), 0.45, 1.0))


def iceberg_risk(
    grid: GridSpec,
    icebergs: Sequence[Mapping],
    settings: Settings,
    horizon_hours: float = 72.0,
) -> tuple[np.ndarray, int]:
    """Risk from tracked icebergs, using their predicted trajectories.

    For each berg the *effective distance* to a cell is the closest approach
    over the forecast window, reduced by that point's positional uncertainty.
    Bergs whose trajectory brings them towards a cell therefore raise its risk
    well before they arrive, while a berg drifting away is discounted.

    Individual berg risks combine as an independent union
    (``1 - prod(1 - r_b)``) rather than a sum, so many distant bergs cannot add
    up to a false alarm.

    Returns ``(risk, n_contributing)`` - the count is of bergs that actually
    raise the risk somewhere on the grid, not of every berg examined.
    """
    risk = np.zeros(grid.shape, dtype=float)
    if not icebergs:
        return risk, 0

    lat2d, lon2d = grid.meshgrid()
    critical = float(settings.iceberg_critical_distance_km)
    influence = float(settings.iceberg_influence_distance_km)
    counted = 0

    for berg in icebergs:
        track = list(berg.get("track") or [])
        if not track:
            lat, lon = berg.get("latitude"), berg.get("longitude")
            if lat is None or lon is None:
                continue
            track = [{"latitude": lat, "longitude": lon, "horizon_hours": 0.0, "uncertainty_radius_km": 0.0}]
        scale = _berg_hazard_scale(berg.get("area_km2"))

        effective = np.full(grid.shape, np.inf)
        for point in track:
            if float(point.get("horizon_hours", 0.0)) > horizon_hours:
                continue
            d = haversine_km(lat2d, lon2d, float(point["latitude"]), float(point["longitude"]))
            d = d - float(point.get("uncertainty_radius_km", 0.0) or 0.0)
            effective = np.minimum(effective, np.maximum(d, 0.0))
        if not np.isfinite(effective).any():
            continue

        berg_risk = np.clip(scale * (1.0 - smoothstep(effective, critical, influence)), 0.0, 1.0)
        if berg_risk.max() <= 0.0:
            # Evaluated, but too far from every cell to matter. Counting it
            # would overstate how much of the grid the iceberg layer explains.
            continue
        counted += 1
        risk = 1.0 - (1.0 - risk) * (1.0 - berg_risk)

    return np.clip(risk, 0.0, 1.0), counted


def weather_risk(
    wind_speed_m_s: np.ndarray | None,
    air_temperature_c: np.ndarray | None = None,
    mean_sea_level_pressure_hpa: np.ndarray | None = None,
) -> np.ndarray | None:
    """Risk from the atmosphere, or ``None`` when no wind field is available.

    * **Wind** - negligible below 10 m/s (Beaufort 5), saturating at 28 m/s
      (Beaufort 10, storm force).
    * **Superstructure icing** - freezing spray becomes a hazard when the air is
      below -2 C and the wind exceeds 10 m/s; the classic icing-severity
      predictors combine exactly these two.
    * **Deep low** - MSL pressure below 970 hPa flags an intense system whose
      wind field is likely to worsen faster than the analysis suggests.
    """
    if not _finite(wind_speed_m_s):
        return None
    wind = np.nan_to_num(np.asarray(wind_speed_m_s, float), nan=0.0)
    risk = smoothstep(wind, 10.0, 28.0)

    if _finite(air_temperature_c):
        t = np.nan_to_num(np.asarray(air_temperature_c, float), nan=0.0)
        icing = (1.0 - smoothstep(t, -8.0, -2.0)) * smoothstep(wind, 10.0, 20.0)
        risk = risk + 0.30 * icing * (1.0 - risk)

    if _finite(mean_sea_level_pressure_hpa):
        p = np.nan_to_num(np.asarray(mean_sea_level_pressure_hpa, float), nan=1013.0)
        deep_low = 1.0 - smoothstep(p, 955.0, 985.0)
        risk = risk + 0.15 * deep_low * (1.0 - risk)

    return np.clip(risk, 0.0, 1.0)


def ocean_risk(
    wave_height_m: np.ndarray | None,
    current_speed_m_s: np.ndarray | None = None,
    sea_surface_temperature_c: np.ndarray | None = None,
) -> np.ndarray | None:
    """Risk from the ocean state, or ``None`` when nothing is available.

    * **Waves** - negligible below 2.5 m, saturating at 9 m significant height.
    * **Currents** - a set above 0.6 m/s meaningfully affects track keeping in
      ice; saturates at 1.5 m/s.
    * **Near-freezing water** - SST below about -1.5 C indicates active new-ice
      formation, which can close a lead behind a vessel.
    """
    layers = []
    if _finite(wave_height_m):
        layers.append(("waves", smoothstep(np.nan_to_num(np.asarray(wave_height_m, float), nan=0.0), 2.5, 9.0), 1.0))
    if _finite(current_speed_m_s):
        layers.append(("current", smoothstep(np.nan_to_num(np.asarray(current_speed_m_s, float), nan=0.0), 0.6, 1.5), 0.4))
    if _finite(sea_surface_temperature_c):
        sst = np.nan_to_num(np.asarray(sea_surface_temperature_c, float), nan=0.0)
        layers.append(("freezing", 1.0 - smoothstep(sst, -1.8, -1.0), 0.3))
    if not layers:
        return None

    risk = np.zeros_like(layers[0][1], dtype=float)
    for _name, values, weight in layers:
        risk = risk + weight * values * (1.0 - risk)
    return np.clip(risk, 0.0, 1.0)


def constraint_risk(grid: GridSpec, vessel: VesselProfile) -> np.ndarray:
    """Navigational constraints from geography.

    POLARIS carries no bathymetry, so proximity to the coast is the proxy for
    shoal water, uncharted ground and ice-shelf calving zones: risk rises as a
    cell approaches land and saturates at the coast, scaled by the vessel's
    draft.  **This is a proxy, not a depth check** - it cannot replace a chart.

    The ramp is scale-aware.  Distance is measured between cell centres, so on a
    coarse grid the nearest possible water cell is already one grid step from
    land; a fixed 60 km ramp would then evaluate to zero everywhere and the
    component would silently vanish.  The upper bound is therefore the larger of
    60 km and 1.5 grid diagonals.
    """
    coast_km = landmask.distance_to_land_km(grid)
    cell_km = math.hypot(grid.dlat * 111.32, grid.dlon * 111.32 * math.cos(math.radians(
        0.5 * (grid.lat_min + grid.lat_max)
    )))
    hi = max(60.0, 1.5 * cell_km)
    proximity = 1.0 - smoothstep(coast_km, 0.08 * hi, hi)
    draft_factor = float(np.clip(vessel.draft_m / 8.0, 0.5, 1.5))
    return np.clip(proximity * draft_factor, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------
def compute_risk_grid(
    grid: GridSpec,
    fields: Mapping[str, np.ndarray],
    icebergs: Sequence[Mapping] | None = None,
    forecast_concentration: np.ndarray | None = None,
    vessel: VesselProfile | None = None,
    weights: RiskWeights | None = None,
    settings: Settings | None = None,
    horizon_hours: int = 0,
    valid_at: datetime | None = None,
    generated_at: datetime | None = None,
) -> RiskGridResult:
    """Assemble the full spatial navigation risk grid."""
    settings = settings or get_settings()
    vessel = vessel or VesselProfile()
    weights = weights or RiskWeights.from_settings(settings)
    generated_at = generated_at or datetime.now(timezone.utc)
    valid_at = valid_at or (generated_at + timedelta(hours=horizon_hours))

    components: dict[str, np.ndarray] = {}
    available: list[str] = []
    missing: list[str] = []

    conc = fields.get("sea_ice_concentration")
    if _finite(conc):
        components["sea_ice"] = sea_ice_risk(
            conc, vessel, forecast_concentration, fields.get("ice_edge_distance_km")
        )
        available.append("sea_ice")
    else:
        components["sea_ice"] = np.zeros(grid.shape)
        missing.append("sea_ice")

    berg_risk, n_bergs = iceberg_risk(grid, icebergs or [], settings, horizon_hours or 72)
    components["iceberg"] = berg_risk
    (available if icebergs else missing).append("iceberg")

    wx = weather_risk(fields.get("wind_speed"), fields.get("t2m"), fields.get("msl"))
    components["weather"] = wx if wx is not None else np.zeros(grid.shape)
    (available if wx is not None else missing).append("weather")

    ocn = ocean_risk(fields.get("wave_height"), fields.get("current_speed"), fields.get("sst"))
    components["ocean"] = ocn if ocn is not None else np.zeros(grid.shape)
    (available if ocn is not None else missing).append("ocean")

    components["constraint"] = constraint_risk(grid, vessel)
    available.append("constraint")

    weight_map = weights.as_dict()
    active = {k: weight_map[k] for k in available if k in weight_map}
    denom = sum(active.values()) or 1.0
    total = np.zeros(grid.shape)
    for key, w in active.items():
        total = total + w * components[key]
    total = np.clip(total / denom, 0.0, 1.0)

    # Navigability: land, an impassable concentration for this hull, or a total
    # risk beyond the configured ceiling all block a cell.
    ocean_mask = landmask.navigable_mask(grid)
    impassable_conc = min(0.99, vessel.ice_capability + 0.12)
    conc_arr = np.nan_to_num(np.asarray(conc, float), nan=0.0) if conc is not None else np.zeros(grid.shape)
    navigable = ocean_mask & (conc_arr < impassable_conc) & (total <= settings.route_max_acceptable_risk)

    result = RiskGridResult(
        grid=grid,
        generated_at=generated_at,
        valid_at=valid_at,
        horizon_hours=int(horizon_hours),
        components=components,
        total=total,
        navigable=navigable,
        weights=weights,
        vessel=vessel,
        available_layers=available,
        missing_layers=missing,
        data_mode=settings.data_mode,
        n_icebergs=n_bergs,
    )
    log.info(
        "Risk grid: mean %.3f over %d navigable cells (layers: %s; missing: %s)",
        result.mean_risk, int(navigable.sum()), ",".join(available) or "-", ",".join(missing) or "-",
    )
    return result


def risk_grid_rows(result: RiskGridResult, ocean_only: bool = True) -> list[dict]:
    """Flatten a risk grid into ``risk_grid`` table rows."""
    lat2d, lon2d = result.grid.meshgrid()
    ocean_mask = landmask.navigable_mask(result.grid)
    rows: list[dict] = []
    for i in range(result.grid.shape[0]):
        for j in range(result.grid.shape[1]):
            if ocean_only and not ocean_mask[i, j]:
                continue
            rows.append(
                {
                    "generated_at": result.generated_at,
                    "valid_at": result.valid_at,
                    "horizon_hours": result.horizon_hours,
                    "latitude": float(lat2d[i, j]),
                    "longitude": float(lon2d[i, j]),
                    "sea_ice_risk": float(result.components["sea_ice"][i, j]),
                    "iceberg_risk": float(result.components["iceberg"][i, j]),
                    "weather_risk": float(result.components["weather"][i, j]),
                    "ocean_risk": float(result.components["ocean"][i, j]),
                    "constraint_risk": float(result.components["constraint"][i, j]),
                    "total_risk": float(result.total[i, j]),
                    "navigable": bool(result.navigable[i, j]),
                    "grid_version": RISK_GRID_VERSION,
                    "data_mode": result.data_mode,
                }
            )
    return rows


def persist_risk_grid(session, result: RiskGridResult) -> int:
    from app.database import repositories as repo

    rows = risk_grid_rows(result)
    n = repo.save_risk_grid(session, rows)
    repo.prune_risk_grids(session, keep=2)
    log.info("Persisted %d risk-grid cells", n)
    return n


# ---------------------------------------------------------------------------
# Point / route assessment
# ---------------------------------------------------------------------------
def assess_point(result: RiskGridResult, lat: float, lon: float) -> dict:
    """Risk breakdown at a single position."""
    if not result.grid.contains(lat, lon):
        raise ValueError(f"({lat}, {lon}) is outside the analysis domain {result.grid.to_dict()}")
    i, j = result.grid.index_of(lat, lon)
    return {
        "latitude": float(result.grid.lats[i]),
        "longitude": float(result.grid.lons[j]),
        "sea_ice_risk": float(result.components["sea_ice"][i, j]),
        "iceberg_risk": float(result.components["iceberg"][i, j]),
        "weather_risk": float(result.components["weather"][i, j]),
        "ocean_risk": float(result.components["ocean"][i, j]),
        "constraint_risk": float(result.components["constraint"][i, j]),
        "total_risk": float(result.total[i, j]),
        "navigable": bool(result.navigable[i, j]),
    }


def closest_approach(
    route: Sequence[tuple[float, float]], berg_track: Sequence[Mapping]
) -> dict | None:
    """Closest approach between a route polyline and one iceberg trajectory.

    Returns the minimum distance, where on the route it occurs, and the
    trajectory time at which the berg is nearest.  This is the quantity a
    watchkeeper actually wants: *how close, and when*.
    """
    if len(route) < 2 or not berg_track:
        return None
    best: dict | None = None
    for point in berg_track:
        blat, blon = float(point["latitude"]), float(point["longitude"])
        for k in range(len(route) - 1):
            lat1, lon1 = route[k]
            lat2, lon2 = route[k + 1]
            d = point_segment_distance_km(blat, blon, lat1, lon1, lat2, lon2)
            if best is None or d < best["distance_km"]:
                best = {
                    "distance_km": float(d),
                    "segment_index": k,
                    "iceberg_latitude": blat,
                    "iceberg_longitude": blon,
                    "horizon_hours": float(point.get("horizon_hours", 0.0)),
                    "uncertainty_radius_km": float(point.get("uncertainty_radius_km", 0.0) or 0.0),
                }
    if best is not None:
        best["margin_km"] = best["distance_km"] - best["uncertainty_radius_km"]
    return best


def assess_route(
    result: RiskGridResult,
    waypoints: Sequence[tuple[float, float]],
    icebergs: Sequence[Mapping] | None = None,
) -> dict:
    """Aggregate risk statistics along a route, plus iceberg encounters."""
    per_point = []
    for lat, lon in waypoints:
        try:
            per_point.append(assess_point(result, lat, lon))
        except ValueError:
            continue
    if not per_point:
        return {"mean_risk": float("nan"), "max_risk": float("nan"), "samples": 0}

    totals = np.array([p["total_risk"] for p in per_point])
    out = {
        "samples": len(per_point),
        "mean_risk": float(totals.mean()),
        "max_risk": float(totals.max()),
        "p90_risk": float(np.percentile(totals, 90)),
        "blocked_samples": int(sum(1 for p in per_point if not p["navigable"])),
        "component_means": {
            key: float(np.mean([p[f"{key}_risk"] for p in per_point]))
            for key in ("sea_ice", "iceberg", "weather", "ocean", "constraint")
        },
    }
    if icebergs:
        encounters = []
        for berg in icebergs:
            track = berg.get("track") or [
                {"latitude": berg.get("latitude"), "longitude": berg.get("longitude"), "horizon_hours": 0.0}
            ]
            approach = closest_approach(waypoints, track)
            if approach and approach["distance_km"] < 250.0:
                encounters.append({"iceberg_id": berg.get("iceberg_id"), **approach})
        encounters.sort(key=lambda e: e["distance_km"])
        out["iceberg_encounters"] = encounters[:10]
        out["closest_iceberg_km"] = encounters[0]["distance_km"] if encounters else None
    return out
