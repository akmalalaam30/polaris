"""Navigation graph construction, route optimisation and endpoints."""

from __future__ import annotations

import numpy as np
import pytest

from app.services import risk_engine as rk
from app.services import route_optimizer as ro


@pytest.fixture()
def vessel():
    return rk.VesselProfile(ice_class="icebreaker", speed_knots=14.0, fuel_consumption_tpd=45.0)


@pytest.fixture()
def open_fields(grid):
    """Ice-free, calm conditions so every ocean cell is navigable."""
    shape = grid.shape
    return {
        "sea_ice_concentration": np.zeros(shape),
        "wind_speed": np.full(shape, 5.0),
        "t2m": np.full(shape, 0.0),
        "msl": np.full(shape, 1005.0),
        "wave_height": np.full(shape, 1.0),
        "current_speed": np.full(shape, 0.1),
        "sst": np.full(shape, 1.0),
        "ice_edge_distance_km": np.full(shape, 900.0),
    }


@pytest.fixture()
def open_risk(grid, open_fields, vessel, settings):
    return rk.compute_risk_grid(grid, open_fields, [], None, vessel, settings=settings)


@pytest.fixture()
def nav(open_risk, open_fields, vessel):
    return ro.build_navigation_graph(open_risk, open_fields, vessel)


def _two_open_points(grid):
    """Two well-separated open-water cells in the north of the domain."""
    from app.services import landmask

    navigable = landmask.navigable_mask(grid)
    row = int(np.argmax(navigable.sum(axis=1)))
    cols = np.where(navigable[row])[0]
    return (
        (float(grid.lats[row]), float(grid.lons[cols[0]])),
        (float(grid.lats[row]), float(grid.lons[cols[-1]])),
    )


# ---------------------------------------------------------------------------
# Speed and cost model
# ---------------------------------------------------------------------------
def test_speed_falls_with_ice_and_waves(vessel):
    open_water = ro.speed_factor(0.0, 0.0, vessel)
    light_ice = ro.speed_factor(0.3, 0.0, vessel)
    heavy_ice = ro.speed_factor(0.95, 0.0, vessel)
    assert open_water == pytest.approx(1.0)
    assert open_water > light_ice > heavy_ice >= 0.08

    rough = ro.speed_factor(0.0, 9.0, vessel)
    assert rough < open_water


def test_a_stronger_hull_keeps_more_speed_in_ice():
    weak = ro.speed_factor(0.5, 0.0, rk.VesselProfile(ice_class="none"))
    strong = ro.speed_factor(0.5, 0.0, rk.VesselProfile(ice_class="icebreaker"))
    assert strong > weak


def test_segment_cost_is_consistent_with_the_speed_model(vessel):
    hours, fuel = ro.segment_cost(100.0, 0.0, 0.0, 0.0, vessel)
    assert hours == pytest.approx(100.0 / vessel.speed_km_h, rel=1e-9)
    assert fuel == pytest.approx(hours * vessel.fuel_consumption_tpd / 24.0, rel=1e-9)

    slow_hours, slow_fuel = ro.segment_cost(100.0, 0.0, 0.9, 0.0, vessel)
    assert slow_hours > hours, "ice must slow the vessel"
    assert slow_fuel > fuel, "more time at sea means more fuel"


def test_profile_weights_differ_as_designed(vessel, settings):
    shortest = ro.profile_weights("shortest", vessel, settings)
    safest = ro.profile_weights("safest", vessel, settings)
    polaris = ro.profile_weights("polaris", vessel, settings)
    assert shortest.risk == 0.0 and shortest.fuel == 0.0
    assert safest.risk > safest.distance
    assert polaris.distance > 0 and polaris.risk > 0 and polaris.fuel > 0


def test_risk_tolerance_shifts_the_polaris_weights(settings):
    cautious = ro.profile_weights("polaris", rk.VesselProfile(risk_tolerance=0.0), settings)
    bold = ro.profile_weights("polaris", rk.VesselProfile(risk_tolerance=1.0), settings)
    assert cautious.risk > bold.risk
    assert bold.fuel > cautious.fuel


def test_unknown_profile_is_rejected(vessel, settings):
    from app.utils.validation import ValidationError

    with pytest.raises(ValidationError):
        ro.profile_weights("cheapest", vessel, settings)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def test_graph_contains_only_navigable_water(nav, grid, open_risk):
    from app.services import landmask

    assert nav.n_nodes > 0 and nav.n_edges > 0
    navigable = landmask.navigable_mask(grid)
    for (i, j) in nav.graph.nodes:
        assert navigable[i, j], "a graph node must not sit on land"
        assert open_risk.navigable[i, j]


def test_graph_is_eight_connected(nav):
    for (i, j), (ni, nj) in nav.graph.edges:
        assert abs(ni - i) <= 1 and abs(nj - j) <= 1
        assert (ni, nj) != (i, j)


def test_every_edge_carries_the_full_cost_vector(nav):
    for _u, _v, data in nav.graph.edges(data=True):
        assert data["distance_km"] > 0
        assert 0.0 <= data["risk"] <= 1.0
        assert data["hours"] > 0
        assert data["fuel_tonnes"] > 0


def test_diagonal_edges_are_longer_than_orthogonal_ones(nav):
    orthogonal = [d["distance_km"] for (i, j), (ni, nj), d in nav.graph.edges(data=True)
                  if (ni == i) ^ (nj == j)]
    diagonal = [d["distance_km"] for (i, j), (ni, nj), d in nav.graph.edges(data=True)
                if ni != i and nj != j]
    assert orthogonal and diagonal
    assert max(orthogonal) < max(diagonal)


def test_nearest_node_snaps_and_reports_the_distance(nav, grid):
    from app.config import STATIONS

    node, snap = nav.nearest_node(STATIONS["maitri"]["latitude"], STATIONS["maitri"]["longitude"])
    assert node in nav.graph.nodes
    assert snap >= 0.0
    # A point already on a node snaps to (almost) zero.
    lat, lon = nav.node_position(node)
    _same, zero = nav.nearest_node(lat, lon)
    assert zero == pytest.approx(0.0, abs=1e-6)


def test_high_risk_cells_are_excluded_unless_explicitly_allowed(grid, open_fields, vessel, settings):
    icy = dict(open_fields)
    icy["sea_ice_concentration"] = np.full(grid.shape, 0.99)
    result = rk.compute_risk_grid(grid, icy, [], None, vessel, settings=settings)
    strict = ro.build_navigation_graph(result, icy, vessel, allow_high_risk=False)
    relaxed = ro.build_navigation_graph(result, icy, vessel, allow_high_risk=True)
    assert strict.n_nodes < relaxed.n_nodes
    assert relaxed.degraded_nodes


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_route_connects_the_endpoints(nav, grid, settings):
    start, end = _two_open_points(grid)
    route = ro.find_route(nav, start, end, "shortest", settings)
    assert route.waypoints[0] == pytest.approx(start, abs=1.5)
    assert route.waypoints[-1] == pytest.approx(end, abs=1.5)
    assert len(route.waypoints) >= 2
    assert route.distance_km > 0
    assert route.duration_hours > 0
    assert route.fuel_tonnes > 0


def test_route_never_crosses_land(nav, grid, settings):
    from app.services import landmask
    from app.utils.geo import densify_great_circle

    start, end = _two_open_points(grid)
    navigable = landmask.navigable_mask(grid)
    for profile in ro.PROFILES:
        route = ro.find_route(nav, start, end, profile, settings)
        for k in range(len(route.waypoints) - 1):
            lat1, lon1 = route.waypoints[k]
            lat2, lon2 = route.waypoints[k + 1]
            for lat, lon in densify_great_circle(lat1, lon1, lat2, lon2, step_km=30.0):
                i, j = grid.index_of(lat, lon)
                assert navigable[i, j], f"{profile} route crosses land near ({lat:.2f}, {lon:.2f})"


def test_shortest_route_is_the_shortest(nav, grid, settings):
    start, end = _two_open_points(grid)
    routes = {p: ro.find_route(nav, start, end, p, settings) for p in ro.PROFILES}
    shortest = routes["shortest"].distance_km
    for profile, route in routes.items():
        assert route.distance_km >= shortest - 1e-6, (
            f"{profile} ({route.distance_km:.1f} km) is shorter than 'shortest' ({shortest:.1f} km)"
        )


def test_safest_route_is_not_riskier_than_the_shortest(grid, settings, vessel):
    """With a genuine risk gradient the safest route must reduce exposure."""
    from app.services import landmask

    shape = grid.shape
    lat2d, _lon2d = grid.meshgrid()
    fields = {
        "sea_ice_concentration": np.zeros(shape),
        # A band of severe weather across the middle of the domain.
        "wind_speed": np.where(np.abs(lat2d + 65.0) < 2.0, 30.0, 4.0),
        "wave_height": np.where(np.abs(lat2d + 65.0) < 2.0, 9.0, 0.5),
        "t2m": np.full(shape, 0.0),
        "msl": np.full(shape, 1005.0),
        "current_speed": np.full(shape, 0.1),
        "sst": np.full(shape, 1.0),
        "ice_edge_distance_km": np.full(shape, 900.0),
    }
    result = rk.compute_risk_grid(grid, fields, [], None, vessel, settings=settings)
    nav = ro.build_navigation_graph(result, fields, vessel)
    navigable = landmask.navigable_mask(grid)

    # Start and end on opposite sides of the hazardous band.
    south_rows = np.where((grid.lats < -66.5))[0]
    north_rows = np.where((grid.lats > -63.5))[0]
    start = end = None
    for i in south_rows:
        cols = np.where(navigable[i])[0]
        if cols.size:
            start = (float(grid.lats[i]), float(grid.lons[cols[len(cols) // 2]]))
            break
    for i in north_rows:
        cols = np.where(navigable[i])[0]
        if cols.size:
            end = (float(grid.lats[i]), float(grid.lons[cols[len(cols) // 2]]))
            break
    assert start and end

    shortest = ro.find_route(nav, start, end, "shortest", settings)
    safest = ro.find_route(nav, start, end, "safest", settings)
    assert safest.mean_risk <= shortest.mean_risk + 1e-9
    assert safest.distance_km >= shortest.distance_km - 1e-6


def test_astar_and_dijkstra_agree_on_cost(nav, grid, settings):
    """A* with an admissible heuristic must find the same optimum as Dijkstra."""
    start, end = _two_open_points(grid)
    a = ro.find_route(nav, start, end, "polaris", settings, algorithm="astar", simplify=False)
    d = ro.find_route(nav, start, end, "polaris", settings, algorithm="dijkstra", simplify=False)
    weights = ro.profile_weights("polaris", nav.vessel, settings)
    fn = ro._edge_weight_fn(weights)

    def total(route):
        return sum(
            fn(None, None, {
                "distance_km": s.distance_km, "risk": s.risk, "fuel_tonnes": s.fuel_tonnes
            })
            for s in route.segments
        )

    assert total(a) == pytest.approx(total(d), rel=1e-9)
    assert a.algorithm == "astar" and d.algorithm == "dijkstra"


def test_route_metrics_are_internally_consistent(nav, grid, settings):
    route = ro.find_route(nav, *_two_open_points(grid), profile="polaris", settings=settings)
    assert route.distance_km == pytest.approx(sum(s.distance_km for s in route.segments))
    assert route.duration_hours == pytest.approx(sum(s.hours for s in route.segments))
    assert route.fuel_tonnes == pytest.approx(sum(s.fuel_tonnes for s in route.segments))
    assert route.max_risk >= route.mean_risk
    # Average speed must be at or below the service speed.
    assert route.distance_km / route.duration_hours <= nav.vessel.speed_km_h + 1e-6


def test_geojson_geometry_is_valid(nav, grid, settings):
    route = ro.find_route(nav, *_two_open_points(grid), profile="shortest", settings=settings)
    geo = route.geojson()
    assert geo["type"] == "Feature"
    assert geo["geometry"]["type"] == "LineString"
    coords = geo["geometry"]["coordinates"]
    assert len(coords) == len(route.waypoints)
    for lon, lat in coords:
        assert -180 <= lon <= 180 and -90 <= lat <= 90


def test_identical_endpoints_are_rejected(nav, grid, settings):
    from app.utils.validation import ValidationError

    point = _two_open_points(grid)[0]
    with pytest.raises(ValidationError):
        ro.find_route(nav, point, point, "shortest", settings)


def test_unreachable_destination_raises(grid, open_fields, vessel, settings):
    """A graph split into two basins must report that no path exists."""
    import networkx as nx

    result = rk.compute_risk_grid(grid, open_fields, [], None, vessel, settings=settings)
    nav = ro.build_navigation_graph(result, open_fields, vessel)
    components = list(nx.connected_components(nav.graph))
    if len(components) < 2:
        # Artificially isolate one node to exercise the failure path.
        node = next(iter(nav.graph.nodes))
        nav.graph.remove_edges_from(list(nav.graph.edges(node)))
        other = next(n for n in nav.graph.nodes if n != node and nav.graph.degree(n) > 0)
        with pytest.raises(ro.RouteNotFoundError):
            ro.find_route(nav, nav.node_position(node), nav.node_position(other), "shortest", settings)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def test_location_resolution_accepts_stations_and_coordinates():
    lat, lon, name = ro.resolve_location("maitri", "start")
    assert name == "Maitri Station"
    assert lat < -70

    lat2, lon2, _ = ro.resolve_location({"latitude": -65.0, "longitude": 40.0}, "start")
    assert (lat2, lon2) == (-65.0, 40.0)


def test_location_resolution_rejects_unknown_stations():
    from app.utils.validation import ValidationError

    with pytest.raises(ValidationError):
        ro.resolve_location({"station": "hogwarts"}, "start")


def test_optimize_routes_returns_all_three_profiles(open_risk, open_fields, vessel, settings, grid):
    start, end = _two_open_points(grid)
    result = ro.optimize_routes(
        open_risk,
        {"latitude": start[0], "longitude": start[1]},
        {"latitude": end[0], "longitude": end[1]},
        fields=open_fields, vessel=vessel, settings=settings,
    )
    assert set(result["routes"]) == set(ro.PROFILES)
    assert result["recommended_profile"] == "polaris"
    assert result["errors"] == {}
    assert result["request_id"]


def test_comparison_is_computed_not_asserted(open_risk, open_fields, vessel, settings, grid):
    start, end = _two_open_points(grid)
    result = ro.optimize_routes(
        open_risk,
        {"latitude": start[0], "longitude": start[1]},
        {"latitude": end[0], "longitude": end[1]},
        fields=open_fields, vessel=vessel, settings=settings,
    )
    routes = result["routes"]
    comparison = result["comparison"]
    reference = routes["shortest"]["distance_km"]
    for profile in ("safest", "polaris"):
        expected = 100.0 * (routes[profile]["distance_km"] - reference) / reference
        assert comparison[profile]["distance_vs_shortest_pct"] == pytest.approx(expected, abs=0.01)


def test_endpoints_outside_the_domain_are_rejected(open_risk, open_fields, vessel, settings):
    from app.utils.validation import ValidationError

    with pytest.raises(ValidationError):
        ro.optimize_routes(
            open_risk, {"latitude": 45.0, "longitude": 10.0}, "maitri",
            fields=open_fields, vessel=vessel, settings=settings,
        )


def test_route_rows_are_database_ready(open_risk, open_fields, vessel, settings, grid):
    start, end = _two_open_points(grid)
    result = ro.optimize_routes(
        open_risk,
        {"latitude": start[0], "longitude": start[1]},
        {"latitude": end[0], "longitude": end[1]},
        fields=open_fields, vessel=vessel, settings=settings,
    )
    rows = ro.route_rows(result, "demo")
    assert len(rows) == len(result["routes"])
    required = {
        "request_id", "profile", "start_latitude", "end_longitude", "distance_km",
        "duration_hours", "fuel_tonnes", "mean_risk", "max_risk", "geometry", "parameters",
    }
    assert required <= set(rows[0])
    assert rows[0]["geometry"]["geometry"]["type"] == "LineString"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def _body(grid, **overrides):
    from app.services import landmask

    navigable = landmask.navigable_mask(grid)
    row = int(np.argmax(navigable.sum(axis=1)))
    cols = np.where(navigable[row])[0]
    payload = {
        "start": {"latitude": float(grid.lats[row]), "longitude": float(grid.lons[cols[0]])},
        "destination": {"latitude": float(grid.lats[row]), "longitude": float(grid.lons[cols[-1]])},
        "vessel": {"ice_class": "icebreaker", "speed_knots": 14, "fuel_consumption_tpd": 45},
        "preferences": {"forecast_hours": 0, "profiles": ["shortest", "safest", "polaris"]},
        "persist": False,
    }
    payload.update(overrides)
    return payload


def test_optimize_endpoint(client, grid):
    r = client.post("/api/route/optimize", json=_body(grid))
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["routes"]) == {"shortest", "safest", "polaris"}
    for route in body["routes"].values():
        assert route["distance_km"] > 0
        assert route["n_waypoints"] >= 2
        assert route["geometry"]["geometry"]["type"] == "LineString"
        assert 0.0 <= route["mean_risk"] <= 1.0
        assert route["status"] in {"ok", "degraded"}
    assert body["provenance"]["data_mode"] == "demo"


def test_optimize_endpoint_accepts_station_names(client):
    r = client.post(
        "/api/route/optimize",
        json={
            "start": {"station": "bharati"},
            "destination": {"station": "maitri"},
            "vessel": {"ice_class": "icebreaker"},
            "preferences": {"forecast_hours": 0},
            "persist": False,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["start"]["name"] == "Bharati Station"
    assert body["destination"]["name"] == "Maitri Station"
    # Both stations are inland; the snap distance must be reported, not hidden.
    assert body["routes"]["polaris"]["end_snap_km"] > 0
    assert any("snapped" in note for note in body["routes"]["polaris"]["notes"])


def test_optimize_endpoint_persists_when_asked(client, grid):
    r = client.post("/api/route/optimize", json=_body(grid, persist=True))
    assert r.status_code == 200
    body = r.json()
    assert body["persisted_route_ids"]

    stored = client.get(f"/api/navigation/routes/{body['request_id']}")
    assert stored.status_code == 200
    assert len(stored.json()["routes"]) == len(body["routes"])


def test_vessel_parameters_change_the_answer(client, grid):
    slow = client.post("/api/route/optimize", json=_body(
        grid, vessel={"ice_class": "icebreaker", "speed_knots": 8, "fuel_consumption_tpd": 20}
    )).json()
    fast = client.post("/api/route/optimize", json=_body(
        grid, vessel={"ice_class": "icebreaker", "speed_knots": 16, "fuel_consumption_tpd": 20}
    )).json()
    assert slow["routes"]["shortest"]["duration_hours"] > fast["routes"]["shortest"]["duration_hours"]
    assert slow["routes"]["shortest"]["estimated_fuel_tonnes"] > fast["routes"]["shortest"]["estimated_fuel_tonnes"]


def test_profile_subset_is_honoured(client, grid):
    body = _body(grid)
    body["preferences"]["profiles"] = ["safest"]
    r = client.post("/api/route/optimize", json=body)
    assert r.status_code == 200
    assert set(r.json()["routes"]) == {"safest"}


def test_risk_weight_overrides_are_applied(client, grid):
    body = _body(grid)
    body["preferences"]["risk_weights"] = {"sea_ice": 1.0, "iceberg": 0.0, "weather": 0.0,
                                           "ocean": 0.0, "constraint": 0.0}
    r = client.post("/api/route/optimize", json=body)
    assert r.status_code == 200
    assert r.json()["risk_grid"]["weights"]["sea_ice"] == 1.0


def test_all_zero_risk_weights_are_rejected(client, grid):
    body = _body(grid)
    body["preferences"]["risk_weights"] = {"sea_ice": 0.0, "iceberg": 0.0, "weather": 0.0,
                                           "ocean": 0.0, "constraint": 0.0}
    assert client.post("/api/route/optimize", json=body).status_code == 422


def test_missing_endpoint_is_rejected(client):
    r = client.post("/api/route/optimize", json={"start": {}, "destination": {"station": "maitri"}})
    assert r.status_code == 422
    assert "station" in str(r.json()["detail"]).lower()


def test_out_of_domain_endpoint_is_rejected(client, grid):
    body = _body(grid)
    body["destination"] = {"latitude": 20.0, "longitude": 40.0}
    r = client.post("/api/route/optimize", json=body)
    assert r.status_code == 422
    assert "domain" in str(r.json()["detail"]).lower()


def test_invalid_vessel_parameters_are_rejected(client, grid):
    body = _body(grid)
    body["vessel"] = {"ice_class": "icebreaker", "speed_knots": -3}
    assert client.post("/api/route/optimize", json=body).status_code == 422

    body["vessel"] = {"ice_class": "starship"}
    assert client.post("/api/route/optimize", json=body).status_code == 422


def test_stations_endpoint(client, grid):
    r = client.get("/api/navigation/stations")
    assert r.status_code == 200
    stations = {s["key"]: s for s in r.json()["stations"]}
    assert {"bharati", "maitri"} <= set(stations)
    assert stations["bharati"]["in_domain"] is True
    assert stations["bharati"]["nearest_navigable_km"] is not None


def test_recent_routes_endpoint(client, grid):
    client.post("/api/route/optimize", json=_body(grid, persist=True))
    r = client.get("/api/navigation/routes?limit=5")
    assert r.status_code == 200
    assert r.json()["count"] > 0


def test_unknown_request_id_returns_404(client):
    assert client.get("/api/navigation/routes/deadbeef").status_code == 404
