"""Risk engine components, grid assembly and endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.services import risk_engine as rk


@pytest.fixture()
def vessel():
    return rk.VesselProfile()


@pytest.fixture()
def calm_fields(grid):
    """A benign environment: no ice, light wind, calm sea."""
    shape = grid.shape
    return {
        "sea_ice_concentration": np.zeros(shape),
        "wind_speed": np.full(shape, 5.0),
        "t2m": np.full(shape, 2.0),
        "msl": np.full(shape, 1005.0),
        "wave_height": np.full(shape, 1.0),
        "current_speed": np.full(shape, 0.1),
        "sst": np.full(shape, 2.0),
        "ice_edge_distance_km": np.full(shape, 500.0),
    }


# ---------------------------------------------------------------------------
# Vessel profile
# ---------------------------------------------------------------------------
def test_ice_class_capability_ordering():
    classes = ["none", "1c", "1b", "1a", "1a_super", "pc5", "icebreaker"]
    capabilities = [rk.VesselProfile(ice_class=c).ice_capability for c in classes]
    assert capabilities == sorted(capabilities), "a stronger hull must never rate lower"


def test_unknown_ice_class_falls_back_safely():
    assert rk.VesselProfile(ice_class="unobtainium").ice_class == rk.DEFAULT_ICE_CLASS


def test_vessel_parameters_are_clamped():
    v = rk.VesselProfile(speed_knots=999, fuel_consumption_tpd=-5, risk_tolerance=2.0, draft_m=0.01)
    assert 1.0 <= v.speed_knots <= 30.0
    assert v.fuel_consumption_tpd > 0
    assert v.risk_tolerance == 1.0
    assert v.draft_m >= 0.5


def test_speed_conversion():
    assert rk.VesselProfile(speed_knots=10).speed_km_h == pytest.approx(18.52)


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------
def test_sea_ice_risk_is_monotonic_in_concentration(vessel):
    conc = np.array([[0.0, 0.1, 0.3, 0.6, 0.9, 1.0]])
    risk = rk.sea_ice_risk(conc, vessel)[0]
    assert list(risk) == sorted(risk)
    assert risk[0] == pytest.approx(0.0), "open water carries no sea-ice risk"
    assert risk[-1] == pytest.approx(1.0, abs=1e-6), "full cover saturates the component"


def test_open_water_below_the_detection_threshold_is_risk_free(vessel):
    assert rk.sea_ice_risk(np.array([[0.14]]), vessel)[0, 0] == pytest.approx(0.0)


def test_stronger_hulls_see_less_risk_in_the_same_ice():
    conc = np.array([[0.7]])
    weak = rk.sea_ice_risk(conc, rk.VesselProfile(ice_class="none"))[0, 0]
    strong = rk.sea_ice_risk(conc, rk.VesselProfile(ice_class="icebreaker"))[0, 0]
    assert weak > strong


def test_a_forecast_increase_in_ice_adds_risk(vessel):
    conc = np.array([[0.4]])
    steady = rk.sea_ice_risk(conc, vessel, forecast_concentration=conc)[0, 0]
    worsening = rk.sea_ice_risk(conc, vessel, forecast_concentration=np.array([[0.8]]))[0, 0]
    assert worsening > steady


def test_marginal_ice_zone_premium(vessel):
    conc = np.array([[0.4]])
    far = rk.sea_ice_risk(conc, vessel, ice_edge_distance_km=np.array([[500.0]]))[0, 0]
    near = rk.sea_ice_risk(conc, vessel, ice_edge_distance_km=np.array([[2.0]]))[0, 0]
    assert near > far


def test_weather_risk_rises_with_wind():
    wind = np.array([[2.0, 8.0, 15.0, 22.0, 30.0]])
    risk = rk.weather_risk(wind)[0]
    assert list(risk) == sorted(risk)
    assert risk[0] == pytest.approx(0.0)
    assert risk[-1] == pytest.approx(1.0, abs=1e-6)


def test_freezing_spray_adds_icing_risk():
    wind = np.array([[15.0]])
    mild = rk.weather_risk(wind, air_temperature_c=np.array([[5.0]]))[0, 0]
    freezing = rk.weather_risk(wind, air_temperature_c=np.array([[-15.0]]))[0, 0]
    assert freezing > mild


def test_deep_low_adds_risk():
    wind = np.array([[12.0]])
    normal = rk.weather_risk(wind, mean_sea_level_pressure_hpa=np.array([[1010.0]]))[0, 0]
    storm = rk.weather_risk(wind, mean_sea_level_pressure_hpa=np.array([[950.0]]))[0, 0]
    assert storm > normal


def test_weather_risk_is_none_without_wind():
    assert rk.weather_risk(None) is None
    assert rk.weather_risk(np.array([[np.nan]])) is None


def test_ocean_risk_rises_with_wave_height():
    waves = np.array([[0.5, 3.0, 6.0, 10.0]])
    risk = rk.ocean_risk(waves)[0]
    assert list(risk) == sorted(risk)
    assert risk[0] == pytest.approx(0.0)


def test_ocean_risk_is_none_with_no_layers():
    assert rk.ocean_risk(None, None, None) is None


def test_constraint_risk_is_highest_near_the_coast(grid, vessel):
    from app.services import landmask

    risk = rk.constraint_risk(grid, vessel)
    distance = landmask.distance_to_land_km(grid)
    navigable = landmask.navigable_mask(grid)
    # Compare by quantile rather than absolute distance, so the assertion holds
    # at any grid resolution.
    offshore = distance[navigable]
    near = navigable & (distance <= np.percentile(offshore, 10))
    far = navigable & (distance >= np.percentile(offshore, 90))
    assert near.any() and far.any()
    assert risk[near].mean() > risk[far].mean()
    assert risk.min() >= 0.0 and risk.max() <= 1.0


def test_deeper_draft_raises_constraint_risk(grid):
    shallow = rk.constraint_risk(grid, rk.VesselProfile(draft_m=3.0))
    deep = rk.constraint_risk(grid, rk.VesselProfile(draft_m=12.0))
    assert deep.mean() > shallow.mean()


# ---------------------------------------------------------------------------
# Iceberg risk
# ---------------------------------------------------------------------------
def test_iceberg_risk_decays_with_distance(grid, settings):
    centre_lat = float(grid.lats[len(grid.lats) // 2])
    centre_lon = float(grid.lons[len(grid.lons) // 2])
    bergs = [{"iceberg_id": "X", "latitude": centre_lat, "longitude": centre_lon, "area_km2": 300.0}]
    risk, counted = rk.iceberg_risk(grid, bergs, settings)
    assert counted == 1
    i, j = grid.index_of(centre_lat, centre_lon)
    assert risk[i, j] > 0.5
    # A cell on the far side of the domain must be unaffected.
    assert risk[0, 0] == pytest.approx(0.0, abs=1e-6) or risk[0, 0] < risk[i, j]


def test_no_icebergs_means_no_iceberg_risk(grid, settings):
    risk, counted = rk.iceberg_risk(grid, [], settings)
    assert counted == 0
    assert np.all(risk == 0.0)


def test_larger_icebergs_are_weighted_more_heavily(grid, settings):
    lat, lon = float(grid.lats[5]), float(grid.lons[5])
    small = rk.iceberg_risk(grid, [{"latitude": lat, "longitude": lon, "area_km2": 5.0}], settings)[0]
    large = rk.iceberg_risk(grid, [{"latitude": lat, "longitude": lon, "area_km2": 4000.0}], settings)[0]
    i, j = grid.index_of(lat, lon)
    assert large[i, j] > small[i, j]


def test_a_trajectory_towards_a_cell_raises_its_risk(grid, settings):
    """A berg drifting towards a cell must raise its risk before it arrives."""
    target_lat, target_lon = float(grid.lats[6]), float(grid.lons[6])
    start_lat = target_lat + 3.0  # several hundred km away
    approaching = {
        "iceberg_id": "APPROACH", "latitude": start_lat, "longitude": target_lon, "area_km2": 200.0,
        "track": [
            {"latitude": start_lat - k * 1.0, "longitude": target_lon,
             "horizon_hours": k * 24.0, "uncertainty_radius_km": 0.0}
            for k in range(4)
        ],
    }
    receding = {
        "iceberg_id": "RECEDE", "latitude": start_lat, "longitude": target_lon, "area_km2": 200.0,
        "track": [
            {"latitude": start_lat + k * 1.0, "longitude": target_lon,
             "horizon_hours": k * 24.0, "uncertainty_radius_km": 0.0}
            for k in range(4)
        ],
    }
    i, j = grid.index_of(target_lat, target_lon)
    towards = rk.iceberg_risk(grid, [approaching], settings)[0][i, j]
    away = rk.iceberg_risk(grid, [receding], settings)[0][i, j]
    assert towards > away


def test_positional_uncertainty_widens_the_hazard(grid, settings):
    lat, lon = float(grid.lats[6]), float(grid.lons[6])
    off_lat = lat + 1.0
    certain = [{"latitude": off_lat, "longitude": lon, "area_km2": 200.0,
                "track": [{"latitude": off_lat, "longitude": lon, "horizon_hours": 0.0,
                           "uncertainty_radius_km": 0.0}]}]
    uncertain = [{"latitude": off_lat, "longitude": lon, "area_km2": 200.0,
                  "track": [{"latitude": off_lat, "longitude": lon, "horizon_hours": 0.0,
                             "uncertainty_radius_km": 80.0}]}]
    i, j = grid.index_of(lat, lon)
    assert rk.iceberg_risk(grid, uncertain, settings)[0][i, j] > rk.iceberg_risk(grid, certain, settings)[0][i, j]


def test_many_distant_bergs_do_not_sum_into_a_false_alarm(grid, settings):
    """Independent-union combination must keep the total bounded by 1."""
    bergs = [
        {"latitude": float(grid.lats[2]), "longitude": float(lon), "area_km2": 100.0}
        for lon in grid.lons
    ]
    risk, counted = rk.iceberg_risk(grid, bergs, settings)
    assert counted == len(bergs)
    assert risk.max() <= 1.0


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------
def test_risk_grid_is_bounded_and_shaped(grid, calm_fields, settings, vessel):
    result = rk.compute_risk_grid(grid, calm_fields, [], None, vessel, settings=settings)
    assert result.total.shape == grid.shape
    assert result.total.min() >= 0.0 and result.total.max() <= 1.0
    for component in result.components.values():
        assert component.shape == grid.shape
        assert component.min() >= 0.0 and component.max() <= 1.0


def test_calm_conditions_produce_low_risk(grid, calm_fields, settings, vessel):
    from app.services import landmask

    result = rk.compute_risk_grid(grid, calm_fields, [], None, vessel, settings=settings)
    offshore = landmask.navigable_mask(grid) & (landmask.distance_to_land_km(grid) > 200)
    assert result.total[offshore].mean() < 0.2


def test_severe_conditions_produce_high_risk(grid, calm_fields, settings, vessel):
    severe = dict(calm_fields)
    severe["sea_ice_concentration"] = np.full(grid.shape, 0.95)
    severe["wind_speed"] = np.full(grid.shape, 30.0)
    severe["wave_height"] = np.full(grid.shape, 10.0)
    severe["t2m"] = np.full(grid.shape, -20.0)
    result = rk.compute_risk_grid(grid, severe, [], None, vessel, settings=settings)
    assert result.total.mean() > 0.6
    assert result.navigable.sum() < np.sum(result.total.size)


def test_missing_layers_are_reported_and_renormalised(grid, settings, vessel):
    """A missing layer must degrade confidence, not silently score zero."""
    partial = {"sea_ice_concentration": np.full(grid.shape, 0.5)}
    result = rk.compute_risk_grid(grid, partial, [], None, vessel, settings=settings)
    assert "weather" in result.missing_layers
    assert "ocean" in result.missing_layers
    assert "sea_ice" in result.available_layers

    full = dict(partial)
    full["wind_speed"] = np.zeros(grid.shape)
    full["wave_height"] = np.zeros(grid.shape)
    complete = rk.compute_risk_grid(grid, full, [], None, vessel, settings=settings)
    # With calm weather/ocean actually present, the total must fall.
    assert complete.total.mean() < result.total.mean()


def test_weights_change_the_outcome(grid, calm_fields, settings, vessel):
    icy = dict(calm_fields)
    icy["sea_ice_concentration"] = np.full(grid.shape, 0.6)
    ice_heavy = rk.compute_risk_grid(
        grid, icy, [], None, vessel, rk.RiskWeights(sea_ice=1.0, iceberg=0.0, weather=0.0, ocean=0.0, constraint=0.0),
        settings=settings,
    )
    ice_light = rk.compute_risk_grid(
        grid, icy, [], None, vessel, rk.RiskWeights(sea_ice=0.05, iceberg=0.0, weather=1.0, ocean=1.0, constraint=1.0),
        settings=settings,
    )
    assert ice_heavy.total.mean() > ice_light.total.mean()


def test_land_is_never_navigable(grid, calm_fields, settings, vessel):
    from app.services import landmask

    result = rk.compute_risk_grid(grid, calm_fields, [], None, vessel, settings=settings)
    land = ~landmask.navigable_mask(grid)
    assert not result.navigable[land].any()


def test_impassable_ice_blocks_cells(grid, calm_fields, settings):
    """At 0.95 cover an open-water hull is stopped where an icebreaker is not."""
    blocked = dict(calm_fields)
    blocked["sea_ice_concentration"] = np.full(grid.shape, 0.95)
    weak = rk.compute_risk_grid(grid, blocked, [], None, rk.VesselProfile(ice_class="none"), settings=settings)
    strong = rk.compute_risk_grid(grid, blocked, [], None, rk.VesselProfile(ice_class="icebreaker"), settings=settings)
    assert weak.navigable.sum() == 0
    assert strong.navigable.sum() > 0


def test_risk_grid_rows_are_database_ready(grid, calm_fields, settings, vessel):
    result = rk.compute_risk_grid(grid, calm_fields, [], None, vessel, settings=settings)
    rows = rk.risk_grid_rows(result)
    assert rows
    required = {
        "generated_at", "valid_at", "horizon_hours", "latitude", "longitude", "sea_ice_risk",
        "iceberg_risk", "weather_risk", "ocean_risk", "constraint_risk", "total_risk", "navigable",
    }
    assert required <= set(rows[0])
    assert all(0.0 <= r["total_risk"] <= 1.0 for r in rows)


# ---------------------------------------------------------------------------
# Point and route assessment
# ---------------------------------------------------------------------------
def test_assess_point_matches_the_grid(grid, calm_fields, settings, vessel):
    result = rk.compute_risk_grid(grid, calm_fields, [], None, vessel, settings=settings)
    lat, lon = float(grid.lats[4]), float(grid.lons[4])
    point = rk.assess_point(result, lat, lon)
    i, j = grid.index_of(lat, lon)
    assert point["total_risk"] == pytest.approx(float(result.total[i, j]))


def test_assess_point_rejects_positions_outside_the_domain(grid, calm_fields, settings, vessel):
    result = rk.compute_risk_grid(grid, calm_fields, [], None, vessel, settings=settings)
    with pytest.raises(ValueError):
        rk.assess_point(result, 45.0, 10.0)


def test_closest_approach_finds_the_nearest_point_and_time():
    route = [(-65.0, 20.0), (-65.0, 30.0), (-65.0, 40.0)]
    track = [
        {"latitude": -66.0, "longitude": 25.0, "horizon_hours": 0.0, "uncertainty_radius_km": 0.0},
        {"latitude": -65.2, "longitude": 30.0, "horizon_hours": 24.0, "uncertainty_radius_km": 5.0},
        {"latitude": -64.0, "longitude": 35.0, "horizon_hours": 48.0, "uncertainty_radius_km": 10.0},
    ]
    approach = rk.closest_approach(route, track)
    assert approach is not None
    assert approach["horizon_hours"] == 24.0, "the closest pass is at +24 h"
    assert approach["distance_km"] < 40.0
    assert approach["margin_km"] == pytest.approx(approach["distance_km"] - 5.0)


def test_assess_route_aggregates_and_flags_encounters(grid, calm_fields, settings, vessel):
    result = rk.compute_risk_grid(grid, calm_fields, [], None, vessel, settings=settings)
    waypoints = [(float(grid.lats[i]), float(grid.lons[i])) for i in range(4)]
    bergs = [{"iceberg_id": "NEAR", "latitude": waypoints[1][0], "longitude": waypoints[1][1],
              "area_km2": 100.0}]
    assessment = rk.assess_route(result, waypoints, bergs)
    assert assessment["samples"] == len(waypoints)
    assert 0.0 <= assessment["mean_risk"] <= assessment["max_risk"] <= 1.0
    assert set(assessment["component_means"]) == {
        "sea_ice", "iceberg", "weather", "ocean", "constraint"
    }
    assert assessment["closest_iceberg_km"] < 100.0


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_risk_map_endpoint(client):
    r = client.get("/api/risk/map")
    assert r.status_code == 200
    body = r.json()
    summary = body["summary"]
    assert summary["cells_ocean"] > 0
    assert summary["cells_navigable"] <= summary["cells_ocean"]
    assert 0.0 <= summary["mean_risk"] <= 1.0
    assert set(summary["weights"]) == {"sea_ice", "iceberg", "weather", "ocean", "constraint"}
    assert body["scale"]["min"] == 0.0 and body["scale"]["max"] == 1.0
    values = [v for row in body["total_risk_field"] for v in row if v is not None]
    assert values and min(values) >= 0.0 and max(values) <= 1.0


def test_risk_map_hotspots_are_ranked(client):
    body = client.get("/api/risk/map?top_hotspots=5").json()
    risks = [h["total_risk"] for h in body["hotspots"]]
    assert risks == sorted(risks, reverse=True)
    for hotspot in body["hotspots"]:
        assert hotspot["dominant_component"] in {
            "sea_ice", "iceberg", "weather", "ocean", "constraint"
        }


def test_risk_map_respects_the_vessel_ice_class(client):
    weak = client.get("/api/risk/map?ice_class=none").json()["summary"]
    strong = client.get("/api/risk/map?ice_class=icebreaker").json()["summary"]
    assert strong["cells_navigable"] >= weak["cells_navigable"]


def test_risk_map_cells_format_and_filter(client):
    body = client.get("/api/risk/map?format=cells&min_risk=0.5&stride=2").json()
    for cell in body["cells"]:
        assert cell["total_risk"] >= 0.5


def test_risk_map_rejects_a_malformed_bbox(client):
    assert client.get("/api/risk/map?bbox=-70,-80,10,40").status_code == 422
    assert client.get("/api/risk/map?bbox=abc").status_code == 422


def test_risk_point_endpoint(client, grid):
    lat, lon = float(grid.lats[3]), float(grid.lons[3])
    r = client.get(f"/api/risk/point?latitude={lat}&longitude={lon}")
    assert r.status_code == 200
    body = r.json()
    total = body["total_risk"]
    assert 0.0 <= total <= 1.0
    assert isinstance(body["navigable"], bool)
    assert body["provenance"]["data_mode"] == "demo"


def test_risk_point_rejects_positions_outside_the_domain(client):
    assert client.get("/api/risk/point?latitude=10&longitude=40").status_code == 422
    assert client.get("/api/risk/point?latitude=-95&longitude=40").status_code == 422


def test_risk_forecast_horizon_is_validated(client):
    assert client.get("/api/risk/map?forecast_hours=13").status_code == 422
    assert client.get("/api/risk/map?forecast_hours=48").status_code == 200
