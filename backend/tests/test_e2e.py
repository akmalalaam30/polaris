"""End-to-end integration: the whole pipeline, wired together.

    data -> preprocess -> forecast -> iceberg trajectory -> risk grid ->
    route optimisation -> API response

These tests do not stub anything.  They run the real services against the real
(demo-mode) database and assert that each stage's output actually reaches the
next one - that the forecast the model produced is the forecast the risk grid
used, that the trajectories the physics produced are the ones the routing
considered, and that what the API returns matches what the services computed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest


@pytest.fixture(scope="module")
def pipeline(seeded_db, trained_models, settings, grid):
    """Run every stage once and hand back all the intermediate products."""
    from app.database.connection import session_scope
    from app.services import iceberg_trajectory as itraj
    from app.services import risk_engine as rk
    from app.services import route_optimizer as ro
    from app.services import sea_ice_forecasting as sif
    from app.services.environment import (
        clear_cache,
        get_environment,
        get_iceberg_context,
        get_risk_grid,
    )

    clear_cache()
    stages: dict = {}
    vessel = rk.VesselProfile(ice_class="icebreaker", speed_knots=14.0, fuel_consumption_tpd=45.0)

    with session_scope() as session:
        # 1. Observed environment ------------------------------------------
        stages["environment"] = get_environment(session, settings, grid)

        # 2. Forecast -------------------------------------------------------
        stages["forecast"] = sif.forecast(24, settings, grid)
        stages["forecast_rows"] = sif.persist_forecast(session, stages["forecast"])

        # 3. Iceberg trajectories -------------------------------------------
        stages["icebergs"] = get_iceberg_context(session, settings, grid, horizon_hours=72)
        stages["trajectory_rows"] = itraj.persist_trajectories(session, stages["icebergs"].trajectories)

        # 4. Risk grid ------------------------------------------------------
        stages["risk"] = get_risk_grid(session, settings, grid, vessel=vessel, horizon_hours=24)
        stages["risk_rows"] = rk.persist_risk_grid(session, stages["risk"])

        # 5. Routes ---------------------------------------------------------
        stages["routes"] = ro.optimize_routes(
            stages["risk"], "bharati", "maitri",
            fields=stages["environment"].fields, vessel=vessel, settings=settings,
            icebergs=stages["icebergs"].records,
        )
        stages["route_ids"] = ro.persist_routes(session, stages["routes"], settings.data_mode)

    stages["vessel"] = vessel
    return stages


# ---------------------------------------------------------------------------
# Stage-by-stage
# ---------------------------------------------------------------------------
def test_stage_1_environment_is_complete(pipeline, grid):
    env = pipeline["environment"]
    assert env.observed_at is not None
    assert env.has_sea_ice
    assert env.sources == ["POLARIS-DEMO"]
    for layer in ("sea_ice_concentration", "u10", "v10", "u_current", "v_current", "wave_height"):
        assert env.fields[layer].shape == grid.shape
        assert np.isfinite(env.fields[layer]).any(), f"{layer} carries no data"


def test_stage_2_forecast_is_produced_and_stored(pipeline, grid):
    fc = pipeline["forecast"]
    assert fc.algorithm == "hist_gradient_boosting"
    assert pipeline["forecast_rows"] > 0
    values = fc.prediction[fc.mask]
    assert values.size > 0
    assert values.min() >= 0.0 and values.max() <= 1.0
    assert fc.metrics["skill_vs_persistence"] > 0


def test_stage_2_forecast_is_anchored_to_the_observations(pipeline):
    """A 24 h forecast must resemble the analysis it started from."""
    env = pipeline["environment"]
    fc = pipeline["forecast"]
    observed = env.fields["sea_ice_concentration"]
    both = fc.mask & np.isfinite(observed)
    assert both.sum() > 50
    difference = np.abs(fc.prediction[both] - observed[both])
    assert difference.mean() < 0.25, "a one-day forecast cannot diverge wildly from the analysis"


def test_stage_3_trajectories_start_from_the_observed_positions(pipeline):
    ctx = pipeline["icebergs"]
    assert ctx.count > 0
    assert pipeline["trajectory_rows"] > 0
    for record in ctx.records:
        result = ctx.trajectories.get(record["iceberg_id"])
        if result is None:
            continue
        assert result.origin == pytest.approx((record["latitude"], record["longitude"]))
        assert record["track"], "each berg's forecast track must be attached to its record"
        assert record["track"][-1]["horizon_hours"] == 72.0


def test_stage_3_trajectories_use_the_ingested_forcing(pipeline, grid):
    """The drift model must read the same fields the environment stage built."""
    from app.utils.geo import bilinear_sample
    from app.utils.geo import fill_nan_nearest

    env = pipeline["environment"]
    ctx = pipeline["icebergs"]
    result = next(iter(ctx.trajectories.values()))
    expected = bilinear_sample(
        grid, fill_nan_nearest(env.fields["u_current"], 10), *result.origin
    )
    assert result.forcing["u_current"] == pytest.approx(expected, abs=1e-3)


def test_stage_4_risk_grid_consumes_the_forecast_and_the_bergs(pipeline, grid):
    risk = pipeline["risk"]
    assert pipeline["risk_rows"] > 0
    assert risk.horizon_hours == 24
    # The count is of bergs that actually raise risk somewhere, so it is at most
    # the number tracked - and at least one, since the demo bergs sit in domain.
    assert 0 < risk.n_icebergs <= pipeline["icebergs"].count
    assert risk.missing_layers == [], f"unexpected missing layers: {risk.missing_layers}"
    assert set(risk.available_layers) == {"sea_ice", "iceberg", "weather", "ocean", "constraint"}
    assert risk.total.min() >= 0.0 and risk.total.max() <= 1.0
    assert risk.components["iceberg"].max() > 0.0, "the tracked bergs must influence the grid"


def test_stage_4_risk_reflects_the_forecast_not_only_the_analysis(pipeline, grid, settings):
    """Switching the horizon changes the sea-ice component, proving it is used."""
    from app.database.connection import session_scope
    from app.services.environment import get_risk_grid

    with session_scope() as session:
        now = get_risk_grid(session, settings, grid, vessel=pipeline["vessel"], horizon_hours=0)
    forecast_conditioned = pipeline["risk"]
    assert now.horizon_hours == 0
    assert not np.allclose(now.components["sea_ice"], forecast_conditioned.components["sea_ice"])


def test_stage_5_routes_are_computed_over_that_risk_grid(pipeline):
    result = pipeline["routes"]
    assert set(result["routes"]) == {"shortest", "safest", "polaris"}
    assert result["risk_grid"]["horizon_hours"] == 24
    assert result["graph"]["nodes"] > 0
    for profile, route in result["routes"].items():
        assert route["distance_km"] > 0, profile
        assert route["duration_hours"] > 0, profile
        assert route["estimated_fuel_tonnes"] > 0, profile
        assert route["n_waypoints"] >= 2, profile


def test_stage_5_routes_avoid_land_end_to_end(pipeline, grid):
    from app.services import landmask
    from app.utils.geo import densify_great_circle

    navigable = landmask.navigable_mask(grid)
    for profile, route in pipeline["routes"]["routes"].items():
        points = [(w["latitude"], w["longitude"]) for w in route["waypoints"]]
        for k in range(len(points) - 1):
            for lat, lon in densify_great_circle(*points[k], *points[k + 1], step_km=30.0):
                i, j = grid.index_of(lat, lon)
                assert navigable[i, j], f"{profile} crosses land at ({lat:.2f}, {lon:.2f})"


def test_stage_5_iceberg_analysis_reaches_the_route(pipeline):
    """Closest-approach analysis must use the predicted tracks, not just fixes."""
    polaris = pipeline["routes"]["routes"]["polaris"]
    assessment = polaris["risk_assessment"]
    assert assessment["samples"] > 0
    assert "component_means" in assessment
    if assessment.get("iceberg_encounters"):
        encounter = assessment["iceberg_encounters"][0]
        assert encounter["distance_km"] >= 0
        assert encounter["horizon_hours"] >= 0
        assert "margin_km" in encounter


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_every_stage_is_persisted_and_readable(pipeline, settings, grid):
    from app.database.connection import session_scope
    from app.database import repositories as repo

    with session_scope() as session:
        summary = repo.database_summary(session)
        counts = summary["row_counts"]
        for table in (
            "sea_ice_observations", "iceberg_observations", "weather_observations",
            "ocean_observations", "forecasts", "iceberg_trajectories", "risk_grid",
            "route_results", "ingestion_log", "model_registry",
        ):
            assert counts[table] > 0, f"{table} is empty after a full pipeline run"

        # The stored forecast must match what the model produced.
        field, _unc, meta = repo.get_forecast_field(session, grid, 24)
        fc = pipeline["forecast"]
        both = fc.mask & np.isfinite(field)
        np.testing.assert_allclose(field[both], fc.prediction[both], atol=1e-6)
        assert meta["model_name"] == settings.sea_ice_model_name

        # The stored routes must match the computed ones.
        stored = repo.get_routes_by_request(session, pipeline["routes"]["request_id"])
        assert len(stored) == len(pipeline["routes"]["routes"])
        by_profile = {r.profile: r for r in stored}
        for profile, route in pipeline["routes"]["routes"].items():
            assert by_profile[profile].distance_km == pytest.approx(route["distance_km"])
            assert by_profile[profile].geometry["geometry"]["type"] == "LineString"


def test_model_registry_records_real_metrics(pipeline, settings):
    from app.database.connection import session_scope
    from app.database import repositories as repo

    with session_scope() as session:
        records = repo.get_model_records(session, settings.sea_ice_model_name)
    assert records
    for record in records:
        assert record.n_train_samples > 0
        assert record.n_val_samples > 0
        assert record.metrics["rmse"] > 0
        assert record.metrics["persistence_baseline"]["rmse"] > 0
        assert record.artifact_path


# ---------------------------------------------------------------------------
# API agreement
# ---------------------------------------------------------------------------
def test_api_reports_the_same_numbers_as_the_services(pipeline, client, grid):
    """The HTTP layer must not transform or re-derive the service results."""
    fc = pipeline["forecast"]
    body = client.get("/api/sea-ice/forecast?forecast_hours=24").json()
    assert body["algorithm"] == fc.algorithm
    assert body["validation_metrics"]["rmse"] == pytest.approx(fc.metrics["rmse"])
    api_values = np.array(
        [[np.nan if v is None else v for v in row] for row in body["field"]["values"]]
    )
    both = np.isfinite(api_values) & np.isfinite(fc.prediction)
    np.testing.assert_allclose(api_values[both], fc.prediction[both], atol=1e-4)


def test_dashboard_summary_covers_every_subsystem(pipeline, client):
    body = client.get("/api/dashboard/summary").json()
    for section in ("sea_ice", "forecast", "icebergs", "risk", "routing", "ingestion", "database"):
        assert section in body and body[section], f"dashboard section {section} is empty"

    assert body["data_mode"] == "demo"
    assert "DEMO" in body["disclaimer"].upper()
    assert body["sea_ice"]["available"] is True
    assert body["icebergs"]["tracked_count"] == pipeline["icebergs"].count
    assert set(body["forecast"]["horizons"]) == {"24", "48", "72"}
    for horizon in body["forecast"]["horizons"].values():
        assert horizon["status"] == "ok"
        assert horizon["skill_vs_persistence"] > 0
    assert body["risk"]["highest_risk_areas"]
    assert body["routing"]["available"] is True
    assert body["routing"]["graph_nodes"] > 0


def test_full_demo_workflow_through_http_only(client, grid):
    """Everything a demonstrator needs, using nothing but the public API."""
    assert client.get("/api/health").json()["status"] in {"ok", "degraded"}

    current = client.get("/api/sea-ice/current").json()
    assert current["observed_at"]

    forecast = client.get("/api/sea-ice/forecast?forecast_hours=72").json()
    assert forecast["valid_at"] > forecast["issued_at"]

    bergs = client.get("/api/icebergs").json()
    assert bergs["count"] > 0
    berg_id = bergs["icebergs"][0]["iceberg_id"]

    trajectory = client.get(f"/api/icebergs/{berg_id}/trajectory").json()
    assert trajectory["points"]

    risk = client.get("/api/risk/map?forecast_hours=24").json()
    assert risk["summary"]["cells_navigable"] > 0

    routes = client.post(
        "/api/route/optimize",
        json={
            "start": {"station": "bharati"},
            "destination": {"station": "maitri"},
            "vessel": {"ice_class": "icebreaker", "speed_knots": 14, "fuel_consumption_tpd": 45},
            "preferences": {"forecast_hours": 24, "profiles": ["shortest", "safest", "polaris"]},
            "persist": True,
        },
    )
    assert routes.status_code == 200, routes.text
    payload = routes.json()
    assert len(payload["routes"]) == 3
    assert payload["persisted_route_ids"]

    summary = client.get("/api/dashboard/summary").json()
    assert summary["routing"]["recent_routes"] > 0


def test_pipeline_is_reproducible(pipeline, settings, grid):
    """Re-running the forecast on unchanged data must give the same answer."""
    from app.services.sea_ice_forecasting import forecast

    again = forecast(24, settings, grid)
    first = pipeline["forecast"]
    assert again.issued_at == first.issued_at
    both = again.mask & first.mask
    np.testing.assert_allclose(again.prediction[both], first.prediction[both], atol=1e-9)
