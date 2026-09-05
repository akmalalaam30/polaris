"""Iceberg trajectory physics and endpoints."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from app.services import iceberg_trajectory as itraj


@pytest.fixture(scope="module")
def issued_at():
    return datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture()
def uniform_fields(grid):
    """A calm, uniform environment: 0.2 m/s eastward current, no wind, no ice."""

    def make(u_current=0.2, v_current=0.0, u10=0.0, v10=0.0, conc=0.0):
        return {
            "u_current": np.full(grid.shape, u_current),
            "v_current": np.full(grid.shape, v_current),
            "u10": np.full(grid.shape, u10),
            "v10": np.full(grid.shape, v10),
            "sea_ice_concentration": np.full(grid.shape, conc),
        }

    return make


def _open_water_point(grid):
    """A cell comfortably offshore, so drift tests do not immediately ground."""
    from app.services import landmask

    distance = landmask.distance_to_land_km(grid)
    i, j = np.unravel_index(int(np.argmax(distance)), grid.shape)
    return float(grid.lats[i]), float(grid.lons[j])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_thickness_estimate_is_monotonic_and_bounded():
    thicknesses = [itraj.estimate_thickness_m(L) for L in (200, 1_000, 5_000, 30_000, 100_000)]
    assert thicknesses == sorted(thicknesses)
    assert all(30.0 <= t <= 350.0 for t in thicknesses)
    # A giant tabular berg is capped at ice-shelf thickness, not extrapolated.
    assert itraj.estimate_thickness_m(100_000) == pytest.approx(350.0)


def test_hydrostatic_balance_of_freeboard_and_draft():
    state = itraj.state_from_observation("T1", -65.0, 30.0, length_nm=10, width_nm=5)
    assert state.freeboard_m + state.draft_m == pytest.approx(state.thickness_m)
    # About 7/8 of the ice sits below the waterline.
    ratio = state.draft_m / state.thickness_m
    assert ratio == pytest.approx(itraj.RHO_ICE / itraj.RHO_WATER, rel=1e-6)
    assert 0.80 < ratio < 0.88


def test_state_reconstruction_from_partial_dimensions():
    from_area = itraj.state_from_observation("T2", -65.0, 30.0, area_km2=200.0)
    assert from_area.length_m == pytest.approx(2 * from_area.width_m)
    assert from_area.plan_area_m2 == pytest.approx(200.0 * 1e6, rel=1e-6)

    from_length = itraj.state_from_observation("T3", -65.0, 30.0, length_nm=20)
    assert from_length.length_m == pytest.approx(20 * itraj.NM_TO_M)
    assert from_length.width_m == pytest.approx(from_length.length_m * 0.5)

    bare = itraj.state_from_observation("T4", -65.0, 30.0)
    assert bare.length_m > 0 and bare.mass_kg > 0


def test_invalid_coordinates_are_rejected():
    from app.utils.validation import ValidationError

    with pytest.raises(ValidationError):
        itraj.state_from_observation("BAD", 200.0, 30.0)


# ---------------------------------------------------------------------------
# Force balance
# ---------------------------------------------------------------------------
def test_berg_at_rest_in_still_water_stays_at_rest():
    state = itraj.state_from_observation("STILL", -65.0, 30.0, length_nm=5, width_nm=3)
    env = {"u10": 0.0, "v10": 0.0, "u_current": 0.0, "v_current": 0.0, "sea_ice_concentration": 0.0}
    ax, ay = itraj._acceleration(state, 0.0, 0.0, env, itraj.DriftParameters())
    assert ax == pytest.approx(0.0, abs=1e-12)
    assert ay == pytest.approx(0.0, abs=1e-12)


def test_coriolis_and_pressure_gradient_cancel_at_the_water_velocity():
    """A berg moving with the water feels no rotational or water-drag force.

    Coriolis acts on the berg's absolute velocity while the sea-surface-tilt
    term acts through the geostrophic current; together they must reduce to a
    force on the velocity *relative to the water*, which is zero here.  Air drag
    is switched off so the cancellation can be checked exactly.
    """
    state = itraj.state_from_observation("DRIFT", -65.0, 30.0, length_nm=5, width_nm=3)
    u_w, v_w = 0.25, -0.1
    env = {"u10": 0.0, "v10": 0.0, "u_current": u_w, "v_current": v_w, "sea_ice_concentration": 0.0}
    ax, ay = itraj._acceleration(state, u_w, v_w, env, itraj.DriftParameters(air_drag=0.0))
    assert ax == pytest.approx(0.0, abs=1e-15)
    assert ay == pytest.approx(0.0, abs=1e-15)


def test_still_air_damps_a_berg_drifting_with_the_current():
    """With air drag active the only residual force opposes the berg's motion."""
    state = itraj.state_from_observation("DAMP", -65.0, 30.0, length_nm=5, width_nm=3)
    u_w, v_w = 0.25, -0.1
    env = {"u10": 0.0, "v10": 0.0, "u_current": u_w, "v_current": v_w, "sea_ice_concentration": 0.0}
    ax, ay = itraj._acceleration(state, u_w, v_w, env, itraj.DriftParameters())
    # Anti-parallel to the velocity, and far too small to matter over 72 hours.
    assert ax * u_w + ay * v_w < 0
    assert math.hypot(ax, ay) < 1e-7


def test_wind_accelerates_a_stationary_berg_downwind():
    state = itraj.state_from_observation("WIND", -65.0, 30.0, length_nm=3, width_nm=2)
    env = {"u10": 15.0, "v10": 0.0, "u_current": 0.0, "v_current": 0.0, "sea_ice_concentration": 0.0}
    ax, ay = itraj._acceleration(state, 0.0, 0.0, env, itraj.DriftParameters())
    assert ax > 0, "an eastward wind must accelerate the berg eastward"
    assert abs(ay) < abs(ax)


def test_drag_scales_with_the_square_of_relative_velocity():
    state = itraj.state_from_observation("DRAG", -65.0, 30.0, length_nm=5, width_nm=3)
    params = itraj.DriftParameters()
    a1 = itraj._acceleration(
        state, 0.0, 0.0, {"u10": 5.0, "v10": 0, "u_current": 0, "v_current": 0, "sea_ice_concentration": 0}, params
    )[0]
    a2 = itraj._acceleration(
        state, 0.0, 0.0, {"u10": 10.0, "v10": 0, "u_current": 0, "v_current": 0, "sea_ice_concentration": 0}, params
    )[0]
    assert a2 / a1 == pytest.approx(4.0, rel=1e-6)


def test_added_mass_reduces_acceleration():
    state = itraj.state_from_observation("AM", -65.0, 30.0, length_nm=5, width_nm=3)
    env = {"u10": 20.0, "v10": 0.0, "u_current": 0.0, "v_current": 0.0, "sea_ice_concentration": 0.0}
    without = itraj._acceleration(state, 0, 0, env, itraj.DriftParameters(added_mass=0.0))[0]
    with_am = itraj._acceleration(state, 0, 0, env, itraj.DriftParameters(added_mass=0.5))[0]
    assert with_am == pytest.approx(without / 1.5, rel=1e-9)


def test_compact_ice_engages_the_pack_drag_term():
    state = itraj.state_from_observation("PACK", -65.0, 30.0, length_nm=5, width_nm=3)
    params = itraj.DriftParameters()
    base_env = {"u10": 10.0, "v10": 0.0, "u_current": 0.0, "v_current": 0.0}
    loose = itraj._acceleration(state, 0, 0, {**base_env, "sea_ice_concentration": 0.5}, params)[0]
    compact = itraj._acceleration(state, 0, 0, {**base_env, "sea_ice_concentration": 0.99}, params)[0]
    assert compact != loose, "a compact pack must change the force balance"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
def test_berg_follows_a_uniform_current(grid, uniform_fields, issued_at):
    """With no wind or ice, drift must converge on the ocean velocity."""
    lat, lon = _open_water_point(grid)
    state = itraj.state_from_observation("FOLLOW", lat, lon, length_nm=4, width_nm=2)
    result = itraj.predict_trajectory(
        state, grid, uniform_fields(u_current=0.2), issued_at,
        horizon_hours=48, output_every_hours=12, ensemble_size=0,
    )
    last = result.points[-1]
    assert not last.grounded
    # 0.2 m/s for 48 h is about 34.5 km, eastward (bearing near 090).
    assert result.total_displacement_km == pytest.approx(0.2 * 48 * 3600 / 1000.0, rel=0.15)
    assert 70 < last.bearing_deg < 110
    assert last.speed_m_s == pytest.approx(0.2, abs=0.05)


def test_trajectory_speeds_stay_physically_plausible(grid, uniform_fields, issued_at):
    lat, lon = _open_water_point(grid)
    state = itraj.state_from_observation("SPEED", lat, lon, length_nm=8, width_nm=4)
    result = itraj.predict_trajectory(
        state, grid, uniform_fields(u_current=0.3, u10=20.0), issued_at,
        horizon_hours=72, output_every_hours=6, ensemble_size=0,
    )
    for point in result.points:
        assert 0.0 <= point.speed_m_s < 1.5, "icebergs do not exceed ~1.5 m/s"


def test_output_cadence_and_horizons(grid, uniform_fields, issued_at):
    lat, lon = _open_water_point(grid)
    state = itraj.state_from_observation("CADENCE", lat, lon, length_nm=5, width_nm=3)
    result = itraj.predict_trajectory(
        state, grid, uniform_fields(), issued_at,
        horizon_hours=72, output_every_hours=24, ensemble_size=0,
    )
    assert [p.horizon_hours for p in result.points] == [24.0, 48.0, 72.0]
    for point in result.points:
        expected = issued_at.timestamp() + point.horizon_hours * 3600
        assert point.valid_at.timestamp() == pytest.approx(expected)


def test_uncertainty_grows_with_lead_time(grid, uniform_fields, issued_at):
    lat, lon = _open_water_point(grid)
    state = itraj.state_from_observation("ENS", lat, lon, length_nm=5, width_nm=3)
    result = itraj.predict_trajectory(
        state, grid, uniform_fields(u_current=0.15, u10=8.0), issued_at,
        horizon_hours=72, output_every_hours=12, ensemble_size=12,
    )
    radii = [p.uncertainty_radius_km for p in result.points]
    assert result.ensemble_size == 12
    assert radii[0] > 0.0
    assert radii == sorted(radii), f"ensemble spread must be non-decreasing, got {radii}"
    assert radii[-1] > radii[0]


def test_zero_ensemble_reports_no_spread(grid, uniform_fields, issued_at):
    lat, lon = _open_water_point(grid)
    state = itraj.state_from_observation("NOENS", lat, lon, length_nm=5, width_nm=3)
    result = itraj.predict_trajectory(
        state, grid, uniform_fields(), issued_at, horizon_hours=24, ensemble_size=0
    )
    assert result.ensemble_size == 0
    assert all(p.uncertainty_radius_km == 0.0 for p in result.points)


def test_berg_grounds_instead_of_crossing_land(grid, uniform_fields, issued_at):
    """A berg driven hard at the coast must stop, not sail over the continent."""
    from app.services import landmask

    navigable = landmask.navigable_mask(grid)
    distance = landmask.distance_to_land_km(grid)
    # Pick the navigable cell closest to land in the southern half of the domain.
    candidates = np.where(navigable & (distance > 0) & (grid.meshgrid()[0] < -66.0))
    order = np.argsort(distance[candidates])
    i, j = candidates[0][order[0]], candidates[1][order[0]]
    state = itraj.state_from_observation(
        "GROUND", float(grid.lats[i]), float(grid.lons[j]), length_nm=3, width_nm=2
    )
    # Drive it southward, into the continent.
    fields = uniform_fields(u_current=0.0, v_current=-0.6, u10=0.0, v10=-25.0)
    result = itraj.predict_trajectory(
        state, grid, fields, issued_at, horizon_hours=120, output_every_hours=12, ensemble_size=0
    )
    for point in result.points:
        assert grid.contains(point.latitude, point.longitude)
        gi, gj = grid.index_of(point.latitude, point.longitude)
        assert navigable[gi, gj], "a predicted position must never lie on land"


def test_trajectory_rows_are_database_ready(grid, uniform_fields, issued_at):
    lat, lon = _open_water_point(grid)
    state = itraj.state_from_observation("ROWS", lat, lon, length_nm=5, width_nm=3)
    result = itraj.predict_trajectory(
        state, grid, uniform_fields(), issued_at, horizon_hours=24, output_every_hours=12, ensemble_size=2
    )
    rows = itraj.trajectory_rows(result)
    assert len(rows) == len(result.points)
    required = {
        "iceberg_id", "issued_at", "valid_at", "horizon_hours", "latitude", "longitude",
        "speed_m_s", "bearing_deg", "uncertainty_radius_km", "model_name", "model_version",
    }
    assert required <= set(rows[0])


def test_predict_many_skips_failures_without_aborting(grid, uniform_fields, issued_at):
    lat, lon = _open_water_point(grid)
    good = itraj.state_from_observation("GOOD", lat, lon, length_nm=5, width_nm=3)
    broken = itraj.IcebergState("BROKEN", lat, lon, length_m=0.0, width_m=0.0, thickness_m=0.0)
    results = itraj.predict_many(
        [good, broken], grid, uniform_fields(), issued_at, horizon_hours=24, ensemble_size=0
    )
    assert "GOOD" in results


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_iceberg_list_endpoint(client):
    r = client.get("/api/icebergs")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    berg = body["icebergs"][0]
    assert berg["iceberg_id"].startswith("DEMO-")
    assert berg["data_mode"] == "demo"
    assert berg["n_observations"] > 1
    assert -90 <= berg["latitude"] <= 0


def test_iceberg_filters(client):
    everything = client.get("/api/icebergs").json()
    filtered = client.get("/api/icebergs?min_area_km2=500").json()
    assert filtered["count"] <= everything["count"]
    for berg in filtered["icebergs"]:
        assert berg["area_km2"] >= 500


def test_trajectory_endpoint(client):
    berg_id = client.get("/api/icebergs").json()["icebergs"][0]["iceberg_id"]
    r = client.get(f"/api/icebergs/{berg_id}/trajectory?forecast_hours=72")
    assert r.status_code == 200
    body = r.json()
    assert body["iceberg_id"] == berg_id
    assert body["points"], "a trajectory must contain predicted positions"
    assert body["points"][-1]["horizon_hours"] == 72.0
    assert body["geometry"]["draft_m"] < body["geometry"]["thickness_m"]
    assert body["ensemble_size"] > 0
    assert body["observed_track"], "the observed history must accompany the forecast"
    assert set(body["forcing"]) >= {"u10", "v10", "u_current", "v_current", "sea_ice_concentration"}


def test_trajectory_endpoint_is_case_insensitive(client):
    berg_id = client.get("/api/icebergs").json()["icebergs"][0]["iceberg_id"]
    assert client.get(f"/api/icebergs/{berg_id.lower()}/trajectory").status_code == 200


def test_unknown_iceberg_returns_404_with_guidance(client):
    r = client.get("/api/icebergs/NOT-A-BERG/trajectory")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "known_icebergs" in detail
    assert detail["known_icebergs"]


def test_trajectory_rejects_an_invalid_horizon(client):
    berg_id = client.get("/api/icebergs").json()["icebergs"][0]["iceberg_id"]
    assert client.get(f"/api/icebergs/{berg_id}/trajectory?forecast_hours=-5").status_code == 422
    assert client.get(f"/api/icebergs/{berg_id}/trajectory?forecast_hours=9999").status_code == 422


def test_batch_trajectories_endpoint(client):
    r = client.get("/api/icebergs/trajectories?forecast_hours=48")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    for entry in body["trajectories"]:
        assert entry["points"]
        assert entry["points"][-1]["horizon_hours"] <= 48.0


def test_out_of_domain_berg_gets_no_fabricated_trajectory(grid, uniform_fields, issued_at):
    """Outside the domain there is no forcing, so refuse rather than invent.

    Many real USNIC bergs sit in the Weddell and Scotia seas, well outside a
    0-100 E domain. Integrating there would silently produce a confident zero
    drift with a non-zero ensemble spread - a fabricated answer.
    """
    from app.utils.validation import ValidationError

    outside_lat = grid.lat_max + 5.0
    outside_lon = grid.lon_min - 30.0
    state = itraj.state_from_observation("OUTSIDE", outside_lat, outside_lon, length_nm=10, width_nm=5)
    with pytest.raises(ValidationError, match="outside the POLARIS analysis domain"):
        itraj.predict_trajectory(
            state, grid, uniform_fields(), issued_at, horizon_hours=24, ensemble_size=0
        )


def test_out_of_domain_berg_returns_422_with_the_domain(client, session, grid):
    """The API must explain why, not return an empty or zeroed trajectory."""
    from datetime import datetime, timezone

    from app.database import repositories as repo
    from app.services.environment import clear_cache

    repo.upsert_icebergs(
        session,
        [{
            "iceberg_id": "DEMO-FAR",
            "observed_at": datetime.now(timezone.utc).replace(microsecond=0),
            "latitude": grid.lat_max + 4.0,
            "longitude": grid.lon_min - 25.0,
            "length_nm": 12.0, "width_nm": 6.0, "area_km2": 247.0,
            "source": "POLARIS-DEMO", "data_mode": "demo", "dataset_version": "demo-1.0",
        }],
    )
    session.commit()
    clear_cache()
    try:
        listed = client.get("/api/icebergs").json()
        entry = next(b for b in listed["icebergs"] if b["iceberg_id"] == "DEMO-FAR")
        assert entry["in_domain"] is False, "an out-of-domain berg is still listed, but flagged"

        r = client.get("/api/icebergs/DEMO-FAR/trajectory")
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "outside the POLARIS analysis domain" in detail["message"]
        assert detail["domain"]["lat_max"] == grid.lat_max

        # The batch endpoint simply omits it rather than erroring.
        batch = client.get("/api/icebergs/trajectories").json()
        assert all(t["iceberg_id"] != "DEMO-FAR" for t in batch["trajectories"])
    finally:
        from app.models.database_models import IcebergObservation

        session.query(IcebergObservation).filter(
            IcebergObservation.iceberg_id == "DEMO-FAR"
        ).delete()
        session.commit()
        clear_cache()


def test_iceberg_context_cache_key_ignores_int_float_horizon(session, settings, grid):
    """72 and 72.0 are the same request and must hit the same cache entry.

    They serialise differently, so without normalisation a warmed cache is
    silently bypassed - the endpoint pays full cost while the cache holds an
    identical result under a near-identical key.
    """
    from app.services.environment import clear_cache, get_iceberg_context

    clear_cache()
    first = get_iceberg_context(session, settings, grid, horizon_hours=72)
    second = get_iceberg_context(session, settings, grid, horizon_hours=72.0)
    assert first is second, "int and float horizons produced separate cache entries"
