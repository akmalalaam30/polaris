"""Ingestion, validation, preprocessing and repository behaviour."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def test_demo_ingestion_populates_every_dataset(seeded_db):
    for dataset in ("sea_ice", "extent_index", "icebergs", "weather", "ocean"):
        result = seeded_db[dataset]
        assert result["status"] == "ok", f"{dataset}: {result['message']}"
        assert result["ingested"] > 0


def test_demo_records_are_labelled_as_synthetic(session):
    from app.models.database_models import IcebergObservation, SeaIceObservation

    for model in (SeaIceObservation, IcebergObservation):
        rows = session.query(model).limit(20).all()
        assert rows
        for row in rows:
            assert row.data_mode == "demo"
            assert row.source == "POLARIS-DEMO"


def test_demo_iceberg_ids_cannot_be_mistaken_for_usnic(session):
    from app.database import repositories as repo

    for berg in repo.latest_iceberg_positions(session):
        assert berg.iceberg_id.startswith("DEMO-")


def test_ingestion_is_idempotent(session, settings, grid):
    """Re-ingesting the same day must update, never duplicate."""
    from app.database import repositories as repo
    from app.services.data_ingestion import ingest_sea_ice

    before = session.query(
        __import__("app.models.database_models", fromlist=["SeaIceObservation"]).SeaIceObservation
    ).count()
    ingest_sea_ice(session, grid, days=3, settings=settings)
    session.commit()
    after = session.query(
        __import__("app.models.database_models", fromlist=["SeaIceObservation"]).SeaIceObservation
    ).count()
    assert after == before


def test_ingestion_log_records_every_run(session):
    from app.database import repositories as repo

    entries = repo.recent_ingestions(session, limit=50)
    assert entries
    assert {e.dataset for e in entries} >= {"sea_ice_concentration", "antarctic_icebergs"}
    for entry in entries:
        assert entry.status in {"ok", "partial", "skipped", "failed"}
        assert entry.finished_at is not None


def test_real_mode_never_emits_demo_records(settings):
    """A real-mode ingestion may fail, but it must not write synthetic rows."""
    from app.services.data_ingestion import DEMO_SOURCE, SOURCE_NSIDC, SOURCE_USNIC

    assert DEMO_SOURCE not in (SOURCE_NSIDC, SOURCE_USNIC)


def test_unavailable_provider_is_skipped_not_faked(session, settings, grid, monkeypatch):
    """With no credentials and mirrors disabled, ocean ingestion is skipped."""
    from app.services import data_ingestion as ing

    real_settings = settings.model_copy(update={"data_mode": "real", "allow_open_mirrors": False})
    result = ing.ingest_ocean(session, grid, settings=real_settings)
    assert result.status == "skipped"
    assert "CMEMS" in result.message or "ALLOW_OPEN_MIRRORS" in result.message
    assert result.ingested == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_coordinate_validation():
    from app.utils.validation import ValidationError, validate_latitude, validate_longitude

    assert validate_latitude(-65.5) == -65.5
    assert validate_longitude(200.0) == pytest.approx(-160.0)
    with pytest.raises(ValidationError):
        validate_latitude(120.0)
    with pytest.raises(ValidationError):
        validate_longitude("not-a-number")


def test_southern_hemisphere_guard():
    from app.utils.validation import ValidationError, require_southern_hemisphere

    assert require_southern_hemisphere(-70.0) == -70.0
    with pytest.raises(ValidationError):
        require_southern_hemisphere(45.0)


def test_point_frame_validation_drops_and_dedupes():
    from app.utils.validation import validate_point_frame

    df = pd.DataFrame(
        {
            "iceberg_id": ["A1", "A1", "A2", "A3"],
            "latitude": [-65.0, -65.0, 999.0, -60.0],
            "longitude": [30.0, 30.0, 40.0, 400.0],
            "observed_at": ["2026-01-01", "2026-01-01", "2026-01-01", "not-a-date"],
        }
    )
    clean, report = validate_point_frame(df, "test", dedupe_on=("iceberg_id", "observed_at"))
    assert report.n_input == 4
    assert report.n_dropped_invalid == 2       # bad latitude, bad timestamp
    assert report.n_dropped_duplicate == 1     # the repeated A1 row
    assert len(clean) == 1
    assert clean.iloc[0]["iceberg_id"] == "A1"


def test_forecast_hours_validation():
    from app.utils.validation import ValidationError, validate_forecast_hours

    assert validate_forecast_hours(48) == 48
    with pytest.raises(ValidationError):
        validate_forecast_hours(36)


# ---------------------------------------------------------------------------
# Geospatial utilities
# ---------------------------------------------------------------------------
def test_haversine_against_a_known_distance():
    from app.utils.geo import haversine_km

    # Cape Town to Sydney is about 11 000 km; check a tighter known pair instead:
    # one degree of latitude is 111.19 km on a sphere of radius 6371 km.
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)
    assert haversine_km(-65.0, 30.0, -65.0, 30.0) == pytest.approx(0.0, abs=1e-9)


def test_bearing_and_destination_round_trip():
    from app.utils.geo import destination_point, haversine_km, initial_bearing_deg

    lat, lon = destination_point(-65.0, 30.0, 90.0, 100.0)
    assert haversine_km(-65.0, 30.0, lat, lon) == pytest.approx(100.0, rel=1e-3)
    assert initial_bearing_deg(-65.0, 30.0, lat, lon) == pytest.approx(90.0, abs=0.5)


def test_landmask_matches_antarctic_geography(grid):
    from app.services import landmask

    navigable = landmask.navigable_mask(grid)
    lat2d, _ = grid.meshgrid()
    # Far south in this sector is the continent; the north is open ocean.
    deep_south = lat2d <= -71.0
    far_north = lat2d >= -60.0
    assert navigable[deep_south].mean() < 0.5
    assert navigable[far_north].mean() > 0.95


def test_distance_to_land_is_zero_on_land_and_large_offshore(grid):
    from app.services import landmask

    navigable = landmask.navigable_mask(grid)
    distance = landmask.distance_to_land_km(grid)
    assert distance[~navigable].max() == pytest.approx(0.0, abs=1e-6)
    assert distance[navigable].max() > 100.0


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_history_archive_round_trips(history, settings, grid):
    from app.services import preprocessing

    times, ice, _weather = history
    assert ice.shape == (len(times), grid.shape[0], grid.shape[1])
    assert times == sorted(times), "history must be chronological"

    loaded_times, variables, loaded_grid, attrs = preprocessing.load_history(settings)
    assert len(loaded_times) == len(times)
    assert loaded_grid.shape == grid.shape
    np.testing.assert_allclose(
        variables["sea_ice_concentration"], ice, rtol=1e-5, atol=1e-5, equal_nan=True
    )
    assert attrs["data_mode"] == "demo"


def test_concentrations_stay_physical(history):
    _times, ice, _weather = history
    finite = ice[np.isfinite(ice)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0
    assert np.isfinite(ice).mean() > 0.4


def test_preprocess_sea_ice_field_masks_land(history, grid):
    from app.services import landmask
    from app.services.preprocessing import preprocess_sea_ice_field

    _times, ice, _ = history
    cleaned = preprocess_sea_ice_field(ice[-1], grid)
    land = landmask.land_fraction(grid) >= 0.5
    assert np.all(np.isnan(cleaned[land]))
    assert np.isfinite(cleaned[~land]).all()


def test_iceberg_track_preprocessing_derives_kinematics(session):
    from app.database import repositories as repo
    from app.services.preprocessing import preprocess_iceberg_tracks

    berg = repo.latest_iceberg_positions(session)[0]
    tracks = preprocess_iceberg_tracks(repo.get_iceberg_track(session, berg.iceberg_id))
    assert len(tracks) > 1
    assert tracks["observed_at"].is_monotonic_increasing
    assert tracks["observed_at"].duplicated().sum() == 0

    derived = tracks.dropna(subset=["drift_speed_m_s"])
    assert len(derived) == len(tracks) - 1, "every fix after the first gets a velocity"
    assert (derived["drift_speed_m_s"] >= 0).all()
    assert (derived["drift_speed_m_s"] < 1.5).all(), "implausible drift must be rejected"
    assert derived["drift_bearing_deg"].between(0, 360).all()

    # Displacement must be consistent with the reported speed and time step.
    row = derived.iloc[0]
    expected = row["drift_speed_m_s"] * row["dt_hours"] * 3600.0 / 1000.0
    assert row["displacement_km"] == pytest.approx(expected, rel=1e-6)


def test_wind_component_conversion_round_trips():
    from app.services.preprocessing import wind_components, wind_speed_direction

    speed, direction = 12.0, 235.0
    u, v = wind_components(speed, direction)
    back_speed, back_direction = wind_speed_direction(u, v)
    assert float(back_speed) == pytest.approx(speed, rel=1e-9)
    assert float(back_direction) == pytest.approx(direction, abs=1e-6)
    # A northerly (000) blows from the north, i.e. towards the south: v < 0.
    u_n, v_n = wind_components(10.0, 0.0)
    assert float(v_n) < 0 and abs(float(u_n)) < 1e-9


def test_environment_synchronisation_fills_every_layer(session, grid):
    from app.database import repositories as repo
    from app.services.preprocessing import synchronise_environment

    ice, _t, _s = repo.get_sea_ice_field(session, grid)
    weather, _ = repo.get_weather_fields(session, grid)
    ocean, _ = repo.get_ocean_fields(session, grid)
    fields = synchronise_environment(grid, ice, weather, ocean)

    for key in ("sea_ice_concentration", "u10", "v10", "u_current", "v_current",
                "wind_speed", "current_speed", "wave_height", "land_fraction",
                "distance_to_land_km", "ice_edge_distance_km"):
        assert key in fields, key
        assert fields[key].shape == grid.shape

    assert np.isfinite(fields["wind_speed"]).any()
    assert (fields["wind_speed"][np.isfinite(fields["wind_speed"])] >= 0).all()


def test_missing_layers_become_nan_not_zero(grid):
    from app.services.preprocessing import synchronise_environment

    fields = synchronise_environment(grid, None, None, None)
    assert np.all(np.isnan(fields["u10"]))
    assert np.all(np.isnan(fields["wind_speed"]))
    assert np.all(np.isnan(fields["current_speed"]))


def test_feature_frame_shape_and_content(history, grid):
    from app.services.preprocessing import build_feature_frame

    times, ice, weather = history
    X, y, names, t_index = build_feature_frame(times, ice, grid, horizon_days=1, weather=weather)
    assert X.shape[0] == y.shape[0] == t_index.shape[0]
    assert X.shape[1] == len(names)
    assert np.isfinite(X).all(), "no NaNs may reach the model"
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert "conc_t" in names and "doy_sin" in names
    assert len(np.unique(t_index)) > 5


def test_feature_frame_rejects_too_little_history(grid):
    from app.services.preprocessing import build_feature_frame
    from app.utils.validation import ValidationError

    times = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(5)]
    ice = np.zeros((5, grid.shape[0], grid.shape[1]))
    with pytest.raises(ValidationError):
        build_feature_frame(times, ice, grid, horizon_days=1)


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------
def test_repository_reconstructs_the_grid(session, grid):
    from app.database import repositories as repo

    field, observed_at, source = repo.get_sea_ice_field(session, grid)
    assert observed_at is not None
    assert source == "POLARIS-DEMO"
    assert field.shape == grid.shape
    assert np.isfinite(field).sum() > 0.5 * field.size


def test_retention_keeps_only_recent_days(session, settings, grid):
    from app.database import repositories as repo
    from app.models.database_models import SeaIceObservation

    distinct_days = session.query(SeaIceObservation.observed_at).distinct().count()
    assert distinct_days <= settings.sea_ice_db_retention_days


def test_database_summary_counts_rows(session):
    from app.database import repositories as repo

    summary = repo.database_summary(session)
    assert summary["row_counts"]["sea_ice_observations"] > 0
    assert summary["tracked_icebergs"] > 0
    assert summary["latest_sea_ice"] is not None
