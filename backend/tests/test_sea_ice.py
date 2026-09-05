"""Sea-ice forecasting: training, validation, persistence, inference, API."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_regression_metrics_are_correct():
    from app.models.ml_models import regression_metrics

    y = np.array([0.0, 0.5, 1.0, 0.25])
    p = np.array([0.1, 0.4, 0.9, 0.25])
    m = regression_metrics(y, p)
    assert m["n"] == 4
    assert m["mae"] == pytest.approx(0.075)
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((p - y) ** 2)))
    assert m["bias"] == pytest.approx(np.mean(p - y))


def test_perfect_prediction_scores_perfectly():
    from app.models.ml_models import regression_metrics

    y = np.linspace(0, 1, 50)
    m = regression_metrics(y, y.copy())
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_skill_score_semantics():
    from app.models.ml_models import skill_score

    assert skill_score(0.05, 0.10) == pytest.approx(0.75)   # better than baseline
    assert skill_score(0.10, 0.10) == pytest.approx(0.0)    # equal to baseline
    assert skill_score(0.20, 0.10) < 0                      # worse than baseline


def test_ice_edge_metrics_at_the_15_percent_threshold():
    from app.models.ml_models import ice_edge_metrics

    y = np.array([0.0, 0.2, 0.9, 0.1])
    p = np.array([0.05, 0.3, 0.8, 0.2])
    m = ice_edge_metrics(y, p)
    assert m["accuracy"] == pytest.approx(0.75)  # last pair disagrees
    assert 0.0 <= m["f1"] <= 1.0


def test_uncertainty_calibration_tracks_real_residual_spread():
    from app.models.ml_models import BinnedUncertainty

    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, 4000)
    # Deliberately heteroscedastic: mid-range predictions are noisier.
    noise = np.where((pred > 0.4) & (pred < 0.7), 0.20, 0.02)
    truth = pred - rng.normal(0, noise)
    unc = BinnedUncertainty().fit(truth, pred)
    sigma = unc.predict(np.array([0.05, 0.5, 0.95]))
    assert sigma[1] > sigma[0] * 3, "the noisy band must report a larger sigma"
    assert sigma[1] > sigma[2] * 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def test_training_produces_every_horizon(trained_models, settings):
    assert set(trained_models) == set(settings.forecast_horizons_hours)


def test_training_metrics_are_real_and_out_of_sample(trained_models):
    for horizon, report in trained_models.items():
        m = report["metrics"]
        assert report["n_train_samples"] > 1000
        assert report["n_val_samples"] > 100
        assert 0.0 <= m["mae"] <= 1.0
        assert 0.0 <= m["rmse"] <= 1.0
        assert np.isfinite(m["r2"])
        assert m["validation_period"]["n_time_steps"] > 5
        assert m["data_mode"] == "demo"


def test_selected_model_is_never_worse_than_persistence(trained_models):
    """Model selection must be gated on measured skill, not assumed.

    Sea-ice concentration is highly persistent, so a learned model does not
    automatically beat "tomorrow looks like today". The pipeline scores every
    candidate against the baseline on the same held-out split and keeps the
    best - so the shipped model can never be worse than persistence, whatever
    the data happens to support.
    """
    for horizon, report in trained_models.items():
        m = report["metrics"]
        assert m["rmse"] <= m["persistence_baseline"]["rmse"] + 1e-9, (
            f"{horizon}h selected model RMSE {m['rmse']:.4f} is worse than persistence "
            f"{m['persistence_baseline']['rmse']:.4f}; selection is broken"
        )
        assert m["skill_vs_persistence"] >= 0.0


def test_every_candidate_is_scored_and_the_choice_is_recorded(trained_models):
    for horizon, report in trained_models.items():
        m = report["metrics"]
        candidates = m["candidates"]
        assert "persistence" in candidates
        assert m["selected_candidate"] in candidates
        # The recorded winner really is the lowest-RMSE candidate.
        best = min(candidates, key=lambda n: candidates[n]["rmse"])
        assert candidates[m["selected_candidate"]]["rmse"] == pytest.approx(
            candidates[best]["rmse"], abs=1e-9
        )
        assert candidates["persistence"]["skill_vs_persistence"] == pytest.approx(0.0, abs=1e-9)


def test_learned_candidates_add_skill_on_this_data(trained_models):
    """On the demo field the gradient-boosting candidates should win outright."""
    for horizon, report in trained_models.items():
        candidates = report["metrics"]["candidates"]
        learned = {k: v for k, v in candidates.items() if k != "persistence"}
        assert learned, "no learned candidate was trained"
        assert max(v["skill_vs_persistence"] for v in learned.values()) > 0.0, (
            f"no learned candidate beat persistence at {horizon}h"
        )


def test_ice_edge_is_predicted_well(trained_models):
    for horizon, report in trained_models.items():
        assert report["metrics"]["edge"]["accuracy"] > 0.85


def test_chronological_split_never_leaks_the_future():
    from app.services.sea_ice_forecasting import _chronological_split

    X = np.arange(100).reshape(-1, 1).astype(float)
    y = X.ravel()
    t = np.repeat(np.arange(10), 10)
    X_tr, y_tr, X_val, y_val = _chronological_split(X, y, t, val_fraction=0.2)
    assert len(y_tr) + len(y_val) == len(y)
    assert y_tr.max() < y_val.min(), "training samples must all precede validation samples"


# ---------------------------------------------------------------------------
# Persistence of artifacts
# ---------------------------------------------------------------------------
def test_model_artifacts_are_saved_and_reloadable(trained_models, settings):
    from pathlib import Path

    from app.models.ml_models import BaseForecaster, model_artifact_path
    from app.services import sea_ice_forecasting as sif

    for horizon in settings.forecast_horizons_hours:
        path = model_artifact_path(Path(settings.model_path), settings.sea_ice_model_name, horizon)
        assert path.exists(), f"missing artifact for {horizon}h"
        assert path.with_suffix(".json").exists(), "human-readable model card must accompany it"

        model = BaseForecaster.load(path)
        assert model.horizon_hours == horizon
        assert model.feature_names
        assert model.metrics["rmse"] > 0

        # A reloaded model must predict identically to the cached one.
        cached = sif.load_model(horizon, settings)
        X = np.zeros((5, len(model.feature_names)), dtype="float32")
        np.testing.assert_allclose(model.predict(X), cached.predict(X))


def test_api_does_not_retrain_on_each_call(trained_models, settings):
    from app.services import sea_ice_forecasting as sif

    sif.clear_model_cache()
    first = sif.load_model(24, settings)
    second = sif.load_model(24, settings)
    assert first is second, "models must be cached, not reloaded or retrained per request"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def test_forecast_produces_a_physical_field(trained_models, settings, grid):
    from app.services.sea_ice_forecasting import forecast

    result = forecast(24, settings, grid)
    assert result.horizon_hours == 24
    assert (result.valid_at - result.issued_at).total_seconds() == 24 * 3600
    assert result.prediction.shape == grid.shape
    values = result.prediction[result.mask]
    assert values.size > 100
    assert values.min() >= 0.0 and values.max() <= 1.0
    assert np.isfinite(result.uncertainty[result.mask]).all()
    assert result.algorithm == "hist_gradient_boosting"


def test_forecast_masks_land(trained_models, settings, grid):
    from app.services import landmask
    from app.services.sea_ice_forecasting import forecast

    result = forecast(48, settings, grid)
    land = landmask.land_fraction(grid) >= 0.5
    assert not result.mask[land].any(), "land cells must never carry a forecast"


def test_longer_horizons_are_not_more_certain(trained_models, settings, grid):
    from app.services.sea_ice_forecasting import forecast

    sigma = {}
    for horizon in (24, 72):
        r = forecast(horizon, settings, grid)
        sigma[horizon] = float(np.nanmean(r.uncertainty[r.mask]))
    assert sigma[72] >= sigma[24] * 0.95, (
        f"72 h uncertainty {sigma[72]:.4f} should not be materially below 24 h {sigma[24]:.4f}"
    )


def test_forecast_rejects_an_unsupported_horizon(settings, grid):
    from app.services.sea_ice_forecasting import forecast
    from app.utils.validation import ValidationError

    with pytest.raises(ValidationError):
        forecast(36, settings, grid)


def test_forecast_falls_back_to_the_baseline_when_untrained(settings, grid, tmp_path):
    """With no artifact present the API must degrade to persistence, not crash."""
    from app.services import sea_ice_forecasting as sif

    sif.clear_model_cache()
    empty = settings.model_copy(update={"model_path": tmp_path})
    result = sif.forecast(24, empty, grid)
    assert result.algorithm == "persistence"
    assert result.prediction[result.mask].size > 0
    assert "baseline" in str(result.metrics).lower()
    sif.clear_model_cache()


def test_forecast_rows_persist_and_reload(trained_models, session, settings, grid):
    from app.database import repositories as repo
    from app.services.sea_ice_forecasting import forecast, persist_forecast

    result = forecast(24, settings, grid)
    n = persist_forecast(session, result)
    session.commit()
    assert n > 0

    field, unc, meta = repo.get_forecast_field(session, grid, 24)
    assert meta["model_name"] == settings.sea_ice_model_name
    assert meta["n_cells"] == n
    stored = field[np.isfinite(field)]
    assert stored.min() >= 0.0 and stored.max() <= 1.0


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_current_sea_ice_endpoint(client):
    r = client.get("/api/sea-ice/current")
    assert r.status_code == 200
    body = r.json()
    assert body["observed_at"] is not None
    assert body["provenance"]["data_mode"] == "demo"
    assert "DEMO" in body["provenance"]["disclaimer"].upper()
    assert 0.0 <= body["statistics"]["mean_concentration"] <= 1.0
    values = [v for row in body["field"]["values"] for v in row if v is not None]
    assert values and min(values) >= 0.0 and max(values) <= 1.0
    assert len(body["field"]["lats"]) == body["grid"]["shape"][0]


def test_current_sea_ice_cells_format_and_stride(client):
    full = client.get("/api/sea-ice/current?format=cells&stride=1").json()
    strided = client.get("/api/sea-ice/current?format=cells&stride=3").json()
    assert len(strided["cells"]) < len(full["cells"])
    for cell in strided["cells"][:20]:
        assert 0.0 <= cell["concentration"] <= 1.0


def test_forecast_endpoint_returns_real_metrics(client):
    r = client.get("/api/sea-ice/forecast?forecast_hours=72")
    assert r.status_code == 200
    body = r.json()
    assert body["forecast_hours"] == 72
    assert body["algorithm"] == "hist_gradient_boosting"
    metrics = body["validation_metrics"]
    assert metrics["rmse"] > 0
    assert metrics["skill_vs_persistence"] > 0
    assert metrics["persistence_baseline"]["rmse"] > metrics["rmse"]
    assert body["uncertainty"] is not None


def test_forecast_endpoint_rejects_bad_horizon(client):
    r = client.get("/api/sea-ice/forecast?forecast_hours=17")
    assert r.status_code == 422
    assert "24" in str(r.json()["detail"])


def test_forecast_endpoint_rejects_bad_region(client):
    r = client.get("/api/sea-ice/forecast?forecast_hours=24&region=1,2")
    assert r.status_code == 422


def test_extent_endpoint(client):
    r = client.get("/api/sea-ice/extent?days=30")
    assert r.status_code == 200
    series = r.json()["series"]
    assert len(series) == 30
    assert all(p["extent_million_km2"] > 0 for p in series)
    stamps = [p["observed_at"] for p in series]
    assert stamps == sorted(stamps)


def test_corrupt_artifact_falls_back_to_the_baseline(settings, grid, tmp_path):
    """A model that cannot be unpickled must not take the endpoint down.

    joblib artifacts are pickled against a specific scikit-learn version, so a
    checkout with a different version cannot load them. The right answer is a
    working persistence baseline plus a clear log line, not a 500 on an
    endpoint that still has perfectly good observations.
    """
    from app.models.ml_models import model_artifact_path
    from app.services import sea_ice_forecasting as sif

    broken = tmp_path / "broken"
    broken.mkdir()
    for horizon in settings.forecast_horizons_hours:
        path = model_artifact_path(broken, settings.sea_ice_model_name, horizon)
        path.write_bytes(b"this is not a joblib artifact")

    sif.clear_model_cache()
    result = sif.forecast(24, settings.model_copy(update={"model_path": broken}), grid)
    assert result.algorithm == "persistence"
    assert result.prediction[result.mask].size > 0
    sif.clear_model_cache()
