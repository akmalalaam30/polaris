"""Sea-ice concentration forecasting: train, validate, save, load, predict.

Pipeline
--------
``build history`` -> ``feature frame`` -> ``chronological split`` -> ``fit`` ->
``validate against persistence`` -> ``calibrate uncertainty`` -> ``save
artifact + registry row``.

Inference loads the saved artifact (models are never retrained inside an API
request), builds features from the latest available observations, and writes one
:class:`~app.models.database_models.Forecast` row per grid cell and horizon.

All reported metrics are computed on a held-out chronological split of the data
that was actually used.  Nothing is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from app.config import Settings, get_settings
from app.database import repositories as repo
from app.models.ml_models import (
    BaseForecaster,
    BinnedUncertainty,
    GradientBoostingForecaster,
    PersistenceForecaster,
    build_forecaster,
    ice_edge_metrics,
    model_artifact_path,
    regression_metrics,
    skill_score,
)
from app.services import preprocessing
from app.utils.geo import GridSpec, grid_from_settings
from app.utils.logging import get_logger
from app.utils.validation import DataUnavailableError, ValidationError, validate_forecast_hours

log = get_logger("services.sea_ice_forecasting")

_MODEL_CACHE: dict[str, BaseForecaster] = {}


@dataclass
class TrainingReport:
    """Result of training one horizon."""

    horizon_hours: int
    algorithm: str
    n_train: int
    n_val: int
    metrics: dict
    artifact_path: str
    feature_importance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "horizon_hours": self.horizon_hours,
            "algorithm": self.algorithm,
            "n_train_samples": self.n_train,
            "n_val_samples": self.n_val,
            "metrics": self.metrics,
            "artifact_path": self.artifact_path,
            "top_features": dict(list(self.feature_importance.items())[:8]),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _chronological_split(
    X: np.ndarray, y: np.ndarray, t_index: np.ndarray, val_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by time, never by shuffling - this is a forecasting problem."""
    unique_t = np.unique(t_index)
    if len(unique_t) < 5:
        raise ValidationError(f"Need at least 5 distinct time steps to split; got {len(unique_t)}")
    cutoff = unique_t[int(len(unique_t) * (1.0 - val_fraction))]
    train = t_index < cutoff
    val = ~train
    if train.sum() == 0 or val.sum() == 0:
        raise ValidationError("Chronological split produced an empty partition")
    return X[train], y[train], X[val], y[val]


def train_horizon(
    horizon_hours: int,
    times: Sequence[datetime],
    ice: np.ndarray,
    grid: GridSpec,
    weather: dict[str, np.ndarray] | None,
    settings: Settings,
    algorithm: str = "gbr",
    subsample: int = 2,
    val_fraction: float = 0.2,
    drift: dict[str, np.ndarray] | None = None,
) -> TrainingReport:
    """Train candidate models, select on measured skill, and persist the winner.

    Sea-ice concentration is strongly persistent, so a learned model does not
    automatically beat "tomorrow looks like today" - and on real observations it
    sometimes does not.  Rather than assert that it does, this trains two
    candidates and scores them against the persistence baseline on the *same*
    held-out chronological split:

    ``gbr_level``
        gradient boosting on the concentration itself, using every available
        feature (sea ice, atmosphere, sea-ice drift).
    ``gbr_lean``
        the same, restricted to the sea-ice features alone.  Extra predictors
        are not free: on a small domain the atmospheric and drift columns can
        cost more in variance than they return in signal, and this candidate
        measures that instead of assuming either way.
    ``gbr_delta``
        gradient boosting on the *change* over the horizon, which makes
        persistence the zero-prediction so the model only moves the answer where
        it found signal.

    Whichever has the lowest validation RMSE is saved - **including the
    persistence baseline itself if no learned model beats it**.  The selection
    and every candidate's score are recorded in the metrics, so the outcome is
    auditable rather than assumed.
    """
    horizon_days = max(1, int(round(horizon_hours / 24)))
    X, y, names, t_index = preprocessing.build_feature_frame(
        times, ice, grid, horizon_days=horizon_days, weather=weather, drift=drift,
        subsample=subsample,
    )
    X_tr, y_tr, X_val, y_val = _chronological_split(X, y, t_index, val_fraction)
    log.info(
        "Horizon %dh: %d training / %d validation samples, %d features",
        horizon_hours, len(y_tr), len(y_val), len(names),
    )

    baseline = PersistenceForecaster(horizon_hours, names).fit(X_tr, y_tr)
    base_pred = baseline.predict(X_val)
    base_metrics = regression_metrics(y_val, base_pred)

    # The lean feature set is a prefix of the full one: feature_names() appends
    # the atmospheric and drift blocks after the sea-ice columns, so the subset
    # is a plain column slice and costs nothing extra to build.
    lean_names = preprocessing.feature_names(False, False)
    lean_slice = slice(0, len(lean_names))
    has_extras = len(names) > len(lean_names)

    candidates: dict[str, BaseForecaster] = {"persistence": baseline}
    feature_view: dict[str, slice] = {"persistence": slice(None)}
    if algorithm.lower() not in ("persistence", "baseline"):
        candidates["gbr_level"] = GradientBoostingForecaster(horizon_hours, names, predict_delta=False)
        feature_view["gbr_level"] = slice(None)
        candidates["gbr_delta"] = GradientBoostingForecaster(horizon_hours, names, predict_delta=True)
        feature_view["gbr_delta"] = slice(None)
        if has_extras:
            candidates["gbr_lean"] = GradientBoostingForecaster(
                horizon_hours, lean_names, predict_delta=False
            )
            feature_view["gbr_lean"] = lean_slice

    scores: dict[str, dict] = {"persistence": base_metrics}
    predictions: dict[str, np.ndarray] = {"persistence": base_pred}
    for name, candidate in candidates.items():
        if name == "persistence":
            continue
        view = feature_view[name]
        candidate.data_mode = settings.data_mode
        candidate.fit(X_tr[:, view], y_tr)
        pred = candidate.predict(X_val[:, view])
        predictions[name] = pred
        scores[name] = regression_metrics(y_val, pred)
        log.info(
            "  candidate %-11s (%2d features) RMSE %.4f (skill %+0.3f)",
            name, len(candidate.feature_names), scores[name]["rmse"],
            skill_score(scores[name]["rmse"], base_metrics["rmse"]),
        )

    # Ties go to the simpler model: persistence, then fewer features, then the
    # level target ahead of the delta target.
    order = ["persistence", "gbr_lean", "gbr_level", "gbr_delta"]
    selected_name = min(
        (n for n in order if n in scores),
        key=lambda n: (round(scores[n]["rmse"], 6), order.index(n)),
    )
    model = candidates[selected_name]
    pred = predictions[selected_name]
    model_metrics = scores[selected_name]

    metrics = {
        **model_metrics,
        "edge": ice_edge_metrics(y_val, pred),
        "persistence_baseline": base_metrics,
        "skill_vs_persistence": skill_score(model_metrics["rmse"], base_metrics["rmse"]),
        "selected_candidate": selected_name,
        "candidates": {
            name: {
                "rmse": score["rmse"],
                "mae": score["mae"],
                "skill_vs_persistence": skill_score(score["rmse"], base_metrics["rmse"]),
            }
            for name, score in scores.items()
        },
        "validation_period": {
            "n_time_steps": int(len(np.unique(t_index))),
            "val_fraction": val_fraction,
            "history_start": times[0].isoformat(),
            "history_end": times[-1].isoformat(),
        },
        "data_mode": settings.data_mode,
    }
    if selected_name == "persistence":
        metrics["note"] = (
            "No learned candidate beat the persistence baseline on this held-out split, so "
            "the baseline itself was selected. This is a measurement, not a training failure: "
            "sea-ice concentration is highly persistent at 24-72 h, so persistence is a strong "
            "competitor and a learned model only wins where it finds real signal beyond it. "
            "Gating selection on measured skill is what keeps the shipped forecast from being "
            "worse than simply assuming no change."
        )
        log.warning("Horizon %dh: persistence selected - no learned candidate beat it", horizon_hours)

    model.metrics = metrics
    model.n_val = int(len(y_val))
    model.uncertainty = BinnedUncertainty().fit(y_val, pred)
    model.data_mode = settings.data_mode

    selected_view = feature_view[selected_name]
    metrics["n_features_used"] = len(model.feature_names)
    metrics["feature_names_used"] = list(model.feature_names)

    importance: dict[str, float] = {}
    if isinstance(model, GradientBoostingForecaster):
        try:
            importance = model.permutation_importance(X_val[:, selected_view], y_val)
            metrics["permutation_importance_rmse_increase"] = {
                k: round(v, 5) for k, v in list(importance.items())[:10]
            }
        except Exception as exc:  # pragma: no cover - diagnostics only
            log.warning("Permutation importance failed: %s", exc)

    artifact = model_artifact_path(Path(settings.model_path), settings.sea_ice_model_name, horizon_hours)
    model.save(artifact)
    _MODEL_CACHE.pop(str(artifact), None)

    log.info(
        "Horizon %dh: selected %s, RMSE %.4f (persistence %.4f), skill %+.3f, edge accuracy %.3f",
        horizon_hours, selected_name, model_metrics["rmse"], base_metrics["rmse"],
        metrics["skill_vs_persistence"], metrics["edge"]["accuracy"],
    )
    return TrainingReport(
        horizon_hours=horizon_hours,
        algorithm=model.algorithm,
        n_train=int(len(y_tr)),
        n_val=int(len(y_val)),
        metrics=metrics,
        artifact_path=str(artifact),
        feature_importance=importance,
    )


def train_all(
    session=None,
    settings: Settings | None = None,
    horizons: Sequence[int] | None = None,
    algorithm: str = "gbr",
    days: int | None = None,
    rebuild_history: bool = False,
    subsample: int = 2,
) -> dict[int, dict]:
    """Train every configured horizon and register the results."""
    settings = settings or get_settings()
    grid = grid_from_settings(settings)
    horizons = horizons or settings.forecast_horizons_hours

    if rebuild_history or not preprocessing.history_path(settings).exists():
        times, ice = preprocessing.build_sea_ice_history(days=days, grid=grid, settings=settings)
        weather = preprocessing.build_weather_history(times, grid, settings)
        drift = preprocessing.build_ice_drift_history(times, grid, settings)
    else:
        times, variables, grid, _attrs = preprocessing.load_history(settings)
        ice = variables["sea_ice_concentration"]
        loaded = preprocessing.load_weather_history(settings)
        weather = loaded[1] if loaded and len(loaded[0]) == len(times) else None
        loaded_drift = preprocessing.load_ice_drift_history(settings)
        drift = loaded_drift[1] if loaded_drift and len(loaded_drift[0]) == len(times) else None
        if drift is None:
            drift = preprocessing.build_ice_drift_history(times, grid, settings)

    if len(times) < settings.sea_ice_min_training_days:
        raise DataUnavailableError(
            f"Only {len(times)} days of history available; at least "
            f"{settings.sea_ice_min_training_days} are required to train. "
            f"Run 'python scripts/preprocess_data.py --days 200' first."
        )

    reports: dict[int, dict] = {}
    for horizon in horizons:
        report = train_horizon(
            horizon, times, ice, grid, weather, settings, algorithm=algorithm,
            subsample=subsample, drift=drift,
        )
        reports[horizon] = report.to_dict()
        if session is not None:
            repo.register_model(
                session,
                {
                    "model_name": settings.sea_ice_model_name,
                    "model_version": report.metrics.get("model_version", "1.0.0"),
                    "horizon_hours": horizon,
                    "algorithm": report.algorithm,
                    "trained_at": datetime.now(timezone.utc),
                    "n_train_samples": report.n_train,
                    "n_val_samples": report.n_val,
                    "metrics": report.metrics,
                    "artifact_path": report.artifact_path,
                    "data_mode": settings.data_mode,
                    "notes": f"history {times[0]:%Y-%m-%d}..{times[-1]:%Y-%m-%d} ({len(times)} days)",
                },
            )
    return reports


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(horizon_hours: int, settings: Settings | None = None, use_cache: bool = True) -> BaseForecaster:
    """Load a trained forecaster (cached per process)."""
    settings = settings or get_settings()
    path = model_artifact_path(Path(settings.model_path), settings.sea_ice_model_name, horizon_hours)
    key = str(path)
    if use_cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = BaseForecaster.load(path)
    _MODEL_CACHE[key] = model
    return model


def available_models(settings: Settings | None = None) -> dict[int, dict]:
    """Describe the trained artifacts currently on disk."""
    settings = settings or get_settings()
    out: dict[int, dict] = {}
    for horizon in settings.forecast_horizons_hours:
        path = model_artifact_path(Path(settings.model_path), settings.sea_ice_model_name, horizon)
        if not path.exists():
            continue
        try:
            model = load_model(horizon, settings)
            out[horizon] = model.describe()
        except Exception as exc:  # pragma: no cover - corrupt artifact
            out[horizon] = {"error": f"{exc.__class__.__name__}: {exc}"}
    return out


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@dataclass
class ForecastResult:
    """A single-horizon gridded forecast plus its provenance."""

    horizon_hours: int
    issued_at: datetime
    valid_at: datetime
    prediction: np.ndarray
    uncertainty: np.ndarray
    mask: np.ndarray
    grid: GridSpec
    model_name: str
    model_version: str
    algorithm: str
    data_mode: str
    metrics: dict = field(default_factory=dict)

    @property
    def mean_concentration(self) -> float:
        vals = self.prediction[self.mask]
        return float(np.nanmean(vals)) if vals.size else float("nan")

    @property
    def ice_covered_fraction(self) -> float:
        vals = self.prediction[self.mask]
        return float(np.mean(vals >= 0.15)) if vals.size else float("nan")


def _latest_history(settings: Settings, grid: GridSpec):
    """Load the history archive, or rebuild a short one if it is missing."""
    try:
        times, variables, hist_grid, _attrs = preprocessing.load_history(settings)
        if hist_grid.shape == grid.shape:
            return times, variables["sea_ice_concentration"], hist_grid
        log.warning("History grid %s differs from configured grid %s; rebuilding", hist_grid.shape, grid.shape)
    except DataUnavailableError:
        log.warning("No history archive found; building a short one for inference")
    days = max(max(preprocessing.LAG_DAYS) + 5, 25)
    times, ice = preprocessing.build_sea_ice_history(days=days, grid=grid, settings=settings)
    return times, ice, grid


def forecast(
    horizon_hours: int,
    settings: Settings | None = None,
    grid: GridSpec | None = None,
    allow_baseline_fallback: bool = True,
) -> ForecastResult:
    """Produce one gridded sea-ice forecast from the latest observations."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    horizon_hours = validate_forecast_hours(horizon_hours, settings.forecast_horizons_hours)
    horizon_days = max(1, int(round(horizon_hours / 24)))

    times, ice, grid = _latest_history(settings, grid)
    weather_pack = preprocessing.load_weather_history(settings)
    weather = weather_pack[1] if weather_pack and len(weather_pack[0]) == len(times) else None
    drift_pack = preprocessing.load_ice_drift_history(settings)
    drift = drift_pack[1] if drift_pack and len(drift_pack[0]) == len(times) else None

    model: BaseForecaster | None
    try:
        model = load_model(horizon_hours, settings)
    except FileNotFoundError:
        if not allow_baseline_fallback:
            raise
        log.warning(
            "No trained artifact for %dh; falling back to the persistence baseline. "
            "Run 'python scripts/train_models.py' for the learned model.",
            horizon_hours,
        )
        model = None
    except Exception as exc:  # noqa: BLE001 - a bad artifact must not take the API down
        # joblib artifacts are pickled against a specific scikit-learn version.
        # A checkout with a different version cannot unpickle them, and the
        # right answer is a working baseline plus a clear instruction - not a
        # 500 on an endpoint that has perfectly good observations to fall back
        # on.
        if not allow_baseline_fallback:
            raise
        log.warning(
            "Trained artifact for %dh could not be loaded (%s: %s); using the persistence "
            "baseline. Retrain with 'python scripts/train_models.py' to rebuild it for this "
            "environment.",
            horizon_hours, exc.__class__.__name__, exc,
        )
        model = None

    X, mask, names = preprocessing.build_inference_features(
        times, ice, grid, horizon_days, weather=weather, drift=drift,
        expected_names=model.feature_names if model else None,
    )
    if model is None:
        # The baseline reads its prediction straight out of the feature matrix,
        # so it adopts whatever column set inference just produced.
        model = PersistenceForecaster(horizon_hours, names)
        model.data_mode = settings.data_mode
        model.metrics = {"note": "untrained persistence baseline; no validation metrics available"}
    elif model.feature_names and X.shape[1] != len(model.feature_names):
        # A model selected on the lean feature set wants only the leading
        # sea-ice columns; the extra blocks are always appended after them, so
        # a prefix slice reproduces exactly what it was trained on.
        if names[: len(model.feature_names)] == list(model.feature_names):
            X = X[:, : len(model.feature_names)]
        else:
            raise ValidationError(
                f"Feature mismatch: model expects {len(model.feature_names)} features "
                f"({model.feature_names[:4]}...), inference produced {X.shape[1]} ({names[:4]}...). "
                f"Retrain with the current data configuration."
            )

    flat_pred = model.predict(X)
    flat_unc = model.predict_uncertainty(flat_pred)
    prediction = np.full(grid.shape, np.nan)
    uncertainty = np.full(grid.shape, np.nan)
    prediction[mask] = flat_pred
    uncertainty[mask] = flat_unc

    issued_at = times[-1]
    return ForecastResult(
        horizon_hours=horizon_hours,
        issued_at=issued_at,
        valid_at=issued_at + timedelta(hours=horizon_hours),
        prediction=prediction,
        uncertainty=uncertainty,
        mask=mask,
        grid=grid,
        model_name=settings.sea_ice_model_name,
        model_version=getattr(model, "version", "1.0.0"),
        algorithm=model.algorithm,
        data_mode=settings.data_mode,
        metrics=model.metrics,
    )


def forecast_all(
    settings: Settings | None = None,
    grid: GridSpec | None = None,
    horizons: Sequence[int] | None = None,
) -> dict[int, ForecastResult]:
    settings = settings or get_settings()
    horizons = horizons or settings.forecast_horizons_hours
    return {h: forecast(h, settings, grid) for h in horizons}


def persist_forecast(session, result: ForecastResult) -> int:
    """Write a forecast into the ``forecasts`` table."""
    lat2d, lon2d = result.grid.meshgrid()
    rows = []
    ii, jj = np.where(result.mask)
    for i, j in zip(ii, jj):
        value = result.prediction[i, j]
        if not np.isfinite(value):
            continue
        unc = result.uncertainty[i, j]
        rows.append(
            {
                "issued_at": result.issued_at,
                "valid_at": result.valid_at,
                "horizon_hours": result.horizon_hours,
                "latitude": float(lat2d[i, j]),
                "longitude": float(lon2d[i, j]),
                "predicted_concentration": float(value),
                "uncertainty": float(unc) if np.isfinite(unc) else None,
                "model_name": result.model_name,
                "model_version": result.model_version,
                "data_mode": result.data_mode,
            }
        )
    n = repo.save_forecasts(session, rows)
    repo.prune_forecasts(session, keep_issues=3)
    log.info("Persisted %d forecast cells for %dh", n, result.horizon_hours)
    return n


def run_forecast_cycle(session, settings: Settings | None = None, grid: GridSpec | None = None) -> dict:
    """Generate and persist forecasts for every configured horizon."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    summary: dict[str, dict] = {}
    for horizon in settings.forecast_horizons_hours:
        try:
            result = forecast(horizon, settings, grid)
            n = persist_forecast(session, result)
            summary[str(horizon)] = {
                "status": "ok",
                "cells": n,
                "algorithm": result.algorithm,
                "issued_at": result.issued_at.isoformat(),
                "valid_at": result.valid_at.isoformat(),
                "mean_concentration": result.mean_concentration,
                "ice_covered_fraction": result.ice_covered_fraction,
            }
        except Exception as exc:  # noqa: BLE001 - reported per horizon
            log.error("Forecast cycle failed for %dh: %s", horizon, exc)
            summary[str(horizon)] = {"status": "failed", "error": f"{exc.__class__.__name__}: {exc}"}
    return summary
