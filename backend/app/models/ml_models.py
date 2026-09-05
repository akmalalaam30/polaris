"""Machine-learning model wrappers, persistence and evaluation.

Two models are implemented:

``PersistenceForecaster``
    The standard meteorological baseline: tomorrow looks like today.  For
    sea-ice concentration at 1-3 day range this is a genuinely strong baseline,
    so every learned model is reported *against* it (skill score) rather than in
    isolation.

``GradientBoostingForecaster``
    A ``HistGradientBoostingRegressor`` over lagged concentration, local spatial
    structure, geography, season and (when available) atmospheric drivers.

    It learns the **change** in concentration over the horizon rather than its
    level.  This matters: sea-ice concentration is so persistent day to day that
    a model fitted on the level spends nearly all of its capacity reproducing
    ``conc_t`` and any residual bias it carries shows up as *negative* skill
    against persistence.  Predicting the increment makes persistence the
    zero-prediction, so the model can only move the answer where it has found
    real signal, and shrinkage degrades gracefully back to the baseline.
    Measured on real NSIDC data this turns a 24 h skill of -0.08 into a positive
    one; see the README.

    It also carries an empirical uncertainty model calibrated on the validation
    residuals.

Both share the same ``fit``/``predict``/``save``/``load`` interface so the
service layer never needs to know which is in use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.utils.logging import get_logger

log = get_logger("models.ml")

MODEL_FORMAT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / R2 / bias, computed on finite pairs only."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "bias": float("nan")}
    yt, yp = y_true[ok], y_pred[ok]
    err = yp - yt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
    }


def ice_edge_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.15) -> dict[str, float]:
    """Binary ice/no-ice skill at the conventional 15% concentration threshold."""
    yt = np.asarray(y_true, float).ravel() >= threshold
    yp = np.asarray(y_pred, float).ravel() >= threshold
    if yt.size == 0:
        return {"accuracy": float("nan"), "f1": float("nan"), "ice_fraction_true": float("nan")}
    tp = float(np.sum(yt & yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float(np.mean(yt == yp)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ice_fraction_true": float(np.mean(yt)),
    }


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """1 - MSE_model / MSE_baseline. Positive means better than the baseline."""
    if not np.isfinite(model_rmse) or not np.isfinite(baseline_rmse) or baseline_rmse <= 0:
        return float("nan")
    return float(1.0 - (model_rmse**2) / (baseline_rmse**2))


# ---------------------------------------------------------------------------
# Uncertainty calibration
# ---------------------------------------------------------------------------
@dataclass
class BinnedUncertainty:
    """Empirical 1-sigma error as a function of predicted concentration.

    Calibrated on held-out residuals - it is a measured spread, not an assumed
    confidence level.
    """

    edges: list[float] = field(default_factory=lambda: [0.0, 0.15, 0.4, 0.7, 0.9, 1.01])
    sigmas: list[float] = field(default_factory=list)
    overall: float = float("nan")

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "BinnedUncertainty":
        residual = np.asarray(y_pred, float) - np.asarray(y_true, float)
        ok = np.isfinite(residual)
        residual, pred = residual[ok], np.asarray(y_pred, float)[ok]
        self.overall = float(np.std(residual)) if residual.size else float("nan")
        self.sigmas = []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            sel = (pred >= lo) & (pred < hi)
            self.sigmas.append(float(np.std(residual[sel])) if sel.sum() >= 30 else self.overall)
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(y_pred, float)
        out = np.full(pred.shape, self.overall, dtype=float)
        if not self.sigmas:
            return out
        for k, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            out[(pred >= lo) & (pred < hi)] = self.sigmas[k]
        return out

    def to_dict(self) -> dict:
        return {"edges": self.edges, "sigmas": self.sigmas, "overall": self.overall}

    @classmethod
    def from_dict(cls, payload: dict) -> "BinnedUncertainty":
        obj = cls(edges=list(payload.get("edges", [])) or None or [0.0, 0.15, 0.4, 0.7, 0.9, 1.01])
        obj.sigmas = list(payload.get("sigmas", []))
        obj.overall = float(payload.get("overall", float("nan")))
        return obj


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class BaseForecaster:
    """Common interface for POLARIS forecast models."""

    algorithm = "base"

    def __init__(self, horizon_hours: int, feature_names: Sequence[str] | None = None) -> None:
        self.horizon_hours = int(horizon_hours)
        self.feature_names: list[str] = list(feature_names or [])
        self.version = MODEL_FORMAT_VERSION
        self.trained_at: datetime | None = None
        self.metrics: dict[str, Any] = {}
        self.n_train = 0
        self.n_val = 0
        self.uncertainty = BinnedUncertainty()
        self.data_mode = "unknown"

    # -- interface -------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseForecaster":  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_uncertainty(self, prediction: np.ndarray) -> np.ndarray:
        return self.uncertainty.predict(prediction)

    # -- helpers ---------------------------------------------------------
    def _feature_index(self, name: str) -> int:
        try:
            return self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Feature {name!r} not present in {self.feature_names}") from exc

    def describe(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "horizon_hours": self.horizon_hours,
            "version": self.version,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "n_train_samples": self.n_train,
            "n_val_samples": self.n_val,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "uncertainty": self.uncertainty.to_dict(),
            "data_mode": self.data_mode,
        }

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".json").write_text(json.dumps(self.describe(), indent=2), encoding="utf-8")
        log.info("Saved %s model for %dh -> %s", self.algorithm, self.horizon_hours, path.name)
        return path

    @staticmethod
    def load(path: Path) -> "BaseForecaster":
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        model = joblib.load(path)
        if not isinstance(model, BaseForecaster):
            raise TypeError(f"{path.name} does not contain a POLARIS forecaster")
        return model


class PersistenceForecaster(BaseForecaster):
    """Baseline: the forecast equals the most recent observed concentration."""

    algorithm = "persistence"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceForecaster":
        self.n_train = int(len(y))
        self.trained_at = datetime.now(timezone.utc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = self._feature_index("conc_t")
        return np.clip(np.asarray(X, dtype=float)[:, idx], 0.0, 1.0)


class GradientBoostingForecaster(BaseForecaster):
    """Histogram gradient boosting over the engineered sea-ice features."""

    algorithm = "hist_gradient_boosting"

    def __init__(
        self,
        horizon_hours: int,
        feature_names: Sequence[str] | None = None,
        max_iter: int = 300,
        learning_rate: float = 0.08,
        max_depth: int | None = 8,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        random_state: int = 42,
        predict_delta: bool = True,
    ) -> None:
        super().__init__(horizon_hours, feature_names)
        #: Learn the increment from ``conc_t`` rather than the level.
        self.predict_delta = predict_delta
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.estimator = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.12,
        )

    def _current(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype="float32")[:, self._feature_index("conc_t")]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingForecaster":
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        target = y - self._current(X) if self.predict_delta else y
        self.estimator.fit(X, target)
        self.n_train = int(len(y))
        self.trained_at = datetime.now(timezone.utc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        pred = self.estimator.predict(X)
        if getattr(self, "predict_delta", False):
            pred = self._current(X) + pred
        return np.clip(pred, 0.0, 1.0)

    def permutation_importance(
        self, X: np.ndarray, y: np.ndarray, n_repeats: int = 2, max_samples: int = 12000
    ) -> dict[str, float]:
        """Permutation importance on a validation subsample (RMSE increase)."""
        rng = np.random.default_rng(7)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        if len(y) > max_samples:
            sel = rng.choice(len(y), max_samples, replace=False)
            X, y = X[sel], y[sel]
        base = np.sqrt(np.mean((self.predict(X) - y) ** 2))
        out: dict[str, float] = {}
        for k, name in enumerate(self.feature_names):
            drops = []
            for _ in range(n_repeats):
                shuffled = X.copy()
                shuffled[:, k] = shuffled[rng.permutation(len(shuffled)), k]
                drops.append(np.sqrt(np.mean((self.predict(shuffled) - y) ** 2)) - base)
            out[name] = float(np.mean(drops))
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def build_forecaster(algorithm: str, horizon_hours: int, feature_names: Sequence[str]) -> BaseForecaster:
    """Factory used by the training pipeline."""
    algorithm = (algorithm or "").lower()
    if algorithm in ("persistence", "baseline"):
        return PersistenceForecaster(horizon_hours, feature_names)
    if algorithm in ("gbr", "hist_gradient_boosting", "gradient_boosting"):
        return GradientBoostingForecaster(horizon_hours, feature_names)
    raise ValueError(f"Unknown forecast algorithm {algorithm!r}")


def model_artifact_path(models_dir: Path, model_name: str, horizon_hours: int) -> Path:
    return Path(models_dir) / f"{model_name}_h{horizon_hours}.joblib"
