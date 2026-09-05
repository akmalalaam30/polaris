"""Input and dataset validation helpers.

These are used both by the ingestion layer (to reject malformed upstream files)
and by the API layer (to produce meaningful 4xx errors instead of tracebacks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when input data fails validation."""


class DataUnavailableError(RuntimeError):
    """Raised when a required dataset is missing or empty."""


@dataclass
class ValidationReport:
    """Outcome of validating an ingested dataset."""

    source: str
    n_input: int = 0
    n_valid: int = 0
    n_dropped_invalid: int = 0
    n_dropped_duplicate: int = 0
    n_missing_filled: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_valid > 0 and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_input": self.n_input,
            "n_valid": self.n_valid,
            "n_dropped_invalid": self.n_dropped_invalid,
            "n_dropped_duplicate": self.n_dropped_duplicate,
            "n_missing_filled": self.n_missing_filled,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


# ---------------------------------------------------------------------------
# Scalar validation
# ---------------------------------------------------------------------------
def validate_latitude(lat: float, name: str = "latitude") -> float:
    try:
        v = float(lat)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric, got {lat!r}") from exc
    if not np.isfinite(v) or not (-90.0 <= v <= 90.0):
        raise ValidationError(f"{name} must be between -90 and 90, got {v}")
    return v


def validate_longitude(lon: float, name: str = "longitude") -> float:
    try:
        v = float(lon)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric, got {lon!r}") from exc
    if not np.isfinite(v) or not (-180.0 <= v <= 360.0):
        raise ValidationError(f"{name} must be between -180 and 360, got {v}")
    return ((v + 180.0) % 360.0) - 180.0


def validate_coordinate(lat: float, lon: float, label: str = "point") -> tuple[float, float]:
    return validate_latitude(lat, f"{label}.latitude"), validate_longitude(lon, f"{label}.longitude")


def require_southern_hemisphere(lat: float, label: str = "point", limit: float = 0.0) -> float:
    """POLARIS is an Antarctic system - reject northern-hemisphere requests."""
    lat = validate_latitude(lat, label)
    if lat > limit:
        raise ValidationError(
            f"{label} latitude {lat:.3f} is outside the Antarctic/Southern Ocean domain "
            f"(latitude must be <= {limit})"
        )
    return lat


def validate_forecast_hours(hours: int, allowed: Sequence[int] = (24, 48, 72)) -> int:
    try:
        h = int(hours)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"forecast_hours must be an integer, got {hours!r}") from exc
    if h not in allowed:
        raise ValidationError(
            f"forecast_hours must be one of {sorted(allowed)}, got {h}"
        )
    return h


def coerce_utc(value: Any, name: str = "timestamp") -> datetime:
    """Coerce assorted timestamp representations to timezone-aware UTC."""
    if value is None:
        raise ValidationError(f"{name} is required")
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)) and np.isfinite(value):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        try:
            dt = pd.to_datetime(value, utc=True, errors="raise").to_pydatetime()
        except Exception as exc:
            raise ValidationError(f"{name} could not be parsed: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# DataFrame validation
# ---------------------------------------------------------------------------
def require_columns(df: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValidationError(f"{source}: missing required column(s) {missing}; got {list(df.columns)}")


def validate_point_frame(
    df: pd.DataFrame,
    source: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str | None = "observed_at",
    dedupe_on: Sequence[str] | None = None,
    value_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate/clean a point-observation frame, returning the clean rows.

    Drops rows with unusable coordinates or timestamps, clamps out-of-range
    values into their physical bounds, and removes duplicates.
    """
    report = ValidationReport(source=source, n_input=len(df))
    if df.empty:
        report.errors.append("input frame is empty")
        return df, report

    require_columns(df, [lat_col, lon_col], source)
    out = df.copy()

    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    bad_coord = out[lat_col].isna() | out[lon_col].isna()
    bad_coord |= ~out[lat_col].between(-90, 90)
    bad_coord |= ~out[lon_col].between(-360, 360)
    report.n_dropped_invalid += int(bad_coord.sum())
    out = out[~bad_coord].copy()
    out[lon_col] = ((out[lon_col] + 180.0) % 360.0) - 180.0

    if time_col and time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
        bad_time = out[time_col].isna()
        report.n_dropped_invalid += int(bad_time.sum())
        out = out[~bad_time].copy()

    if value_ranges:
        for col, (lo, hi) in value_ranges.items():
            if col not in out.columns:
                continue
            numeric = pd.to_numeric(out[col], errors="coerce")
            n_missing = int(numeric.isna().sum())
            clamped = numeric.clip(lo, hi)
            report.n_missing_filled += n_missing
            out[col] = clamped

    if dedupe_on:
        keys = [c for c in dedupe_on if c in out.columns]
        if keys:
            before = len(out)
            out = out.drop_duplicates(subset=keys, keep="last")
            report.n_dropped_duplicate += before - len(out)

    if time_col and time_col in out.columns:
        sort_cols = [c for c in (dedupe_on or []) if c in out.columns and c != time_col]
        out = out.sort_values(sort_cols + [time_col]) if sort_cols else out.sort_values(time_col)

    out = out.reset_index(drop=True)
    report.n_valid = len(out)
    if report.n_valid == 0:
        report.errors.append("no valid rows remained after validation")
    return out, report


def validate_grid_field(
    field: np.ndarray,
    name: str,
    expected_shape: tuple[int, int] | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """Validate a 2-D field, clamping to physical bounds."""
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        raise ValidationError(f"{name}: expected a 2-D field, got shape {arr.shape}")
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValidationError(f"{name}: expected shape {expected_shape}, got {arr.shape}")
    if lo is not None or hi is not None:
        arr = np.clip(arr, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
    return arr


def clamp01(x):
    """Clamp to the unit interval (used everywhere risk scores are produced)."""
    return float(np.clip(x, 0.0, 1.0)) if np.isscalar(x) else np.clip(x, 0.0, 1.0)
