"""Shared FastAPI dependencies and response helpers."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database.connection import get_db
from app.schemas.common import GridInfo, Provenance
from app.services import environment as envsvc
from app.utils.geo import GridSpec, grid_from_settings
from app.utils.validation import DataUnavailableError, ValidationError

#: Starlette deprecated its 422 constant name; the numeric code is stable.
HTTP_422 = 422

SessionDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_grid(settings: SettingsDep) -> GridSpec:
    return grid_from_settings(settings)


GridDep = Annotated[GridSpec, Depends(get_grid)]


def grid_info(grid: GridSpec) -> GridInfo:
    return GridInfo(**grid.to_dict())


def provenance(
    settings: Settings,
    sources: list[str] | None = None,
    observed_at=None,
    generated_at=None,
) -> Provenance:
    return Provenance(
        data_mode=settings.data_mode,
        sources=sorted({s for s in (sources or []) if s}),
        disclaimer=envsvc.disclaimer(settings),
        observed_at=observed_at,
        generated_at=generated_at,
    )


def bbox_param(
    bbox: str | None = Query(
        default=None,
        description="Optional filter 'lat_min,lat_max,lon_min,lon_max'",
        examples=["-70,-60,20,60"],
    ),
) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise HTTPException(
            status_code=HTTP_422,
            detail="bbox must have exactly four comma-separated values: lat_min,lat_max,lon_min,lon_max",
        )
    try:
        lat_min, lat_max, lon_min, lon_max = (float(p) for p in parts)
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTP_422,
            detail=f"bbox values must be numeric: {bbox!r}",
        ) from exc
    if lat_min >= lat_max or lon_min >= lon_max:
        raise HTTPException(
            status_code=HTTP_422,
            detail="bbox requires lat_min < lat_max and lon_min < lon_max",
        )
    return lat_min, lat_max, lon_min, lon_max


BBoxDep = Annotated[tuple[float, float, float, float] | None, Depends(bbox_param)]


def field_to_lists(field: np.ndarray, decimals: int = 4) -> list[list[float | None]]:
    """JSON-safe nested lists with NaN rendered as ``null``."""
    arr = np.asarray(field, dtype=float)
    rounded = np.round(arr, decimals)
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in rounded]


def subsample(grid: GridSpec, stride: int) -> tuple[list[int], list[int]]:
    """Row/column indices for a strided view of the grid."""
    stride = max(1, int(stride))
    return list(range(0, grid.shape[0], stride)), list(range(0, grid.shape[1], stride))


def raise_unavailable(exc: Exception, hint: str | None = None):
    """Turn a data-availability failure into a 503 with actionable text."""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"message": str(exc), "hint": hint or "Run the ingestion scripts, then retry."},
    )


def raise_bad_request(exc: Exception):
    raise HTTPException(status_code=HTTP_422, detail=str(exc))
