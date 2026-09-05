"""Sea-ice observation and forecast endpoints."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import (
    BBoxDep,
    GridDep,
    SessionDep,
    SettingsDep,
    field_to_lists,
    grid_info,
    provenance,
    raise_unavailable,
)
from app.database import repositories as repo
from app.schemas.sea_ice import (
    ExtentPoint,
    SeaIceCell,
    SeaIceCurrentResponse,
    SeaIceExtentResponse,
    SeaIceForecastResponse,
    SeaIceGrid,
    SeaIceStatistics,
)
from app.services import environment as envsvc
from app.services import landmask
from app.utils.logging import get_logger
from app.utils.validation import DataUnavailableError, ValidationError, validate_forecast_hours

#: Starlette deprecated its 422 constant name; the numeric code is stable.
HTTP_422 = 422

router = APIRouter(prefix="/sea-ice", tags=["sea-ice"])
log = get_logger("api.sea_ice")


def _cells_from_field(grid, field, bbox, stride, limit, land=None) -> list[SeaIceCell]:
    rows = range(0, grid.shape[0], max(1, stride))
    cols = range(0, grid.shape[1], max(1, stride))
    out: list[SeaIceCell] = []
    for i in rows:
        lat = float(grid.lats[i])
        if bbox and not (bbox[0] <= lat <= bbox[1]):
            continue
        for j in cols:
            lon = float(grid.lons[j])
            if bbox and not (bbox[2] <= lon <= bbox[3]):
                continue
            value = field[i, j]
            is_land = bool(land[i, j]) if land is not None else False
            if not np.isfinite(value) and not is_land:
                continue
            out.append(
                SeaIceCell(
                    latitude=lat,
                    longitude=lon,
                    concentration=round(float(value), 4) if np.isfinite(value) else 0.0,
                    is_land=is_land,
                    quality_flag="land" if is_land else "ok",
                )
            )
            if len(out) >= limit:
                return out
    return out


@router.get(
    "/current",
    response_model=SeaIceCurrentResponse,
    summary="Latest observed sea-ice conditions",
    description=(
        "Returns the most recent gridded sea-ice concentration analysis held by POLARIS.\n\n"
        "Use `format=grid` (default) for a compact array payload suitable for map rendering, or "
        "`format=cells` for one object per grid cell. `stride` subsamples the grid, and `bbox` "
        "restricts the area."
    ),
)
def current_sea_ice(
    session: SessionDep,
    settings: SettingsDep,
    grid: GridDep,
    bbox: BBoxDep,
    response_format: Literal["grid", "cells", "both"] = Query(
        default="grid", alias="format", description="Payload shape"
    ),
    stride: int = Query(default=1, ge=1, le=20, description="Return every Nth row/column"),
    limit: int = Query(default=6000, ge=1, le=60000, description="Maximum cells when format includes 'cells'"),
) -> SeaIceCurrentResponse:
    env = envsvc.get_environment(session, settings, grid)
    if not env.has_sea_ice:
        raise_unavailable(
            DataUnavailableError("No sea-ice observations have been ingested yet."),
            hint="Run 'python scripts/download_data.py' (or seed_demo_data.py) to populate the database.",
        )

    field = env.fields["sea_ice_concentration"]
    stats = env.statistics()
    land = landmask.land_fraction(grid) >= 0.5

    payload_cells = None
    payload_field = None
    if response_format in ("cells", "both"):
        payload_cells = _cells_from_field(grid, field, bbox, stride, limit, land)
    if response_format in ("grid", "both"):
        rows = list(range(0, grid.shape[0], stride))
        cols = list(range(0, grid.shape[1], stride))
        payload_field = SeaIceGrid(
            lats=[float(grid.lats[i]) for i in rows],
            lons=[float(grid.lons[j]) for j in cols],
            values=field_to_lists(field[np.ix_(rows, cols)]),
        )

    extent = repo.get_extent_series(session, days=1)
    return SeaIceCurrentResponse(
        observed_at=env.sea_ice_observed_at,
        grid=grid_info(grid),
        statistics=SeaIceStatistics(**{k: v for k, v in stats.items() if k in SeaIceStatistics.model_fields}),
        cells=payload_cells,
        field=payload_field,
        hemispheric_extent_million_km2=extent[-1].extent_million_km2 if extent else None,
        provenance=provenance(settings, env.sources, env.sea_ice_observed_at),
    )


@router.get(
    "/forecast",
    response_model=SeaIceForecastResponse,
    summary="Sea-ice concentration forecast",
    description=(
        "Machine-learning forecast of Antarctic sea-ice concentration at 24, 48 or 72 hours.\n\n"
        "`validation_metrics` carries the model's **real** held-out scores, including its skill "
        "against a persistence baseline. If no artifact has been trained the API falls back to that "
        "persistence baseline and says so in `algorithm`."
    ),
)
def sea_ice_forecast(
    session: SessionDep,
    settings: SettingsDep,
    grid: GridDep,
    forecast_hours: int = Query(default=24, description="Forecast lead time in hours (24, 48 or 72)"),
    region: str | None = Query(
        default=None, description="Optional 'lat_min,lat_max,lon_min,lon_max' subset of the domain"
    ),
    response_format: Literal["grid", "cells", "both"] = Query(default="grid", alias="format"),
    stride: int = Query(default=1, ge=1, le=20),
    limit: int = Query(default=6000, ge=1, le=60000),
    include_uncertainty: bool = Query(default=True),
) -> SeaIceForecastResponse:
    try:
        horizon = validate_forecast_hours(forecast_hours, settings.forecast_horizons_hours)
    except ValidationError as exc:
        raise HTTPException(status_code=HTTP_422, detail=str(exc)) from exc

    bbox = None
    if region:
        try:
            lat_min, lat_max, lon_min, lon_max = (float(p) for p in region.split(","))
            bbox = (lat_min, lat_max, lon_min, lon_max)
        except ValueError as exc:
            raise HTTPException(
                status_code=HTTP_422,
                detail="region must be 'lat_min,lat_max,lon_min,lon_max'",
            ) from exc

    try:
        fc = envsvc.get_forecast(horizon, settings, grid)
    except (DataUnavailableError, ValidationError) as exc:
        raise_unavailable(
            exc,
            hint="Build the history archive with 'python scripts/preprocess_data.py', then train with "
                 "'python scripts/train_models.py'.",
        )

    ocean = landmask.navigable_mask(grid)
    values = fc.prediction[ocean & np.isfinite(fc.prediction)]
    stats = SeaIceStatistics(
        mean_concentration=float(values.mean()) if values.size else None,
        max_concentration=float(values.max()) if values.size else None,
        ice_covered_fraction=float(np.mean(values >= 0.15)) if values.size else None,
        ocean_cells=int(ocean.sum()),
    )

    rows = list(range(0, grid.shape[0], stride))
    cols = list(range(0, grid.shape[1], stride))
    payload_field = payload_cells = payload_unc = None
    if response_format in ("grid", "both"):
        payload_field = SeaIceGrid(
            lats=[float(grid.lats[i]) for i in rows],
            lons=[float(grid.lons[j]) for j in cols],
            values=field_to_lists(fc.prediction[np.ix_(rows, cols)]),
        )
        if include_uncertainty:
            payload_unc = SeaIceGrid(
                lats=payload_field.lats,
                lons=payload_field.lons,
                values=field_to_lists(fc.uncertainty[np.ix_(rows, cols)]),
            )
    if response_format in ("cells", "both"):
        payload_cells = _cells_from_field(grid, fc.prediction, bbox, stride, limit)

    return SeaIceForecastResponse(
        issued_at=fc.issued_at,
        valid_at=fc.valid_at,
        forecast_hours=horizon,
        model_name=fc.model_name,
        model_version=fc.model_version,
        algorithm=fc.algorithm,
        grid=grid_info(grid),
        statistics=stats,
        cells=payload_cells,
        field=payload_field,
        uncertainty=payload_unc,
        validation_metrics=fc.metrics,
        provenance=provenance(settings, [fc.model_name], fc.issued_at, fc.valid_at),
    )


@router.get(
    "/extent",
    response_model=SeaIceExtentResponse,
    summary="Hemispheric sea-ice extent index",
    description="Daily Antarctic sea-ice extent/area time series (NSIDC Sea Ice Index in real mode).",
)
def sea_ice_extent(
    session: SessionDep,
    settings: SettingsDep,
    days: int = Query(default=90, ge=1, le=1000),
) -> SeaIceExtentResponse:
    series = repo.get_extent_series(session, days=days)
    if not series:
        raise_unavailable(
            DataUnavailableError("No sea-ice extent index has been ingested."),
            hint="Run 'python scripts/download_data.py'.",
        )
    return SeaIceExtentResponse(
        hemisphere="south",
        series=[
            ExtentPoint(
                observed_at=s.observed_at,
                extent_million_km2=s.extent_million_km2,
                area_million_km2=s.area_million_km2,
            )
            for s in series
        ],
        provenance=provenance(settings, [s.source for s in series], series[-1].observed_at),
    )
