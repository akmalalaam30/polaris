"""Navigation risk endpoints."""

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
    provenance,
    raise_unavailable,
)
from app.schemas.risk import (
    PointRiskResponse,
    RiskCell,
    RiskGridSummary,
    RiskHotspot,
    RiskMapResponse,
)
from app.services import environment as envsvc
from app.services import landmask
from app.services import risk_engine as rk
from app.utils.logging import get_logger
from app.utils.validation import DataUnavailableError, ValidationError

#: Starlette deprecated its 422 constant name; the numeric code is stable.
HTTP_422 = 422

router = APIRouter(prefix="/risk", tags=["risk"])
log = get_logger("api.risk")


#: 0 means "use the current analysis"; the rest are the trained forecast leads.
def _validate_horizon(forecast_hours: int, settings) -> int:
    """Query strings arrive as text, which a Literal[int] annotation rejects.

    Validating explicitly keeps the parameter usable and produces a message that
    names the accepted values.
    """
    allowed = (0,) + tuple(settings.forecast_horizons_hours)
    if forecast_hours not in allowed:
        raise HTTPException(
            status_code=HTTP_422,
            detail=(
                f"forecast_hours must be one of {sorted(allowed)} "
                f"(0 uses the current analysis), got {forecast_hours}"
            ),
        )
    return int(forecast_hours)


def _vessel(ice_class: str, draft_m: float, risk_tolerance: float) -> rk.VesselProfile:
    try:
        return rk.VesselProfile(ice_class=ice_class, draft_m=draft_m, risk_tolerance=risk_tolerance)
    except Exception as exc:  # pragma: no cover - VesselProfile is forgiving
        raise HTTPException(status_code=HTTP_422, detail=str(exc)) from exc


@router.get(
    "/map",
    response_model=RiskMapResponse,
    summary="Spatial navigation risk grid",
    description=(
        "Generates the navigation risk grid over the Antarctic domain.\n\n"
        "Risk is normalised to **0-1** (0 = negligible, 1 = extreme) and combines sea-ice, iceberg, "
        "weather, ocean and navigational-constraint components under configurable weights. If a "
        "layer has no data it is listed in `summary.missing_layers` and the total is renormalised "
        "over the layers that were available.\n\n"
        "Set `forecast_hours` to condition the sea-ice component on the forecast instead of the "
        "current analysis. This is decision support, not a safety guarantee."
    ),
)
def risk_map(
    session: SessionDep,
    settings: SettingsDep,
    grid: GridDep,
    bbox: BBoxDep,
    forecast_hours: int = Query(
        default=0, description="0 = current analysis; otherwise a trained forecast lead (24, 48, 72)"
    ),
    response_format: Literal["grid", "cells", "both"] = Query(default="grid", alias="format"),
    stride: int = Query(default=1, ge=1, le=20),
    limit: int = Query(default=6000, ge=1, le=60000),
    min_risk: float | None = Query(default=None, ge=0.0, le=1.0, description="Only cells at or above this risk"),
    ice_class: str = Query(default="1a_super", description="Vessel ice class used for the assessment"),
    draft_m: float = Query(default=7.0, gt=0, le=25),
    risk_tolerance: float = Query(default=0.3, ge=0.0, le=1.0),
    top_hotspots: int = Query(default=10, ge=0, le=100),
    persist: bool = Query(default=False, description="Store the generated grid in the database"),
) -> RiskMapResponse:
    forecast_hours = _validate_horizon(forecast_hours, settings)
    vessel = _vessel(ice_class, draft_m, risk_tolerance)
    try:
        result = envsvc.get_risk_grid(session, settings, grid, vessel, horizon_hours=forecast_hours)
    except (DataUnavailableError, ValidationError) as exc:
        raise_unavailable(exc, hint="Ingest observations first, then retry.")

    if persist:
        rk.persist_risk_grid(session, result)
        session.commit()

    ocean = landmask.navigable_mask(grid)
    rows = list(range(0, grid.shape[0], stride))
    cols = list(range(0, grid.shape[1], stride))

    payload_cells = None
    if response_format in ("cells", "both"):
        payload_cells = []
        for i in rows:
            lat = float(grid.lats[i])
            if bbox and not (bbox[0] <= lat <= bbox[1]):
                continue
            for j in cols:
                lon = float(grid.lons[j])
                if bbox and not (bbox[2] <= lon <= bbox[3]):
                    continue
                if not ocean[i, j]:
                    continue
                total = float(result.total[i, j])
                if min_risk is not None and total < min_risk:
                    continue
                payload_cells.append(
                    RiskCell(
                        latitude=lat,
                        longitude=lon,
                        sea_ice_risk=round(float(result.components["sea_ice"][i, j]), 4),
                        iceberg_risk=round(float(result.components["iceberg"][i, j]), 4),
                        weather_risk=round(float(result.components["weather"][i, j]), 4),
                        ocean_risk=round(float(result.components["ocean"][i, j]), 4),
                        constraint_risk=round(float(result.components["constraint"][i, j]), 4),
                        total_risk=round(total, 4),
                        navigable=bool(result.navigable[i, j]),
                    )
                )
                if len(payload_cells) >= limit:
                    break
            if payload_cells is not None and len(payload_cells) >= limit:
                break

    total_field = lats_out = lons_out = None
    if response_format in ("grid", "both"):
        masked = np.where(ocean, result.total, np.nan)
        total_field = field_to_lists(masked[np.ix_(rows, cols)])
        lats_out = [float(grid.lats[i]) for i in rows]
        lons_out = [float(grid.lons[j]) for j in cols]

    hotspots: list[RiskHotspot] = []
    if top_hotspots:
        flat = np.where(ocean, result.total, -np.inf).ravel()
        for idx in np.argsort(flat)[::-1][:top_hotspots]:
            if not np.isfinite(flat[idx]):
                continue
            i, j = divmod(int(idx), grid.shape[1])
            contributions = {k: float(v[i, j]) for k, v in result.components.items()}
            hotspots.append(
                RiskHotspot(
                    latitude=float(grid.lats[i]),
                    longitude=float(grid.lons[j]),
                    total_risk=round(float(result.total[i, j]), 4),
                    dominant_component=max(contributions, key=contributions.get),
                )
            )

    return RiskMapResponse(
        summary=RiskGridSummary(**result.summary()),
        cells=payload_cells,
        total_risk_field=total_field,
        lats=lats_out,
        lons=lons_out,
        hotspots=hotspots,
        provenance=provenance(
            settings,
            envsvc.get_environment(session, settings, grid).sources,
            result.valid_at,
            result.generated_at,
        ),
    )


@router.get(
    "/point",
    response_model=PointRiskResponse,
    summary="Risk breakdown at a single position",
    description="Component-by-component risk at one coordinate, for a chosen vessel profile.",
)
def risk_at_point(
    session: SessionDep,
    settings: SettingsDep,
    grid: GridDep,
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=360),
    forecast_hours: int = Query(default=0, description="0 = current analysis; else 24, 48 or 72"),
    ice_class: str = Query(default="1a_super"),
    draft_m: float = Query(default=7.0, gt=0, le=25),
    risk_tolerance: float = Query(default=0.3, ge=0.0, le=1.0),
) -> PointRiskResponse:
    forecast_hours = _validate_horizon(forecast_hours, settings)
    vessel = _vessel(ice_class, draft_m, risk_tolerance)
    if not grid.contains(latitude, longitude):
        raise HTTPException(
            status_code=HTTP_422,
            detail=(
                f"({latitude}, {longitude}) is outside the POLARIS domain "
                f"(lat {grid.lat_min}..{grid.lat_max}, lon {grid.lon_min}..{grid.lon_max})"
            ),
        )
    try:
        result = envsvc.get_risk_grid(session, settings, grid, vessel, horizon_hours=forecast_hours)
        point = rk.assess_point(result, latitude, longitude)
    except (DataUnavailableError, ValidationError) as exc:
        raise_unavailable(exc)
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_422, detail=str(exc)) from exc

    return PointRiskResponse(
        **point,
        generated_at=result.generated_at,
        valid_at=result.valid_at,
        horizon_hours=result.horizon_hours,
        provenance=provenance(settings, result.available_layers, result.valid_at, result.generated_at),
    )
