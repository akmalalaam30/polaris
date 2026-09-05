"""Route optimisation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import GridDep, SessionDep, SettingsDep, provenance, raise_unavailable
from app.config import STATIONS
from app.database import repositories as repo
from app.schemas.route import (
    RouteOptimizeRequest,
    RouteOptimizeResponse,
    StationListResponse,
    StationOut,
)
from app.services import environment as envsvc
from app.services import risk_engine as rk
from app.services import route_optimizer as ro
from app.utils.logging import get_logger
from app.utils.validation import DataUnavailableError, ValidationError

#: Starlette deprecated its 422 constant name; the numeric code is stable.
HTTP_422 = 422

router = APIRouter(tags=["navigation"])
log = get_logger("api.navigation")


def _risk_weights(settings, overrides) -> rk.RiskWeights:
    weights = rk.RiskWeights.from_settings(settings)
    if overrides is None:
        return weights
    for key in ("sea_ice", "iceberg", "weather", "ocean", "constraint"):
        value = getattr(overrides, key, None)
        if value is not None:
            setattr(weights, key, float(value))
    if sum(weights.as_dict().values()) <= 0:
        raise HTTPException(
            status_code=HTTP_422,
            detail="risk_weights must not all be zero",
        )
    return weights


@router.post(
    "/route/optimize",
    response_model=RouteOptimizeResponse,
    summary="Optimise navigation routes",
    description=(
        "Computes up to three routes between two points over the current (or forecast) risk grid:\n\n"
        "* **shortest** - minimum distance\n"
        "* **safest** - minimum accumulated risk exposure\n"
        "* **polaris** - balances distance, risk and estimated fuel, shaded by the vessel's risk "
        "tolerance\n\n"
        "Endpoints may be given as coordinates or as a station key (`bharati`, `maitri`, "
        "`cape_town`). A land endpoint is snapped to the nearest navigable water and the snap "
        "distance is reported.\n\n"
        "Duration and fuel are **model estimates** from the documented speed model. Any comparison "
        "between profiles is computed from these same numbers. A route whose `status` is `degraded` "
        "could not be found within the navigability constraints and crosses restricted water."
    ),
    responses={
        422: {"description": "Invalid request (bad coordinates, unknown station, outside the domain)"},
        503: {"description": "Required data has not been ingested yet"},
    },
)
def optimize_route(
    payload: RouteOptimizeRequest,
    session: SessionDep,
    settings: SettingsDep,
    grid: GridDep,
) -> RouteOptimizeResponse:
    prefs = payload.preferences
    vessel = rk.VesselProfile(**payload.vessel.model_dump())
    weights = _risk_weights(settings, prefs.risk_weights)

    # Per-request cost-weight overrides, applied without mutating global config.
    tuned = settings.model_copy(
        update={
            k: v
            for k, v in {
                "route_distance_weight": prefs.distance_weight,
                "route_risk_weight": prefs.risk_weight,
                "route_fuel_weight": prefs.fuel_weight,
            }.items()
            if v is not None
        }
    )

    try:
        risk_result = envsvc.get_risk_grid(
            session, settings, grid, vessel, weights, horizon_hours=prefs.forecast_hours
        )
        env = envsvc.get_environment(session, settings, grid)
        icebergs = None
        if prefs.include_iceberg_analysis:
            icebergs = envsvc.get_iceberg_context(
                session, settings, grid, horizon_hours=max(prefs.forecast_hours, 72)
            ).records
        result = ro.optimize_routes(
            risk_result,
            payload.start.model_dump(exclude_none=True),
            payload.destination.model_dump(exclude_none=True),
            fields=env.fields,
            vessel=vessel,
            profiles=list(prefs.profiles),
            settings=tuned,
            icebergs=icebergs,
            algorithm=prefs.algorithm,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=HTTP_422, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        raise_unavailable(exc, hint="Ingest sea-ice, weather and ocean data, then retry.")

    if not result["routes"]:
        raise HTTPException(
            status_code=HTTP_422,
            detail={
                "message": "No route could be computed between the requested points.",
                "errors": result["errors"],
                "hint": "Check that both endpoints are inside the domain and that navigable water connects them.",
            },
        )

    persisted: list[int] = []
    if payload.persist:
        try:
            persisted = ro.persist_routes(session, result, settings.data_mode)
            session.commit()
        except Exception as exc:  # noqa: BLE001 - persistence must not fail the response
            session.rollback()
            log.error("Route persistence failed: %s", exc)

    return RouteOptimizeResponse(
        **result,
        persisted_route_ids=persisted,
        provenance=provenance(settings, env.sources, env.observed_at, result["computed_at"]),
    )


@router.get(
    "/navigation/stations",
    response_model=StationListResponse,
    summary="Known reference stations",
    description="Stations that can be used as route endpoints by key, with their distance to navigable water.",
)
def stations(session: SessionDep, settings: SettingsDep, grid: GridDep) -> StationListResponse:
    out: list[StationOut] = []
    nav = None
    try:
        risk_result = envsvc.get_risk_grid(session, settings, grid, horizon_hours=0)
        nav = ro.build_navigation_graph(risk_result, None, risk_result.vessel, allow_high_risk=True)
    except Exception as exc:  # noqa: BLE001 - the list is still useful without the graph
        log.warning("Station snap distances unavailable: %s", exc)

    for key, meta in STATIONS.items():
        in_domain = grid.contains(meta["latitude"], meta["longitude"])
        snap = None
        if nav is not None and in_domain and nav.n_nodes:
            _node, snap_km = nav.nearest_node(meta["latitude"], meta["longitude"])
            snap = round(snap_km, 1)
        out.append(
            StationOut(
                key=key,
                name=meta["name"],
                operator=meta["operator"],
                region=meta["region"],
                latitude=meta["latitude"],
                longitude=meta["longitude"],
                in_domain=in_domain,
                nearest_navigable_km=snap,
            )
        )
    return StationListResponse(stations=out, domain=grid.to_dict())


@router.get(
    "/navigation/landmask",
    summary="Antarctic land mask as GeoJSON",
    description=(
        "The land/ice-shelf mask used by the router and the risk engine, as GeoJSON "
        "polygons on the analysis grid. "
        "It is derived from the land and coast flags published with the NSIDC G02135 sea-ice "
        "product. Serving it lets a client draw the coastline without contacting a tile server, "
        "which matters in the field and guarantees the map shows the *same* coastline the "
        "routing actually respected."
    ),
)
def landmask_geojson(settings: SettingsDep, grid: GridDep) -> dict:
    from app.services import landmask

    land = ~landmask.navigable_mask(grid)
    half_lat, half_lon = grid.dlat / 2.0, grid.dlon / 2.0
    features = []
    # Merge consecutive land cells in each row into a single rectangle, which
    # cuts the payload by roughly an order of magnitude.
    for i in range(grid.shape[0]):
        j = 0
        while j < grid.shape[1]:
            if not land[i, j]:
                j += 1
                continue
            start = j
            while j < grid.shape[1] and land[i, j]:
                j += 1
            lat = float(grid.lats[i])
            lon0 = float(grid.lons[start]) - half_lon
            lon1 = float(grid.lons[j - 1]) + half_lon
            features.append(
                {
                    "type": "Feature",
                    "properties": {"kind": "land"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [round(lon0, 4), round(lat - half_lat, 4)],
                            [round(lon1, 4), round(lat - half_lat, 4)],
                            [round(lon1, 4), round(lat + half_lat, 4)],
                            [round(lon0, 4), round(lat + half_lat, 4)],
                            [round(lon0, 4), round(lat - half_lat, 4)],
                        ]],
                    },
                }
            )
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "source": "NSIDC G02135 land/coast flags, regridded to the POLARIS analysis grid",
            "grid": grid.to_dict(),
            "land_cells": int(land.sum()),
        },
    }


@router.get(
    "/navigation/routes/{request_id}",
    summary="Retrieve stored routes by request id",
    responses={404: {"description": "No stored routes for this request id"}},
)
def stored_routes(request_id: str, session: SessionDep) -> dict:
    rows = repo.get_routes_by_request(session, request_id)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No stored routes for request_id {request_id!r}",
        )
    return {
        "request_id": request_id,
        "routes": [
            {
                "id": r.id,
                "profile": r.profile,
                "computed_at": r.computed_at,
                "distance_km": r.distance_km,
                "duration_hours": r.duration_hours,
                "fuel_tonnes": r.fuel_tonnes,
                "mean_risk": r.mean_risk,
                "max_risk": r.max_risk,
                "status": r.status,
                "geometry": r.geometry,
                "parameters": r.parameters,
                "notes": r.notes,
            }
            for r in rows
        ],
    }


@router.get("/navigation/routes", summary="Recently computed routes")
def recent_routes(session: SessionDep, limit: int = Query(default=10, ge=1, le=100)) -> dict:
    rows = repo.recent_routes(session, limit=limit)
    return {
        "count": len(rows),
        "routes": [
            {
                "id": r.id,
                "request_id": r.request_id,
                "profile": r.profile,
                "computed_at": r.computed_at,
                "start": {"latitude": r.start_latitude, "longitude": r.start_longitude, "name": r.start_name},
                "destination": {"latitude": r.end_latitude, "longitude": r.end_longitude, "name": r.end_name},
                "distance_km": r.distance_km,
                "duration_hours": r.duration_hours,
                "fuel_tonnes": r.fuel_tonnes,
                "mean_risk": r.mean_risk,
                "status": r.status,
            }
            for r in rows
        ],
    }
