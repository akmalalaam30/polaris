"""Live weather and marine conditions.

Distinct from the sea-ice forecast endpoints: this is the *observational now*,
valid within the hour, rather than a model prediction of ice.

The distinction that matters here is between a reanalysis and a forecast. ERA5
is assimilated after the fact and arrives about five days late - excellent for
training on a year of consistent history, useless for telling a master what the
wind is doing tonight. These endpoints serve the ECMWF IFS forecast, and each
response says which product it came from.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.api.deps import GridDep, SessionDep, SettingsDep, provenance
from app.config import STATIONS
from app.schemas.weather import (
    LiveConditionsResponse,
    StationConditions,
    StationWeatherResponse,
)
from app.services import environment as envsvc
from app.utils.logging import get_logger
from app.utils.validation import DataUnavailableError, validate_coordinate

HTTP_422 = 422

router = APIRouter(prefix="/weather", tags=["weather"])
log = get_logger("api.weather")


@router.get(
    "/stations",
    response_model=StationWeatherResponse,
    summary="Live conditions at the research stations",
    description=(
        "Current atmospheric and marine conditions at each known station, plus an hourly "
        "outlook.\n\n"
        "Values are valid **now** (ECMWF IFS forecast), not a reanalysis. Marine fields are "
        "null where the point lies inside the pack - waves are not defined under sea ice, so "
        "that is physics rather than a gap.\n\n"
        "`freezing_spray_risk` is a qualitative flag from air temperature, wind and sea "
        "temperature; POLARIS does not compute an icing accretion rate."
    ),
    responses={503: {"description": "No live weather source is reachable"}},
)
def station_weather(
    settings: SettingsDep,
    forecast_hours: int = Query(default=72, ge=1, le=168, description="Length of the hourly outlook"),
    station: str | None = Query(default=None, description="Restrict to one station key"),
) -> StationWeatherResponse:
    from app.services.data_ingestion import LiveWeatherClient

    wanted = dict(STATIONS)
    if station:
        key = station.strip().lower().replace(" ", "_").replace("_station", "")
        if key not in wanted:
            raise HTTPException(
                status_code=404,
                detail={"message": f"Unknown station {station!r}", "known": sorted(STATIONS)},
            )
        wanted = {key: wanted[key]}

    points = [(k, v["latitude"], v["longitude"]) for k, v in wanted.items()]
    try:
        readings = LiveWeatherClient(settings).fetch_points(points, forecast_hours)
    except DataUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "hint": "Live weather needs outbound network access; the endpoint requires no key.",
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001 - upstream failure, not a bug
        log.error("Live station weather failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"message": f"Live weather provider unavailable ({exc.__class__.__name__})"},
        ) from exc

    stations = []
    for reading in readings:
        meta = wanted[reading["name"]]
        stations.append(
            StationConditions(
                **{k: v for k, v in reading.items() if k in StationConditions.model_fields},
                station=reading["name"],
                display_name=meta["name"],
                operator=meta["operator"],
                region=meta["region"],
            )
        )
    newest = max((s.observed_at for s in stations if s.observed_at), default=None)
    return StationWeatherResponse(
        count=len(stations),
        forecast_hours=forecast_hours,
        stations=stations,
        provenance=provenance(
            settings,
            sorted({s.source for s in stations if s.source}),
            newest,
            datetime.now(timezone.utc),
        ),
    )


@router.get(
    "/point",
    response_model=StationConditions,
    summary="Live conditions at an arbitrary position",
    description="Current atmospheric and marine conditions at any coordinate, with an hourly outlook.",
)
def point_weather(
    settings: SettingsDep,
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=360),
    forecast_hours: int = Query(default=48, ge=1, le=168),
) -> StationConditions:
    from app.services.data_ingestion import LiveWeatherClient

    lat, lon = validate_coordinate(latitude, longitude, "point")
    try:
        readings = LiveWeatherClient(settings).fetch_points([("point", lat, lon)], forecast_hours)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"message": f"Live weather provider unavailable ({exc.__class__.__name__})"},
        ) from exc
    if not readings:
        raise HTTPException(status_code=503, detail={"message": "No live data returned for that point"})
    reading = readings[0]
    return StationConditions(
        **{k: v for k, v in reading.items() if k in StationConditions.model_fields},
        station="point",
        display_name=f"{lat:.3f}, {lon:.3f}",
        operator=None,
        region=None,
    )


@router.get(
    "/current",
    response_model=LiveConditionsResponse,
    summary="Age and freshness of every environmental layer",
    description=(
        "How old each layer actually is. Different products have very different latencies, and "
        "a decision-support system should state that rather than implying everything is current."
    ),
)
def current_freshness(session: SessionDep, settings: SettingsDep, grid: GridDep) -> LiveConditionsResponse:
    from app.database import repositories as repo

    now = datetime.now(timezone.utc)
    env = envsvc.get_environment(session, settings, grid)
    bergs = repo.latest_iceberg_positions(session)
    berg_time = max((b.observed_at for b in bergs if b.observed_at), default=None)

    def age(stamp):
        if stamp is None:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return round((now - stamp).total_seconds() / 3600.0, 1)

    layers = {
        "sea_ice": {"observed_at": env.sea_ice_observed_at, "age_hours": age(env.sea_ice_observed_at)},
        "weather": {"observed_at": env.weather_observed_at, "age_hours": age(env.weather_observed_at)},
        "ocean": {"observed_at": env.ocean_observed_at, "age_hours": age(env.ocean_observed_at)},
        "icebergs": {"observed_at": berg_time, "age_hours": age(berg_time)},
    }
    return LiveConditionsResponse(
        generated_at=now,
        data_mode=settings.data_mode,
        layers=layers,
        provenance=provenance(settings, env.sources, env.observed_at, now),
    )
