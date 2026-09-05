"""Health, readiness and system-information endpoints."""

from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Response, status

from app.api.deps import SessionDep, SettingsDep
from app.config import APP_VERSION
from app.database import repositories as repo
from app.database.connection import check_connection, dialect_name, has_postgis
from app.schemas.common import ComponentHealth, HealthResponse
from app.services import sea_ice_forecasting as sif
from app.services.preprocessing import history_path
from app.utils.logging import get_logger

router = APIRouter(tags=["health"])
log = get_logger("api.health")

_STARTED_AT = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Backend health and version",
    description=(
        "Reports overall status plus the state of each subsystem: database, ingested data, "
        "the processed history archive and the trained forecast models. Returns HTTP 503 when "
        "the database is unreachable so orchestrators can act on it."
    ),
)
def health(session: SessionDep, settings: SettingsDep, response: Response) -> HealthResponse:
    components: list[ComponentHealth] = []

    db_ok, db_detail = check_connection()
    components.append(
        ComponentHealth(
            name="database",
            status="ok" if db_ok else "unavailable",
            detail=db_detail,
            metadata={"dialect": dialect_name(), "postgis": has_postgis()},
        )
    )

    db_summary: dict = {}
    if db_ok:
        try:
            db_summary = repo.database_summary(session)
            counts = db_summary.get("row_counts", {})
            has_ice = counts.get("sea_ice_observations", 0) > 0
            has_bergs = counts.get("iceberg_observations", 0) > 0
            components.append(
                ComponentHealth(
                    name="ingested_data",
                    status="ok" if (has_ice and has_bergs) else ("degraded" if has_ice or has_bergs else "unavailable"),
                    detail=(
                        f"sea-ice cells={counts.get('sea_ice_observations', 0)}, "
                        f"icebergs={db_summary.get('tracked_icebergs', 0)}"
                    ),
                    metadata={"latest_sea_ice": str(db_summary.get("latest_sea_ice"))},
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            components.append(
                ComponentHealth(name="ingested_data", status="unavailable", detail=str(exc))
            )

    archive = history_path(settings)
    components.append(
        ComponentHealth(
            name="history_archive",
            status="ok" if archive.exists() else "unavailable",
            detail=str(archive) if archive.exists() else "not built; run scripts/preprocess_data.py",
        )
    )

    try:
        models = sif.available_models(settings)
        trained = [h for h, m in models.items() if "error" not in m]
        components.append(
            ComponentHealth(
                name="forecast_models",
                status="ok" if len(trained) == len(settings.forecast_horizons_hours) else
                       ("degraded" if trained else "unavailable"),
                detail=(
                    f"trained horizons: {sorted(trained)}"
                    if trained
                    else "no trained artifacts; the API falls back to a persistence baseline"
                ),
                metadata={
                    str(h): {
                        "algorithm": m.get("algorithm"),
                        "rmse": (m.get("metrics") or {}).get("rmse"),
                        "skill_vs_persistence": (m.get("metrics") or {}).get("skill_vs_persistence"),
                    }
                    for h, m in models.items()
                },
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        components.append(ComponentHealth(name="forecast_models", status="unavailable", detail=str(exc)))

    if not db_ok:
        overall = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif any(c.status == "unavailable" for c in components) or any(c.status == "degraded" for c in components):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        app=settings.app_name,
        version=APP_VERSION,
        data_mode=settings.data_mode,
        timestamp=datetime.now(timezone.utc),
        uptime_seconds=round(time.time() - _STARTED_AT, 1),
        components=components,
        database=db_summary,
    )


@router.get("/health/live", summary="Liveness probe", tags=["health"])
def liveness() -> dict:
    """Cheap probe: the process is up. Does not touch the database."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc)}


@router.get("/health/ready", summary="Readiness probe", tags=["health"])
def readiness(session: SessionDep, response: Response) -> dict:
    """Ready when the database answers and at least one observation exists."""
    ok, detail = check_connection()
    ready = ok
    counts = {}
    if ok:
        try:
            counts = repo.database_summary(session).get("row_counts", {})
            ready = counts.get("sea_ice_observations", 0) > 0
        except Exception as exc:  # pragma: no cover
            ready, detail = False, str(exc)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready, "detail": detail, "row_counts": counts}


@router.get("/system", summary="Runtime and configuration information", tags=["health"])
def system_info(settings: SettingsDep) -> dict:
    """Non-secret runtime configuration. Credentials are reported only as booleans."""
    return {
        "app": settings.app_name,
        "version": APP_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "data_mode": settings.data_mode,
        "database_dialect": dialect_name(),
        "domain": {
            "lat_min": settings.domain_lat_min,
            "lat_max": settings.domain_lat_max,
            "lon_min": settings.domain_lon_min,
            "lon_max": settings.domain_lon_max,
            "dlat": settings.grid_resolution_lat,
            "dlon": settings.grid_resolution_lon,
        },
        "forecast_horizons_hours": list(settings.forecast_horizons_hours),
        "risk_weights": settings.risk_weights,
        "credentials_configured": {
            "era5_cds": bool(settings.era5_api_key),
            "copernicus_marine": bool(settings.copernicus_username and settings.copernicus_password),
            "open_mirrors_allowed": settings.allow_open_mirrors,
        },
    }
