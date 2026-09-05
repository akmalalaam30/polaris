"""POLARIS FastAPI application.

Wires the routers together, installs uniform error handling, and serves the
minimal test frontend from ``/ui`` when it is present.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import (
    routes_dashboard,
    routes_health,
    routes_icebergs,
    routes_navigation,
    routes_risk,
    routes_sea_ice,
)
from app.config import APP_TITLE, APP_VERSION, PROJECT_ROOT, get_settings
from app.database.connection import check_connection
from app.database.init_db import create_all
from app.utils.logging import get_logger, setup_logging
from app.utils.validation import DataUnavailableError, ValidationError

#: Starlette deprecated its 422 constant name; the numeric code is stable.
HTTP_422 = 422

settings = get_settings()
setup_logging(settings.log_level, settings.log_file)
log = get_logger("main")

DESCRIPTION = """
**POLARIS** - AI-enabled Antarctic sea-ice, iceberg trajectory and navigation decision support.

Smart India Hackathon 2026 - Problem statement **SIH26059**
Ministry of Earth Sciences / National Centre for Polar and Ocean Research.

### Pipeline
`data ingestion -> validation -> preprocessing -> sea-ice forecast -> iceberg trajectory ->
risk engine -> risk grid -> route optimisation -> API`

### Data modes
* `DATA_MODE=real` - NSIDC Sea Ice Index (G02135), US National Ice Center Antarctic iceberg
  bulletin, ERA5 atmospheric reanalysis and Copernicus Marine ocean analysis.
* `DATA_MODE=demo` - clearly labelled synthetic fields generated locally, running through the
  identical pipeline. Every record carries its `data_mode` and `source`.

### What this is, and is not
POLARIS produces **decision support**. Forecasts, trajectories and risk scores carry uncertainty
and do **not** guarantee safe navigation. Reported fuel and duration are model estimates from a
documented speed model, not measurements.
"""

TAGS_METADATA = [
    {"name": "health", "description": "Service health, readiness and runtime configuration."},
    {"name": "sea-ice", "description": "Observed sea-ice conditions and machine-learning forecasts."},
    {"name": "icebergs", "description": "Tracked Antarctic icebergs and physics-based drift trajectories."},
    {"name": "risk", "description": "Spatial navigation risk grid and point assessments."},
    {"name": "navigation", "description": "Graph-based route optimisation between Antarctic locations."},
    {"name": "dashboard", "description": "Aggregated status, model registry and ingestion audit trail."},
]


def _warm_caches() -> None:
    """Precompute the expensive shared products in the background.

    The first request that needs the land-mask distance field, the iceberg
    trajectories or the risk grid pays several seconds to build them; every
    later request is served from cache in milliseconds.  Doing that work at
    startup instead means nobody's first click is the one that waits.

    Failures here are logged and ignored: a warm cache is an optimisation, and
    the endpoints rebuild whatever is missing on demand.
    """
    import time

    started = time.perf_counter()
    try:
        from app.database.connection import session_scope
        from app.services import environment as envsvc
        from app.services import landmask
        from app.utils.geo import grid_from_settings

        grid = grid_from_settings(settings)
        landmask.navigable_mask(grid)
        landmask.distance_to_land_km(grid)

        with session_scope() as session:
            envsvc.get_environment(session, settings, grid)
            for horizon in settings.forecast_horizons_hours:
                try:
                    envsvc.get_forecast(horizon, settings, grid)
                except Exception as exc:  # noqa: BLE001 - untrained model is fine
                    log.debug("Warm-up skipped forecast %dh: %s", horizon, exc)
            envsvc.get_iceberg_context(session, settings, grid, horizon_hours=72.0)
            envsvc.get_risk_grid(session, settings, grid, horizon_hours=0)
        log.info("Cache warm-up finished in %.1fs", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - never let warm-up break startup
        log.warning("Cache warm-up skipped (%s); first request will build on demand", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s v%s (data_mode=%s)", settings.app_name, APP_VERSION, settings.data_mode)
    try:
        create_all()
    except SQLAlchemyError as exc:
        log.error("Schema initialisation failed at startup: %s", exc.__class__.__name__)
    ok, detail = check_connection()
    log.info("Database: %s (%s)", "ok" if ok else "UNAVAILABLE", detail)
    if not ok:
        log.error("The API will start, but data endpoints will return 503 until the database is reachable.")
    elif settings.warm_cache_on_startup:
        # In a daemon thread so the server accepts connections immediately -
        # health checks answer while the heavy products are still building.
        threading.Thread(target=_warm_caches, name="polaris-warmup", daemon=True).start()
        log.info("Cache warm-up running in the background")
    yield
    log.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    contact={"name": "Team Brookies - SIH26059"},
    license_info={"name": "Hackathon prototype"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_and_logging(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Unhandled error while serving %s %s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
    response.headers["X-POLARIS-Data-Mode"] = settings.data_mode
    if elapsed_ms > 5000:
        log.warning("Slow request: %s %s took %.0f ms", request.method, request.url.path, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Error handling: never leak a traceback, always explain what to do
# ---------------------------------------------------------------------------
_ERROR_NAMES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    HTTP_422: "invalid_input",
    500: "internal_error",
    503: "data_unavailable",
}
def _error(request: Request, code: int, error: str, detail, hint: str | None = None) -> JSONResponse:
    payload = {
        "error": error,
        "detail": detail if isinstance(detail, str) else detail,
        "status_code": code,
        "path": request.url.path,
    }
    if hint:
        payload["hint"] = hint
    return JSONResponse(status_code=code, content=payload)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Give explicitly raised HTTP errors the same envelope as everything else."""
    detail = exc.detail
    hint = None
    if isinstance(detail, dict):
        hint = detail.pop("hint", None)
        detail = detail.get("message", detail) if len(detail) == 1 else detail
    return _error(request, exc.status_code, _ERROR_NAMES.get(exc.status_code, "http_error"), detail, hint)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(p) for p in e.get("loc", [])), "message": e.get("msg"), "type": e.get("type")}
        for e in exc.errors()
    ]
    return _error(
        request, HTTP_422, "validation_error", errors,
        hint="Check the request against the schema at /docs.",
    )


@app.exception_handler(ValidationError)
async def polaris_validation_handler(request: Request, exc: ValidationError):
    return _error(request, HTTP_422, "invalid_input", str(exc))


@app.exception_handler(DataUnavailableError)
async def data_unavailable_handler(request: Request, exc: DataUnavailableError):
    return _error(
        request, status.HTTP_503_SERVICE_UNAVAILABLE, "data_unavailable", str(exc),
        hint="Run the ingestion/preprocessing scripts, then retry.",
    )


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError):
    log.error("Database error on %s: %s", request.url.path, exc.__class__.__name__)
    return _error(
        request, status.HTTP_503_SERVICE_UNAVAILABLE, "database_error",
        f"A database error occurred ({exc.__class__.__name__}).",
        hint="Check DATABASE_URL and that the schema has been initialised.",
    )


@app.exception_handler(FileNotFoundError)
async def missing_artifact_handler(request: Request, exc: FileNotFoundError):
    return _error(
        request, status.HTTP_503_SERVICE_UNAVAILABLE, "artifact_missing", str(exc),
        hint="Run 'python scripts/train_models.py' to produce the model artifacts.",
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url.path)
    return _error(
        request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error",
        f"An unexpected error occurred ({exc.__class__.__name__}). See the server log for details.",
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
prefix = settings.api_prefix
app.include_router(routes_health.router, prefix=prefix)
app.include_router(routes_sea_ice.router, prefix=prefix)
app.include_router(routes_icebergs.router, prefix=prefix)
app.include_router(routes_risk.router, prefix=prefix)
app.include_router(routes_navigation.router, prefix=prefix)
app.include_router(routes_dashboard.router, prefix=prefix)


@app.get("/", include_in_schema=False)
def root():
    frontend = PROJECT_ROOT / "frontend" / "src" / "index.html"
    if frontend.exists():
        return RedirectResponse(url="/ui/")
    return RedirectResponse(url="/docs")


@app.get("/api", include_in_schema=False)
def api_index():
    return {
        "name": settings.app_name,
        "version": APP_VERSION,
        "data_mode": settings.data_mode,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": [
            f"{prefix}/health",
            f"{prefix}/sea-ice/current",
            f"{prefix}/sea-ice/forecast?forecast_hours=72",
            f"{prefix}/sea-ice/extent",
            f"{prefix}/icebergs",
            f"{prefix}/icebergs/{{iceberg_id}}/trajectory",
            f"{prefix}/risk/map",
            f"{prefix}/risk/point?latitude=-65&longitude=40",
            f"{prefix}/route/optimize  (POST)",
            f"{prefix}/navigation/stations",
            f"{prefix}/dashboard/summary",
        ],
    }


# The demo frontend is a static page; mounting it is optional.
_frontend_dir = PROJECT_ROOT / "frontend" / "src"
if _frontend_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
    log.info("Test frontend mounted at /ui")
