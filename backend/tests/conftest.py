"""Shared pytest fixtures.

The suite runs against a **reduced analysis domain** (1.0 x 2.0 degree cells over
a smaller box) and a throwaway SQLite database, so the whole pipeline - ingest,
preprocess, train, forecast, drift, risk, route - executes for real in seconds
rather than minutes.  Nothing is mocked: these are the same code paths the
production configuration uses, only over a smaller grid.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

#: Reduced domain: still covers Maitri (11.7E) to Bharati (76.2E).
TEST_DOMAIN = {
    "DOMAIN_LAT_MIN": "-72.0",
    "DOMAIN_LAT_MAX": "-58.0",
    "DOMAIN_LON_MIN": "6.0",
    "DOMAIN_LON_MAX": "82.0",
    "GRID_RESOLUTION_LAT": "1.0",
    "GRID_RESOLUTION_LON": "2.0",
}
HISTORY_DAYS = 70


@pytest.fixture(scope="session", autouse=True)
def polaris_test_environment(tmp_path_factory):
    """Point the whole application at a disposable, demo-mode configuration."""
    workspace = tmp_path_factory.mktemp("polaris")
    processed = workspace / "processed"
    models = workspace / "models"
    processed.mkdir()
    models.mkdir()

    # Point the suite at PostgreSQL by exporting POLARIS_TEST_DATABASE_URL, e.g.
    #   POLARIS_TEST_DATABASE_URL=postgresql+psycopg2://polaris:pw@localhost:5432/polaris
    # Everything is created and dropped inside that database, so the same 189
    # tests validate the PostgreSQL/PostGIS code path rather than only SQLite.
    database_url = os.environ.get(
        "POLARIS_TEST_DATABASE_URL", f"sqlite:///{(workspace / 'test.db').as_posix()}"
    )

    previous = dict(os.environ)
    os.environ.update(
        {
            "DATA_MODE": "demo",
            "DATABASE_URL": database_url,
            "PROCESSED_DATA_DIR": str(processed),
            "MODEL_PATH": str(models),
            "LOG_LEVEL": "WARNING",
            "SEA_ICE_HISTORY_DAYS": str(HISTORY_DAYS),
            "SEA_ICE_MIN_TRAINING_DAYS": "30",
            "ICEBERG_ENSEMBLE_MEMBERS": "6",
            "ALLOW_OPEN_MIRRORS": "false",
            **TEST_DOMAIN,
        }
    )

    from app.config import Settings, reload_settings
    from app.database.connection import reset_engine
    from app.services import landmask
    from app.services.environment import clear_cache

    # Ignore any developer .env for the duration of the suite. Without this the
    # tests inherit whatever credentials and domain settings happen to be
    # configured on the machine, so the same code passes here and fails there -
    # exactly the kind of environment-dependent result a test must never have.
    saved_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None

    reset_engine()
    landmask.clear_cache()
    clear_cache()
    settings = reload_settings()

    yield settings

    Settings.model_config["env_file"] = saved_env_file
    os.environ.clear()
    os.environ.update(previous)
    reset_engine()
    landmask.clear_cache()
    clear_cache()
    reload_settings()


@pytest.fixture(scope="session")
def settings(polaris_test_environment):
    return polaris_test_environment


@pytest.fixture(scope="session")
def grid(settings):
    from app.utils.geo import grid_from_settings

    return grid_from_settings(settings)


@pytest.fixture(scope="session")
def initialised_db(settings):
    """A fresh schema in the throwaway database.

    PostGIS is enabled when the target is PostgreSQL so the generated geography
    columns and their GiST indexes are exercised too, not just the portable
    lat/lon schema.
    """
    from app.database.connection import dialect_name
    from app.database.init_db import init_database

    return init_database(reset=True, postgis=dialect_name() == "postgresql")


@pytest.fixture(scope="session")
def seeded_db(initialised_db, settings, grid):
    """Ingest demo observations once for the whole session."""
    from app.database.connection import session_scope
    from app.services.data_ingestion import ingest_all
    from app.services.environment import clear_cache

    with session_scope() as session:
        results = ingest_all(session, days=3, settings=settings, grid=grid)
    clear_cache()
    return results


@pytest.fixture(scope="session")
def history(seeded_db, settings, grid):
    """The processed history archives the forecast model trains on."""
    from app.services import preprocessing

    times, ice = preprocessing.build_sea_ice_history(days=HISTORY_DAYS, grid=grid, settings=settings)
    weather = preprocessing.build_weather_history(times, grid, settings)
    return times, ice, weather


@pytest.fixture(scope="session")
def trained_models(history, settings):
    """Train every horizon once; returns the real validation reports."""
    from app.database.connection import session_scope
    from app.services import sea_ice_forecasting as sif
    from app.services.environment import clear_cache

    with session_scope() as session:
        reports = sif.train_all(session=session, settings=settings, subsample=1)
    clear_cache()
    return reports


@pytest.fixture()
def session(seeded_db):
    """A short-lived database session."""
    from app.database.connection import get_session_factory

    s = get_session_factory()()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


@pytest.fixture(scope="session")
def client(seeded_db, trained_models):
    """A TestClient bound to the fully seeded application."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def api(seeded_db):
    """A TestClient without trained models (exercises the baseline fallback)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
