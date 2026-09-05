"""Database engine / session management.

Works unchanged against SQLite (default) and PostgreSQL(+PostGIS).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("database.connection")

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def _build_engine(url: str, echo: bool) -> Engine:
    kwargs: dict = {"echo": echo, "future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # check_same_thread=False lets FastAPI's threadpool share the engine.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs.update(pool_size=5, max_overflow=10, pool_recycle=1800)

    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def get_engine() -> Engine:
    """Process-wide SQLAlchemy engine (created lazily)."""
    global _engine, _SessionFactory
    if _engine is None:
        settings = get_settings()
        _engine = _build_engine(settings.database_url, settings.database_echo)
        _SessionFactory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
        log.info("Database engine created (dialect=%s)", _engine.dialect.name)
    return _engine


def get_session_factory() -> sessionmaker:
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


def reset_engine() -> None:
    """Dispose the engine (used by tests that switch DATABASE_URL)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def check_connection() -> tuple[bool, str]:
    """Ping the database. Returns ``(ok, detail)`` and never raises."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, f"connected ({engine.dialect.name})"
    except SQLAlchemyError as exc:
        log.error("Database connection failed: %s", exc.__class__.__name__)
        return False, f"{exc.__class__.__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Unexpected database error: %s", exc)
        return False, str(exc)


def dialect_name() -> str:
    try:
        return get_engine().dialect.name
    except Exception:  # pragma: no cover
        return "unknown"


def has_postgis() -> bool:
    """True when the connected PostgreSQL server exposes PostGIS."""
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        return False
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT PostGIS_Version()")).scalar()
        return bool(row)
    except Exception:
        return False
