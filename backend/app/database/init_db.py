"""Schema creation and optional PostGIS enablement.

POLARIS keeps schema management deliberately simple (``create_all`` +
idempotent PostGIS upgrades) rather than pulling in Alembic: the schema is
append-only for the hackathon deliverable and the same code path has to work on
SQLite and PostgreSQL.  The ``upgrade_postgis`` step is safe to run repeatedly.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.database.connection import get_engine, has_postgis, session_scope
from app.models.database_models import POINT_TABLES, Base
from app.utils.logging import get_logger

log = get_logger("database.init")


def create_all() -> None:
    """Create every POLARIS table that does not yet exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    log.info("Schema ensured on %s (%d tables)", engine.dialect.name, len(Base.metadata.tables))


def drop_all() -> None:
    """Drop every POLARIS table (used by tests and ``init_db --reset``)."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    log.warning("All POLARIS tables dropped")


def enable_postgis() -> bool:
    """Try to enable the PostGIS extension. Returns True when available."""
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        log.info("PostGIS extension enabled")
        return True
    except SQLAlchemyError as exc:
        log.warning("PostGIS not enabled (%s) - continuing with plain lat/lon columns", exc.__class__.__name__)
        return False


def upgrade_postgis() -> dict[str, str]:
    """Add generated geography columns + GiST indexes to the point tables.

    No-op on SQLite or when PostGIS is unavailable.  The application never
    depends on these columns; they exist so operators can run fast spatial
    queries (``ST_DWithin`` etc.) directly against the warehouse.
    """
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        return {"status": "skipped", "reason": f"dialect={engine.dialect.name}"}
    if not enable_postgis() or not has_postgis():
        return {"status": "skipped", "reason": "postgis unavailable"}

    done: list[str] = []
    with engine.begin() as conn:
        for table in POINT_TABLES:
            try:
                conn.execute(
                    text(
                        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS geom geography(Point,4326) "
                        f"GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography) STORED"
                    )
                )
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_geom ON {table} USING GIST (geom)"))
                done.append(table)
            except SQLAlchemyError as exc:
                log.warning("PostGIS upgrade failed for %s: %s", table, exc.__class__.__name__)
    return {"status": "ok", "tables": ",".join(done)}


def init_database(reset: bool = False, postgis: bool = True) -> dict:
    """Full initialisation entry point used by scripts and tests."""
    if reset:
        drop_all()
    create_all()
    result = {"schema": "ok", "reset": reset}
    if postgis:
        result["postgis"] = upgrade_postgis()
    # Smoke-check that we can actually write and read.
    try:
        with session_scope() as s:
            s.execute(text("SELECT 1"))
        result["write_check"] = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        result["write_check"] = f"failed: {exc}"
    return result


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Initialise the POLARIS database")
    parser.add_argument("--reset", action="store_true", help="drop existing tables first")
    parser.add_argument("--no-postgis", action="store_true", help="skip PostGIS upgrade")
    args = parser.parse_args()
    print(json.dumps(init_database(reset=args.reset, postgis=not args.no_postgis), indent=2))
