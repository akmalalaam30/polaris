"""Health endpoints and database connectivity."""

from __future__ import annotations

import pytest


def test_database_connects(initialised_db):
    from app.database.connection import check_connection

    ok, detail = check_connection()
    assert ok, detail
    assert initialised_db["schema"] == "ok"
    assert initialised_db["write_check"] == "ok"


def test_every_table_is_created(initialised_db):
    from sqlalchemy import inspect

    from app.database.connection import get_engine
    from app.models.database_models import ALL_TABLES

    existing = set(inspect(get_engine()).get_table_names())
    for model in ALL_TABLES:
        assert model.__tablename__ in existing, f"missing table {model.__tablename__}"


def test_health_reports_each_component(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["data_mode"] == "demo"
    names = {c["name"] for c in body["components"]}
    assert {"database", "ingested_data", "history_archive", "forecast_models"} <= names
    db = next(c for c in body["components"] if c["name"] == "database")
    assert db["status"] == "ok"
    assert body["database"]["row_counts"]["sea_ice_observations"] > 0


def test_liveness_does_not_need_the_database(client):
    r = client.get("/api/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_readiness_requires_ingested_data(client):
    r = client.get("/api/health/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_system_info_never_leaks_credentials(client):
    r = client.get("/api/system")
    assert r.status_code == 200
    body = r.json()
    assert set(body["credentials_configured"]) == {
        "era5_cds", "copernicus_marine", "open_mirrors_allowed"
    }
    for value in body["credentials_configured"].values():
        assert isinstance(value, bool)
    serialised = r.text.lower()
    for secret_marker in ("password", "api_key", "secret"):
        assert secret_marker not in serialised


def test_openapi_and_docs_are_served(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    for path in (
        "/api/health",
        "/api/sea-ice/current",
        "/api/sea-ice/forecast",
        "/api/icebergs",
        "/api/icebergs/{iceberg_id}/trajectory",
        "/api/risk/map",
        "/api/route/optimize",
        "/api/dashboard/summary",
    ):
        assert path in spec["paths"], f"{path} missing from the OpenAPI schema"


def test_unknown_route_returns_the_uniform_error_envelope(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert body["status_code"] == 404
    assert body["path"] == "/api/does-not-exist"


def test_log_filter_redacts_credentials():
    import logging

    from app.utils.logging import SecretRedactingFilter

    record = logging.LogRecord(
        "t", logging.INFO, __file__, 1,
        "connecting with password=hunter2 and api_key=abcdef", None, None,
    )
    SecretRedactingFilter().filter(record)
    assert "hunter2" not in record.getMessage()
    assert "abcdef" not in record.getMessage()

    url_record = logging.LogRecord(
        "t", logging.INFO, __file__, 1,
        "postgresql://polaris:s3cret@db:5432/polaris", None, None,
    )
    SecretRedactingFilter().filter(url_record)
    assert "s3cret" not in url_record.getMessage()
