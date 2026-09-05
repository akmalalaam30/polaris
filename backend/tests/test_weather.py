"""Live weather: derived hazard logic, endpoints, and offline degradation.

These tests never call the live provider. The suite runs with
``ALLOW_OPEN_MIRRORS=false``, so the live endpoints are expected to refuse
cleanly - which is exactly the behaviour that matters when a venue's network is
down mid-demonstration. The physics-derived helpers are pure functions and are
tested directly.
"""

from __future__ import annotations

import pytest

from app.services.data_ingestion import _beaufort, _freezing_spray, _hourly_outlook


# ---------------------------------------------------------------------------
# Beaufort scale
# ---------------------------------------------------------------------------
def test_beaufort_boundaries_match_the_scale():
    # Published thresholds: F0 <0.5, F4 5.5-7.9, F8 17.2-20.7, F12 >=32.7 m/s.
    assert _beaufort(0.2) == 0
    assert _beaufort(3.0) == 2
    assert _beaufort(6.0) == 4
    assert _beaufort(18.0) == 8
    assert _beaufort(35.0) == 12
    assert _beaufort(None) is None


def test_beaufort_is_monotonic():
    speeds = [0, 1, 3, 5, 8, 11, 14, 18, 22, 26, 30, 40]
    forces = [_beaufort(s) for s in speeds]
    assert forces == sorted(forces)
    assert all(0 <= f <= 12 for f in forces)


# ---------------------------------------------------------------------------
# Freezing spray
# ---------------------------------------------------------------------------
def test_freezing_spray_needs_cold_and_wind():
    """Icing requires both cold air and enough wind to generate spray."""
    assert _freezing_spray({"air_temperature_c": 5.0, "wind_speed_m_s": 25.0}) == "none"
    assert _freezing_spray({"air_temperature_c": -20.0, "wind_speed_m_s": 2.0}) == "none"


def test_freezing_spray_severity_increases():
    light = _freezing_spray({"air_temperature_c": -5.0, "wind_speed_m_s": 11.0})
    moderate = _freezing_spray({"air_temperature_c": -9.0, "wind_speed_m_s": 15.0})
    severe = _freezing_spray({"air_temperature_c": -18.0, "wind_speed_m_s": 22.0})
    assert (light, moderate, severe) == ("light", "moderate", "severe")


def test_warm_sea_suppresses_freezing_spray():
    """Spray from warm water does not accrete however cold the air is."""
    cold_sea = _freezing_spray(
        {"air_temperature_c": -15.0, "wind_speed_m_s": 20.0, "sea_surface_temperature_c": -1.5}
    )
    warm_sea = _freezing_spray(
        {"air_temperature_c": -15.0, "wind_speed_m_s": 20.0, "sea_surface_temperature_c": 12.0}
    )
    assert cold_sea == "severe"
    assert warm_sea == "none"


def test_freezing_spray_is_none_without_inputs():
    assert _freezing_spray({"air_temperature_c": None, "wind_speed_m_s": 10.0}) is None
    assert _freezing_spray({}) is None


# ---------------------------------------------------------------------------
# Outlook trimming
# ---------------------------------------------------------------------------
def test_hourly_outlook_trims_and_aligns():
    hourly = {
        "time": ["2026-09-05T00:00", "2026-09-05T01:00", "2026-09-05T02:00"],
        "temperature_2m": [-10.0, -11.0, -12.0],
        "wind_speed_10m": [5.0, 6.0],  # deliberately short
    }
    out = _hourly_outlook(hourly, hours=2)
    assert len(out) == 2
    assert out[0]["air_temperature_c"] == -10.0
    assert out[1]["wind_speed_m_s"] == 6.0
    # A series shorter than the time axis must yield None, not raise.
    assert _hourly_outlook(hourly, hours=3)[2]["wind_speed_m_s"] is None


def test_hourly_outlook_handles_empty_input():
    assert _hourly_outlook({}, hours=24) == []
    assert _hourly_outlook(None, hours=24) == []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def test_station_weather_refuses_cleanly_when_offline(client):
    """With no live source the endpoint must explain, not crash or invent."""
    r = client.get("/api/weather/stations")
    assert r.status_code == 503
    detail = r.json()["detail"]
    text = str(detail).lower()
    assert "live" in text or "unavailable" in text


def test_unknown_station_returns_404_with_the_known_list(client):
    r = client.get("/api/weather/stations?station=atlantis")
    assert r.status_code == 404
    assert "bharati" in str(r.json()["detail"]["known"])


def test_point_weather_validates_coordinates(client):
    assert client.get("/api/weather/point?latitude=200&longitude=40").status_code == 422


def test_freshness_endpoint_works_without_network(client):
    """Layer ages come from the database, so this must work offline."""
    r = client.get("/api/weather/current")
    assert r.status_code == 200
    body = r.json()
    assert body["data_mode"] == "demo"
    assert {"sea_ice", "weather", "ocean", "icebergs"} <= set(body["layers"])
    for name, layer in body["layers"].items():
        if layer["observed_at"] is not None:
            assert layer["age_hours"] is not None
            assert layer["age_hours"] >= 0, f"{name} reports a negative age"


def test_freshness_is_in_the_openapi_schema(client):
    spec = client.get("/openapi.json").json()
    for path in ("/api/weather/stations", "/api/weather/point", "/api/weather/current"):
        assert path in spec["paths"], f"{path} missing from the OpenAPI schema"
