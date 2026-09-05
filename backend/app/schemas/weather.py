"""Live weather and marine-condition schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import PolarisModel, Provenance


class ForecastHour(PolarisModel):
    """One hour of the atmospheric outlook."""

    valid_at: datetime
    air_temperature_c: float | None = None
    wind_speed_m_s: float | None = None
    wind_direction_deg: float | None = None
    mean_sea_level_pressure_hpa: float | None = None


class StationConditions(PolarisModel):
    """Live conditions at one station or coordinate."""

    station: str
    display_name: str
    operator: str | None = None
    region: str | None = None
    latitude: float
    longitude: float
    observed_at: datetime | None = Field(
        default=None, description="Valid time of these conditions (UTC), typically within the hour"
    )

    # -- atmosphere -------------------------------------------------------
    air_temperature_c: float | None = None
    wind_speed_m_s: float | None = None
    wind_direction_deg: float | None = Field(
        default=None, description="Meteorological convention: the direction the wind blows FROM"
    )
    beaufort_force: int | None = Field(default=None, ge=0, le=12)
    mean_sea_level_pressure_hpa: float | None = None
    relative_humidity_pct: float | None = None
    cloud_cover_pct: float | None = None

    # -- ocean ------------------------------------------------------------
    significant_wave_height_m: float | None = Field(
        default=None,
        description="Null inside the pack: waves are not defined under sea ice, so this is physics rather than missing data",
    )
    wave_period_s: float | None = None
    ocean_current_speed_m_s: float | None = None
    ocean_current_direction_deg: float | None = Field(
        default=None, description="Marine convention: the direction the current flows TOWARDS"
    )
    sea_surface_temperature_c: float | None = None

    # -- derived ----------------------------------------------------------
    freezing_spray_risk: Literal["none", "light", "moderate", "severe"] | None = Field(
        default=None,
        description=(
            "Qualitative superstructure-icing hazard from air temperature, wind and sea "
            "temperature. Not an accretion rate - POLARIS does not compute one."
        ),
    )

    source: str | None = None
    marine_source: str | None = None
    forecast: list[ForecastHour] = Field(default_factory=list)


class StationWeatherResponse(PolarisModel):
    count: int
    forecast_hours: int
    stations: list[StationConditions]
    provenance: Provenance


class LayerFreshness(PolarisModel):
    observed_at: datetime | None = None
    age_hours: float | None = None


class LiveConditionsResponse(PolarisModel):
    """How current each environmental layer actually is."""

    generated_at: datetime
    data_mode: str
    layers: dict[str, LayerFreshness]
    provenance: Provenance
