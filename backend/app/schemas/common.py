"""Schemas shared by every POLARIS endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PolarisModel(BaseModel):
    """Base model with a consistent configuration."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


class Provenance(PolarisModel):
    """Where a payload came from, and how far it may be trusted."""

    data_mode: Literal["demo", "real"] = Field(
        description="'real' = ingested from the upstream providers; 'demo' = locally generated synthetic data"
    )
    sources: list[str] = Field(default_factory=list, description="Upstream providers contributing to this payload")
    disclaimer: str = Field(description="Explicit statement of what these values are and are not")
    observed_at: datetime | None = Field(default=None, description="Valid time of the underlying observations (UTC)")
    generated_at: datetime | None = Field(default=None, description="When POLARIS produced this payload (UTC)")


class GridInfo(PolarisModel):
    """Definition of the regular lat/lon analysis grid."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    dlat: float
    dlon: float
    shape: list[int] = Field(description="[n_lat, n_lon]")


class Coordinate(PolarisModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=360)


class NamedLocation(PolarisModel):
    """A route endpoint: explicit coordinates, or a known station key."""

    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=360)
    station: str | None = Field(
        default=None, description="Station key, e.g. 'bharati' or 'maitri' (see /api/navigation/stations)"
    )
    name: str | None = None


class ErrorResponse(PolarisModel):
    """Uniform error payload for every 4xx/5xx response."""

    error: str = Field(description="Machine-readable error class")
    detail: str = Field(description="Human-readable explanation")
    status_code: int
    path: str | None = None
    hint: str | None = Field(default=None, description="What the caller can do about it")


class ComponentHealth(PolarisModel):
    name: str
    status: Literal["ok", "degraded", "unavailable"]
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(PolarisModel):
    status: Literal["ok", "degraded", "unavailable"]
    app: str
    version: str
    data_mode: str
    timestamp: datetime
    uptime_seconds: float
    components: list[ComponentHealth]
    database: dict[str, Any] = Field(default_factory=dict)
