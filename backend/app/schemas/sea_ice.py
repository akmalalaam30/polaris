"""Sea-ice observation and forecast schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import GridInfo, PolarisModel, Provenance


class SeaIceCell(PolarisModel):
    """One analysis-grid cell of observed sea-ice concentration."""

    latitude: float
    longitude: float
    concentration: float = Field(ge=0.0, le=1.0, description="Sea-ice concentration as a fraction")
    is_land: bool = False
    quality_flag: str | None = None


class SeaIceGrid(PolarisModel):
    """Compact array form of a gridded field (far smaller than per-cell JSON)."""

    lats: list[float]
    lons: list[float]
    values: list[list[float | None]] = Field(
        description="values[i][j] corresponds to lats[i], lons[j]; null where no data exists"
    )


class SeaIceStatistics(PolarisModel):
    mean_concentration: float | None = None
    max_concentration: float | None = None
    ice_covered_fraction: float | None = Field(
        default=None, description="Fraction of ocean cells with concentration >= 0.15"
    )
    ocean_cells: int = 0
    ice_edge_latitude_mean: float | None = Field(
        default=None, description="Mean latitude of the 15% ice edge across the domain"
    )


class SeaIceCurrentResponse(PolarisModel):
    """Latest observed sea-ice state over the domain."""

    observed_at: datetime | None
    grid: GridInfo
    statistics: SeaIceStatistics
    cells: list[SeaIceCell] | None = None
    field: SeaIceGrid | None = None
    hemispheric_extent_million_km2: float | None = Field(
        default=None, description="Latest hemispheric sea-ice extent index value"
    )
    provenance: Provenance


class SeaIceForecastResponse(PolarisModel):
    """Sea-ice concentration forecast for one horizon."""

    issued_at: datetime
    valid_at: datetime
    forecast_hours: int
    model_name: str
    model_version: str
    algorithm: str
    grid: GridInfo
    statistics: SeaIceStatistics
    cells: list[SeaIceCell] | None = None
    field: SeaIceGrid | None = None
    uncertainty: SeaIceGrid | None = Field(
        default=None, description="1-sigma error calibrated on held-out validation residuals"
    )
    validation_metrics: dict = Field(
        default_factory=dict,
        description="Real metrics from the model's held-out chronological split; empty for an untrained baseline",
    )
    provenance: Provenance


class ExtentPoint(PolarisModel):
    observed_at: datetime
    extent_million_km2: float | None = None
    area_million_km2: float | None = None


class SeaIceExtentResponse(PolarisModel):
    hemisphere: str = "south"
    series: list[ExtentPoint]
    provenance: Provenance
