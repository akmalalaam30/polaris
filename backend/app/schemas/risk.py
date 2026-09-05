"""Navigation risk schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import GridInfo, PolarisModel, Provenance


class RiskComponents(PolarisModel):
    sea_ice_risk: float = Field(ge=0.0, le=1.0)
    iceberg_risk: float = Field(ge=0.0, le=1.0)
    weather_risk: float = Field(ge=0.0, le=1.0)
    ocean_risk: float = Field(ge=0.0, le=1.0)
    constraint_risk: float = Field(ge=0.0, le=1.0)


class RiskCell(RiskComponents):
    latitude: float
    longitude: float
    total_risk: float = Field(ge=0.0, le=1.0)
    navigable: bool


class RiskHotspot(PolarisModel):
    latitude: float
    longitude: float
    total_risk: float
    dominant_component: str = Field(description="Which component contributes most at this cell")


class RiskGridSummary(PolarisModel):
    generated_at: datetime
    valid_at: datetime
    horizon_hours: int
    grid: GridInfo
    cells_total: int
    cells_ocean: int
    cells_navigable: int
    cells_blocked: int
    mean_risk: float | None = None
    max_risk: float | None = None
    weights: dict[str, float]
    vessel: dict
    available_layers: list[str]
    missing_layers: list[str] = Field(
        default_factory=list,
        description="Risk layers with no data; the total is renormalised over the layers that were available",
    )
    n_icebergs_considered: int = Field(
        default=0,
        description=(
            "Icebergs that actually contribute risk somewhere on this grid. Bergs too distant to "
            "affect any cell, and bergs outside the analysis domain, are excluded from the count."
        ),
    )
    grid_version: str
    data_mode: str


class RiskMapResponse(PolarisModel):
    summary: RiskGridSummary
    cells: list[RiskCell] | None = None
    total_risk_field: list[list[float | None]] | None = Field(
        default=None, description="Compact array form: values[i][j] for grid lats[i], lons[j]"
    )
    lats: list[float] | None = None
    lons: list[float] | None = None
    hotspots: list[RiskHotspot] = Field(default_factory=list)
    scale: dict = Field(
        default_factory=lambda: {
            "min": 0.0,
            "max": 1.0,
            "interpretation": "0 = negligible, 1 = extreme. Decision support only; not a safety guarantee.",
        }
    )
    provenance: Provenance


class PointRiskResponse(RiskCell):
    generated_at: datetime
    valid_at: datetime
    horizon_hours: int
    provenance: Provenance
