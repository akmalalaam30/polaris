"""Route optimisation request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import NamedLocation, PolarisModel, Provenance


class VesselParameters(PolarisModel):
    """Vessel model parameters.

    These describe a *model* vessel used to score routes; they are not a claim
    about any real ship's capability.  Defaults approximate a polar
    research/resupply vessel.
    """

    name: str = Field(default="generic polar research vessel", max_length=64)
    ice_class: Literal["none", "1c", "1b", "1a", "1a_super", "pc5", "icebreaker"] = Field(
        default="1a_super", description="Determines the ice concentration the hull can work in"
    )
    speed_knots: float = Field(default=12.0, gt=0, le=30, description="Service speed in open water")
    fuel_consumption_tpd: float = Field(
        default=25.0, gt=0, le=500, description="Fuel burn at service speed, tonnes per day"
    )
    draft_m: float = Field(default=7.0, gt=0, le=25)
    risk_tolerance: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="0 = strongly prefers safety over distance; 1 = prefers the direct route",
    )

    @field_validator("ice_class", mode="before")
    @classmethod
    def _normalise(cls, v):
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v


class RiskWeightOverrides(PolarisModel):
    """Optional per-request override of the risk-component weights."""

    sea_ice: float | None = Field(default=None, ge=0.0, le=1.0)
    iceberg: float | None = Field(default=None, ge=0.0, le=1.0)
    weather: float | None = Field(default=None, ge=0.0, le=1.0)
    ocean: float | None = Field(default=None, ge=0.0, le=1.0)
    constraint: float | None = Field(default=None, ge=0.0, le=1.0)


class OptimizationPreferences(PolarisModel):
    """Which routes to compute, over which horizon, with which cost weights."""

    profiles: list[Literal["shortest", "safest", "polaris"]] = Field(
        default=["shortest", "safest", "polaris"], min_length=1
    )
    forecast_hours: Literal[0, 24, 48, 72] = Field(
        default=24,
        description="Horizon whose forecast conditions the risk grid; 0 uses current observations only",
    )
    algorithm: Literal["astar", "dijkstra"] = "astar"
    distance_weight: float | None = Field(default=None, ge=0.0, le=10.0)
    risk_weight: float | None = Field(default=None, ge=0.0, le=10.0)
    fuel_weight: float | None = Field(default=None, ge=0.0, le=10.0)
    risk_weights: RiskWeightOverrides | None = None
    include_iceberg_analysis: bool = True
    simplify_geometry: bool = True


class RouteOptimizeRequest(PolarisModel):
    """POST /api/route/optimize body."""

    start: NamedLocation
    destination: NamedLocation
    vessel: VesselParameters = Field(default_factory=VesselParameters)
    preferences: OptimizationPreferences = Field(default_factory=OptimizationPreferences)
    persist: bool = Field(default=True, description="Store the computed routes in the database")

    @model_validator(mode="after")
    def _require_location(self):
        for label, loc in (("start", self.start), ("destination", self.destination)):
            has_coords = loc.latitude is not None and loc.longitude is not None
            if not has_coords and not loc.station:
                raise ValueError(
                    f"{label} must provide either latitude+longitude or a station key "
                    f"(see GET /api/navigation/stations)"
                )
        return self


class Waypoint(PolarisModel):
    latitude: float
    longitude: float


class RouteOut(PolarisModel):
    profile: str
    status: Literal["ok", "degraded"]
    algorithm: str
    distance_km: float
    duration_hours: float
    estimated_fuel_tonnes: float = Field(
        description="Model estimate from the documented speed/consumption model, not a measurement"
    )
    mean_risk: float
    max_risk: float
    max_sea_ice_concentration: float
    n_waypoints: int
    waypoints: list[Waypoint]
    geometry: dict = Field(description="GeoJSON Feature with a LineString geometry")
    cost_weights: dict[str, float]
    start_snap_km: float = Field(description="Distance the start was moved to reach navigable water")
    end_snap_km: float
    risk_assessment: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RouteOptimizeResponse(PolarisModel):
    request_id: str
    computed_at: datetime
    start: dict
    destination: dict
    vessel: dict
    graph: dict
    fallback_graph: dict | None = None
    risk_grid: dict
    routes: dict[str, RouteOut]
    recommended_profile: str | None
    comparison: dict = Field(
        default_factory=dict,
        description="Differences between profiles, computed from this run's own numbers",
    )
    errors: dict[str, str] = Field(default_factory=dict)
    persisted_route_ids: list[int] = Field(default_factory=list)
    provenance: Provenance


class StationOut(PolarisModel):
    key: str
    name: str
    operator: str
    region: str
    latitude: float
    longitude: float
    in_domain: bool
    nearest_navigable_km: float | None = None


class StationListResponse(PolarisModel):
    stations: list[StationOut]
    domain: dict
