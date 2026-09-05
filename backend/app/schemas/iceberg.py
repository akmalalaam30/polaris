"""Iceberg observation and trajectory schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import PolarisModel, Provenance


class IcebergObservationOut(PolarisModel):
    """A tracked iceberg's most recent reported position."""

    iceberg_id: str = Field(description="Designator, e.g. 'A76C' (USNIC) or 'DEMO-01' (synthetic)")
    observed_at: datetime
    latitude: float
    longitude: float
    length_nm: float | None = None
    width_nm: float | None = None
    area_km2: float | None = None
    drift_speed_m_s: float | None = Field(
        default=None, description="Derived from the previous fix of the same berg, when one exists"
    )
    drift_bearing_deg: float | None = None
    n_observations: int = Field(default=1, description="Number of stored fixes for this berg")
    in_domain: bool = Field(
        default=True,
        description=(
            "True when the berg lies inside the POLARIS analysis domain. Outside it there is no "
            "forcing field, so no trajectory is computed for that berg."
        ),
    )
    source: str
    data_mode: str


class IcebergListResponse(PolarisModel):
    count: int
    icebergs: list[IcebergObservationOut]
    provenance: Provenance


class TrackPoint(PolarisModel):
    observed_at: datetime
    latitude: float
    longitude: float
    drift_speed_m_s: float | None = None
    drift_bearing_deg: float | None = None


class TrajectoryPointOut(PolarisModel):
    """One predicted position along a drift trajectory."""

    valid_at: datetime
    horizon_hours: float
    latitude: float
    longitude: float
    speed_m_s: float
    bearing_deg: float
    uncertainty_radius_km: float = Field(
        description="RMS spread of the Monte-Carlo forcing ensemble at this lead time"
    )
    grounded: bool = False


class IcebergGeometry(PolarisModel):
    length_m: float
    width_m: float
    thickness_m: float = Field(description="Estimated from waterline length; rarely observed directly")
    draft_m: float
    mass_kg: float


class IcebergTrajectoryResponse(PolarisModel):
    iceberg_id: str
    issued_at: datetime
    origin: dict
    geometry: IcebergGeometry
    forecast_hours: float
    ensemble_size: int
    total_displacement_km: float
    points: list[TrajectoryPointOut]
    observed_track: list[TrackPoint] = Field(default_factory=list)
    forcing: dict = Field(default_factory=dict, description="Environmental forcing sampled at the origin")
    model_name: str
    model_version: str
    notes: list[str] = Field(default_factory=list)
    provenance: Provenance
