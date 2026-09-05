"""SQLAlchemy ORM models for POLARIS.

Design notes
------------
* Coordinates are stored as plain ``latitude``/``longitude`` columns so the
  schema works identically on SQLite (default, zero-setup) and PostgreSQL.
* When the backend runs on PostgreSQL **with PostGIS available**, an additional
  ``geom`` geography column plus GiST index is added to the point tables by
  :func:`app.database.init_db.enable_postgis`.  Nothing in the application
  requires PostGIS; it is a spatial-query accelerator.
* Every observation row records ``source`` and ``data_mode`` so demo/synthetic
  records can never be mistaken for real observations.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all POLARIS tables."""


class TimestampMixin:
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class ProvenanceMixin:
    #: Upstream provider, e.g. "NSIDC/G02135", "USNIC", "CMEMS", "ERA5",
    #: or "POLARIS-DEMO" for locally generated synthetic data.
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    #: "real" or "demo" - never mix these silently.
    data_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="demo", index=True)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
class SeaIceObservation(Base, TimestampMixin, ProvenanceMixin):
    """Gridded Antarctic sea-ice concentration observation (one grid cell)."""

    __tablename__ = "sea_ice_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    #: Sea-ice concentration as a fraction in [0, 1].
    concentration: Mapped[float] = mapped_column(Float, nullable=False)
    #: True where the cell is land / ice shelf on the source grid.
    is_land: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    quality_flag: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        UniqueConstraint("observed_at", "latitude", "longitude", "source", name="uq_seaice_cell"),
        Index("ix_seaice_time_pos", "observed_at", "latitude", "longitude"),
    )


class SeaIceExtentIndex(Base, TimestampMixin, ProvenanceMixin):
    """Hemispheric daily sea-ice extent/area index (NSIDC Sea Ice Index)."""

    __tablename__ = "sea_ice_extent_index"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hemisphere: Mapped[str] = mapped_column(String(8), nullable=False, default="south")
    extent_million_km2: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_million_km2: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("observed_at", "hemisphere", "source", name="uq_extent_day"),
    )


class IcebergObservation(Base, TimestampMixin, ProvenanceMixin):
    """A tracked Antarctic iceberg position report (e.g. USNIC bulletin row)."""

    __tablename__ = "iceberg_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: USNIC designator, e.g. "A76C", "B09B".
    iceberg_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    length_nm: Mapped[float | None] = mapped_column(Float, nullable=True)
    width_nm: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_km2: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Derived from the previous observation of the same berg (km/day, degrees).
    drift_speed_km_per_day: Mapped[float | None] = mapped_column(Float, nullable=True)
    drift_bearing_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_update: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("iceberg_id", "observed_at", "source", name="uq_iceberg_obs"),
        Index("ix_iceberg_track", "iceberg_id", "observed_at"),
    )


class WeatherObservation(Base, TimestampMixin, ProvenanceMixin):
    """Atmospheric state at a grid point (ERA5 reanalysis or demo field)."""

    __tablename__ = "weather_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    u10_m_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    v10_m_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_m_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    air_temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_sea_level_pressure_hpa: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("observed_at", "latitude", "longitude", "source", name="uq_weather_cell"),
        Index("ix_weather_time_pos", "observed_at", "latitude", "longitude"),
    )


class OceanObservation(Base, TimestampMixin, ProvenanceMixin):
    """Ocean state at a grid point (Copernicus Marine analysis or demo field)."""

    __tablename__ = "ocean_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    u_current_m_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    v_current_m_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    sea_surface_temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    salinity_psu: Mapped[float | None] = mapped_column(Float, nullable=True)
    significant_wave_height_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("observed_at", "latitude", "longitude", "source", name="uq_ocean_cell"),
        Index("ix_ocean_time_pos", "observed_at", "latitude", "longitude"),
    )


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------
class Forecast(Base, TimestampMixin):
    """A sea-ice concentration forecast for one grid cell and horizon."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    valid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_concentration: Mapped[float] = mapped_column(Float, nullable=False)
    #: 1-sigma model uncertainty derived from validation residuals.
    uncertainty: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    data_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="demo")

    __table_args__ = (
        UniqueConstraint(
            "issued_at", "horizon_hours", "latitude", "longitude", "model_name",
            name="uq_forecast_cell",
        ),
        Index("ix_forecast_lookup", "issued_at", "horizon_hours"),
    )


class IcebergTrajectory(Base, TimestampMixin):
    """One predicted position along an iceberg drift trajectory."""

    __tablename__ = "iceberg_trajectories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iceberg_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    valid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    horizon_hours: Mapped[float] = mapped_column(Float, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    speed_m_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    bearing_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Radius of the ensemble spread (km), i.e. positional uncertainty.
    uncertainty_radius_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, default="iceberg_dynamics")
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    data_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="demo")

    __table_args__ = (
        UniqueConstraint("iceberg_id", "issued_at", "horizon_hours", name="uq_iceberg_traj"),
    )


class RiskGrid(Base, TimestampMixin):
    """One cell of a generated navigation risk grid."""

    __tablename__ = "risk_grid"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    valid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    sea_ice_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    iceberg_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weather_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ocean_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    constraint_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    navigable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    grid_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    data_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="demo")

    __table_args__ = (
        UniqueConstraint("generated_at", "horizon_hours", "latitude", "longitude", name="uq_risk_cell"),
        Index("ix_risk_lookup", "generated_at", "horizon_hours"),
    )


class RouteResult(Base, TimestampMixin):
    """A computed route (one optimisation profile of one request)."""

    __tablename__ = "route_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    #: "shortest" | "safest" | "polaris"
    profile: Mapped[str] = mapped_column(String(32), nullable=False)
    start_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    end_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    start_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    start_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    end_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    end_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    duration_hours: Mapped[float] = mapped_column(Float, nullable=False)
    fuel_tonnes: Mapped[float] = mapped_column(Float, nullable=False)
    mean_risk: Mapped[float] = mapped_column(Float, nullable=False)
    max_risk: Mapped[float] = mapped_column(Float, nullable=False)
    n_waypoints: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="ok")
    #: GeoJSON LineString of the route.
    geometry: Mapped[dict] = mapped_column(JSON, nullable=False)
    #: Vessel parameters + cost weights actually used.
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="demo")


# ---------------------------------------------------------------------------
# Operations / bookkeeping
# ---------------------------------------------------------------------------
class IngestionLog(Base, TimestampMixin):
    """Audit trail of every ingestion attempt (success or failure)."""

    __tablename__ = "ingestion_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dataset: Mapped[str] = mapped_column(String(128), nullable=False)
    data_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="demo")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    records_ingested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ModelRegistry(Base, TimestampMixin):
    """Trained-model bookkeeping: what was trained, when, and how well."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_hours: Mapped[int | None] = mapped_column(Integer, nullable=True)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    n_train_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_val_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Real metrics computed on the held-out split. Never hand-written.
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    data_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="demo")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", "horizon_hours", name="uq_model_version"),
    )


ALL_TABLES = [
    SeaIceObservation,
    SeaIceExtentIndex,
    IcebergObservation,
    WeatherObservation,
    OceanObservation,
    Forecast,
    IcebergTrajectory,
    RiskGrid,
    RouteResult,
    IngestionLog,
    ModelRegistry,
]

#: Tables that carry a lat/lon point and can be accelerated with PostGIS.
POINT_TABLES = [
    "sea_ice_observations",
    "iceberg_observations",
    "weather_observations",
    "ocean_observations",
    "forecasts",
    "iceberg_trajectories",
    "risk_grid",
]
