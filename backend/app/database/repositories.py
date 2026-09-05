"""Repository layer: all SQL access lives here.

Every write is an idempotent UPSERT keyed on the table's natural key, so
re-running an ingestion never duplicates records.  The implementation picks the
right dialect-specific ``INSERT ... ON CONFLICT`` at runtime and falls back to a
portable merge loop for any other backend.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.database_models import (
    Forecast,
    IcebergObservation,
    IcebergTrajectory,
    IngestionLog,
    ModelRegistry,
    OceanObservation,
    RiskGrid,
    RouteResult,
    SeaIceExtentIndex,
    SeaIceObservation,
    WeatherObservation,
)
from app.utils.geo import GridSpec
from app.utils.logging import get_logger

log = get_logger("database.repositories")

_BATCH = 2000


# ---------------------------------------------------------------------------
# Generic upsert
# ---------------------------------------------------------------------------
def _natural_key(model) -> list[str]:
    """Column names of the model's first UniqueConstraint."""
    from sqlalchemy import UniqueConstraint

    for c in model.__table__.constraints:
        if isinstance(c, UniqueConstraint):
            return [col.name for col in c.columns]
    return [c.name for c in model.__table__.primary_key.columns]


def bulk_upsert(
    session: Session,
    model,
    rows: Sequence[dict[str, Any]],
    update: bool = True,
) -> int:
    """Insert ``rows``, updating existing records with the same natural key.

    Returns the number of rows submitted (not necessarily changed).
    """
    if not rows:
        return 0
    keys = _natural_key(model)
    dialect = session.bind.dialect.name if session.bind is not None else "sqlite"

    if dialect in ("sqlite", "postgresql"):
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        else:
            from sqlalchemy.dialects.postgresql import insert as dialect_insert

        total = 0
        for start in range(0, len(rows), _BATCH):
            chunk = rows[start : start + _BATCH]
            stmt = dialect_insert(model).values(chunk)
            if update:
                updatable = {
                    c.name: getattr(stmt.excluded, c.name)
                    for c in model.__table__.columns
                    if c.name not in keys and not c.primary_key
                }
                stmt = stmt.on_conflict_do_update(index_elements=keys, set_=updatable)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=keys)
            session.execute(stmt)
            total += len(chunk)
        session.flush()
        return total

    # Portable fallback (MySQL, MSSQL, ...): select-then-merge.
    total = 0
    for row in rows:
        conditions = [getattr(model, k) == row.get(k) for k in keys]
        existing = session.execute(select(model).where(*conditions)).scalar_one_or_none()
        if existing is None:
            session.add(model(**row))
        elif update:
            for k, v in row.items():
                setattr(existing, k, v)
        total += 1
    session.flush()
    return total


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Sea ice
# ---------------------------------------------------------------------------
def upsert_sea_ice_observations(session: Session, rows: Sequence[dict]) -> int:
    return bulk_upsert(session, SeaIceObservation, rows)


def upsert_sea_ice_extent(session: Session, rows: Sequence[dict]) -> int:
    return bulk_upsert(session, SeaIceExtentIndex, rows)


def latest_sea_ice_time(session: Session) -> datetime | None:
    return _as_utc(session.execute(select(func.max(SeaIceObservation.observed_at))).scalar())


def sea_ice_observation_days(session: Session) -> int:
    return int(
        session.execute(
            select(func.count(func.distinct(SeaIceObservation.observed_at)))
        ).scalar()
        or 0
    )


def get_sea_ice_field(
    session: Session, grid: GridSpec, observed_at: datetime | None = None
) -> tuple[np.ndarray, datetime | None, str | None]:
    """Reconstruct a gridded concentration field from stored observations.

    Returns ``(field, observed_at, source)``; the field is NaN where no
    observation exists for that cell.
    """
    if observed_at is None:
        observed_at = latest_sea_ice_time(session)
    if observed_at is None:
        return np.full(grid.shape, np.nan), None, None

    rows = session.execute(
        select(
            SeaIceObservation.latitude,
            SeaIceObservation.longitude,
            SeaIceObservation.concentration,
            SeaIceObservation.source,
        ).where(SeaIceObservation.observed_at == observed_at)
    ).all()
    field = np.full(grid.shape, np.nan)
    source = None
    for lat, lon, conc, src in rows:
        i, j = grid.index_of(lat, lon)
        field[i, j] = conc
        source = source or src
    return field, _as_utc(observed_at), source


def get_sea_ice_cells(
    session: Session,
    observed_at: datetime | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    limit: int | None = None,
) -> list[SeaIceObservation]:
    if observed_at is None:
        observed_at = latest_sea_ice_time(session)
    if observed_at is None:
        return []
    stmt = select(SeaIceObservation).where(SeaIceObservation.observed_at == observed_at)
    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        stmt = stmt.where(
            SeaIceObservation.latitude.between(lat_min, lat_max),
            SeaIceObservation.longitude.between(lon_min, lon_max),
        )
    if limit:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def prune_sea_ice_observations(session: Session, keep_days: int) -> int:
    """Keep only the most recent ``keep_days`` distinct observation dates."""
    dates = [
        d
        for (d,) in session.execute(
            select(SeaIceObservation.observed_at)
            .distinct()
            .order_by(SeaIceObservation.observed_at.desc())
        ).all()
    ]
    if len(dates) <= keep_days:
        return 0
    cutoff = dates[keep_days - 1]
    result = session.execute(
        delete(SeaIceObservation).where(SeaIceObservation.observed_at < cutoff)
    )
    return int(result.rowcount or 0)


def get_extent_series(
    session: Session, days: int = 90, hemisphere: str = "south"
) -> list[SeaIceExtentIndex]:
    stmt = (
        select(SeaIceExtentIndex)
        .where(SeaIceExtentIndex.hemisphere == hemisphere)
        .order_by(SeaIceExtentIndex.observed_at.desc())
        .limit(days)
    )
    return list(reversed(list(session.execute(stmt).scalars())))


# ---------------------------------------------------------------------------
# Icebergs
# ---------------------------------------------------------------------------
def upsert_icebergs(session: Session, rows: Sequence[dict]) -> int:
    return bulk_upsert(session, IcebergObservation, rows)


def latest_iceberg_positions(
    session: Session,
    bbox: tuple[float, float, float, float] | None = None,
    min_area_km2: float | None = None,
) -> list[IcebergObservation]:
    """Most recent observation per tracked iceberg."""
    sub = (
        select(
            IcebergObservation.iceberg_id,
            func.max(IcebergObservation.observed_at).label("latest"),
        )
        .group_by(IcebergObservation.iceberg_id)
        .subquery()
    )
    stmt = select(IcebergObservation).join(
        sub,
        (IcebergObservation.iceberg_id == sub.c.iceberg_id)
        & (IcebergObservation.observed_at == sub.c.latest),
    )
    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        stmt = stmt.where(
            IcebergObservation.latitude.between(lat_min, lat_max),
            IcebergObservation.longitude.between(lon_min, lon_max),
        )
    if min_area_km2 is not None:
        stmt = stmt.where(IcebergObservation.area_km2 >= min_area_km2)
    rows = list(session.execute(stmt).scalars())
    # Guard against duplicate joins if two sources share a timestamp.
    seen: dict[str, IcebergObservation] = {}
    for r in rows:
        seen.setdefault(r.iceberg_id, r)
    return sorted(seen.values(), key=lambda r: r.iceberg_id)


def get_iceberg_track(session: Session, iceberg_id: str) -> list[IcebergObservation]:
    stmt = (
        select(IcebergObservation)
        .where(func.upper(IcebergObservation.iceberg_id) == iceberg_id.upper())
        .order_by(IcebergObservation.observed_at.asc())
    )
    return list(session.execute(stmt).scalars())


def count_icebergs(session: Session) -> int:
    return int(
        session.execute(select(func.count(func.distinct(IcebergObservation.iceberg_id)))).scalar() or 0
    )


# ---------------------------------------------------------------------------
# Weather / ocean
# ---------------------------------------------------------------------------
def upsert_weather(session: Session, rows: Sequence[dict]) -> int:
    return bulk_upsert(session, WeatherObservation, rows)


def upsert_ocean(session: Session, rows: Sequence[dict]) -> int:
    return bulk_upsert(session, OceanObservation, rows)


def latest_weather_time(session: Session) -> datetime | None:
    return _as_utc(session.execute(select(func.max(WeatherObservation.observed_at))).scalar())


def latest_ocean_time(session: Session) -> datetime | None:
    return _as_utc(session.execute(select(func.max(OceanObservation.observed_at))).scalar())


def get_weather_fields(
    session: Session, grid: GridSpec, observed_at: datetime | None = None
) -> tuple[dict[str, np.ndarray], datetime | None]:
    if observed_at is None:
        observed_at = latest_weather_time(session)
    keys = ("u10", "v10", "t2m", "msl")
    fields = {k: np.full(grid.shape, np.nan) for k in keys}
    if observed_at is None:
        return fields, None
    rows = session.execute(
        select(
            WeatherObservation.latitude,
            WeatherObservation.longitude,
            WeatherObservation.u10_m_s,
            WeatherObservation.v10_m_s,
            WeatherObservation.air_temperature_c,
            WeatherObservation.mean_sea_level_pressure_hpa,
        ).where(WeatherObservation.observed_at == observed_at)
    ).all()
    for lat, lon, u, v, t, p in rows:
        i, j = grid.index_of(lat, lon)
        fields["u10"][i, j] = u if u is not None else np.nan
        fields["v10"][i, j] = v if v is not None else np.nan
        fields["t2m"][i, j] = t if t is not None else np.nan
        fields["msl"][i, j] = p if p is not None else np.nan
    return fields, _as_utc(observed_at)


def get_ocean_fields(
    session: Session, grid: GridSpec, observed_at: datetime | None = None
) -> tuple[dict[str, np.ndarray], datetime | None]:
    if observed_at is None:
        observed_at = latest_ocean_time(session)
    keys = ("u_current", "v_current", "sst", "salinity", "wave_height")
    fields = {k: np.full(grid.shape, np.nan) for k in keys}
    if observed_at is None:
        return fields, None
    rows = session.execute(
        select(
            OceanObservation.latitude,
            OceanObservation.longitude,
            OceanObservation.u_current_m_s,
            OceanObservation.v_current_m_s,
            OceanObservation.sea_surface_temperature_c,
            OceanObservation.salinity_psu,
            OceanObservation.significant_wave_height_m,
        ).where(OceanObservation.observed_at == observed_at)
    ).all()
    for lat, lon, u, v, sst, sal, hs in rows:
        i, j = grid.index_of(lat, lon)
        fields["u_current"][i, j] = u if u is not None else np.nan
        fields["v_current"][i, j] = v if v is not None else np.nan
        fields["sst"][i, j] = sst if sst is not None else np.nan
        fields["salinity"][i, j] = sal if sal is not None else np.nan
        fields["wave_height"][i, j] = hs if hs is not None else np.nan
    return fields, _as_utc(observed_at)


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------
def save_forecasts(session: Session, rows: Sequence[dict]) -> int:
    return bulk_upsert(session, Forecast, rows)


def latest_forecast_issue(session: Session, model_name: str | None = None) -> datetime | None:
    stmt = select(func.max(Forecast.issued_at))
    if model_name:
        stmt = stmt.where(Forecast.model_name == model_name)
    return _as_utc(session.execute(stmt).scalar())


def get_forecast_field(
    session: Session,
    grid: GridSpec,
    horizon_hours: int,
    issued_at: datetime | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(prediction, uncertainty, metadata)`` grids for one horizon."""
    if issued_at is None:
        issued_at = latest_forecast_issue(session)
    pred = np.full(grid.shape, np.nan)
    unc = np.full(grid.shape, np.nan)
    meta: dict = {"issued_at": None, "valid_at": None, "model_name": None, "model_version": None}
    if issued_at is None:
        return pred, unc, meta

    rows = session.execute(
        select(
            Forecast.latitude,
            Forecast.longitude,
            Forecast.predicted_concentration,
            Forecast.uncertainty,
            Forecast.valid_at,
            Forecast.model_name,
            Forecast.model_version,
        ).where(Forecast.issued_at == issued_at, Forecast.horizon_hours == horizon_hours)
    ).all()
    for lat, lon, p, u, valid_at, mname, mver in rows:
        i, j = grid.index_of(lat, lon)
        pred[i, j] = p
        unc[i, j] = u if u is not None else np.nan
        meta["valid_at"] = _as_utc(valid_at)
        meta["model_name"] = mname
        meta["model_version"] = mver
    meta["issued_at"] = _as_utc(issued_at)
    meta["n_cells"] = len(rows)
    return pred, unc, meta


def prune_forecasts(session: Session, keep_issues: int = 3) -> int:
    issues = [
        d
        for (d,) in session.execute(
            select(Forecast.issued_at).distinct().order_by(Forecast.issued_at.desc())
        ).all()
    ]
    if len(issues) <= keep_issues:
        return 0
    cutoff = issues[keep_issues - 1]
    return int(session.execute(delete(Forecast).where(Forecast.issued_at < cutoff)).rowcount or 0)


# ---------------------------------------------------------------------------
# Iceberg trajectories
# ---------------------------------------------------------------------------
def save_trajectory(session: Session, rows: Sequence[dict]) -> int:
    return bulk_upsert(session, IcebergTrajectory, rows)


def get_trajectory(
    session: Session, iceberg_id: str, issued_at: datetime | None = None
) -> list[IcebergTrajectory]:
    stmt = select(IcebergTrajectory).where(
        func.upper(IcebergTrajectory.iceberg_id) == iceberg_id.upper()
    )
    if issued_at is None:
        issued_at = _as_utc(
            session.execute(
                select(func.max(IcebergTrajectory.issued_at)).where(
                    func.upper(IcebergTrajectory.iceberg_id) == iceberg_id.upper()
                )
            ).scalar()
        )
    if issued_at is None:
        return []
    stmt = stmt.where(IcebergTrajectory.issued_at == issued_at).order_by(
        IcebergTrajectory.horizon_hours.asc()
    )
    return list(session.execute(stmt).scalars())


def latest_trajectories(session: Session) -> list[IcebergTrajectory]:
    latest = _as_utc(session.execute(select(func.max(IcebergTrajectory.issued_at))).scalar())
    if latest is None:
        return []
    return list(
        session.execute(
            select(IcebergTrajectory)
            .where(IcebergTrajectory.issued_at == latest)
            .order_by(IcebergTrajectory.iceberg_id, IcebergTrajectory.horizon_hours)
        ).scalars()
    )


# ---------------------------------------------------------------------------
# Risk grid
# ---------------------------------------------------------------------------
def save_risk_grid(session: Session, rows: Sequence[dict]) -> int:
    return bulk_upsert(session, RiskGrid, rows)


def latest_risk_generation(session: Session) -> datetime | None:
    return _as_utc(session.execute(select(func.max(RiskGrid.generated_at))).scalar())


def get_risk_cells(
    session: Session,
    generated_at: datetime | None = None,
    horizon_hours: int = 0,
    bbox: tuple[float, float, float, float] | None = None,
    min_total_risk: float | None = None,
    limit: int | None = None,
) -> list[RiskGrid]:
    if generated_at is None:
        generated_at = latest_risk_generation(session)
    if generated_at is None:
        return []
    stmt = select(RiskGrid).where(
        RiskGrid.generated_at == generated_at, RiskGrid.horizon_hours == horizon_hours
    )
    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        stmt = stmt.where(
            RiskGrid.latitude.between(lat_min, lat_max),
            RiskGrid.longitude.between(lon_min, lon_max),
        )
    if min_total_risk is not None:
        stmt = stmt.where(RiskGrid.total_risk >= min_total_risk)
    stmt = stmt.order_by(RiskGrid.total_risk.desc())
    if limit:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def prune_risk_grids(session: Session, keep: int = 2) -> int:
    gens = [
        d
        for (d,) in session.execute(
            select(RiskGrid.generated_at).distinct().order_by(RiskGrid.generated_at.desc())
        ).all()
    ]
    if len(gens) <= keep:
        return 0
    cutoff = gens[keep - 1]
    return int(session.execute(delete(RiskGrid).where(RiskGrid.generated_at < cutoff)).rowcount or 0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def save_routes(session: Session, rows: Sequence[dict]) -> list[int]:
    objs = [RouteResult(**r) for r in rows]
    session.add_all(objs)
    session.flush()
    return [o.id for o in objs]


def get_routes_by_request(session: Session, request_id: str) -> list[RouteResult]:
    return list(
        session.execute(
            select(RouteResult).where(RouteResult.request_id == request_id)
        ).scalars()
    )


def recent_routes(session: Session, limit: int = 10) -> list[RouteResult]:
    return list(
        session.execute(
            select(RouteResult).order_by(RouteResult.computed_at.desc()).limit(limit)
        ).scalars()
    )


# ---------------------------------------------------------------------------
# Ingestion log / model registry
# ---------------------------------------------------------------------------
def start_ingestion(session: Session, source: str, dataset: str, data_mode: str) -> IngestionLog:
    entry = IngestionLog(source=source, dataset=dataset, data_mode=data_mode, status="running")
    session.add(entry)
    session.flush()
    return entry


def finish_ingestion(
    session: Session,
    entry: IngestionLog,
    status: str,
    ingested: int = 0,
    rejected: int = 0,
    message: str | None = None,
    details: dict | None = None,
) -> None:
    entry.status = status
    entry.records_ingested = ingested
    entry.records_rejected = rejected
    entry.message = (message or "")[:4000] or None
    entry.details = details
    entry.finished_at = datetime.now(timezone.utc)
    session.flush()


def recent_ingestions(session: Session, limit: int = 20) -> list[IngestionLog]:
    return list(
        session.execute(
            select(IngestionLog).order_by(IngestionLog.started_at.desc()).limit(limit)
        ).scalars()
    )


def register_model(session: Session, row: dict) -> int:
    bulk_upsert(session, ModelRegistry, [row])
    rec = session.execute(
        select(ModelRegistry).where(
            ModelRegistry.model_name == row["model_name"],
            ModelRegistry.model_version == row["model_version"],
            ModelRegistry.horizon_hours == row.get("horizon_hours"),
        )
    ).scalar_one_or_none()
    return int(rec.id) if rec else 0


def get_model_records(session: Session, model_name: str | None = None) -> list[ModelRegistry]:
    stmt = select(ModelRegistry).order_by(ModelRegistry.trained_at.desc())
    if model_name:
        stmt = stmt.where(ModelRegistry.model_name == model_name)
    return list(session.execute(stmt).scalars())


def database_summary(session: Session) -> dict[str, Any]:
    """Row counts per table plus latest timestamps (used by /api/health)."""
    from app.models.database_models import ALL_TABLES

    counts: dict[str, int] = {}
    for model in ALL_TABLES:
        try:
            counts[model.__tablename__] = int(
                session.execute(select(func.count()).select_from(model)).scalar() or 0
            )
        except Exception:  # pragma: no cover - table may not exist yet
            counts[model.__tablename__] = -1
    return {
        "row_counts": counts,
        "latest_sea_ice": latest_sea_ice_time(session),
        "latest_weather": latest_weather_time(session),
        "latest_ocean": latest_ocean_time(session),
        "latest_forecast": latest_forecast_issue(session),
        "latest_risk_grid": latest_risk_generation(session),
        "tracked_icebergs": count_icebergs(session),
    }
