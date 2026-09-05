"""Shared, cached access to the current environmental state.

Every API endpoint needs some slice of the same picture: the latest observed
fields, the sea-ice forecast, the iceberg trajectories, and the risk grid built
from them.  Recomputing that per request would make the API slow and would also
let two endpoints disagree with each other.

This module owns one small cache keyed on *the data itself* - the latest
observation timestamps plus the parameters that affect the result - so a cache
entry is invalidated automatically as soon as new data is ingested, and two
requests made against the same data always see the same numbers.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from app.config import DEMO_DATA_DISCLAIMER, REAL_DATA_DISCLAIMER, Settings, get_settings
from app.database import repositories as repo
from app.services import iceberg_trajectory as itraj
from app.services import risk_engine as rk
from app.services import sea_ice_forecasting as sif
from app.services.preprocessing import preprocess_iceberg_tracks, synchronise_environment
from app.utils.geo import GridSpec, grid_from_settings
from app.utils.logging import get_logger

log = get_logger("services.environment")

#: Entries older than this are refreshed even if the data key is unchanged.
CACHE_TTL_SECONDS = 900.0

_LOCK = threading.RLock()
_CACHE: dict[str, tuple[float, Any]] = {}
#: One build lock per cache key, so concurrent callers wait for a single build
#: instead of each doing the same expensive work.
_BUILD_LOCKS: dict[str, threading.Lock] = {}


def _cached(key: str, builder: Callable[[], Any], ttl: float = CACHE_TTL_SECONDS) -> Any:
    """Return a cached product, building it at most once across threads.

    Without the per-key build lock every request that arrives while a product
    is still being built starts its own copy.  That is not just wasted work: on
    a cold start the background warm-up and the first user request end up
    computing the same iceberg ensemble simultaneously and competing for CPU,
    which makes the request *slower* than having no warm-up at all.
    """
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and (now - hit[0]) < ttl:
            return hit[1]
        build_lock = _BUILD_LOCKS.setdefault(key, threading.Lock())

    with build_lock:
        # Re-check: another thread may have finished while we waited.
        with _LOCK:
            hit = _CACHE.get(key)
            if hit and (time.monotonic() - hit[0]) < ttl:
                return hit[1]
        value = builder()
        with _LOCK:
            _CACHE[key] = (time.monotonic(), value)
    return value


def clear_cache() -> None:
    """Drop every cached product (used by tests and after ingestion runs)."""
    with _LOCK:
        _CACHE.clear()
        _BUILD_LOCKS.clear()
    sif.clear_model_cache()


def _key(*parts: Any) -> str:
    payload = json.dumps(parts, default=str, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def disclaimer(settings: Settings) -> str:
    return DEMO_DATA_DISCLAIMER if settings.is_demo else REAL_DATA_DISCLAIMER


# ---------------------------------------------------------------------------
# Observed environment
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentSnapshot:
    """The latest observed environmental fields on the analysis grid."""

    grid: GridSpec
    fields: dict[str, np.ndarray]
    sea_ice_observed_at: datetime | None
    weather_observed_at: datetime | None
    ocean_observed_at: datetime | None
    sources: list[str]
    data_mode: str

    @property
    def observed_at(self) -> datetime | None:
        stamps = [t for t in (self.sea_ice_observed_at, self.weather_observed_at, self.ocean_observed_at) if t]
        return max(stamps) if stamps else None

    @property
    def has_sea_ice(self) -> bool:
        return bool(np.isfinite(self.fields.get("sea_ice_concentration", np.array([np.nan]))).any())

    def statistics(self) -> dict:
        from app.services import landmask

        conc = self.fields.get("sea_ice_concentration")
        ocean = landmask.navigable_mask(self.grid)
        out: dict[str, Any] = {"ocean_cells": int(ocean.sum())}
        if conc is not None and np.isfinite(conc).any():
            vals = conc[ocean & np.isfinite(conc)]
            out["mean_concentration"] = float(np.mean(vals)) if vals.size else None
            out["max_concentration"] = float(np.max(vals)) if vals.size else None
            out["ice_covered_fraction"] = float(np.mean(vals >= 0.15)) if vals.size else None
            # Mean latitude of the 15% edge, computed per longitude column.
            edges = []
            for j in range(self.grid.shape[1]):
                col = conc[:, j]
                idx = np.where(np.isfinite(col) & (col >= 0.15))[0]
                if idx.size:
                    edges.append(self.grid.lats[idx.max()])
            out["ice_edge_latitude_mean"] = float(np.mean(edges)) if edges else None
        return out


def get_environment(session, settings: Settings | None = None, grid: GridSpec | None = None) -> EnvironmentSnapshot:
    """Latest observed fields, synchronised onto the analysis grid."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)

    ice_t = repo.latest_sea_ice_time(session)
    wx_t = repo.latest_weather_time(session)
    ocn_t = repo.latest_ocean_time(session)
    key = "env:" + _key(grid.to_dict(), ice_t, wx_t, ocn_t, settings.data_mode)

    def build() -> EnvironmentSnapshot:
        ice, ice_time, ice_source = repo.get_sea_ice_field(session, grid, ice_t)
        weather, weather_time = repo.get_weather_fields(session, grid, wx_t)
        ocean, ocean_time = repo.get_ocean_fields(session, grid, ocn_t)
        fields = synchronise_environment(grid, ice if np.isfinite(ice).any() else None, weather, ocean)
        sources = sorted({s for s in (ice_source,) if s})
        return EnvironmentSnapshot(
            grid=grid,
            fields=fields,
            sea_ice_observed_at=ice_time,
            weather_observed_at=weather_time,
            ocean_observed_at=ocean_time,
            sources=sources,
            data_mode=settings.data_mode,
        )

    return _cached(key, build)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
def get_forecast(horizon_hours: int, settings: Settings | None = None, grid: GridSpec | None = None):
    """Cached sea-ice forecast for one horizon."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    from app.services.preprocessing import history_path

    path = history_path(settings)
    stamp = path.stat().st_mtime if path.exists() else 0.0
    key = "fc:" + _key(horizon_hours, grid.to_dict(), settings.data_mode, stamp)
    return _cached(key, lambda: sif.forecast(horizon_hours, settings, grid))


# ---------------------------------------------------------------------------
# Icebergs
# ---------------------------------------------------------------------------
@dataclass
class IcebergContext:
    """Latest iceberg positions with their predicted trajectories."""

    issued_at: datetime
    horizon_hours: float
    records: list[dict] = field(default_factory=list)
    trajectories: dict[str, itraj.TrajectoryResult] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.records)


def get_iceberg_context(
    session,
    settings: Settings | None = None,
    grid: GridSpec | None = None,
    horizon_hours: float = 72.0,
    with_trajectories: bool = True,
) -> IcebergContext:
    """Latest bergs plus their drift forecasts, cached per data state."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    env = get_environment(session, settings, grid)
    bergs = repo.latest_iceberg_positions(session)
    stamps = [b.observed_at for b in bergs if b.observed_at]
    # Normalise the horizon: 72 and 72.0 are the same request, but they
    # serialise differently and would otherwise be two cache entries - which is
    # exactly how a warmed cache silently fails to be used.
    horizon_hours = float(horizon_hours)
    key = "berg:" + _key(
        max(stamps) if stamps else None, len(bergs), horizon_hours, with_trajectories,
        env.observed_at, grid.to_dict(),
    )

    def build() -> IcebergContext:
        issued_at = datetime.now(timezone.utc).replace(microsecond=0)
        states, records = [], []
        for b in bergs:
            track_rows = repo.get_iceberg_track(session, b.iceberg_id)
            derived = preprocess_iceberg_tracks(track_rows)
            last = derived.iloc[-1] if not derived.empty else None
            record = {
                "iceberg_id": b.iceberg_id,
                "observed_at": b.observed_at,
                "latitude": b.latitude,
                "longitude": b.longitude,
                "length_nm": b.length_nm,
                "width_nm": b.width_nm,
                "area_km2": b.area_km2,
                "source": b.source,
                "data_mode": b.data_mode,
                "n_observations": len(track_rows),
                "drift_speed_m_s": (
                    float(last["drift_speed_m_s"])
                    if last is not None and np.isfinite(last.get("drift_speed_m_s", np.nan))
                    else None
                ),
                "drift_bearing_deg": (
                    float(last["drift_bearing_deg"])
                    if last is not None and np.isfinite(last.get("drift_bearing_deg", np.nan))
                    else None
                ),
                "observed_track": [
                    {
                        "observed_at": r["observed_at"],
                        "latitude": float(r["latitude"]),
                        "longitude": float(r["longitude"]),
                        "drift_speed_m_s": (
                            float(r["drift_speed_m_s"]) if np.isfinite(r["drift_speed_m_s"]) else None
                        ),
                        "drift_bearing_deg": (
                            float(r["drift_bearing_deg"]) if np.isfinite(r["drift_bearing_deg"]) else None
                        ),
                    }
                    for _, r in derived.iterrows()
                ],
                "track": [],
            }
            record["in_domain"] = bool(grid.contains(b.latitude, b.longitude))
            records.append(record)
            if not record["in_domain"]:
                # Listed, but not integrated: there is no forcing field outside
                # the domain, so a trajectory would be fabricated.
                continue
            state = itraj.state_from_observation(
                b.iceberg_id, b.latitude, b.longitude, b.length_nm, b.width_nm, b.area_km2
            )
            # Seed the integration with the observed drift when we have one.
            if record["drift_speed_m_s"] and record["drift_bearing_deg"] is not None:
                brg = np.radians(record["drift_bearing_deg"])
                state.u_m_s = float(record["drift_speed_m_s"] * np.sin(brg))
                state.v_m_s = float(record["drift_speed_m_s"] * np.cos(brg))
            states.append(state)

        trajectories: dict[str, itraj.TrajectoryResult] = {}
        if with_trajectories and states:
            trajectories = itraj.predict_many(
                states, grid, env.fields, issued_at,
                horizon_hours=horizon_hours, output_every_hours=6.0, settings=settings,
            )
            by_id = {r["iceberg_id"]: r for r in records}
            for berg_id, result in trajectories.items():
                by_id[berg_id]["track"] = [p.to_dict() for p in result.points]
        return IcebergContext(
            issued_at=issued_at, horizon_hours=horizon_hours, records=records, trajectories=trajectories
        )

    return _cached(key, build)


# ---------------------------------------------------------------------------
# Risk grid
# ---------------------------------------------------------------------------
def get_risk_grid(
    session,
    settings: Settings | None = None,
    grid: GridSpec | None = None,
    vessel: rk.VesselProfile | None = None,
    weights: rk.RiskWeights | None = None,
    horizon_hours: int = 0,
) -> rk.RiskGridResult:
    """Cached risk grid for a vessel/weight/horizon combination."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    vessel = vessel or rk.VesselProfile()
    weights = weights or rk.RiskWeights.from_settings(settings)
    horizon_hours = int(horizon_hours)

    env = get_environment(session, settings, grid)
    berg_ctx = get_iceberg_context(session, settings, grid, horizon_hours=max(horizon_hours, 72))
    key = "risk:" + _key(
        env.observed_at, berg_ctx.issued_at.date().isoformat(), berg_ctx.count,
        vessel.to_dict(), weights.as_dict(), horizon_hours, grid.to_dict(),
    )

    def build() -> rk.RiskGridResult:
        forecast_field = None
        valid_at = env.observed_at
        if horizon_hours:
            try:
                fc = get_forecast(horizon_hours, settings, grid)
                forecast_field = fc.prediction
                valid_at = fc.valid_at
            except Exception as exc:  # noqa: BLE001 - risk grid still works without it
                log.warning("Risk grid: forecast for %dh unavailable (%s)", horizon_hours, exc)
        return rk.compute_risk_grid(
            grid, env.fields, berg_ctx.records, forecast_field, vessel, weights,
            settings=settings, horizon_hours=horizon_hours, valid_at=valid_at,
        )

    return _cached(key, build)
