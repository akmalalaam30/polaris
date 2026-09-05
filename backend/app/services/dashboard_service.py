"""Dashboard aggregation: one compact snapshot of the whole system.

Everything here reads from the same cached products the individual endpoints
use, so the dashboard can never disagree with the detail views.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.database import repositories as repo
from app.services import environment as envsvc
from app.services import risk_engine as rk
from app.services import sea_ice_forecasting as sif
from app.utils.geo import GridSpec, grid_from_settings
from app.utils.logging import get_logger

log = get_logger("services.dashboard")


def _safe(fn, default=None, label: str = ""):
    """Run a sub-query, degrading to ``default`` instead of failing the page."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the dashboard must stay available
        log.warning("Dashboard section %r unavailable: %s", label, exc)
        return default


def sea_ice_section(session, settings: Settings, grid: GridSpec) -> dict:
    env = envsvc.get_environment(session, settings, grid)
    stats = env.statistics()
    extent = repo.get_extent_series(session, days=2)
    return {
        "observed_at": env.sea_ice_observed_at,
        "available": env.has_sea_ice,
        "mean_concentration": stats.get("mean_concentration"),
        "ice_covered_fraction": stats.get("ice_covered_fraction"),
        "ice_edge_latitude_mean": stats.get("ice_edge_latitude_mean"),
        "ocean_cells": stats.get("ocean_cells"),
        "hemispheric_extent_million_km2": extent[-1].extent_million_km2 if extent else None,
        "sources": env.sources,
    }


def forecast_section(session, settings: Settings, grid: GridSpec) -> dict:
    horizons: dict[str, Any] = {}
    for h in settings.forecast_horizons_hours:
        def build(h=h):
            fc = envsvc.get_forecast(h, settings, grid)
            return {
                "status": "ok",
                "algorithm": fc.algorithm,
                "issued_at": fc.issued_at,
                "valid_at": fc.valid_at,
                "mean_concentration": fc.mean_concentration,
                "ice_covered_fraction": fc.ice_covered_fraction,
                "validation_rmse": fc.metrics.get("rmse"),
                "skill_vs_persistence": fc.metrics.get("skill_vs_persistence"),
            }

        horizons[str(h)] = _safe(build, {"status": "unavailable"}, f"forecast_{h}")
    return {
        "horizons": horizons,
        "trained_models": _safe(lambda: list(sif.available_models(settings).keys()), [], "models"),
    }


def iceberg_section(session, settings: Settings, grid: GridSpec) -> dict:
    ctx = envsvc.get_iceberg_context(session, settings, grid, horizon_hours=72)
    areas = [r["area_km2"] for r in ctx.records if r.get("area_km2")]
    moving = [
        r for r in ctx.records
        if r.get("drift_speed_m_s") is not None and r["drift_speed_m_s"] > 0.01
    ]
    largest = max(ctx.records, key=lambda r: r.get("area_km2") or 0.0, default=None)
    return {
        "tracked_count": ctx.count,
        "with_trajectories": len(ctx.trajectories),
        "total_area_km2": round(float(np.sum(areas)), 1) if areas else None,
        "largest": (
            {
                "iceberg_id": largest["iceberg_id"],
                "area_km2": largest.get("area_km2"),
                "latitude": largest["latitude"],
                "longitude": largest["longitude"],
            }
            if largest
            else None
        ),
        "with_derived_drift": len(moving),
        "issued_at": ctx.issued_at,
        "sources": sorted({r.get("source") for r in ctx.records if r.get("source")}),
    }


def risk_section(session, settings: Settings, grid: GridSpec, top_n: int = 5) -> dict:
    result = envsvc.get_risk_grid(session, settings, grid, horizon_hours=0)
    summary = result.summary()
    lat2d, lon2d = grid.meshgrid()
    from app.services import landmask

    ocean = landmask.navigable_mask(grid)
    masked = np.where(ocean, result.total, -np.inf)
    flat = masked.ravel()
    order = np.argsort(flat)[::-1][:top_n]
    hotspots = []
    for idx in order:
        i, j = divmod(int(idx), grid.shape[1])
        if not np.isfinite(flat[idx]):
            continue
        contributions = {k: float(v[i, j]) for k, v in result.components.items()}
        hotspots.append(
            {
                "latitude": float(lat2d[i, j]),
                "longitude": float(lon2d[i, j]),
                "total_risk": float(result.total[i, j]),
                "dominant_component": max(contributions, key=contributions.get),
            }
        )
    return {
        "generated_at": summary["generated_at"],
        "mean_risk": summary["mean_risk"],
        "max_risk": summary["max_risk"],
        "cells_navigable": summary["cells_navigable"],
        "cells_blocked": summary["cells_blocked"],
        "available_layers": summary["available_layers"],
        "missing_layers": summary["missing_layers"],
        "weights": summary["weights"],
        "highest_risk_areas": hotspots,
    }


def route_availability(session, settings: Settings, grid: GridSpec) -> dict:
    """Can routes be produced right now, and between which stations?"""
    from app.config import STATIONS
    from app.services import landmask
    from app.services.route_optimizer import build_navigation_graph

    result = envsvc.get_risk_grid(session, settings, grid, horizon_hours=0)
    nav = build_navigation_graph(result, None, result.vessel, allow_high_risk=False)
    stations = {}
    for key, meta in STATIONS.items():
        in_domain = grid.contains(meta["latitude"], meta["longitude"])
        entry = {"name": meta["name"], "in_domain": in_domain}
        if in_domain and nav.n_nodes:
            _node, snap = nav.nearest_node(meta["latitude"], meta["longitude"])
            entry["nearest_navigable_km"] = round(snap, 1)
        stations[key] = entry
    return {
        "available": nav.n_nodes > 0,
        "graph_nodes": nav.n_nodes,
        "graph_edges": nav.n_edges,
        "stations": stations,
        "recent_routes": len(repo.recent_routes(session, limit=5)),
    }


def ingestion_section(session) -> dict:
    entries = repo.recent_ingestions(session, limit=8)
    return {
        "recent": [
            {
                "source": e.source,
                "dataset": e.dataset,
                "status": e.status,
                "records": e.records_ingested,
                "started_at": e.started_at,
                "message": (e.message or "")[:200] or None,
            }
            for e in entries
        ],
        "last_success": next((e.started_at for e in entries if e.status == "ok"), None),
    }


def build_summary(session, settings: Settings | None = None, grid: GridSpec | None = None) -> dict:
    """Compose the full dashboard payload."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    return {
        "generated_at": datetime.now(timezone.utc),
        "data_mode": settings.data_mode,
        "domain": grid.to_dict(),
        "sea_ice": _safe(lambda: sea_ice_section(session, settings, grid), {}, "sea_ice"),
        "forecast": _safe(lambda: forecast_section(session, settings, grid), {}, "forecast"),
        "icebergs": _safe(lambda: iceberg_section(session, settings, grid), {}, "icebergs"),
        "risk": _safe(lambda: risk_section(session, settings, grid), {}, "risk"),
        "routing": _safe(lambda: route_availability(session, settings, grid), {}, "routing"),
        "ingestion": _safe(lambda: ingestion_section(session), {}, "ingestion"),
        "database": _safe(lambda: repo.database_summary(session), {}, "database"),
        "disclaimer": envsvc.disclaimer(settings),
    }
