"""One-command DEMO bootstrap.

    python scripts/seed_demo_data.py

Runs the whole pipeline in demo mode so the backend is immediately usable:

    init schema -> ingest synthetic observations -> build history archives ->
    train the forecast models -> generate forecasts, trajectories and a risk
    grid -> compute the Bharati <-> Maitri demonstration routes

Everything written carries ``source="POLARIS-DEMO"`` and ``data_mode="demo"``.
The synthetic fields are *physically shaped* but they are **not observations**,
and no endpoint will ever present them as such.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time
from datetime import datetime, timezone


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed POLARIS with the demo dataset")
    parser.add_argument("--days", type=int, default=120, help="Days of synthetic history to generate")
    parser.add_argument("--ingest-days", type=int, default=5, help="Days of gridded observations to store")
    parser.add_argument("--skip-training", action="store_true", help="Do not train the forecast models")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables first")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ["DATA_MODE"] = "demo"

    from app.config import reload_settings
    from app.database.connection import session_scope
    from app.database.init_db import init_database
    from app.services import data_ingestion as ing
    from app.services import iceberg_trajectory as itraj
    from app.services import preprocessing
    from app.services import risk_engine as rk
    from app.services import route_optimizer as ro
    from app.services import sea_ice_forecasting as sif
    from app.services.environment import (
        clear_cache,
        get_environment,
        get_iceberg_context,
        get_risk_grid,
    )
    from app.utils.geo import grid_from_settings
    from app.utils.logging import setup_logging

    settings = reload_settings()
    setup_logging(settings.log_level, settings.log_file)
    grid = grid_from_settings(settings)
    report: dict = {"data_mode": settings.data_mode, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.time()

    def step(n: int, label: str) -> None:
        print(f"\n[{n}/7] {label}")

    step(1, "Initialising the database schema")
    report["database"] = init_database(reset=args.reset)

    step(2, f"Ingesting {args.ingest_days} day(s) of synthetic observations")
    with session_scope() as session:
        report["ingestion"] = ing.ingest_all(session, days=args.ingest_days, settings=settings, grid=grid)
    for name, r in report["ingestion"].items():
        print(f"    {name:14s} {r['status']:8s} {r['ingested']:6d} rows")

    step(3, f"Building the {args.days}-day history archives")
    times, ice = preprocessing.build_sea_ice_history(days=args.days, grid=grid, settings=settings)
    weather = preprocessing.build_weather_history(times, grid, settings)
    report["history"] = {
        "days": len(times),
        "start": times[0].isoformat(),
        "end": times[-1].isoformat(),
        "weather_variables": sorted(weather) if weather else [],
    }
    print(f"    {len(times)} days, {times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d}")

    if args.skip_training:
        step(4, "Skipping model training (--skip-training)")
        report["training"] = {"status": "skipped"}
    else:
        step(4, "Training the sea-ice forecast models")
        with session_scope() as session:
            reports = sif.train_all(session=session, settings=settings, subsample=2)
        report["training"] = reports
        for horizon, r in reports.items():
            m = r["metrics"]
            print(
                f"    {horizon}h: RMSE {m['rmse']:.4f} (persistence {m['persistence_baseline']['rmse']:.4f}), "
                f"skill {m['skill_vs_persistence']:+.3f}, ice-edge accuracy {m['edge']['accuracy']:.3f}"
            )

    clear_cache()

    step(5, "Generating and storing forecasts")
    with session_scope() as session:
        report["forecasts"] = sif.run_forecast_cycle(session, settings, grid)
    for horizon, r in report["forecasts"].items():
        print(f"    {horizon}h: {r.get('status')} ({r.get('cells', 0)} cells, {r.get('algorithm', '-')})")

    step(6, "Predicting iceberg trajectories and generating the risk grid")
    with session_scope() as session:
        ctx = get_iceberg_context(session, settings, grid, horizon_hours=72)
        n_traj = itraj.persist_trajectories(session, ctx.trajectories)
        risk = get_risk_grid(session, settings, grid, horizon_hours=24)
        n_risk = rk.persist_risk_grid(session, risk)
    report["trajectories"] = {"icebergs": len(ctx.trajectories), "points": n_traj}
    report["risk_grid"] = {"cells": n_risk, "mean_risk": risk.mean_risk}
    print(f"    {len(ctx.trajectories)} trajectories ({n_traj} points); risk grid {n_risk} cells, "
          f"mean risk {risk.mean_risk:.3f}")

    step(7, "Computing the Bharati <-> Maitri demonstration routes")
    with session_scope() as session:
        env = get_environment(session, settings, grid)
        result = ro.optimize_routes(
            risk, "bharati", "maitri", fields=env.fields, vessel=risk.vessel,
            settings=settings, icebergs=ctx.records,
        )
        ids = ro.persist_routes(session, result, settings.data_mode)
    report["demo_route"] = {"request_id": result["request_id"], "persisted_ids": ids,
                            "comparison": result["comparison"]}
    for profile, route in result["routes"].items():
        print(
            f"    {profile:9s} {route['distance_km']:8.1f} km  {route['duration_hours']:7.1f} h  "
            f"{route['estimated_fuel_tonnes']:8.1f} t  risk {route['mean_risk']:.3f}  [{route['status']}]"
        )
    if any(r["status"] == "degraded" for r in result["routes"].values()):
        # This is the physically correct answer at peak sea-ice extent: an
        # ice-strengthened but non-icebreaking hull cannot reach either station
        # through consolidated pack. Rather than leave the demo looking broken,
        # re-run the same request for a vessel that *can* make the passage, so
        # the constraint and its resolution are both visible.
        print("    NOTE: no path satisfies every navigability constraint for a "
              f"{risk.vessel.ice_class} hull in these ice conditions -")
        print("          that is the correct answer at this point in the sea-ice season.")
        print("    Re-running for an icebreaker-capable vessel:")
        icebreaker = rk.VesselProfile(
            name="icebreaker-capable research vessel", ice_class="icebreaker",
            speed_knots=14.0, fuel_consumption_tpd=45.0, risk_tolerance=0.3,
        )
        with session_scope() as session:
            ib_risk = get_risk_grid(session, settings, grid, vessel=icebreaker, horizon_hours=24)
            ib_result = ro.optimize_routes(
                ib_risk, "bharati", "maitri", fields=env.fields, vessel=icebreaker,
                settings=settings, icebergs=ctx.records,
            )
            ib_ids = ro.persist_routes(session, ib_result, settings.data_mode)
        report["demo_route_icebreaker"] = {
            "request_id": ib_result["request_id"], "persisted_ids": ib_ids,
            "comparison": ib_result["comparison"],
        }
        for profile, route in ib_result["routes"].items():
            print(
                f"    {profile:9s} {route['distance_km']:8.1f} km  {route['duration_hours']:7.1f} h  "
                f"{route['estimated_fuel_tonnes']:8.1f} t  risk {route['mean_risk']:.3f}  [{route['status']}]"
            )

    report["elapsed_seconds"] = round(time.time() - started, 1)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    print(f"\nDemo data ready in {report['elapsed_seconds']}s. Start the API with 'python run.py'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
