"""Step 3 of the pipeline: train and validate the sea-ice forecast models.

    python scripts/train_models.py                     # train 24/48/72 h
    python scripts/train_models.py --horizons 24       # one horizon
    python scripts/train_models.py --algorithm persistence
    python scripts/train_models.py --rebuild-history --days 250

Artifacts land in ``models/`` and a registry row is written for each one.

Every metric printed here is computed on a **held-out chronological split** of
the data that was actually used - the split never shuffles across time.  Two
learned candidates are trained and scored against a persistence baseline, and
the one with the lowest validation RMSE is saved; if neither beats the baseline,
the baseline itself is saved and the run says so.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the POLARIS sea-ice forecast models")
    parser.add_argument("--mode", choices=("real", "demo"), help="Override DATA_MODE for this run")
    parser.add_argument("--horizons", type=int, nargs="+", help="Horizons in hours (default 24 48 72)")
    parser.add_argument(
        "--algorithm", default="gbr", choices=("gbr", "persistence"),
        help="'gbr' trains the gradient-boosting candidates; 'persistence' fits only the baseline",
    )
    parser.add_argument("--days", type=int, default=None, help="History length when rebuilding")
    parser.add_argument("--rebuild-history", action="store_true", help="Rebuild the archives before training")
    parser.add_argument(
        "--subsample", type=int, default=2,
        help="Use every Nth time step when building the design matrix (2 keeps training fast)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.mode:
        os.environ["DATA_MODE"] = args.mode

    from app.config import reload_settings
    from app.database.connection import session_scope
    from app.database.init_db import create_all
    from app.services import sea_ice_forecasting as sif
    from app.services.environment import clear_cache
    from app.utils.logging import setup_logging

    settings = reload_settings()
    setup_logging(settings.log_level, settings.log_file)
    create_all()

    horizons = args.horizons or list(settings.forecast_horizons_hours)
    print(f"Training {settings.sea_ice_model_name} ({args.algorithm}) for horizons {horizons}h "
          f"| DATA_MODE={settings.data_mode}")
    started = time.time()
    with session_scope() as session:
        reports = sif.train_all(
            session=session,
            settings=settings,
            horizons=horizons,
            algorithm=args.algorithm,
            days=args.days,
            rebuild_history=args.rebuild_history,
            subsample=args.subsample,
        )
    clear_cache()

    if args.json:
        print(json.dumps(reports, indent=2, default=str))
        print(f"Completed in {time.time() - started:.1f}s")
        return 0

    print()
    print(
        f"{'horizon':>8s} {'selected':>12s} {'train':>9s} {'val':>8s} "
        f"{'MAE':>8s} {'RMSE':>8s} {'R2':>7s} {'persist':>8s} {'skill':>7s} {'edgeAcc':>8s}"
    )
    print("-" * 96)
    for horizon, report in reports.items():
        m = report["metrics"]
        print(
            f"{horizon:6d}h {m.get('selected_candidate', '-'):>12s} "
            f"{report['n_train_samples']:9d} {report['n_val_samples']:8d} "
            f"{m['mae']:8.4f} {m['rmse']:8.4f} {m['r2']:7.3f} "
            f"{m['persistence_baseline']['rmse']:8.4f} {m['skill_vs_persistence']:+7.3f} "
            f"{m['edge']['accuracy']:8.3f}"
        )

    print()
    print("skill    = 1 - MSE(model)/MSE(persistence); positive means it beats the baseline.")
    print("edgeAcc  = ice/no-ice agreement at the 15% concentration threshold.")
    print("selected = the candidate with the lowest held-out RMSE. 'persistence' means no learned")
    print("           candidate beat the baseline on this data - a real measurement, not a failure.")

    print()
    print("All candidates (held-out RMSE / skill vs persistence):")
    for horizon, report in reports.items():
        parts = [
            f"{name}={c['rmse']:.4f}/{c['skill_vs_persistence']:+.3f}"
            for name, c in report["metrics"].get("candidates", {}).items()
        ]
        print(f"  {horizon:>4}h  " + "  ".join(parts))

    print()
    for horizon, report in reports.items():
        top = list(report.get("top_features", {}).items())[:5]
        if top:
            print(f"  {horizon}h top predictors (RMSE increase when shuffled): "
                  + ", ".join(f"{k}={v:.4f}" for k, v in top))
        note = report["metrics"].get("note")
        if note:
            print(f"  {horizon}h note: {note}")

    print(f"\nArtifacts written to {settings.model_path}")
    print(f"Completed in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
