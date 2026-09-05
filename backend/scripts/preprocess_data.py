"""Step 2 of the pipeline: build the model-ready history archives.

    python scripts/preprocess_data.py                 # default history length
    python scripts/preprocess_data.py --days 200      # explicit length
    python scripts/preprocess_data.py --no-weather    # sea-ice predictors only

Produces ``data/processed/sea_ice_history.nc`` (and ``weather_history.nc``),
the ``(time, lat, lon)`` archives the forecast model trains on.

In real mode this downloads one NSIDC GeoTIFF per day, caching them under
``data/raw/nsidc`` - the first run over a long period takes a while; subsequent
runs reuse the cache.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time
from datetime import date

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the POLARIS history archives")
    parser.add_argument("--mode", choices=("real", "demo"), help="Override DATA_MODE for this run")
    parser.add_argument("--days", type=int, default=None, help="Days of history (default: SEA_ICE_HISTORY_DAYS)")
    parser.add_argument("--end-date", help="Last day of the archive (YYYY-MM-DD)")
    parser.add_argument("--no-weather", action="store_true", help="Skip the atmospheric-driver archive")
    parser.add_argument("--no-drift", action="store_true", help="Skip the sea-ice drift archive")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.mode:
        os.environ["DATA_MODE"] = args.mode

    from app.config import reload_settings
    from app.services import preprocessing
    from app.utils.geo import grid_from_settings
    from app.utils.logging import setup_logging

    settings = reload_settings()
    setup_logging(settings.log_level, settings.log_file)
    grid = grid_from_settings(settings)
    days = args.days or settings.sea_ice_history_days
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    print(f"Building sea-ice history: {days} day(s), DATA_MODE={settings.data_mode}, grid {grid.shape}")
    started = time.time()
    times, ice = preprocessing.build_sea_ice_history(
        days=days, end_date=end_date, grid=grid, settings=settings
    )
    finite = np.isfinite(ice)
    summary = {
        "sea_ice_history": {
            "path": str(preprocessing.history_path(settings)),
            "days": len(times),
            "start": times[0].isoformat(),
            "end": times[-1].isoformat(),
            "shape": list(ice.shape),
            "valid_cell_fraction": round(float(finite.mean()), 4),
            "mean_concentration": round(float(np.nanmean(ice)), 4),
        }
    }

    if not args.no_weather:
        print("Building weather history ...")
        weather = preprocessing.build_weather_history(times, grid, settings)
        summary["weather_history"] = (
            {
                "path": str(preprocessing.history_path(settings, preprocessing.WEATHER_HISTORY_FILE)),
                "variables": sorted(weather.keys()),
                "shape": list(next(iter(weather.values())).shape),
            }
            if weather
            else {"status": "unavailable", "note": "training will use sea-ice predictors only"}
        )
    else:
        summary["weather_history"] = {"status": "skipped"}

    if not args.no_drift:
        print("Building sea-ice drift history ...")
        drift = preprocessing.build_ice_drift_history(times, grid, settings)
        summary["ice_drift_history"] = (
            {
                "path": str(preprocessing.history_path(settings, preprocessing.ICE_DRIFT_HISTORY_FILE)),
                "variables": sorted(drift),
                "shape": list(next(iter(drift.values())).shape),
            }
            if drift
            else {"status": "unavailable", "note": "training will run without the dynamic terms"}
        )
    else:
        summary["ice_drift_history"] = {"status": "skipped"}

    elapsed = time.time() - started
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for section, payload in summary.items():
            print(f"\n[{section}]")
            for k, v in payload.items():
                print(f"  {k}: {v}")
    print(f"\nCompleted in {elapsed:.1f}s")

    if len(times) < settings.sea_ice_min_training_days:
        print(
            f"\nWARNING: only {len(times)} day(s) of history; training needs at least "
            f"{settings.sea_ice_min_training_days}. Re-run with a larger --days."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
