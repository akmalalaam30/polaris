"""Step 1 of the pipeline: ingest environmental data into the database.

    python scripts/download_data.py                 # honour DATA_MODE from the environment
    python scripts/download_data.py --mode real     # force real providers
    python scripts/download_data.py --days 7        # more days of sea-ice grids
    python scripts/download_data.py --only icebergs # one dataset
    python scripts/download_data.py --landmask      # rebuild the land mask asset

In real mode this contacts NSIDC, the US National Ice Center, and the ERA5 /
Copernicus Marine services (or their key-free mirrors when no credentials are
configured).  Each dataset is reported separately; a failure in one does not
abort the others, and nothing is ever silently replaced by demo data.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (path setup)

import argparse
import json
import os
import sys
import time
from datetime import date, datetime

DATASETS = ("sea_ice", "extent_index", "icebergs", "weather", "ocean")


def rebuild_landmask(settings) -> str:
    """Re-derive the land mask from a freshly downloaded NSIDC GeoTIFF."""
    from datetime import timedelta

    from app.services.data_ingestion import NSIDCClient
    from app.services.landmask import build_land_mask_from_nsidc, clear_cache, save_land_mask

    client = NSIDCClient(settings)
    last_error = None
    for back in range(2, 12):
        day = datetime.utcnow().date() - timedelta(days=back)
        try:
            path = client.fetch_concentration(day)
            land = build_land_mask_from_nsidc(path)
            out = save_land_mask(land)
            clear_cache()
            return f"land mask rebuilt from {day} -> {out} ({int(land.sum())} land cells)"
        except Exception as exc:  # noqa: BLE001 - try the previous day
            last_error = exc
    return f"land mask rebuild FAILED: {last_error}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest POLARIS environmental data")
    parser.add_argument("--mode", choices=("real", "demo"), help="Override DATA_MODE for this run")
    parser.add_argument("--days", type=int, default=5, help="Days of gridded sea-ice data to ingest")
    parser.add_argument("--end-date", help="Last day to ingest (YYYY-MM-DD); defaults to the latest available")
    parser.add_argument("--only", choices=DATASETS, action="append", help="Restrict to specific datasets")
    parser.add_argument("--landmask", action="store_true", help="Rebuild the NSIDC land mask asset first")
    parser.add_argument("--init-db", action="store_true", help="Create the schema before ingesting")
    parser.add_argument("--json", action="store_true", help="Emit the raw result as JSON")
    args = parser.parse_args()

    if args.mode:
        os.environ["DATA_MODE"] = args.mode

    from app.config import reload_settings
    from app.database.connection import session_scope
    from app.database.init_db import init_database
    from app.services import data_ingestion as ing
    from app.services.environment import clear_cache
    from app.utils.geo import grid_from_settings
    from app.utils.logging import setup_logging

    settings = reload_settings()
    setup_logging(settings.log_level, settings.log_file)
    grid = grid_from_settings(settings)

    print(f"POLARIS ingestion  |  DATA_MODE={settings.data_mode}  |  domain {grid.shape[0]}x{grid.shape[1]} cells")
    if args.init_db:
        print(json.dumps(init_database(), indent=2, default=str))
    if args.landmask:
        print(rebuild_landmask(settings))

    wanted = args.only or list(DATASETS)
    end_date = date.fromisoformat(args.end_date) if args.end_date else None
    started = time.time()
    results: dict[str, dict] = {}

    with session_scope() as session:
        if "sea_ice" in wanted:
            results["sea_ice"] = ing.ingest_sea_ice(
                session, grid, days=args.days, end_date=end_date, settings=settings
            ).to_dict()
        if "extent_index" in wanted:
            results["extent_index"] = ing.ingest_extent_index(session, settings=settings).to_dict()
        if "icebergs" in wanted:
            results["icebergs"] = ing.ingest_icebergs(session, settings=settings).to_dict()
        if "weather" in wanted:
            results["weather"] = ing.ingest_weather(session, grid, settings=settings).to_dict()
        if "ocean" in wanted:
            results["ocean"] = ing.ingest_ocean(session, grid, settings=settings).to_dict()

    clear_cache()

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"\n{'dataset':14s} {'status':9s} {'source':22s} {'rows':>8s}  message")
        print("-" * 100)
        for name, r in results.items():
            print(f"{name:14s} {r['status']:9s} {r['source']:22s} {r['ingested']:8d}  {r['message'][:44]}")
    print(f"\nCompleted in {time.time() - started:.1f}s")

    failed = [n for n, r in results.items() if r["status"] == "failed"]
    if failed:
        print(f"FAILED datasets: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
