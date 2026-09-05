"""POLARIS backend entry point.

    python run.py                 # start the API in whatever DATA_MODE is set
    python run.py --mode real     # start in real-data mode for this run only
    python run.py --mode demo     # start in demo mode for this run only
    python run.py --init-db       # create the schema, then start
    python run.py --reload        # development auto-reload

Demo and real keep separate databases, archives and models, so switching
between them needs no rebuild.

Equivalent to ``uvicorn app.main:app``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the POLARIS backend")
    parser.add_argument("--host", default=None, help="Bind address (default from API_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Port (default from API_PORT)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--init-db", action="store_true", help="Create the schema before starting")
    parser.add_argument("--log-level", default=None, help="uvicorn log level")
    parser.add_argument(
        "--mode", choices=("demo", "real"),
        help="Override DATA_MODE for this run. Each mode has its own database, "
             "processed archive and models, so switching is instant.",
    )
    args = parser.parse_args()

    if args.mode:
        os.environ["DATA_MODE"] = args.mode

    import uvicorn

    from app.config import get_settings
    from app.utils.logging import setup_logging

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_file)

    if args.init_db:
        from app.database.init_db import init_database

        print("Initialising database ...")
        for key, value in init_database().items():
            print(f"  {key}: {value}")

    host = args.host or settings.api_host
    port = args.port or settings.api_port
    banner = "REAL observational data" if not settings.is_demo else "DEMO / SYNTHETIC data"
    print(f"POLARIS backend starting on http://{host}:{port}")
    print(f"  DATA MODE  : {settings.data_mode.upper()}  <- {banner}")
    print(f"  database   : {settings.database_url.split('/')[-1]}")
    print(f"  models     : {settings.model_path}")
    print(f"  Swagger UI : http://localhost:{port}/docs")
    print(f"  Test UI    : http://localhost:{port}/ui/")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=(args.log_level or settings.log_level).lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
