"""Dashboard and operational-status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import GridDep, SessionDep, SettingsDep
from app.database import repositories as repo
from app.services import dashboard_service
from app.services import sea_ice_forecasting as sif
from app.utils.logging import get_logger

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
log = get_logger("api.dashboard")


@router.get(
    "/summary",
    summary="Compact status of the whole system",
    description=(
        "One call returning the latest sea-ice status, the number of tracked icebergs, the "
        "highest-risk areas, a forecast summary, routing availability and recent ingestion "
        "activity.\n\n"
        "Every section degrades independently: if one subsystem has no data the rest still "
        "return, and the failing section reports why."
    ),
)
def summary(session: SessionDep, settings: SettingsDep, grid: GridDep) -> dict:
    return dashboard_service.build_summary(session, settings, grid)


@router.get(
    "/models",
    summary="Trained model registry",
    description=(
        "Artifacts on disk and their registry rows. All metrics are real held-out scores produced "
        "at training time."
    ),
)
def models(session: SessionDep, settings: SettingsDep) -> dict:
    artifacts = sif.available_models(settings)
    records = repo.get_model_records(session)
    return {
        "artifacts": artifacts,
        "registry": [
            {
                "model_name": r.model_name,
                "model_version": r.model_version,
                "horizon_hours": r.horizon_hours,
                "algorithm": r.algorithm,
                "trained_at": r.trained_at,
                "n_train_samples": r.n_train_samples,
                "n_val_samples": r.n_val_samples,
                "metrics": r.metrics,
                "artifact_path": r.artifact_path,
                "data_mode": r.data_mode,
                "notes": r.notes,
            }
            for r in records
        ],
    }


@router.get(
    "/ingestion",
    summary="Recent ingestion runs",
    description="Audit trail of ingestion attempts, including skipped and failed ones with their reasons.",
)
def ingestion(session: SessionDep, limit: int = Query(default=20, ge=1, le=200)) -> dict:
    entries = repo.recent_ingestions(session, limit=limit)
    return {
        "count": len(entries),
        "runs": [
            {
                "id": e.id,
                "source": e.source,
                "dataset": e.dataset,
                "data_mode": e.data_mode,
                "status": e.status,
                "started_at": e.started_at,
                "finished_at": e.finished_at,
                "records_ingested": e.records_ingested,
                "records_rejected": e.records_rejected,
                "message": e.message,
                "details": e.details,
            }
            for e in entries
        ],
    }
