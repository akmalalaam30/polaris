"""Iceberg tracking and trajectory endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.api.deps import BBoxDep, GridDep, SessionDep, SettingsDep, provenance, raise_unavailable
from app.database import repositories as repo
from app.schemas.iceberg import (
    IcebergGeometry,
    IcebergListResponse,
    IcebergObservationOut,
    IcebergTrajectoryResponse,
    TrackPoint,
    TrajectoryPointOut,
)
from app.services import environment as envsvc
from app.utils.logging import get_logger
from app.utils.validation import DataUnavailableError

#: Starlette deprecated its 422 constant name; the numeric code is stable.
HTTP_422 = 422

router = APIRouter(prefix="/icebergs", tags=["icebergs"])
log = get_logger("api.icebergs")


@router.get(
    "",
    response_model=IcebergListResponse,
    summary="Tracked Antarctic icebergs",
    description=(
        "Most recent reported position of every tracked iceberg, with drift speed and bearing "
        "derived from the previous fix where one exists.\n\n"
        "In real mode these are US National Ice Center designators (A/B/C/D series). In demo mode "
        "the IDs are prefixed `DEMO-` and the records are synthetic."
    ),
)
def list_icebergs(
    session: SessionDep,
    settings: SettingsDep,
    grid: GridDep,
    bbox: BBoxDep,
    min_area_km2: float | None = Query(default=None, ge=0, description="Only bergs at least this large"),
    in_domain_only: bool = Query(default=False, description="Restrict to the POLARIS analysis domain"),
    limit: int = Query(default=500, ge=1, le=5000),
) -> IcebergListResponse:
    ctx = envsvc.get_iceberg_context(session, settings, grid, with_trajectories=False)
    if not ctx.records:
        raise_unavailable(
            DataUnavailableError("No iceberg observations have been ingested."),
            hint="Run 'python scripts/download_data.py' (real) or 'scripts/seed_demo_data.py' (demo).",
        )

    records = ctx.records
    if min_area_km2 is not None:
        records = [r for r in records if (r.get("area_km2") or 0.0) >= min_area_km2]
    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        records = [
            r for r in records
            if lat_min <= r["latitude"] <= lat_max and lon_min <= r["longitude"] <= lon_max
        ]
    if in_domain_only:
        records = [r for r in records if grid.contains(r["latitude"], r["longitude"])]
    records = records[:limit]

    return IcebergListResponse(
        count=len(records),
        icebergs=[IcebergObservationOut(**{k: v for k, v in r.items() if k in IcebergObservationOut.model_fields}) for r in records],
        provenance=provenance(
            settings,
            [r.get("source") for r in records],
            max((r["observed_at"] for r in records), default=None),
        ),
    )


@router.get(
    "/{iceberg_id}/trajectory",
    response_model=IcebergTrajectoryResponse,
    summary="Predicted iceberg drift trajectory",
    description=(
        "Physics-based drift forecast for one iceberg.\n\n"
        "The momentum balance combines air drag on the sail, water drag on the keel, sea-ice drag in "
        "compact pack, Coriolis and the sea-surface pressure gradient, integrated with RK4. "
        "`uncertainty_radius_km` is the spread of a Monte-Carlo ensemble that perturbs the wind, "
        "current and drag coefficients - a measured spread, not an assumed confidence interval.\n\n"
        "Forcing fields are held at their analysis values for the whole forecast; that persistence "
        "assumption is the dominant error source at these lead times."
    ),
    responses={404: {"description": "Unknown iceberg ID"}},
)
def iceberg_trajectory(
    session: SessionDep,
    settings: SettingsDep,
    grid: GridDep,
    iceberg_id: str = Path(description="Iceberg designator, e.g. 'A76C' or 'DEMO-01'"),
    forecast_hours: float = Query(default=72.0, gt=0, le=240, description="Lead time in hours"),
    persist: bool = Query(default=False, description="Store the trajectory in the database"),
) -> IcebergTrajectoryResponse:
    ctx = envsvc.get_iceberg_context(session, settings, grid, horizon_hours=forecast_hours)
    wanted = iceberg_id.strip().upper()
    record = next((r for r in ctx.records if r["iceberg_id"].upper() == wanted), None)
    if record is None:
        known = sorted(r["iceberg_id"] for r in ctx.records)[:20]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"Iceberg {iceberg_id!r} is not tracked.",
                "known_icebergs": known,
                "hint": "List available icebergs with GET /api/icebergs",
            },
        )

    result = ctx.trajectories.get(record["iceberg_id"])
    if result is None:
        if not record.get("in_domain", True):
            raise HTTPException(
                status_code=HTTP_422,
                detail={
                    "message": (
                        f"Iceberg {record['iceberg_id']} is at "
                        f"({record['latitude']:.2f}, {record['longitude']:.2f}), outside the POLARIS "
                        f"analysis domain, where no environmental forcing is available. "
                        f"No trajectory is computed rather than returning a fabricated one."
                    ),
                    "domain": grid.to_dict(),
                    "hint": (
                        "Widen DOMAIN_LAT_MIN/MAX and DOMAIN_LON_MIN/MAX to cover this berg, "
                        "or list only in-domain bergs with GET /api/icebergs?in_domain_only=true"
                    ),
                },
            )
        raise_unavailable(
            DataUnavailableError(
                f"Trajectory for {iceberg_id} could not be computed; check that ocean and wind "
                f"fields have been ingested."
            ),
            hint="Run the ingestion for weather and ocean, then retry.",
        )

    if persist:
        from app.services.iceberg_trajectory import persist_trajectories

        persist_trajectories(session, {record["iceberg_id"]: result})
        session.commit()

    state = result.state
    return IcebergTrajectoryResponse(
        iceberg_id=result.iceberg_id,
        issued_at=result.issued_at,
        origin={"latitude": result.origin[0], "longitude": result.origin[1]},
        geometry=IcebergGeometry(
            length_m=round(state.length_m, 1),
            width_m=round(state.width_m, 1),
            thickness_m=round(state.thickness_m, 1),
            draft_m=round(state.draft_m, 1),
            mass_kg=state.mass_kg,
        ),
        forecast_hours=forecast_hours,
        ensemble_size=result.ensemble_size,
        total_displacement_km=round(result.total_displacement_km, 2),
        points=[TrajectoryPointOut(**p.to_dict()) for p in result.points],
        observed_track=[TrackPoint(**t) for t in record.get("observed_track", [])],
        forcing=result.forcing,
        model_name="iceberg_dynamics",
        model_version="1.0.0",
        notes=result.notes,
        provenance=provenance(settings, [record.get("source")], record.get("observed_at"), result.issued_at),
    )


@router.get(
    "/trajectories",
    summary="Trajectories for every tracked iceberg",
    description="Batch version of the per-iceberg endpoint, for map rendering.",
)
def all_trajectories(
    session: SessionDep,
    settings: SettingsDep,
    grid: GridDep,
    forecast_hours: float = Query(default=72.0, gt=0, le=240),
    in_domain_only: bool = Query(default=True),
) -> dict:
    ctx = envsvc.get_iceberg_context(session, settings, grid, horizon_hours=forecast_hours)
    out = []
    for record in ctx.records:
        result = ctx.trajectories.get(record["iceberg_id"])
        if result is None:
            continue
        if in_domain_only and not grid.contains(record["latitude"], record["longitude"]):
            continue
        out.append(
            {
                "iceberg_id": record["iceberg_id"],
                "origin": {"latitude": record["latitude"], "longitude": record["longitude"]},
                "area_km2": record.get("area_km2"),
                "total_displacement_km": round(result.total_displacement_km, 2),
                "points": [p.to_dict() for p in result.points],
            }
        )
    return {
        "count": len(out),
        "issued_at": ctx.issued_at,
        "forecast_hours": forecast_hours,
        "trajectories": out,
        "provenance": provenance(settings, [r.get("source") for r in ctx.records], generated_at=ctx.issued_at).model_dump(),
    }
