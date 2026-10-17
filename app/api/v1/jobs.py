from __future__ import annotations

import uuid

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import DB, error_responses, owned_job
from app.core.ratelimit import LimitedUser
from app.db.models import Recognition
from app.schemas import JobOut

router = APIRouter(tags=["jobs"])


@router.get(
    "/jobs/{job_id}",
    response_model=JobOut,
    responses=error_responses(404),
)
def get_job(job_id: uuid.UUID, user: LimitedUser, db: DB) -> JobOut:
    job = owned_job(db, job_id, user)
    ids = db.scalars(
        select(Recognition.id).where(Recognition.job_id == job.id).order_by(Recognition.created_at)
    ).all()
    return JobOut(
        id=job.id,
        type=job.type,
        status=job.status,  # type: ignore[arg-type]
        total=job.total,
        done_count=job.done_count,
        failed_count=job.failed_count,
        error=job.error,
        item_errors=job.item_errors or [],  # type: ignore[arg-type]
        recognition_ids=list(ids),
        created_at=job.created_at,
        finished_at=job.finished_at,
    )
