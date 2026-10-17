from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.api.deps import DB
from app.core.errors import AppError
from app.core.ratelimit import get_redis
from app.schemas import ErrorResponse, HealthOut

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthOut, responses={503: {"model": ErrorResponse}})
def healthz(request: Request, db: DB) -> HealthOut:
    """不需登入。資料庫、Redis、模型任何一個不可用就回 503。

    503 也用統一錯誤格式（code SERVICE_UNAVAILABLE），各元件狀態放在 error.details。
    """
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    try:
        redis_ok = bool(get_redis().ping())
    except Exception:
        redis_ok = False
    registry = request.app.state.registry
    body = HealthOut(status="ok", database=db_ok, redis=redis_ok, models=registry.status())
    if not (db_ok and redis_ok and registry.ready):
        raise AppError(
            "SERVICE_UNAVAILABLE",
            "服務尚未就緒",
            503,
            details=body.model_copy(update={"status": "degraded"}).model_dump(),
        )
    return body
