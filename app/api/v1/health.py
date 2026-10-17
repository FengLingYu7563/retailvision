from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.deps import DB
from app.core.ratelimit import get_redis
from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthOut, responses={503: {"model": HealthOut}})
def healthz(request: Request, db: DB) -> JSONResponse:
    """不需登入。資料庫、Redis、模型任何一個不可用就回 503。"""
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
    ok = db_ok and redis_ok and registry.ready
    body = HealthOut(
        status="ok" if ok else "degraded", database=db_ok, redis=redis_ok, models=registry.status()
    )
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)
