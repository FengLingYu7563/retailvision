from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.api.deps import DB
from app.core.errors import AppError
from app.core.ratelimit import get_redis
from app.schemas import ErrorResponse, HealthOut

router = APIRouter(tags=["health"])


@router.get("/livez")
def livez() -> dict[str, str]:
    """Liveness：process 活著就回 200，不檢查資料庫與模型。

    給 docker healthcheck 用。/healthz 是 readiness：沒放權重時回 503 是正確的降級狀態，
    如果 healthcheck 也用它，容器會被標成 unhealthy，依賴它的 nginx 就起不來。
    """
    return {"status": "alive"}


@router.get("/healthz", response_model=HealthOut, responses={503: {"model": ErrorResponse}})
def healthz(request: Request, db: DB) -> HealthOut:
    """不需登入。資料庫、Redis、模型任何一個不可用就回 503。

    503 也用統一錯誤格式（code SERVICE_UNAVAILABLE），各元件狀態放在 error.details。
    """
    registry = request.app.state.registry
    try:
        db.execute(text("SELECT 1"))
        # 先和資料庫的使用中模型對齊，再回報狀態：/healthz 可能是切換模型後這個 process 收到的
        # 第一個請求，不同步的話會回報舊模型的狀態（Codex 第 7 輪 L1）
        registry.sync(db)
        db_ok = True
    except Exception:
        db_ok = False
    try:
        redis_ok = bool(get_redis().ping())
    except Exception:
        redis_ok = False
    body = HealthOut(status="ok", database=db_ok, redis=redis_ok, models=registry.status())
    if not (db_ok and redis_ok and registry.ready):
        raise AppError(
            "SERVICE_UNAVAILABLE",
            "服務尚未就緒",
            503,
            details=body.model_copy(update={"status": "degraded"}).model_dump(),
        )
    return body
