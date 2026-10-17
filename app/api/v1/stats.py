from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from app.api.deps import DB
from app.core.ratelimit import LimitedUser
from app.schemas import ErrorResponse, StatsOut
from app.services import stats

router = APIRouter(tags=["stats"])


@router.get("/stats/summary", response_model=StatsOut, responses={401: {"model": ErrorResponse}})
def stats_summary(
    user: LimitedUser, db: DB, days: Annotated[int, Query(ge=1, le=365)] = 30
) -> dict[str, Any]:
    """各類別數量、平均信心度、缺貨率、每日趨勢。一般使用者只統計自己的紀錄，admin 統計全部。"""
    return stats.summary(db, user, days)
