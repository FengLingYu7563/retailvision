"""Rate limit：Redis 固定時間窗計數（驗收 2.11）。

為什麼自己寫而不用 slowapi：slowapi 要在每個端點的參數裡帶 Request，還要額外的裝飾器；
這裡只需要「每個使用者（或 IP）每分鐘 N 次」，用 Redis INCR + EXPIRE 十幾行就能做到，
而且放在 Redis，多個 API process 共用同一份計數。
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated

import redis
from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.security import CurrentUser
from app.db.models import User


@lru_cache
def get_redis(url: str | None = None) -> redis.Redis:
    return redis.Redis.from_url(url or get_settings().redis_url, socket_timeout=2)


def check_rate(key: str, limit: int, window: int = 60) -> None:
    bucket = f"rl:{key}:{int(time.time() // window)}"
    r = get_redis()
    pipe = r.pipeline()
    pipe.incr(bucket)
    pipe.expire(bucket, window + 5)
    count = int(pipe.execute()[0])
    if count > limit:
        raise AppError("RATE_LIMITED", "請求太頻繁，請稍後再試", 429, {"Retry-After": str(window)})


def rate_limit_user(
    user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]
) -> User:
    """已登入的端點依使用者限制：同一個帳號換 IP 也共用同一份額度。"""
    check_rate(f"user:{user.id}", settings.rate_limit_per_minute)
    return user


LimitedUser = Annotated[User, Depends(rate_limit_user)]


def rate_limit_by_ip(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> None:
    """未登入的端點（換 token）依 IP 限制，減緩密碼暴力嘗試。"""
    ip = request.client.host if request.client else "unknown"
    check_rate(f"ip:{ip}", settings.rate_limit_per_minute)
