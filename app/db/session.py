"""資料庫連線與 session。"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    # pool_pre_ping：資料庫重啟後，連線池裡的舊連線會先被檢查，避免第一個請求失敗
    return create_engine(url or get_settings().database_url, pool_pre_ping=True)


def session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency：每個請求一個 session，結束時關閉。"""
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()
