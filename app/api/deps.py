"""路由共用的 dependency 與轉換函式。"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import not_found
from app.db.models import Job, Recognition, User
from app.db.session import get_db
from app.services.registry import ModelRegistry
from app.services.uploads import Storage

DB = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def get_storage(request: Request) -> Storage:
    storage: Storage = request.app.state.storage
    return storage


def get_registry(request: Request, db: DB) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.registry
    # 每個請求先和資料庫的使用中模型對齊：別的 process 切換過模型時，這裡會重新載入
    registry.sync(db)
    return registry


StorageDep = Annotated[Storage, Depends(get_storage)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]


def owned_recognition(db: Session, rec_id: Any, user: User) -> Recognition:
    rec = db.get(Recognition, rec_id)
    # 不是自己的紀錄一律回 404 而不是 403：不讓人用猜 ID 的方式確認別人的紀錄存在
    if rec is None or (user.role != "admin" and rec.user_id != user.id):
        raise not_found("找不到辨識紀錄")
    return rec


def owned_job(db: Session, job_id: Any, user: User) -> Job:
    job = db.get(Job, job_id)
    if job is None or (user.role != "admin" and job.user_id != user.id):
        raise not_found("找不到批次任務")
    return job


def recognition_out(rec: Recognition) -> dict[str, Any]:
    return {
        "id": rec.id,
        "type": rec.type,
        "model_version": rec.model_version,
        "latency_ms": rec.latency_ms,
        "created_at": rec.created_at,
        "job_id": rec.job_id,
        **rec.result,
    }
