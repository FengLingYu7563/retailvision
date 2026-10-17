"""Celery worker：批次辨識（驗收 2.8）。

啟動：celery -A app.worker worker --loglevel=info（開發與測試一律在 docker-compose 裡跑，
規格第 12 節：Windows 上跑 Celery 不穩定）。

一張圖失敗只記在 job.item_errors，繼續處理下一張；每處理完一張就更新 done_count，
前端輪詢 /jobs/{id} 可以看到進度。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from celery import Celery
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError
from app.db.models import Job
from app.db.session import session_factory
from app.services.recognition import RECOGNIZERS
from app.services.registry import ModelRegistry
from app.services.uploads import Storage, validate_image_bytes

log = logging.getLogger("app.worker")
_settings = get_settings()

celery_app = Celery("retailvision", broker=_settings.redis_url)
celery_app.conf.update(
    task_always_eager=_settings.celery_eager,
    # 同步測試時讓例外直接丟出來，不要被 Celery 吞掉
    task_eager_propagates=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

_registry: ModelRegistry | None = None


def get_worker_registry() -> ModelRegistry:
    """worker process 自己的模型註冊表，第一次用到時才載入。測試可以用 set_worker_registry 替換。"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry.from_settings(_settings)
    return _registry


def set_worker_registry(registry: ModelRegistry | None) -> None:
    global _registry
    _registry = registry


def run_batch(db: Session, job_id: uuid.UUID, items: list[dict[str, Any]]) -> None:
    settings = get_settings()
    storage = Storage(settings.storage_dir)
    registry = get_worker_registry()
    registry.sync(db)
    job = db.get(Job, job_id)
    if job is None:
        log.warning("找不到批次任務 %s", job_id)
        return
    job.status = "running"
    db.commit()
    recognize = RECOGNIZERS[job.type]
    # 上傳時就驗證失敗的項目已經記在 item_errors（也已計入 done_count），這裡接著累加
    errors: list[dict[str, Any]] = list(job.item_errors or [])
    done = job.done_count
    for pos, item in enumerate(items):
        index = item.get("index", pos)  # 在原本上傳順序裡的位置，錯誤訊息才對得上是哪一張
        try:
            img = validate_image_bytes(storage.read(item["path"]), item.get("filename"), settings)
            recognize(db, registry, storage, settings, img, job.user_id, False, job.id)
        except AppError as exc:
            db.rollback()
            errors.append(
                {
                    "index": index,
                    "filename": item.get("filename"),
                    "code": exc.code,
                    "message": exc.message,
                }
            )
        except Exception:
            db.rollback()
            log.exception("批次第 %d 張處理失敗", index)
            errors.append(
                {
                    "index": index,
                    "filename": item.get("filename"),
                    "code": "INTERNAL_ERROR",
                    "message": "處理時發生內部錯誤",
                }
            )
        finally:
            storage.delete(item["path"])
        done += 1
        job = db.get(Job, job_id)
        assert job is not None
        job.done_count = done
        job.failed_count = len(errors)
        job.item_errors = sorted(errors, key=lambda e: e["index"])
        db.commit()
    job.status = "done"
    job.finished_at = datetime.now(UTC)
    db.commit()


@celery_app.task(name="retailvision.process_batch")
def process_batch(job_id: str, items: list[dict[str, Any]]) -> None:
    db = session_factory()()
    try:
        run_batch(db, uuid.UUID(job_id), items)
    except Exception as exc:
        db.rollback()
        job = db.get(Job, uuid.UUID(job_id))
        if job is not None:
            job.status = "failed"
            job.error = f"{type(exc).__name__}"
            job.finished_at = datetime.now(UTC)
            db.commit()
        raise
    finally:
        db.close()
