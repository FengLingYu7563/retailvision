from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import DB, RegistryDep, StorageDep, error_responses
from app.core.errors import AppError, not_found
from app.core.ratelimit import LimitedUser
from app.core.security import AdminUser
from app.db.models import ModelVersion
from app.schemas import ActivateModelIn, ActivateModelOut, ModelOut
from app.services import catalog

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelOut], responses=error_responses())
def list_models(user: LimitedUser, db: DB) -> list[ModelVersion]:
    return list(db.scalars(select(ModelVersion).order_by(ModelVersion.task, ModelVersion.name)))


@router.put(
    "/models/active",
    response_model=ActivateModelOut,
    responses=error_responses(403, 404, 409),
)
def activate_model(
    body: ActivateModelIn, user: AdminUser, db: DB, registry: RegistryDep, storage: StorageDep
) -> ActivateModelOut:
    """切換使用中的分類模型（限 admin），不需重啟服務（驗收 3.6）。

    順序很重要：先載入新模型、用它把 catalog 參考圖全部重算 embedding，最後才切換。
    中途失敗的話資料庫交易會回滾，舊模型繼續使用，不會出現「新模型配舊向量」的錯誤搜尋。
    目前只支援切換分類模型；偵測模型只有一個版本。
    """
    # 鎖住分類模型的列：兩個 admin 同時切換時排隊執行，不會兩個都變成 active
    rows = db.scalars(
        select(ModelVersion).where(ModelVersion.task == "classification").with_for_update()
    ).all()
    target = next((m for m in rows if m.name == body.name), None)
    if target is None:
        if db.scalar(select(ModelVersion.id).where(ModelVersion.name == body.name)):
            raise AppError("VALIDATION_ERROR", "目前只支援切換分類模型", 422)
        raise not_found(f"找不到模型 {body.name}")
    try:
        clf = registry.build_classifier(target.name)
    except Exception as exc:  # 檔案不存在或損壞：都不切換，舊模型繼續使用
        db.rollback()
        raise AppError("MODEL_NOT_READY", f"模型檔不存在或無法載入：{target.name}", 409) from exc
    added = catalog.reembed(db, clf, storage)
    for m in rows:
        m.is_active = m.name == target.name
    db.commit()
    registry.set_classifier(clf)
    return ActivateModelOut(active=target.name, reembedded=added)
