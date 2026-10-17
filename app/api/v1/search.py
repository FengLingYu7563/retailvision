from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile

from app.api.deps import DB, AppSettings, RegistryDep, error_responses
from app.core.errors import AppError
from app.core.ratelimit import LimitedUser
from app.schemas import SearchOut, SimilarHit
from app.services import catalog
from app.services.uploads import read_and_validate

router = APIRouter(tags=["search"])


@router.post(
    "/search/similar",
    response_model=SearchOut,
    responses=error_responses(400, 413, 415, 503),
)
async def search_similar(
    file: Annotated[UploadFile, File()],
    user: LimitedUser,
    db: DB,
    registry: RegistryDep,
    settings: AppSettings,
    k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> SearchOut:
    """以圖搜圖，回傳前 k 個最相似的已登錄商品。

    對訓練時沒看過的商品，第一名的準確率約 47%、前 5 名約 93%（Phase 1 保留類別實驗），
    所以預設回傳前 5 名讓人確認，而不是只給一個答案。
    """
    clf = registry.classifier
    if clf is None:
        raise AppError("MODEL_NOT_READY", "分類模型尚未載入", 503)
    img = await read_and_validate(file, settings)
    hits = catalog.search(db, clf, img.image, k)
    return SearchOut(
        model_version=clf.version,
        results=[
            SimilarHit(product_id=h.product_id, name=h.name, similarity=h.similarity) for h in hits
        ],
    )
