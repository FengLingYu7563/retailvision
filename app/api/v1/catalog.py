from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile

from app.api.deps import DB, AppSettings, RegistryDep, StorageDep, error_responses
from app.core.errors import AppError
from app.core.ratelimit import LimitedUser
from app.core.security import AdminUser
from app.db.models import CatalogProduct
from app.schemas import Page, ProductOut
from app.services import catalog
from app.services.uploads import read_and_validate

router = APIRouter(tags=["catalog"])


def _out(db: DB, p: CatalogProduct, version: str) -> ProductOut:
    return ProductOut(
        id=p.id,
        name=p.name,
        sku=p.sku,
        reference_images=catalog.reference_count(db, p.id, version),
        created_at=p.created_at,
    )


@router.post(
    "/catalog/products",
    status_code=201,
    response_model=ProductOut,
    responses=error_responses(400, 409, 413, 415, 503),
)
async def create_product(
    user: LimitedUser,
    db: DB,
    registry: RegistryDep,
    storage: StorageDep,
    settings: AppSettings,
    name: Annotated[str, Form(min_length=1, max_length=128)],
    files: Annotated[list[UploadFile], File(description="1 到 5 張參考圖，建議用店內實拍")],
    sku: Annotated[str | None, Form(max_length=64)] = None,
) -> ProductOut:
    """新品登錄：不重新訓練，用使用中的模型計算參考圖的 embedding（驗收 2.9）。"""
    clf = registry.classifier
    if clf is None:
        raise AppError("MODEL_NOT_READY", "分類模型尚未載入", 503)
    if not 1 <= len(files) <= 5:
        raise AppError("VALIDATION_ERROR", "參考圖必須是 1 到 5 張", 422)
    images = [await read_and_validate(f, settings) for f in files]
    product = catalog.create_product(db, clf, storage, name, sku or None, images)
    return _out(db, product, clf.version)


@router.get("/catalog/products", response_model=Page[ProductOut], responses=error_responses())
def list_products(
    user: LimitedUser,
    db: DB,
    registry: RegistryDep,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Page[ProductOut]:
    rows, total = catalog.list_products(db, page, size)
    version = registry.classifier.version if registry.classifier else ""
    return Page[ProductOut](
        items=[_out(db, p, version) for p in rows], page=page, size=size, total=total
    )


@router.delete(
    "/catalog/products/{product_id}",
    status_code=204,
    responses=error_responses(403, 404),
)
def delete_product(product_id: int, user: AdminUser, db: DB, storage: StorageDep) -> Response:
    """刪除商品與參考圖。限 admin：刪除會影響所有使用者的辨識結果。"""
    catalog.delete_product(db, storage, product_id)
    return Response(status_code=204)
