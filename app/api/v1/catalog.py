from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response

from app.api.deps import DB, AppSettings, RegistryDep, StorageDep, error_responses, multipart_doc
from app.core.errors import AppError
from app.core.ratelimit import LimitedUser
from app.core.security import AdminUser
from app.db.models import CatalogProduct
from app.schemas import Page, ProductOut
from app.services import catalog
from app.services.multipart import read_multipart
from app.services.uploads import validate_part

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
    openapi_extra=multipart_doc(
        "files",
        multiple=True,
        description="1 到 5 張參考圖（每張 10 MB 以內），建議用店內實拍",
        fields={
            "name": {"type": "string", "minLength": 1, "maxLength": 128, "required": True},
            "sku": {"type": "string", "maxLength": 64},
        },
    ),
)
async def create_product(
    request: Request,
    user: LimitedUser,
    db: DB,
    registry: RegistryDep,
    storage: StorageDep,
    settings: AppSettings,
) -> ProductOut:
    """新品登錄：不重新訓練，用使用中的模型計算參考圖的 embedding（驗收 2.9）。

    multipart 串流解析，每張參考圖在讀取時就限制大小（見 app/services/multipart.py）。
    """
    clf = registry.classifier
    if clf is None:
        raise AppError("MODEL_NOT_READY", "分類模型尚未載入", 503)
    form = await read_multipart(request, settings.max_upload_bytes, max_files=5)
    name = (form.field("name") or "").strip()
    sku = (form.field("sku") or "").strip() or None
    if not 1 <= len(name) <= 128:
        raise AppError("VALIDATION_ERROR", "name 必須是 1 到 128 個字元", 422)
    if sku is not None and len(sku) > 64:
        raise AppError("VALIDATION_ERROR", "sku 最多 64 個字元", 422)
    files = form.files_named("files")
    if not 1 <= len(files) <= 5:
        raise AppError("VALIDATION_ERROR", "參考圖必須是 1 到 5 張", 422)
    images = [validate_part(f, settings) for f in files]
    product = catalog.create_product(db, clf, storage, name, sku, images)
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
