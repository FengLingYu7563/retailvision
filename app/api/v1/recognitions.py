from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select

from app.api.deps import (
    DB,
    AppSettings,
    RegistryDep,
    StorageDep,
    owned_recognition,
    recognition_out,
)
from app.core.errors import AppError
from app.core.ratelimit import LimitedUser
from app.db.models import Job, Recognition
from app.schemas import (
    ErrorResponse,
    JobAccepted,
    Page,
    ProductRecognition,
    RecognitionOut,
    RecognitionSummary,
    ShelfRecognition,
)
from app.services.recognition import recognize_product, recognize_shelf
from app.services.uploads import read_and_validate, read_limited, sniff_format

router = APIRouter(tags=["recognitions"])

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 401, 413, 415, 422, 429, 503)
}
Strict = Annotated[bool, Query(description="品質不佳時直接回 422 LOW_QUALITY_IMAGE")]


@router.post(
    "/recognitions/product", status_code=201, response_model=ProductRecognition, responses=_ERRORS
)
async def create_product_recognition(
    file: Annotated[UploadFile, File(description="商品照片（jpg / png / webp，10 MB 以內）")],
    user: LimitedUser,
    db: DB,
    registry: RegistryDep,
    storage: StorageDep,
    settings: AppSettings,
    strict: Strict = False,
) -> dict[str, Any]:
    img = await read_and_validate(file, settings)
    rec = recognize_product(db, registry, storage, settings, img, user.id, strict)
    return recognition_out(rec)


@router.post(
    "/recognitions/shelf", status_code=201, response_model=ShelfRecognition, responses=_ERRORS
)
async def create_shelf_recognition(
    file: Annotated[UploadFile, File(description="貨架照片（jpg / png / webp，10 MB 以內）")],
    user: LimitedUser,
    db: DB,
    registry: RegistryDep,
    storage: StorageDep,
    settings: AppSettings,
    strict: Strict = False,
) -> dict[str, Any]:
    img = await read_and_validate(file, settings)
    rec = recognize_shelf(db, registry, storage, settings, img, user.id, strict)
    return recognition_out(rec)


@router.post("/recognitions/batch", status_code=202, response_model=JobAccepted, responses=_ERRORS)
async def create_batch(
    files: Annotated[list[UploadFile], File(description="多張圖片")],
    user: LimitedUser,
    db: DB,
    storage: StorageDep,
    settings: AppSettings,
    type: Annotated[Literal["product", "shelf"], Form()] = "product",
) -> JobAccepted:
    """批次辨識：先存檔、建立 job，交給 Celery worker 處理，立即回 202。

    每張圖的格式驗證在 worker 裡做：一張壞圖只會記錄在 item_errors，不會讓整批被拒絕（驗收 2.8）。
    大小上限仍在這裡檢查，避免把超大檔案寫進硬碟。
    """
    if not 1 <= len(files) <= settings.max_batch_files:
        raise AppError("VALIDATION_ERROR", f"一次最多 {settings.max_batch_files} 張", 422)
    items = []
    for f in files:
        data = await read_limited(f, settings.max_upload_bytes)
        # 存檔用 UUID 與依內容判斷的副檔名；格式不明的先用 .bin 存，worker 會判定為錯誤
        path = storage.save(data, sniff_format(data) or ".bin", "batch")
        items.append({"path": path, "filename": f.filename})
    job = Job(user_id=user.id, type=type, status="pending", total=len(items), item_errors=[])
    db.add(job)
    db.commit()

    from app.worker import process_batch

    process_batch.delay(str(job.id), items)
    db.refresh(job)
    return JobAccepted(job_id=job.id, status=job.status, total=job.total)


@router.get("/recognitions", response_model=Page[RecognitionSummary], responses=_ERRORS)
def list_recognitions(
    user: LimitedUser,
    db: DB,
    type: Literal["product", "shelf"] | None = None,
    label: Annotated[str | None, Query(max_length=128)] = None,
    from_: Annotated[datetime | None, Query(alias="from")] = None,
    to: datetime | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Page[RecognitionSummary]:
    conds = []
    if user.role != "admin":
        conds.append(Recognition.user_id == user.id)
    if type:
        conds.append(Recognition.type == type)
    if label:
        conds.append(Recognition.top_label == label)
    if from_:
        conds.append(Recognition.created_at >= from_)
    if to:
        conds.append(Recognition.created_at <= to)
    total = db.scalar(select(func.count()).select_from(Recognition).where(*conds)) or 0
    rows = db.scalars(
        select(Recognition)
        .where(*conds)
        .order_by(Recognition.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
    ).all()
    return Page[RecognitionSummary](
        items=[RecognitionSummary.model_validate(r) for r in rows],
        page=page,
        size=size,
        total=total,
    )


@router.get(
    "/recognitions/{rec_id}",
    response_model=RecognitionOut,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def get_recognition(rec_id: uuid.UUID, user: LimitedUser, db: DB) -> dict[str, Any]:
    return recognition_out(owned_recognition(db, rec_id, user))


@router.get(
    "/recognitions/{rec_id}/annotated",
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
def get_annotated(rec_id: uuid.UUID, user: LimitedUser, db: DB, storage: StorageDep) -> Response:
    rec = owned_recognition(db, rec_id, user)
    if not rec.annotated_path:
        raise AppError("NOT_FOUND", "這筆紀錄沒有標註圖", 404)
    return Response(storage.read(rec.annotated_path), media_type="image/jpeg")
