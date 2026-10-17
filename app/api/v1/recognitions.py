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
    error_responses,
    owned_recognition,
    recognition_out,
)
from app.core.errors import AppError
from app.core.ratelimit import LimitedUser
from app.db.models import Job, Recognition
from app.schemas import (
    JobAccepted,
    Page,
    ProductRecognition,
    RecognitionOut,
    RecognitionSummary,
    ShelfRecognition,
)
from app.services.recognition import recognize_product, recognize_shelf
from app.services.uploads import read_and_validate, read_limited, validate_image_bytes

router = APIRouter(tags=["recognitions"])

_ERRORS = error_responses(400, 413, 415, 503)
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
    """批次辨識：先驗證、存檔、建立 job，交給 Celery worker 處理，立即回 202。

    每張圖在這裡就做完整驗證（大小、格式、像素數），**只有驗證通過的圖才寫進硬碟**；
    不合格的圖記在 item_errors，不會讓整批被拒絕（驗收 2.8）。
    佇列送不出去時，job 標成 failed、刪掉已存的檔案，回 503，不留下永遠 pending 的 job。
    """
    if not 1 <= len(files) <= settings.max_batch_files:
        raise AppError("VALIDATION_ERROR", f"一次最多 {settings.max_batch_files} 張", 422)
    items: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, f in enumerate(files):
        try:
            img = validate_image_bytes(
                await read_limited(f, settings.max_upload_bytes), f.filename, settings
            )
        except AppError as exc:
            errors.append(
                {"index": index, "filename": f.filename, "code": exc.code, "message": exc.message}
            )
            continue
        path = storage.save(img.data, img.ext, "batch")
        items.append({"index": index, "path": path, "filename": f.filename})

    job = Job(
        user_id=user.id,
        type=type,
        status="pending" if items else "done",
        total=len(files),
        done_count=len(errors),
        failed_count=len(errors),
        item_errors=errors,
    )
    db.add(job)
    db.commit()
    if items:
        from app.worker import process_batch

        try:
            process_batch.delay(str(job.id), items)
        except Exception as exc:
            for it in items:
                storage.delete(it["path"])
            job.status, job.error = "failed", "QUEUE_UNAVAILABLE"
            db.commit()
            raise AppError("QUEUE_UNAVAILABLE", "批次佇列暫時無法使用，請稍後再試", 503) from exc
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
    responses=error_responses(404),
)
def get_recognition(rec_id: uuid.UUID, user: LimitedUser, db: DB) -> dict[str, Any]:
    return recognition_out(owned_recognition(db, rec_id, user))


@router.get(
    "/recognitions/{rec_id}/annotated",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, **error_responses(404)},
)
def get_annotated(rec_id: uuid.UUID, user: LimitedUser, db: DB, storage: StorageDep) -> Response:
    rec = owned_recognition(db, rec_id, user)
    if not rec.annotated_path:
        raise AppError("NOT_FOUND", "這筆紀錄沒有標註圖", 404)
    return Response(storage.read(rec.annotated_path), media_type="image/jpeg")
