from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response
from sqlalchemy import func, select

from app.api.deps import (
    DB,
    AppSettings,
    RegistryDep,
    StorageDep,
    error_responses,
    multipart_doc,
    owned_recognition,
    recognition_out,
    single_image,
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
from app.services.multipart import read_multipart
from app.services.recognition import RECOGNIZERS, recognize_product, recognize_shelf
from app.services.uploads import validate_part

router = APIRouter(tags=["recognitions"])

_ERRORS = error_responses(400, 413, 415, 503)
Strict = Annotated[bool, Query(description="品質不佳時直接回 422 LOW_QUALITY_IMAGE")]


@router.post(
    "/recognitions/product",
    status_code=201,
    response_model=ProductRecognition,
    responses=_ERRORS,
    openapi_extra=multipart_doc(
        "file", multiple=False, description="商品照片（jpg / png / webp，10 MB 以內）"
    ),
)
async def create_product_recognition(
    request: Request,
    user: LimitedUser,
    db: DB,
    registry: RegistryDep,
    storage: StorageDep,
    settings: AppSettings,
    strict: Strict = False,
) -> dict[str, Any]:
    # 先驗證登入與限流（dependency），通過後才開始讀上傳內容
    img = await single_image(request, settings)
    rec = recognize_product(db, registry, storage, settings, img, user.id, strict)
    return recognition_out(rec)


@router.post(
    "/recognitions/shelf",
    status_code=201,
    response_model=ShelfRecognition,
    responses=_ERRORS,
    openapi_extra=multipart_doc(
        "file", multiple=False, description="貨架照片（jpg / png / webp，10 MB 以內）"
    ),
)
async def create_shelf_recognition(
    request: Request,
    user: LimitedUser,
    db: DB,
    registry: RegistryDep,
    storage: StorageDep,
    settings: AppSettings,
    strict: Strict = False,
) -> dict[str, Any]:
    img = await single_image(request, settings)
    rec = recognize_shelf(db, registry, storage, settings, img, user.id, strict)
    return recognition_out(rec)


@router.post(
    "/recognitions/batch",
    status_code=202,
    response_model=JobAccepted,
    responses=_ERRORS,
    openapi_extra=multipart_doc(
        "files",
        multiple=True,
        description="多張圖片，每張 10 MB 以內",
        fields={"type": {"type": "string", "enum": ["product", "shelf"], "default": "product"}},
    ),
)
async def create_batch(
    request: Request,
    user: LimitedUser,
    db: DB,
    storage: StorageDep,
    settings: AppSettings,
) -> JobAccepted:
    """批次辨識：先驗證、存檔、建立 job，交給 Celery worker 處理，立即回 202。

    multipart 串流解析：每張圖在讀取時就累計大小，超過 10 MB 的部分直接丟掉（不進記憶體、
    不寫硬碟），該張記成 FILE_TOO_LARGE。其他驗證（格式、像素數）也在這裡做，
    **只有驗證通過的圖才寫進硬碟**；不合格的圖記在 item_errors，不會讓整批被拒絕（驗收 2.8）。
    佇列送不出去時，job 標成 failed、刪掉已存的檔案，回 503，不留下永遠 pending 的 job。
    """
    form = await read_multipart(request, settings.max_upload_bytes, settings.max_batch_files)
    files = form.files_named("files")
    type_ = form.field("type") or "product"
    if type_ not in RECOGNIZERS:
        raise AppError("VALIDATION_ERROR", "type 必須是 product 或 shelf", 422)
    if not files:
        raise AppError("VALIDATION_ERROR", "至少要有一張圖片", 422)
    items: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, f in enumerate(files):
        try:
            img = validate_part(f, settings)
        except AppError as exc:
            errors.append(
                {"index": index, "filename": f.filename, "code": exc.code, "message": exc.message}
            )
            continue
        path = storage.save(img.data, img.ext, "batch")
        items.append({"index": index, "path": path, "filename": f.filename})

    job = Job(
        user_id=user.id,
        type=type_,
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
