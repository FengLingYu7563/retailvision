"""單品辨識與貨架分析（API 同步端點與 Celery 批次共用）。"""

from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.db.models import Recognition
from app.services import catalog
from app.services.registry import ModelRegistry
from app.services.uploads import Storage, ValidatedImage
from vision.draw import draw_label, draw_shelf
from vision.io import encode_jpeg
from vision.quality import QualityReport, check_quality
from vision.shelf import analyze_shelf


def _quality_dict(q: QualityReport) -> dict[str, Any]:
    return {
        "blur_score": q.blur_score,
        "brightness": q.brightness,
        "passed": q.passed,
        "warnings": q.warnings,
    }


def _check_quality(img: ValidatedImage, strict: bool) -> QualityReport:
    q = check_quality(img.image)
    # 品質不佳預設只是警告；strict=true 時才擋下（規格第 6 節 LOW_QUALITY_IMAGE）
    if strict and not q.passed:
        raise AppError("LOW_QUALITY_IMAGE", f"影像品質不佳：{', '.join(q.warnings)}", 422)
    return q


def recognize_product(
    db: Session,
    registry: ModelRegistry,
    storage: Storage,
    settings: Settings,
    img: ValidatedImage,
    user_id: int,
    strict: bool = False,
    job_id: uuid.UUID | None = None,
) -> Recognition:
    clf = registry.classifier
    if clf is None:
        raise AppError("MODEL_NOT_READY", "分類模型尚未載入", 503)
    quality = _check_quality(img, strict)

    start = time.perf_counter()
    pred = clf.predict(img.image)
    hits = catalog.search(db, clf, img.image, k=1)
    latency = (time.perf_counter() - start) * 1000

    # 相似度低於門檻就不硬塞一個最像的商品（Phase 2 調整第 5 點）
    match = None
    if hits and hits[0].similarity >= settings.catalog_match_threshold:
        h = hits[0]
        match = {"product_id": h.product_id, "name": h.name, "similarity": h.similarity}

    image_path = storage.save(img.data, img.ext, "uploads")
    annotated = draw_label(img.image, f"{pred.fine.label} {pred.fine.confidence:.2f}")
    annotated_path = storage.save(encode_jpeg(annotated), ".jpg", "annotated")
    result = {
        "quality": _quality_dict(quality),
        "prediction": {
            "fine": {"label": pred.fine.label, "confidence": pred.fine.confidence},
            "coarse": {"label": pred.coarse.label, "confidence": pred.coarse.confidence},
            "top_k": [{"label": s.label, "confidence": s.confidence} for s in pred.top_k],
        },
        "catalog_match": match,
    }
    rec = Recognition(
        type="product",
        user_id=user_id,
        job_id=job_id,
        image_path=image_path,
        annotated_path=annotated_path,
        result=result,
        top_label=pred.fine.label,
        confidence=pred.fine.confidence,
        model_version=clf.version,
        latency_ms=round(latency, 2),
        quality_passed=quality.passed,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def recognize_shelf(
    db: Session,
    registry: ModelRegistry,
    storage: Storage,
    settings: Settings,
    img: ValidatedImage,
    user_id: int,
    strict: bool = False,
    job_id: uuid.UUID | None = None,
) -> Recognition:
    det = registry.detector
    if det is None:
        raise AppError("MODEL_NOT_READY", "偵測模型尚未載入", 503)
    quality = _check_quality(img, strict)

    start = time.perf_counter()
    boxes = det.detect(img.image)
    analysis = analyze_shelf(
        boxes, img.image.shape, image=img.image, classifier=registry.gap_classifier
    )
    latency = (time.perf_counter() - start) * 1000

    image_path = storage.save(img.data, img.ext, "uploads")
    annotated = draw_shelf(img.image, analysis.items, analysis.gaps)
    annotated_path = storage.save(encode_jpeg(annotated), ".jpg", "annotated")
    items = analysis.items
    result = {
        "quality": _quality_dict(quality),
        "item_count": analysis.item_count,
        "row_count": analysis.row_count,
        "boxes": [b.to_dict() for b in items],
        "gaps": [g.to_dict() for g in analysis.gaps],
        "gap_ratio": analysis.gap_ratio,
        "density": analysis.density,
    }
    rec = Recognition(
        type="shelf",
        user_id=user_id,
        job_id=job_id,
        image_path=image_path,
        annotated_path=annotated_path,
        result=result,
        top_label=None,
        confidence=round(sum(b.confidence for b in items) / len(items), 4) if items else None,
        model_version=det.version,
        latency_ms=round(latency, 2),
        quality_passed=quality.passed,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


RECOGNIZERS = {"product": recognize_product, "shelf": recognize_shelf}
