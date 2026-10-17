"""新品登錄與以圖搜圖（驗收 2.9）。

每個 embedding 都記錄是哪個模型算的（model_version）：不同模型的向量不在同一個空間，
搜尋時只用使用中模型的向量；切換模型前先用新模型把所有參考圖重新算一次（reembed）。

索引每次從資料庫現建：catalog 只有數百到數千個向量，查詢 + 組矩陣在毫秒內完成；
不做跨請求快取，就不會有「另一個 process 新增了商品但這邊的快取沒更新」的問題。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import AppError, not_found
from app.db.models import CatalogEmbedding, CatalogProduct
from app.services.registry import ClassifierLike
from app.services.uploads import Storage, ValidatedImage
from vision.embedder import EmbeddingIndex
from vision.io import Image, decode_image

log = logging.getLogger("app.catalog")


@dataclass(frozen=True)
class CatalogHit:
    product_id: int
    name: str
    similarity: float


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


def create_product(
    db: Session,
    clf: ClassifierLike,
    storage: Storage,
    name: str,
    sku: str | None,
    images: list[ValidatedImage],
) -> CatalogProduct:
    if not 1 <= len(images) <= 5:
        raise AppError("VALIDATION_ERROR", "參考圖必須是 1 到 5 張", 422)
    if sku and db.scalar(select(CatalogProduct.id).where(CatalogProduct.sku == sku)):
        raise AppError("CONFLICT", f"SKU {sku} 已存在", 409)
    saved: list[str] = []
    try:
        product = CatalogProduct(name=name, sku=sku)
        db.add(product)
        db.flush()
        for img in images:
            path = storage.save(img.data, img.ext, "catalog")
            saved.append(path)
            db.add(
                CatalogEmbedding(
                    product_id=product.id,
                    vector=to_bytes(clf.embed(img.image)),
                    image_path=path,
                    model_version=clf.version,
                )
            )
        db.commit()
    except Exception:
        # 資料庫寫入失敗（例如 SKU 同時被別人登錄）：回滾並刪掉已存的參考圖，不留孤兒檔案
        db.rollback()
        for path in saved:
            storage.delete(path)
        raise
    db.refresh(product)
    return product


def list_products(db: Session, page: int, size: int) -> tuple[list[CatalogProduct], int]:
    total = db.scalar(select(func.count()).select_from(CatalogProduct)) or 0
    rows = db.scalars(
        select(CatalogProduct).order_by(CatalogProduct.id).offset((page - 1) * size).limit(size)
    ).all()
    return list(rows), total


def reference_count(db: Session, product_id: int, version: str) -> int:
    return (
        db.scalar(
            select(func.count())
            .select_from(CatalogEmbedding)
            .where(
                CatalogEmbedding.product_id == product_id,
                CatalogEmbedding.model_version == version,
            )
        )
        or 0
    )


def delete_product(db: Session, storage: Storage, product_id: int) -> None:
    product = db.get(CatalogProduct, product_id)
    if product is None:
        raise not_found("找不到商品")
    paths = {e.image_path for e in product.embeddings}
    db.delete(product)
    db.commit()
    # 先刪資料庫再刪檔案：檔案刪不掉頂多留下孤兒檔案（記 log），不會出現資料庫指向不存在的檔案
    for p in paths:
        try:
            storage.delete(p)
        except OSError:
            log.warning("刪除參考圖失敗：%s", p, exc_info=True)


def build_index(db: Session, version: str) -> tuple[EmbeddingIndex | None, dict[int, str]]:
    rows = db.execute(
        select(CatalogEmbedding.product_id, CatalogEmbedding.vector, CatalogProduct.name)
        .join(CatalogProduct)
        .where(CatalogEmbedding.model_version == version)
    ).all()
    if not rows:
        return None, {}
    dim = len(from_bytes(rows[0].vector))
    index = EmbeddingIndex(dim)
    names: dict[int, str] = {}
    for pid, vec, name in rows:
        index.add(pid, from_bytes(vec)[None])
        names[pid] = name
    return index, names


def search(db: Session, clf: ClassifierLike, image: Image, k: int) -> list[CatalogHit]:
    index, names = build_index(db, clf.version)
    if index is None:
        return []
    return [
        CatalogHit(m.item_id, names[m.item_id], m.similarity)
        for m in index.search(clf.embed(image), k)
    ]


def reembed(db: Session, clf: ClassifierLike, storage: Storage) -> int:
    """用 clf 對所有參考圖補算 embedding（已經有這個版本的就跳過），回傳新增的數量。

    舊版本的向量保留：切回舊模型時不用再算一次。
    """
    have = set(
        db.scalars(
            select(CatalogEmbedding.image_path).where(CatalogEmbedding.model_version == clf.version)
        )
    )
    refs = db.execute(
        select(CatalogEmbedding.product_id, CatalogEmbedding.image_path).distinct()
    ).all()
    added = 0
    for pid, path in refs:
        if path in have:
            continue
        vec = clf.embed(decode_image(storage.read(path)))
        db.add(
            CatalogEmbedding(
                product_id=pid, vector=to_bytes(vec), image_path=path, model_version=clf.version
            )
        )
        have.add(path)
        added += 1
    db.flush()
    return added
