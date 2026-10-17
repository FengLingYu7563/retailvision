"""SQLAlchemy 2.0 資料表（規格第 7 節）。

和規格的差異（都是 Phase 1 結果帶來的需要，記錄在 docs/ACCEPTANCE_PHASE2.md）：
- recognitions 加 job_id：批次任務要能查到它產生了哪些辨識紀錄。
- jobs 加 type、failed_count、item_errors：一張壞圖不能讓整批失敗（2.8），失敗的項目要能查。
- catalog_embeddings 加 model_version：不同模型的 embedding 不在同一個空間，切換模型時要重新計算。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, ExcludeConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    total: Mapped[int] = mapped_column(Integer, nullable=False)
    done_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    item_errors: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Recognition(Base):
    __tablename__ = "recognitions"
    __table_args__ = (Index("ix_recognitions_type_created_at", "type", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), index=True
    )
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)
    annotated_path: Mapped[str | None] = mapped_column(String(255))
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    top_label: Mapped[str | None] = mapped_column(String(128), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    quality_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CatalogProduct(Base):
    __tablename__ = "catalog_products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    sku: Mapped[str | None] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    embeddings: Mapped[list[CatalogEmbedding]] = relationship(
        back_populates="product", cascade="all, delete-orphan", passive_deletes=True
    )


class CatalogEmbedding(Base):
    __tablename__ = "catalog_embeddings"
    __table_args__ = (Index("ix_catalog_embeddings_model_version", "model_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # float32 bytes：資料量小（每個商品 1 到 5 個向量），不需要 pgvector
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    product: Mapped[CatalogProduct] = relationship(back_populates="embeddings")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    # 每個任務最多一個使用中的版本（Codex 第 9 輪 L1）。
    # 用可延遲到 commit 才檢查的 exclusion constraint，而不是 partial unique index：
    # 切換時同一個交易裡會先後更新新舊兩列，unique index 逐列檢查，
    # 更新順序剛好先設新的 active 時會誤判衝突；延遲到 commit 時只剩一個 active，就不會誤判。
    __table_args__ = (
        ExcludeConstraint(
            ("task", "="),
            where=text("is_active"),
            name="ex_model_versions_one_active_per_task",
            using="btree",  # 和 migration 一致，之後比對 schema 才不會有落差
            deferrable=True,
            initially="DEFERRED",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    onnx_path: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
