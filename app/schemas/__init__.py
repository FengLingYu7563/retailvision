"""Pydantic v2 schemas：定義 OpenAPI 文件，也用來驗證回應格式（驗收 2.1）。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class Quality(BaseModel):
    blur_score: float
    brightness: float
    passed: bool
    warnings: list[str]


class LabelScore(BaseModel):
    label: str
    confidence: float


class Prediction(BaseModel):
    fine: LabelScore
    coarse: LabelScore
    top_k: list[LabelScore]


class CatalogMatch(BaseModel):
    product_id: int
    name: str
    similarity: float


class _RecognitionBase(BaseModel):
    id: uuid.UUID
    quality: Quality
    model_version: str
    latency_ms: float
    created_at: datetime
    job_id: uuid.UUID | None = None


class ProductRecognition(_RecognitionBase):
    type: Literal["product"]
    prediction: Prediction
    catalog_match: CatalogMatch | None


class BoxOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class GapOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    row: int


class ShelfRecognition(_RecognitionBase):
    type: Literal["shelf"]
    item_count: int
    row_count: int
    boxes: list[BoxOut]
    gaps: list[GapOut]
    gap_ratio: float
    density: float


RecognitionOut = Annotated[ProductRecognition | ShelfRecognition, Field(discriminator="type")]


class RecognitionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    type: str
    top_label: str | None
    confidence: float | None
    model_version: str
    latency_ms: float
    quality_passed: bool
    created_at: datetime
    job_id: uuid.UUID | None


class Page(BaseModel, Generic[T]):
    items: list[T]
    page: int
    size: int
    total: int


class JobAccepted(BaseModel):
    job_id: uuid.UUID
    status: str
    total: int


class JobItemError(BaseModel):
    index: int
    filename: str | None
    code: str
    message: str


class JobOut(BaseModel):
    id: uuid.UUID
    type: str
    status: Literal["pending", "running", "done", "failed"]
    total: int
    done_count: int
    failed_count: int
    error: str | None
    item_errors: list[JobItemError]
    recognition_ids: list[uuid.UUID]
    created_at: datetime
    finished_at: datetime | None


class ProductOut(BaseModel):
    id: int
    name: str
    sku: str | None
    reference_images: int
    created_at: datetime


class SimilarHit(BaseModel):
    product_id: int
    name: str
    similarity: float


class SearchOut(BaseModel):
    model_version: str
    results: list[SimilarHit]


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    task: str
    is_active: bool
    metrics: dict[str, Any]
    created_at: datetime


class ActivateModelIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class ActivateModelOut(BaseModel):
    active: str
    reembedded: int


class LabelStat(BaseModel):
    label: str | None
    count: int
    avg_confidence: float | None


class ProductStats(BaseModel):
    count: int
    avg_confidence: float | None
    by_label: list[LabelStat]


class ShelfStats(BaseModel):
    count: int
    avg_gap_ratio: float | None
    stockout_rate: float | None


class DailyCount(BaseModel):
    date: str
    type: str
    count: int


class StatsOut(BaseModel):
    days: int
    product: ProductStats
    shelf: ShelfStats
    daily: list[DailyCount]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    redis: bool
    models: dict[str, str | None]
