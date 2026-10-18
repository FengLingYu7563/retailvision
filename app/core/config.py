"""設定：全部從環境變數或 .env 讀（pydantic-settings）。

JWT_SECRET、DATABASE_URL 沒有預設值：少設的時候啟動就失敗，
比默默用一個寫死在程式裡的 secret 安全（驗收 2.5）。
文件裡出現過的範例值（replace-with...、change-me...）也拒絕，避免照抄範例就上線。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    redis_url: str = "redis://127.0.0.1:6379/0"
    jwt_secret: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30

    storage_dir: Path = Path("storage")
    models_dir: Path = Path("models")
    classifier_version: str = "cls-dinov2-v2"
    detector_version: str = "det-yolov8n-v1"
    gap_classifier_file: str = "gap_classifier.json"

    max_upload_bytes: int = 10 * 1024 * 1024
    # 解碼後像素上限：防止 decompression bomb（很小的檔案解碼後吃光記憶體）
    max_image_pixels: int = 40_000_000
    max_batch_files: int = 50
    # 批次請求的總大小上限（在 multipart 解析前就檢查，見 app/core/bodylimit.py）
    max_batch_bytes: int = 100 * 1024 * 1024
    # catalog_match 的相似度門檻：training/tune_catalog_threshold.py 只用 Grocery val 選定，
    # cls-dinov2-v2 在 0.675 時「該配對的配對正確、沒登錄的回 null」正確率最高（91.5%）。
    # 門檻依模型而異，換模型時要重跑那支腳本。
    catalog_match_threshold: float = 0.675

    rate_limit_per_minute: int = 60
    celery_eager: bool = False
    onnx_threads: int | None = None

    @field_validator("jwt_secret")
    @classmethod
    def _reject_placeholder_secret(cls, v: str) -> str:
        if is_placeholder(v):
            raise ValueError("JWT_SECRET 不可以用範例值，請產生隨機字串")
        return v


PLACEHOLDER_PREFIXES = ("replace-with", "change-me", "changeme")


def is_placeholder(value: str) -> bool:
    """文件或範例檔裡的佔位值：拿來當密碼或金鑰等於公開。"""
    return value.strip().lower().startswith(PLACEHOLDER_PREFIXES)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
