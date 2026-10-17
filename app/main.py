"""FastAPI 進入點：uvicorn app.main:create_app --factory

用 factory 而不是模組層級的 app 物件：import 這個模組不會讀設定、不會載入模型，
測試可以自己用假模型建立 app。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import health
from app.api.v1 import router as v1_router
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.core.logging import request_id_middleware, setup_logging
from app.services.registry import ModelRegistry
from app.services.uploads import Storage


def create_app(registry: ModelRegistry | None = None, settings: Settings | None = None) -> FastAPI:
    """registry 可以從外部傳入：測試用假的 vision 模型，CI 不需要 ONNX 權重。"""
    settings = settings or get_settings()
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 模型只在啟動時載入一次，所有請求共用（規格第 4 節設計原則）
        app.state.registry = registry or ModelRegistry.from_settings(settings)
        app.state.storage = Storage(settings.storage_dir)
        yield

    app = FastAPI(
        title="RetailVision API",
        version="1.0.0",
        description="零售商品影像辨識：單品辨識、貨架分析、新品登錄與以圖搜圖。",
        lifespan=lifespan,
    )
    app.middleware("http")(request_id_middleware)
    install_error_handlers(app)
    app.include_router(v1_router)
    # 根路徑也提供 /healthz，給 docker healthcheck 與負載平衡器用
    app.include_router(health.router, include_in_schema=False)
    return app
