"""/api/v1 路由。"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import auth, catalog, health, jobs, models, recognitions, search, stats

router = APIRouter(prefix="/api/v1")
for module in (auth, recognitions, jobs, catalog, search, models, stats, health):
    router.include_router(module.router)
