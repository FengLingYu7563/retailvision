"""統計：各類別數量、平均信心度、缺貨率、每日趨勢（全部用 ORM 聚合查詢，驗收 2.7）。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Float, case, cast, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from app.db.models import Recognition, User


def _scoped(stmt: Select[Any], user: User) -> Select[Any]:
    # 一般使用者只看得到自己的資料，admin 看全部
    return stmt if user.role == "admin" else stmt.where(Recognition.user_id == user.id)


def summary(db: Session, user: User, days: int = 30, top: int = 20) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(days=days)
    base = Recognition.created_at >= since

    by_label = db.execute(
        _scoped(
            select(
                Recognition.top_label,
                func.count(),
                func.avg(Recognition.confidence),
            )
            .where(base, Recognition.type == "product")
            .group_by(Recognition.top_label)
            .order_by(func.count().desc())
            .limit(top),
            user,
        )
    ).all()

    product_stats = db.execute(
        _scoped(
            select(func.count(), func.avg(Recognition.confidence)).where(
                base, Recognition.type == "product"
            ),
            user,
        )
    ).one()

    gap_ratio = cast(Recognition.result["gap_ratio"].astext, Float)
    gap_count = func.jsonb_array_length(Recognition.result["gaps"])
    shelf_stats = db.execute(
        _scoped(
            select(
                func.count(),
                func.avg(gap_ratio),
                # 缺貨率：至少偵測到一個空位的貨架照比例
                # PostgreSQL 不能把 boolean 直接轉成浮點數，用 CASE 轉成 1 / 0
                func.avg(case((gap_count > 0, 1.0), else_=0.0)),
            ).where(base, Recognition.type == "shelf"),
            user,
        )
    ).one()

    day = func.date_trunc("day", Recognition.created_at)
    trend = db.execute(
        _scoped(
            select(day, Recognition.type, func.count())
            .where(base)
            .group_by(day, Recognition.type)
            .order_by(day),
            user,
        )
    ).all()

    def r(v: Any, n: int = 4) -> float | None:
        return round(float(v), n) if v is not None else None

    return {
        "days": days,
        "product": {
            "count": product_stats[0],
            "avg_confidence": r(product_stats[1]),
            "by_label": [
                {"label": lbl, "count": c, "avg_confidence": r(a)} for lbl, c, a in by_label
            ],
        },
        "shelf": {
            "count": shelf_stats[0],
            "avg_gap_ratio": r(shelf_stats[1]),
            "stockout_rate": r(shelf_stats[2]),
        },
        "daily": [{"date": d.date().isoformat(), "type": t, "count": c} for d, t, c in trend],
    }
