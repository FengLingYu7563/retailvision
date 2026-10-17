"""one active model per task

每個任務最多一個 is_active 的模型版本（Codex 第 9 輪 L1）。
用可延遲到 commit 才檢查的 exclusion constraint：切換模型時同一個交易會先後更新新舊兩列，
逐列檢查的 partial unique index 會因為更新順序而誤判衝突。
（Alembic autogenerate 偵測不到 exclusion constraint，這支 migration 是手寫的。）

Revision ID: 86839bae6e29
Revises: 14d6301e83f0
Create Date: 2026-10-18 01:11:59.677880
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "86839bae6e29"
down_revision: str | None = "14d6301e83f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NAME = "ex_model_versions_one_active_per_task"


def upgrade() -> None:
    # 固定的 DDL 字串，沒有任何外部輸入（不是字串拼接 SQL）
    op.execute(
        "ALTER TABLE model_versions ADD CONSTRAINT ex_model_versions_one_active_per_task "
        "EXCLUDE USING btree (task WITH =) WHERE (is_active) DEFERRABLE INITIALLY DEFERRED"
    )


def downgrade() -> None:
    op.drop_constraint(NAME, "model_versions")
