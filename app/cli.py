"""管理用 CLI（規格第 11 節：不做註冊流程，admin 用 CLI 建立）。

python -m app.cli create-user <username> --role admin
    密碼從互動提示輸入，不放在命令列（命令列參數會留在 shell 歷史紀錄裡）
python -m app.cli seed-models
    依 models/*.json 與 metrics.json 登錄模型版本
python -m app.cli bootstrap-admin
    依 ADMIN_USERNAME / ADMIN_PASSWORD 環境變數建立 admin（容器啟動時自動執行）
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings, is_placeholder
from app.core.security import hash_password
from app.db.models import ModelVersion, User
from app.db.session import session_factory

MIN_PASSWORD = 12


def create_user(db: Session, username: str, password: str, role: str) -> User:
    if role not in {"admin", "user"}:
        raise ValueError("role 只能是 admin 或 user")
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"密碼至少 {MIN_PASSWORD} 個字元")
    if is_placeholder(password):
        raise ValueError("密碼不可以用範例值")
    if db.scalar(select(User.id).where(User.username == username)):
        raise ValueError(f"帳號 {username} 已存在")
    user = User(username=username, password_hash=hash_password(password), role=role)
    db.add(user)
    db.commit()
    return user


def seed_models(db: Session) -> list[str]:
    """把 models/ 底下有 .onnx 的版本登錄進 model_versions；已存在的只更新指標。"""
    settings = get_settings()
    metrics_path = settings.models_dir / "metrics.json"
    metrics: dict[str, Any] = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    section = {"classification": "cnn", "detection": "detector"}
    done = []
    for meta_path in sorted(settings.models_dir.glob("*.json")):
        onnx = meta_path.with_suffix(".onnx")
        if not onnx.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task = meta.get("task")
        if task not in section:
            continue
        m = metrics.get(section[task], {})
        # metrics.json 只記錄最新評估的那個版本，版本不同時不套用，避免把別的模型的指標掛上去
        m = m if m.get("model_version") == meta["version"] else {}
        row = db.scalar(select(ModelVersion).where(ModelVersion.name == meta["version"]))
        if row is None:
            row = ModelVersion(name=meta["version"], task=task, onnx_path=onnx.name, metrics=m)
            db.add(row)
        elif m:
            row.metrics = m
        done.append(meta["version"])
    db.flush()
    # 每個任務都要有一個使用中的版本：沒有的話用設定檔指定的預設版本
    defaults = {
        "classification": settings.classifier_version,
        "detection": settings.detector_version,
    }
    for task, name in defaults.items():
        rows = db.scalars(select(ModelVersion).where(ModelVersion.task == task)).all()
        if rows and not any(r.is_active for r in rows):
            for r in rows:
                r.is_active = r.name == name
    db.commit()
    return done


def bootstrap_admin(db: Session) -> str:
    """容器啟動時建立 admin（Phase 3 調整第 2 點）：讀 ADMIN_USERNAME / ADMIN_PASSWORD 環境變數。

    - 沒設定就略過（不強制每個部署都有 admin）。
    - 帳號已存在就略過，**不會**用環境變數覆蓋既有密碼（避免重啟容器就重設密碼）。
    - 密碼不符合規則時直接失敗（ValueError -> exit 1），不默默略過，設定錯誤要讓人看到。
    """
    username = os.environ.get("ADMIN_USERNAME", "").strip()
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not username or not password:
        return "未設定 ADMIN_USERNAME / ADMIN_PASSWORD，略過建立 admin"
    if db.scalar(select(User.id).where(User.username == username)):
        return f"admin {username} 已存在，略過"
    create_user(db, username, password, "admin")
    return f"已建立 admin {username}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cu = sub.add_parser("create-user")
    cu.add_argument("username")
    cu.add_argument("--role", choices=["admin", "user"], default="user")
    sub.add_parser("seed-models")
    sub.add_parser("bootstrap-admin")
    args = ap.parse_args(argv)

    db = session_factory()()
    try:
        if args.cmd == "create-user":
            password = getpass.getpass("密碼：")
            if password != getpass.getpass("再輸入一次："):
                print("兩次密碼不一致", file=sys.stderr)
                return 1
            user = create_user(db, args.username, password, args.role)
            print(f"已建立 {user.username}（{user.role}）")
        elif args.cmd == "bootstrap-admin":
            print(bootstrap_admin(db))
        else:
            print("已登錄：", ", ".join(seed_models(db)))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
