"""管理 CLI：建立帳號的檢查、模型版本登錄。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cli import create_user, main, seed_models
from app.core.config import get_settings
from app.db.models import ModelVersion, User


def test_create_user_validation(db: Session) -> None:
    with pytest.raises(ValueError, match="至少"):
        create_user(db, "a", "short", "user")
    with pytest.raises(ValueError, match="role"):
        create_user(db, "a", "long-enough-password", "superuser")
    create_user(db, "a", "long-enough-password", "admin")
    with pytest.raises(ValueError, match="已存在"):
        create_user(db, "a", "long-enough-password", "user")


def test_cli_create_user_reads_password_from_prompt(
    db: Session, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["prompted-password-1", "prompted-password-1"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": next(answers))
    assert main(["create-user", "carol", "--role", "admin"]) == 0
    user = db.scalars(select(User).where(User.username == "carol")).one()
    assert user.role == "admin"

    answers = iter(["one-password-xxx", "another-password"])
    assert main(["create-user", "dave"]) == 1
    assert "不一致" in capsys.readouterr().err


def test_seed_models(db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, task in (
        ("cls-x", "classification"),
        ("cls-y", "classification"),
        ("det-x", "detection"),
    ):
        (tmp_path / f"{name}.onnx").write_bytes(b"onnx")
        (tmp_path / f"{name}.json").write_text(json.dumps({"version": name, "task": task}))
    (tmp_path / "orphan.json").write_text(json.dumps({"version": "orphan", "task": "detection"}))
    (tmp_path / "metrics.json").write_text(
        json.dumps({"cnn": {"model_version": "cls-y", "fine_top1": 0.9}})
    )
    settings = get_settings().model_copy(
        update={"models_dir": tmp_path, "classifier_version": "cls-y", "detector_version": "det-x"}
    )
    monkeypatch.setattr("app.cli.get_settings", lambda: settings)
    db.execute(ModelVersion.__table__.delete())
    db.commit()

    assert sorted(seed_models(db)) == ["cls-x", "cls-y", "det-x"]
    rows = {m.name: m for m in db.scalars(select(ModelVersion))}
    assert set(rows) == {"cls-x", "cls-y", "det-x"}  # 沒有 .onnx 的不登錄
    assert rows["cls-y"].is_active and not rows["cls-x"].is_active and rows["det-x"].is_active
    assert rows["cls-y"].metrics == {"model_version": "cls-y", "fine_top1": 0.9}
    assert rows["cls-x"].metrics == {}  # 別的版本的指標不會掛上去
    # 再跑一次不會重複新增
    seed_models(db)
    assert len(db.scalars(select(ModelVersion)).all()) == 3
