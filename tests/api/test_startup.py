"""乾淨 checkout 的啟動行為（Codex 第 5 輪 L1）。

權重檔不進 git，缺檔或壞檔都不能讓 API 啟動失敗。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.services.registry import ModelRegistry
from tests.api.conftest import ROOT, assert_error


def _settings(models_dir: Path):  # type: ignore[no-untyped-def]
    return get_settings().model_copy(update={"models_dir": models_dir})


def test_clean_checkout_only_tracked_metadata(tmp_path: Path) -> None:
    """只有 git 追蹤的 .json（例如 gap_classifier.json），沒有任何 .onnx / .joblib。"""
    shutil.copy(ROOT / "models" / "gap_classifier.json", tmp_path / "gap_classifier.json")
    reg = ModelRegistry.from_settings(_settings(tmp_path))
    assert reg.classifier is None and reg.detector is None and reg.gap_classifier is None
    assert not reg.ready


def test_corrupt_model_files_do_not_crash(tmp_path: Path) -> None:
    s = get_settings()
    (tmp_path / f"{s.classifier_version}.onnx").write_bytes(b"not an onnx model")
    (tmp_path / f"{s.classifier_version}.json").write_text(json.dumps({"version": "x"}))
    (tmp_path / f"{s.detector_version}.onnx").write_bytes(b"garbage")
    (tmp_path / "gap_classifier.json").write_text(json.dumps({"feature_names": []}))
    (tmp_path / "gap_classifier.joblib").write_bytes(b"garbage")
    reg = ModelRegistry.from_settings(_settings(tmp_path))
    assert reg.classifier is None and reg.detector is None and reg.gap_classifier is None


def test_api_starts_and_reports_degraded_without_models(tmp_path: Path) -> None:
    from app.main import create_app

    settings = _settings(tmp_path)
    with TestClient(create_app(settings=settings)) as c:  # 用真的 from_settings，不注入假模型
        r = c.get("/healthz")
        assert_error(r, 503, "SERVICE_UNAVAILABLE")
        assert r.json()["error"]["details"]["models"]["classifier"] is None


def test_db_active_model_missing_returns_model_not_ready(
    user_token: str, admin_token: str, db
) -> None:  # type: ignore[no-untyped-def]
    """資料庫的使用中分類模型指向不存在或損壞的檔案（例如另一台機器切換了模型、這台沒放檔案）：
    需要分類模型的端點回 503 MODEL_NOT_READY，不是 500；貨架分析與健康檢查仍正常回應。"""
    from sqlalchemy import text

    from app.main import create_app
    from tests.api.conftest import RED, FakeClassifier, FakeDetector, auth, jpeg, textured

    def loader(name: str, onnx_path: str | None = None) -> FakeClassifier:
        if name == "cls-fake-b":
            raise RuntimeError("corrupt onnx")
        return FakeClassifier(name)

    reg = ModelRegistry(loader, FakeDetector(), None)
    reg.load_classifier("cls-fake-a")
    with TestClient(create_app(registry=reg, settings=get_settings())) as c:
        db.execute(
            text(
                "UPDATE model_versions SET is_active = (name = 'cls-fake-b') "
                "WHERE task = 'classification'"
            )
        )
        db.commit()
        img = {"file": ("a.jpg", jpeg(textured(RED)), "image/jpeg")}
        for path in ("/api/v1/recognitions/product", "/api/v1/search/similar"):
            assert_error(c.post(path, files=img, headers=auth(user_token)), 503, "MODEL_NOT_READY")
        # 貨架分析只需要偵測模型，照常運作
        r = c.post("/api/v1/recognitions/shelf", files=img, headers=auth(user_token))
        assert r.status_code == 201
        assert_error(c.get("/healthz"), 503, "SERVICE_UNAVAILABLE")
        # 切換到損壞的模型：409，不切換
        r = c.put("/api/v1/models/active", json={"name": "cls-fake-b"}, headers=auth(admin_token))
        assert_error(r, 409, "MODEL_NOT_READY")


def test_healthz_as_first_request_after_switch_to_broken_model(db) -> None:  # type: ignore[no-untyped-def]
    """另一個 process 把使用中模型切到壞掉的版本後，這個 process 收到的第一個請求就是 /healthz：
    必須回 503，classifier 為 null，不能回報舊模型的狀態。"""
    from sqlalchemy import text

    from app.main import create_app
    from tests.api.conftest import FakeClassifier, FakeDetector

    def loader(name: str, onnx_path: str | None = None) -> FakeClassifier:
        if name == "cls-fake-b":
            raise RuntimeError("corrupt onnx")
        return FakeClassifier(name)

    reg = ModelRegistry(loader, FakeDetector(), None)
    reg.load_classifier("cls-fake-a")
    with TestClient(create_app(registry=reg, settings=get_settings())) as c:
        db.execute(
            text(
                "UPDATE model_versions SET is_active = (name = 'cls-fake-b') "
                "WHERE task = 'classification'"
            )
        )
        db.commit()
        r = c.get("/healthz")  # 第一個請求
    assert_error(r, 503, "SERVICE_UNAVAILABLE")
    assert r.json()["error"]["details"]["models"]["classifier"] is None


def test_no_active_classifier_in_db_disables_stale_model(user_token: str, db) -> None:  # type: ignore[no-untyped-def]
    """資料庫沒有任何使用中的分類模型：不能繼續用記憶體裡的舊模型。"""
    from sqlalchemy import text

    from app.main import create_app
    from tests.api.conftest import RED, auth, fake_registry, jpeg, textured

    with TestClient(create_app(registry=fake_registry(), settings=get_settings())) as c:
        db.execute(
            text("UPDATE model_versions SET is_active = false WHERE task = 'classification'")
        )
        db.commit()
        assert_error(c.get("/healthz"), 503, "SERVICE_UNAVAILABLE")
        img = {"file": ("a.jpg", jpeg(textured(RED)), "image/jpeg")}
        r = c.post("/api/v1/recognitions/product", files=img, headers=auth(user_token))
        assert_error(r, 503, "MODEL_NOT_READY")
        # 重新設定使用中模型後恢復
        db.execute(
            text(
                "UPDATE model_versions SET is_active = (name = 'cls-fake-a') "
                "WHERE task = 'classification'"
            )
        )
        db.commit()
        r = c.post("/api/v1/recognitions/product", files=img, headers=auth(user_token))
        assert r.status_code == 201 and r.json()["model_version"] == "cls-fake-a"


def test_loader_uses_onnx_path_from_db(tmp_path: Path) -> None:
    """模型檔名可以和版本名稱不同：以 model_versions.onnx_path 為準（Codex 第 8 輪 L1）。"""
    import pytest

    from app.services.registry import onnx_classifier_loader

    src = ROOT / "models" / "cls-dinov2-v2.onnx"
    if not src.exists():
        pytest.skip("需要本機的 ONNX 權重")
    shutil.copy(src, tmp_path / "renamed-file.onnx")
    shutil.copy(src.with_suffix(".json"), tmp_path / "renamed-file.json")
    load = onnx_classifier_loader(_settings(tmp_path))

    assert load("cls-dinov2-v2", "renamed-file.onnx").version == "cls-dinov2-v2"
    # metadata 記錄的版本和資料庫不同：拒絕，避免列到錯的檔案
    with pytest.raises(ValueError, match="版本"):
        load("cls-other", "renamed-file.onnx")
    with pytest.raises(FileNotFoundError):
        load("cls-dinov2-v2", "missing.onnx")


def test_loader_rejects_path_outside_models_dir(tmp_path: Path) -> None:
    import pytest

    from app.services.registry import onnx_classifier_loader

    load = onnx_classifier_loader(_settings(tmp_path / "models"))
    for bad in ("../secret.onnx", "../../etc/passwd", str(tmp_path / "abs.onnx")):
        with pytest.raises(ValueError, match="模型資料夾"):
            load("x", bad)


def test_switch_uses_onnx_path(db, admin_token: str) -> None:  # type: ignore[no-untyped-def]
    """PUT /models/active 與 sync 都把資料庫的 onnx_path 傳給 loader。"""
    from sqlalchemy import text

    from app.main import create_app
    from tests.api.conftest import FakeClassifier, FakeDetector, auth

    calls: list[tuple[str, str | None]] = []

    def loader(name: str, onnx_path: str | None = None) -> FakeClassifier:
        calls.append((name, onnx_path))
        return FakeClassifier(name, onnx_path)

    reg = ModelRegistry(loader, FakeDetector(), None)
    reg.load_classifier("cls-fake-a")
    db.execute(
        text("UPDATE model_versions SET onnx_path = 'files/b-v2.onnx' WHERE name = 'cls-fake-b'")
    )
    db.commit()
    with TestClient(create_app(registry=reg, settings=get_settings())) as c:
        r = c.put("/api/v1/models/active", json={"name": "cls-fake-b"}, headers=auth(admin_token))
        assert r.status_code == 200, r.text
    assert ("cls-fake-b", "files/b-v2.onnx") in calls


def test_missing_gap_classifier_falls_back_to_v1(tmp_path: Path) -> None:
    """分類與偵測模型都在、只缺空位分類器的 .joblib：服務正常，貨架分析用 v1 規則。"""
    s = get_settings()
    for name in (s.classifier_version, s.detector_version):
        for ext in (".onnx", ".json"):
            src = ROOT / "models" / f"{name}{ext}"
            if src.exists():
                shutil.copy(src, tmp_path / src.name)
    shutil.copy(ROOT / "models" / "gap_classifier.json", tmp_path / "gap_classifier.json")
    if not (tmp_path / f"{s.classifier_version}.onnx").exists():
        import pytest

        pytest.skip("需要本機的 ONNX 權重")
    reg = ModelRegistry.from_settings(_settings(tmp_path))
    assert reg.ready and reg.gap_classifier is None
