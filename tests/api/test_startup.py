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
