"""設計約束與 repo 衛生檢查（Codex 審查提出的驗收補充項目）。"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import ROOT

# vision/ 要能放進不含 PyTorch 的 API 映像檔，所以不能 import 這些套件
FORBIDDEN_IN_VISION = {"torch", "torchvision", "ultralytics", "fastapi", "starlette"}


def test_vision_has_no_heavy_or_web_dependencies() -> None:
    offenders = []
    for path in (ROOT / "vision").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [
                f"{path.name}: {n}" for n in names if n.split(".")[0] in FORBIDDEN_IN_VISION
            ]
    assert offenders == []


# app/（API 與 worker）要能放進不含 PyTorch 的映像檔：不能 import 訓練用套件或 training/ 模組
FORBIDDEN_IN_APP = {"torch", "torchvision", "ultralytics", "timm", "training"}


def test_app_has_no_training_dependencies() -> None:
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [
                f"{path.relative_to(ROOT)}: {n}"
                for n in names
                if n.split(".")[0] in FORBIDDEN_IN_APP
            ]
    assert offenders == []


def test_default_dependencies_exclude_torch() -> None:
    """pyproject 的主要依賴（API 映像檔會裝的）不含 torch 系列；那些只在 [train] extra。"""
    import tomllib

    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    names = {d.split("[")[0].split(">")[0].split("=")[0].strip().lower() for d in deps}
    assert not names & {"torch", "torchvision", "ultralytics", "timm"}


FORBIDDEN_TRACKED_SUFFIXES = {".onnx", ".pt", ".pth", ".joblib", ".tar", ".gz", ".npy", ".log"}
FORBIDDEN_TRACKED_PREFIXES = (
    "training/data/raw/",
    "training/data/processed/",
    "checkpoints/",
    "runs/",
    "storage/",
)


def test_no_datasets_weights_or_env_tracked() -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("不是 git repo")
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    bad = [
        f
        for f in files
        if Path(f).suffix in FORBIDDEN_TRACKED_SUFFIXES
        or f.startswith(FORBIDDEN_TRACKED_PREFIXES)
        or Path(f).name == ".env"
    ]
    assert bad == []
