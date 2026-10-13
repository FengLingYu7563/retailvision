"""共用 fixture。

CI 沒有資料集也沒有權重，所以單元測試用程式產生的合成圖片；
需要真實資料或權重的測試用 marker 標記，缺少時自動 skip。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GROCERY_DIR = ROOT / "training" / "data" / "raw" / "GroceryStoreDataset" / "dataset"
MODELS_DIR = ROOT / "models"
CHECKPOINT_DIR = ROOT / "checkpoints"
SKU_IMAGES = ROOT / "training" / "data" / "raw" / "SKU110K_fixed" / "images"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_data = GROCERY_DIR.exists()
    has_torch = importlib.util.find_spec("torch") is not None
    has_weights = has_torch and any(CHECKPOINT_DIR.glob("classifier*/best.pt"))
    has_detector = (
        importlib.util.find_spec("ultralytics") is not None
        and (MODELS_DIR / "det-yolov8n-v1.onnx").exists()
        and SKU_IMAGES.exists()
    )
    for item in items:
        if "needs_data" in item.keywords and not has_data:
            item.add_marker(pytest.mark.skip(reason="缺少本機資料集"))
        if "needs_weights" in item.keywords and not has_weights:
            item.add_marker(pytest.mark.skip(reason="缺少 torch 或本機訓練權重"))
        if "needs_detector" in item.keywords and not has_detector:
            item.add_marker(pytest.mark.skip(reason="缺少 ultralytics、偵測模型或 SKU-110K"))


@pytest.fixture
def sharp_image() -> np.ndarray:
    """有清楚邊緣與文字的合成「商品」圖。"""
    rng = np.random.default_rng(0)
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    for _ in range(25):
        x1, y1 = rng.integers(0, 600), rng.integers(0, 440)
        x2, y2 = x1 + rng.integers(10, 120), y1 + rng.integers(10, 120)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    for i in range(6):
        cv2.putText(
            img, "RETAIL 123", (20, 60 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
        )
    return img


@pytest.fixture
def blurred_image(sharp_image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(sharp_image, (21, 21), 6)
