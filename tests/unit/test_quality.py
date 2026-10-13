from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from tests.conftest import GROCERY_DIR, ROOT
from vision.io import load_image
from vision.quality import blur_score, check_quality


def test_sharp_passes(sharp_image: np.ndarray) -> None:
    report = check_quality(sharp_image)
    assert report.passed, report
    assert report.warnings == []


def test_blurred_fails(blurred_image: np.ndarray) -> None:
    report = check_quality(blurred_image)
    assert not report.passed
    assert "BLURRY" in report.warnings


def test_blur_score_resolution_invariant(sharp_image: np.ndarray) -> None:
    # 同一張圖放大兩倍，分數不應差到跨越門檻（分析前會正規化尺寸）
    big = cv2.resize(sharp_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    small, large = blur_score(sharp_image), blur_score(big)
    assert large / small > 0.5


def test_dark_and_bright() -> None:
    dark = np.full((100, 100, 3), 3, dtype=np.uint8)
    bright = np.full((100, 100, 3), 253, dtype=np.uint8)
    assert {"TOO_DARK", "UNDEREXPOSED"} <= set(check_quality(dark).warnings)
    assert {"TOO_BRIGHT", "OVEREXPOSED"} <= set(check_quality(bright).warnings)


def test_grayscale_input(sharp_image: np.ndarray) -> None:
    gray = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
    assert check_quality(gray).blur_score > 0


@pytest.mark.needs_data
def test_real_photos_blur_detection() -> None:
    """真實商品照（測試集前 50 張，排序固定）：原圖通過模糊檢查，GaussianBlur 後判為未通過。"""
    with (ROOT / "training" / "data" / "splits" / "grocery_test.csv").open(encoding="utf-8") as f:
        paths = [row["path"] for row in csv.DictReader(f)][::50][:50]
    orig_pass = blur_fail = 0
    for p in paths:
        img = load_image(GROCERY_DIR / p)
        orig_pass += "BLURRY" not in check_quality(img).warnings
        blur_fail += "BLURRY" in check_quality(cv2.GaussianBlur(img, (21, 21), 6)).warnings
    assert blur_fail == len(paths)
    assert orig_pass / len(paths) >= 0.9, f"原圖通過率 {orig_pass}/{len(paths)}"
