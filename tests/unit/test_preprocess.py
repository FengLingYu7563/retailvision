from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.io import InvalidImageError, decode_image, encode_jpeg, load_image
from vision.preprocess import (
    clahe,
    denoise,
    find_quad,
    letterbox,
    order_quad,
    perspective_correct,
    to_classifier_input,
)


def test_clahe_raises_contrast_keeps_shape() -> None:
    rng = np.random.default_rng(1)
    low = (rng.integers(100, 130, (200, 300, 3))).astype(np.uint8)
    out = clahe(low)
    assert out.shape == low.shape and out.dtype == np.uint8
    assert out.std() > low.std()


def test_denoise_reduces_noise(sharp_image: np.ndarray) -> None:
    rng = np.random.default_rng(2)
    noisy = np.clip(sharp_image + rng.normal(0, 20, sharp_image.shape), 0, 255).astype(np.uint8)
    diff_before = np.abs(noisy.astype(int) - sharp_image).mean()
    diff_after = np.abs(denoise(noisy).astype(int) - sharp_image).mean()
    assert diff_after < diff_before


def test_classifier_input_shape_and_norm(sharp_image: np.ndarray) -> None:
    x = to_classifier_input(sharp_image)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]
    assert x.min() > -3 and x.max() < 3


def test_letterbox_keeps_aspect() -> None:
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    out, lb = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert lb.scale == pytest.approx(640 / 600)
    assert lb.pad_x == 0 and lb.pad_y == (640 - 320) // 2


def test_order_quad() -> None:
    pts = np.array([[100, 0], [0, 0], [0, 50], [100, 50]], np.float32)
    assert order_quad(pts).tolist() == [[0, 0], [100, 0], [100, 50], [0, 50]]


def test_perspective_correct_recovers_rectangle() -> None:
    # 畫一個白色矩形，用透視變換弄斜，再校正回來，尺寸應接近原本
    flat = np.zeros((200, 300, 3), np.uint8)
    flat[:] = 255
    src = np.array([[0, 0], [299, 0], [299, 199], [0, 199]], np.float32)
    dst = np.array([[40, 20], [330, 0], [350, 240], [20, 210]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(flat, m, (400, 260))
    out = perspective_correct(canvas, dst)
    # 輸出尺寸由四邊形的邊長估計，不會剛好等於原尺寸；重點是整張都落在白色矩形內（已拉正）
    assert abs(out.shape[1] / 300 - 1) < 0.25 and abs(out.shape[0] / 200 - 1) < 0.25
    assert out.mean() > 240


def test_perspective_correct_rejects_degenerate() -> None:
    with pytest.raises(ValueError):
        perspective_correct(np.zeros((10, 10, 3), np.uint8), np.zeros((4, 2), np.float32))


def test_find_quad() -> None:
    img = np.zeros((400, 600, 3), np.uint8)
    quad = np.array([[80, 60], [520, 40], [540, 360], [60, 340]], np.int32)
    cv2.fillConvexPoly(img, quad, (255, 255, 255))
    found = find_quad(img)
    assert found is not None
    assert np.abs(order_quad(found) - order_quad(quad.astype(np.float32))).max() < 15
    assert find_quad(np.zeros((400, 600, 3), np.uint8)) is None


def test_decode_rejects_non_image() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"%PDF-1.4 not an image")
    with pytest.raises(InvalidImageError):
        decode_image(b"")


def test_roundtrip_unicode_path(tmp_path, sharp_image: np.ndarray) -> None:
    path = tmp_path / "中文路徑.jpg"
    path.write_bytes(encode_jpeg(sharp_image))
    assert load_image(path).shape == sharp_image.shape
