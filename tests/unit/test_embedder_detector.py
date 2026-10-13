from __future__ import annotations

import numpy as np
import pytest

from vision.classifier import softmax
from vision.detector import Box, decode_yolo
from vision.draw import draw_shelf
from vision.embedder import EmbeddingIndex, l2_normalize
from vision.preprocess import Letterbox
from vision.shelf import Gap


def test_index_search_max_over_refs() -> None:
    idx = EmbeddingIndex(3)
    idx.add(1, np.array([[1, 0, 0], [0, 1, 0]], np.float32))
    idx.add(2, np.array([[0, 0, 1]], np.float32))
    res = idx.search(np.array([0.1, 0.9, 0], np.float32), k=2)
    assert [m.item_id for m in res] == [1, 2]
    assert res[0].similarity == pytest.approx(0.9 / np.hypot(0.1, 0.9), abs=1e-4)
    assert len(idx) == 3 and idx.item_ids == [1, 2]


def test_index_remove_and_empty() -> None:
    idx = EmbeddingIndex(2)
    assert idx.search(np.ones(2, np.float32)) == []
    idx.add(5, np.ones((1, 2), np.float32))
    idx.remove(5)
    assert len(idx) == 0
    with pytest.raises(ValueError):
        idx.add(1, np.ones((1, 3), np.float32))


def test_l2_normalize_zero_safe() -> None:
    assert np.allclose(np.linalg.norm(l2_normalize(np.ones((2, 4), np.float32)), axis=1), 1)
    assert np.all(np.isfinite(l2_normalize(np.zeros((1, 4), np.float32))))


def test_softmax() -> None:
    p = softmax(np.array([[1000.0, 1000.0, -1000.0]], np.float32))
    assert np.allclose(p, [[0.5, 0.5, 0]])


def test_decode_yolo_nms_and_unletterbox() -> None:
    # 兩個高度重疊的框（NMS 留一個）+ 一個低分框（被濾掉）+ 一個獨立框
    preds = np.array(
        [
            [100, 100, 40, 40, 0.9],
            [102, 101, 40, 40, 0.8],
            [300, 300, 40, 40, 0.1],
            [500, 200, 20, 60, 0.7],
        ],
        np.float32,
    ).T[None]
    lb = Letterbox(scale=0.5, pad_x=0, pad_y=80)
    boxes = decode_yolo(preds, lb, (960, 1280, 3), conf=0.25, iou=0.5)
    assert len(boxes) == 2
    b = max(boxes, key=lambda b: b.confidence)
    # letterbox 座標 (80, 80)-(120, 120)，扣掉 pad_y=80 再除以 scale=0.5
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((160, 0, 240, 80))
    assert decode_yolo(np.zeros((1, 5, 10), np.float32), lb, (960, 1280, 3)) == []


def test_draw_shelf(sharp_image: np.ndarray) -> None:
    out = draw_shelf(sharp_image, [Box(10, 10, 100, 100, 0.9)], [Gap(200, 10, 300, 100, 0)])
    assert out.shape == sharp_image.shape
    assert not np.array_equal(out, sharp_image)
