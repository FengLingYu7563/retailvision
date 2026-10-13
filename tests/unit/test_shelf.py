from __future__ import annotations

import numpy as np
import pytest

from vision.detector import Box
from vision.shelf import analyze_shelf, box_iou, gap_recall, group_rows


def row_of(y: float, xs: list[float], w: float = 50, h: float = 80) -> list[Box]:
    return [Box(x, y, x + w, y + h, 0.9) for x in xs]


def test_empty() -> None:
    a = analyze_shelf([], (100, 100, 3))
    assert a.item_count == 0 and a.gaps == [] and a.gap_ratio == 0


def test_full_shelf_has_no_gap() -> None:
    boxes = row_of(0, [0, 52, 104, 156, 208]) + row_of(100, [0, 52, 104, 156, 208])
    a = analyze_shelf(boxes, (300, 300, 3))
    assert a.item_count == 10 and a.row_count == 2
    assert a.gaps == []
    assert a.density > 0.9


def test_detects_middle_gap() -> None:
    # 第二層中間少了兩個商品
    boxes = row_of(0, [0, 52, 104, 156, 208, 260]) + row_of(100, [0, 52, 208, 260])
    a = analyze_shelf(boxes, (300, 400, 3))
    assert len(a.gaps) == 1
    g = a.gaps[0]
    assert g.row == 1 and g.x1 == pytest.approx(102) and g.x2 == pytest.approx(208)
    assert 0 < a.gap_ratio < 0.5


def test_detects_edge_gap() -> None:
    # 第二層右邊缺貨：貨架寬度由第一層決定
    boxes = row_of(0, [0, 52, 104, 156, 208, 260]) + row_of(100, [0, 52, 104])
    a = analyze_shelf(boxes, (300, 400, 3))
    assert len(a.gaps) == 1 and a.gaps[0].x1 == pytest.approx(154)


def test_row_with_too_few_items_is_ignored() -> None:
    # 只有 2 個框的「層」多半是誤偵測或價格牌，不拿來判斷空位
    boxes = row_of(0, [0, 52, 104, 156, 208]) + row_of(100, [0, 52])
    a = analyze_shelf(boxes, (300, 300, 3))
    assert a.row_count == 1 and a.gaps == []


def _textured_shelf(gap_textured: bool) -> tuple[np.ndarray, list[Box]]:
    """六格貨架：有商品的格子畫條紋（高邊緣密度），第 3、4 格是空隙。"""
    img = np.full((100, 312, 3), 90, np.uint8)
    xs = [0, 52, 208, 260]
    for x in xs:
        img[10:90, x : x + 50 : 6] = 230
    if gap_textured:  # 偵測漏掉的商品：空隙裡也有條紋
        img[10:90, 104:204:6] = 230
    return img, [Box(x, 10, x + 50, 90, 0.9) for x in xs]


def test_texture_check_keeps_real_gap() -> None:
    img, boxes = _textured_shelf(gap_textured=False)
    a = analyze_shelf(boxes, img.shape, image=img)
    assert len(a.gaps) == 1


def test_texture_check_rejects_missed_product() -> None:
    img, boxes = _textured_shelf(gap_textured=True)
    assert len(analyze_shelf(boxes, img.shape).gaps) == 1  # 沒有影像時無法分辨
    assert analyze_shelf(boxes, img.shape, image=img).gaps == []


def test_small_spacing_is_not_gap() -> None:
    boxes = row_of(0, [0, 70, 140, 210])  # 每個間距 20 < 0.8 * 50
    assert analyze_shelf(boxes, (100, 300, 3)).gaps == []


def test_group_rows_tolerates_jitter() -> None:
    boxes = [Box(0, 0, 50, 80, 1), Box(60, 10, 110, 90, 1), Box(0, 100, 50, 180, 1)]
    rows = group_rows(boxes)
    assert [len(r) for r in rows] == [2, 1]


def test_iou_and_recall() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    hit, total = gap_recall([(0, 0, 10, 10)], [(1, 1, 10, 10), (50, 50, 60, 60)])
    assert (hit, total) == (1, 2)
    assert gap_recall([], [(0, 0, 1, 1)]) == (0, 1)
    assert gap_recall([(0, 0, 1, 1)], []) == (0, 0)


def test_recall_is_one_to_one() -> None:
    # 一個橫跨兩個標註空位的大預測框，只能抵其中一個
    gt = [(0, 0, 10, 10), (10, 0, 20, 10)]
    assert gap_recall([(0, 0, 20, 10)], gt) == (1, 2)
    # 各自有對應的預測框時兩個都算命中
    assert gap_recall([(0, 0, 11, 10), (9, 0, 20, 10)], gt) == (2, 2)
