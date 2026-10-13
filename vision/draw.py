"""在影像上畫偵測框、缺貨空位與分類標籤。"""

from __future__ import annotations

from collections.abc import Sequence

import cv2

from vision.detector import Box
from vision.io import Image
from vision.shelf import Gap

BOX_COLOR = (60, 200, 60)
GAP_COLOR = (40, 40, 230)


def _thickness(img: Image) -> int:
    # 線寬隨圖片大小調整，高解析度貨架照的框才看得清楚
    return max(1, round(max(img.shape[:2]) / 600))


def draw_boxes(img: Image, boxes: Sequence[Box], color: tuple[int, int, int] = BOX_COLOR) -> Image:
    out = img.copy()
    t = _thickness(img)
    for b in boxes:
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, t)
    return out


def draw_gaps(img: Image, gaps: Sequence[Gap], alpha: float = 0.35) -> Image:
    """半透明紅色填滿空位，再加外框，和綠色商品框區分。"""
    overlay = img.copy()
    for g in gaps:
        cv2.rectangle(overlay, (int(g.x1), int(g.y1)), (int(g.x2), int(g.y2)), GAP_COLOR, -1)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    t = _thickness(img) + 1
    for g in gaps:
        cv2.rectangle(out, (int(g.x1), int(g.y1)), (int(g.x2), int(g.y2)), GAP_COLOR, t)
    return out


def draw_label(img: Image, text: str, origin: tuple[int, int] = (10, 10)) -> Image:
    """左上角畫文字標籤（白字黑底）。只用 ASCII：OpenCV 內建字型不支援中文。"""
    out = img.copy()
    scale = max(0.6, max(img.shape[:2]) / 1000)
    t = max(1, round(scale * 2))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    x, y = origin
    cv2.rectangle(out, (x, y), (x + w + 10, y + h + base + 10), (0, 0, 0), -1)
    cv2.putText(out, text, (x + 5, y + h + 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t)
    return out


def draw_shelf(img: Image, boxes: Sequence[Box], gaps: Sequence[Gap]) -> Image:
    out = draw_gaps(draw_boxes(img, boxes), gaps)
    return draw_label(out, f"items: {len(boxes)}  gaps: {len(gaps)}")
