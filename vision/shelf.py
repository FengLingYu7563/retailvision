"""貨架分析：商品數量、缺貨空位、陳列密度。

偵測模型只會框出「有商品的地方」，缺貨空位要從框之間的空隙推算：
1. 依照框的垂直中心把商品分成一層一層（貨架層）。
2. 每一層內把框依 x 排序、合併重疊區間，找出水平空隙。
3. 空隙寬度超過該層商品中位寬度的一定比例，才算缺貨（商品之間本來就有小縫）。

4. （有傳入影像時）紋理驗證：比較候選空位與同層商品框的邊緣密度。
   偵測模型漏掉的商品，空隙裡仍有包裝的輪廓與文字，邊緣密度和商品差不多；
   真正缺貨的地方通常只剩貨架背板，邊緣很少。不做這一步的話，偵測漏掉的商品全部會被當成空位。

貨架的左右邊界取所有層的最左與最右商品，避免把畫面外側的牆面或走道當成空位。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.detector import Box
from vision.io import Image

EDGE_LONG_SIDE = 1024


@dataclass(frozen=True)
class Gap:
    x1: float
    y1: float
    x2: float
    y2: float
    row: int

    def to_dict(self) -> dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "row": self.row,
        }


@dataclass(frozen=True)
class ShelfAnalysis:
    item_count: int
    row_count: int
    gaps: list[Gap] = field(default_factory=list)
    gap_ratio: float = 0.0
    density: float = 0.0


def group_rows(boxes: list[Box], row_tolerance: float = 0.5) -> list[list[Box]]:
    """依垂直中心分層。

    兩個框的中心高度差小於「中位框高 x row_tolerance」就視為同一層。
    用中位數而不是平均：少數特別高的瓶子或誤偵測的大框不會把門檻拉歪。
    """
    if not boxes:
        return []
    med_h = float(np.median([b.height for b in boxes]))
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if rows and abs(b.cy - float(np.mean([r.cy for r in rows[-1]]))) <= med_h * row_tolerance:
            rows[-1].append(b)
        else:
            rows.append([b])
    return rows


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


class EdgeDensity:
    """用積分影像在 O(1) 時間算任意矩形內的邊緣像素比例。"""

    def __init__(self, img: Image) -> None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        self.scale = min(1.0, EDGE_LONG_SIDE / max(gray.shape[:2]))
        if self.scale < 1.0:
            gray = cv2.resize(
                gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
            )
        edges = (cv2.Canny(gray, 50, 150) > 0).astype(np.float64)
        self.integral: NDArray[np.float64] = np.asarray(cv2.integral(edges), dtype=np.float64)

    def __call__(self, x1: float, y1: float, x2: float, y2: float) -> float:
        h, w = self.integral.shape[0] - 1, self.integral.shape[1] - 1
        a = int(np.clip(round(x1 * self.scale), 0, w))
        b = int(np.clip(round(y1 * self.scale), 0, h))
        c = int(np.clip(round(x2 * self.scale), 0, w))
        d = int(np.clip(round(y2 * self.scale), 0, h))
        area = (c - a) * (d - b)
        if area <= 0:
            return 0.0
        ii = self.integral
        return float(ii[d, c] - ii[b, c] - ii[d, a] + ii[b, a]) / area


def analyze_shelf(
    boxes: list[Box],
    image_shape: tuple[int, ...],
    image: Image | None = None,
    min_gap_factor: float = 0.8,
    row_tolerance: float = 0.5,
    min_row_items: int = 3,
    texture_ratio: float = 0.5,
) -> ShelfAnalysis:
    """texture_ratio：候選空位的邊緣密度低於「同層商品中位密度 x texture_ratio」才算空位。"""
    if not boxes:
        return ShelfAnalysis(item_count=0, row_count=0)
    rows = [r for r in group_rows(boxes, row_tolerance) if len(r) >= min_row_items]
    if not rows:
        return ShelfAnalysis(item_count=len(boxes), row_count=0)

    edge_density = EdgeDensity(image) if image is not None else None
    shelf_x1 = min(b.x1 for r in rows for b in r)
    shelf_x2 = max(b.x2 for r in rows for b in r)
    gaps: list[Gap] = []
    gap_width = row_width = covered_area = row_area = 0.0
    for idx, row in enumerate(rows):
        med_w = float(np.median([b.width for b in row]))
        # 以中位數決定這一層的上下緣，比 min/max 更不受單一異常框影響
        y1 = float(np.median([b.y1 for b in row]))
        y2 = float(np.median([b.y2 for b in row]))
        ref = (
            float(np.median([edge_density(b.x1, b.y1, b.x2, b.y2) for b in row]))
            if edge_density
            else 0.0
        )
        occupied = _merge_intervals([(b.x1, b.x2) for b in row])
        cursor = shelf_x1
        for a, b in [*occupied, (shelf_x2, shelf_x2)]:
            if a - cursor >= med_w * min_gap_factor and not (
                edge_density and ref > 0 and edge_density(cursor, y1, a, y2) >= ref * texture_ratio
            ):
                gaps.append(Gap(cursor, y1, a, y2, idx))
                gap_width += a - cursor
            cursor = max(cursor, b)
        row_width += shelf_x2 - shelf_x1
        covered_area += sum((b - a) for a, b in occupied) * (y2 - y1)
        row_area += (shelf_x2 - shelf_x1) * (y2 - y1)

    return ShelfAnalysis(
        item_count=len(boxes),
        row_count=len(rows),
        gaps=gaps,
        gap_ratio=round(gap_width / row_width, 4) if row_width > 0 else 0.0,
        # 密度：各層商品實際佔用的面積比例（0 到 1）
        density=round(covered_area / row_area, 4) if row_area > 0 else 0.0,
    )


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gap_recall(
    predicted: list[tuple[float, float, float, float]],
    ground_truth: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.3,
) -> tuple[int, int]:
    """回傳 (命中數, 標註總數)。空位邊界本身很模糊，所以 IoU 門檻設 0.3 而不是 0.5。

    一對一配對：所有 (標註, 預測) 組合依 IoU 由高到低貪婪配對，每個預測框最多抵一個標註。
    否則一個橫跨整層的大預測框會同時「命中」好幾個標註空位，高估召回率。
    """
    pairs = sorted(
        (
            (box_iou(g, p), gi, pi)
            for gi, g in enumerate(ground_truth)
            for pi, p in enumerate(predicted)
        ),
        reverse=True,
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for iou, gi, pi in pairs:
        if iou < iou_threshold:
            break
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
    return len(used_gt), len(ground_truth)
