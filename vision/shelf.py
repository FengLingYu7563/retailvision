"""貨架分析：商品數量、缺貨空位、陳列密度。

偵測模型只會框出「有商品的地方」，缺貨空位要從框之間的空隙推算：
1. 依照框的垂直中心把商品分成一層一層（貨架層）。
2. 每一層內把框依 x 排序、合併重疊區間，找出水平空隙。
3. 空隙寬度超過該層商品中位寬度的一定比例，才算缺貨（商品之間本來就有小縫）。

貨架的左右邊界取所有層的最左與最右商品，避免把畫面外側的牆面或走道當成空位。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vision.detector import Box


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


def analyze_shelf(
    boxes: list[Box],
    image_shape: tuple[int, ...],
    min_gap_factor: float = 0.8,
    row_tolerance: float = 0.5,
    min_row_items: int = 2,
) -> ShelfAnalysis:
    if not boxes:
        return ShelfAnalysis(item_count=0, row_count=0)
    rows = [r for r in group_rows(boxes, row_tolerance) if len(r) >= min_row_items]
    if not rows:
        return ShelfAnalysis(item_count=len(boxes), row_count=0)

    shelf_x1 = min(b.x1 for r in rows for b in r)
    shelf_x2 = max(b.x2 for r in rows for b in r)
    gaps: list[Gap] = []
    gap_width = row_width = covered_area = row_area = 0.0
    for idx, row in enumerate(rows):
        med_w = float(np.median([b.width for b in row]))
        # 以中位數決定這一層的上下緣，比 min/max 更不受單一異常框影響
        y1 = float(np.median([b.y1 for b in row]))
        y2 = float(np.median([b.y2 for b in row]))
        occupied = _merge_intervals([(b.x1, b.x2) for b in row])
        cursor = shelf_x1
        for a, b in [*occupied, (shelf_x2, shelf_x2)]:
            if a - cursor >= med_w * min_gap_factor:
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
