"""貨架分析：商品數量、缺貨空位、陳列密度。

偵測模型只會框出「有商品的地方」，缺貨空位要從框之間的空隙推算：
1. 依照框的垂直中心把商品分成一層一層（貨架層）。
2. 每一層內把框依 x 排序、合併重疊區間，找出水平空隙，當成空位候選。
3. 判斷候選是不是真的空位，有兩種模式：
   - 規則（v1，沒有傳入分類器時）：空隙寬度 >= 中位商品寬 x min_gap_factor，
     而且邊緣密度低於同層商品的 texture_ratio 倍。
   - 分類器（v2，傳入 GapClassifier 時）：候選放寬到 0.4 個商品寬，並把**偵測框本身**也當候選
     （偵測器會把空格裡的暗色背板誤框成商品）。對每個候選算出「和同層商品比較」的外觀特徵
     （亮度、飽和度、邊緣密度、寬度、偵測信心等），用梯度提升樹判斷。
     分類器只用調參集訓練，見 training/fit_gap_classifier.py。

為什麼特徵都用「和同層商品的比值」：不同照片的曝光、白平衡差很多，
絕對亮度沒有意義；同一層的商品就是最好的參考值。

貨架的左右邊界取所有層的最左與最右商品，避免把畫面外側的牆面或走道當成空位。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from numpy.typing import NDArray

from vision.detector import Box
from vision.io import Image

STATS_LONG_SIDE = 1024
FEATURE_NAMES = [
    "brightness_ratio",
    "saturation_ratio",
    "edge_ratio",
    "brightness_abs",
    "width_ratio",
    "from_detection",
    # 亮度標準差：空位多半是一片均勻的背板，深色商品上有印刷與反光，標準差較大
    "brightness_std_ratio",
    "edge_abs",
    "aspect_ratio",
    # 偵測器的信心分數（空隙候選為 0，由 from_detection 區分）：空格被誤框時信心通常較低
    "det_confidence",
    # 碰到貨架左右邊界：邊界外常是側板、走道，不是貨架空位
    "at_shelf_edge",
    "log_width_ratio",
]


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


@dataclass(frozen=True)
class GapCandidate:
    x1: float
    y1: float
    x2: float
    y2: float
    row: int
    box_index: int | None  # 來自偵測框時是該框在 boxes 中的索引；來自空隙時是 None
    confidence: float = 0.0
    at_edge: bool = False


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


class RegionStats:
    """用積分影像在 O(1) 時間算任意矩形內的平均亮度、飽和度、邊緣像素比例。"""

    def __init__(self, img: Image) -> None:
        self.scale = min(1.0, STATS_LONG_SIDE / max(img.shape[:2]))
        if self.scale < 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = (cv2.Canny(gray, 50, 150) > 0).astype(np.float64)
        v = hsv[:, :, 2] / 255.0
        self._ii: list[NDArray[np.float64]] = [
            np.asarray(cv2.integral(ch.astype(np.float64)), dtype=np.float64)
            for ch in (v, hsv[:, :, 1] / 255.0, edges, v * v)
        ]

    def mean(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        """回傳 (亮度, 飽和度, 邊緣比例, 亮度標準差)，都在 0 到 1。"""
        ii = self._ii[0]
        h, w = ii.shape[0] - 1, ii.shape[1] - 1
        a = int(np.clip(round(x1 * self.scale), 0, w))
        b = int(np.clip(round(y1 * self.scale), 0, h))
        c = int(np.clip(round(x2 * self.scale), 0, w))
        d = int(np.clip(round(y2 * self.scale), 0, h))
        area = (c - a) * (d - b)
        if area <= 0:
            return 0.0, 0.0, 0.0, 0.0
        v, s, e, v2 = (float(m[d, c] - m[b, c] - m[d, a] + m[b, a]) / area for m in self._ii)
        return v, s, e, float(np.sqrt(max(v2 - v * v, 0.0)))

    def edge(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return self.mean(x1, y1, x2, y2)[2]


@dataclass(frozen=True)
class GapClassifier:
    """sklearn 分類器（有 predict_proba 即可）+ 門檻，由 training/fit_gap_classifier.py 產生。

    存成兩個檔：`.joblib` 是模型本身（和其他權重一樣不 commit），
    `.json` 是門檻、候選設定與特徵名稱（可以 commit，方便追蹤上線用的是哪組設定）。
    改用梯度提升樹的原因：40 張調參集的交叉驗證中，召回 70% 時精確率 37%，高於 LR 的 32%。
    """

    estimator: Any
    threshold: float
    min_gap_factor: float = 0.4
    kind: str = "gbt"

    def proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(features) == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self.estimator.predict_proba(features)[:, 1], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        joblib.dump(self.estimator, path.with_suffix(".joblib"))
        meta = {
            "kind": self.kind,
            "threshold": self.threshold,
            "min_gap_factor": self.min_gap_factor,
            "feature_names": FEATURE_NAMES,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GapClassifier:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("feature_names") != FEATURE_NAMES:
            raise ValueError("分類器的特徵定義和程式不一致，需要重新訓練")
        return cls(
            estimator=joblib.load(path.with_suffix(".joblib")),
            threshold=float(meta["threshold"]),
            min_gap_factor=float(meta["min_gap_factor"]),
            kind=str(meta["kind"]),
        )


def shelf_rows(
    boxes: list[Box], row_tolerance: float = 0.5, min_row_items: int = 3
) -> list[list[Box]]:
    return [r for r in group_rows(boxes, row_tolerance) if len(r) >= min_row_items]


def gap_candidates(
    boxes: list[Box], rows: list[list[Box]], min_gap_factor: float, include_detections: bool
) -> list[GapCandidate]:
    """每一層內框與框之間（以及到貨架左右邊界）的空隙；可選擇把偵測框本身也列為候選。"""
    if not rows:
        return []
    index = {id(b): i for i, b in enumerate(boxes)}
    shelf_x1 = min(b.x1 for r in rows for b in r)
    shelf_x2 = max(b.x2 for r in rows for b in r)
    out: list[GapCandidate] = []
    for idx, row in enumerate(rows):
        med_w = float(np.median([b.width for b in row]))
        # 以中位數決定這一層的上下緣，比 min/max 更不受單一異常框影響
        y1 = float(np.median([b.y1 for b in row]))
        y2 = float(np.median([b.y2 for b in row]))
        cursor = shelf_x1
        for a, b in [*_merge_intervals([(b.x1, b.x2) for b in row]), (shelf_x2, shelf_x2)]:
            if a - cursor >= med_w * min_gap_factor:
                edge = cursor <= shelf_x1 + 1 or a >= shelf_x2 - 1
                out.append(GapCandidate(cursor, y1, a, y2, idx, None, at_edge=edge))
            cursor = max(cursor, b)
        if include_detections:
            out += [
                GapCandidate(
                    b.x1,
                    b.y1,
                    b.x2,
                    b.y2,
                    idx,
                    index[id(b)],
                    b.confidence,
                    at_edge=b.x1 <= shelf_x1 + 1 or b.x2 >= shelf_x2 - 1,
                )
                for b in row
            ]
    return out


def candidate_features(
    cands: list[GapCandidate], rows: list[list[Box]], stats: RegionStats
) -> NDArray[np.float64]:
    """每個候選和「同層商品中位數」比較的特徵，欄位順序見 FEATURE_NAMES。"""
    refs = []
    for row in rows:
        vals = np.array([stats.mean(b.x1, b.y1, b.x2, b.y2) for b in row])
        aspect = float(np.median([b.height / max(b.width, 1e-3) for b in row]))
        refs.append((np.median(vals, axis=0), float(np.median([b.width for b in row])), aspect))
    eps = 1e-3
    feats = []
    for c in cands:
        (rv, rs, re, rsd), med_w, med_aspect = refs[c.row]
        v, s, e, sd = stats.mean(c.x1, c.y1, c.x2, c.y2)
        feats.append(
            [
                v / (rv + eps),
                s / (rs + eps),
                e / (re + eps),
                v,
                (c.x2 - c.x1) / max(med_w, eps),
                float(c.box_index is not None),
                sd / (rsd + eps),
                e,
                ((c.y2 - c.y1) / max(c.x2 - c.x1, eps)) / max(med_aspect, eps),
                c.confidence,
                float(c.at_edge),
                float(np.log(max((c.x2 - c.x1) / max(med_w, eps), eps))),
            ]
        )
    return np.array(feats, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))


def select_gaps(
    cands: list[GapCandidate], probs: NDArray[np.float64], threshold: float, nms_iou: float = 0.3
) -> list[GapCandidate]:
    """機率 >= 門檻的候選，再依機率由高到低做 NMS。

    同一個空位常同時出現「空隙候選」和「偵測框候選」兩個重疊的框，不去重的話一個空位會回報兩次。
    """
    order = [i for i in np.argsort(-probs) if probs[i] >= threshold]
    kept: list[GapCandidate] = []
    for i in order:
        c = cands[i]
        box = (c.x1, c.y1, c.x2, c.y2)
        if all(box_iou(box, (k.x1, k.y1, k.x2, k.y2)) < nms_iou for k in kept):
            kept.append(c)
    return kept


def analyze_shelf(
    boxes: list[Box],
    image_shape: tuple[int, ...],
    image: Image | None = None,
    min_gap_factor: float = 0.8,
    row_tolerance: float = 0.5,
    min_row_items: int = 3,
    texture_ratio: float = 0.5,
    classifier: GapClassifier | None = None,
) -> ShelfAnalysis:
    """有 image 與 classifier 時用分類器（v2）；只有 image 時用紋理規則（v1）；
    兩者都沒有時只看空隙寬度。

    texture_ratio（v1）：候選空位的邊緣密度低於「同層商品中位密度 x texture_ratio」才算空位。
    """
    if not boxes:
        return ShelfAnalysis(item_count=0, row_count=0)
    rows = shelf_rows(boxes, row_tolerance, min_row_items)
    if not rows:
        return ShelfAnalysis(item_count=len(boxes), row_count=0)

    stats = RegionStats(image) if image is not None else None
    if classifier is not None and stats is not None:
        cands = gap_candidates(boxes, rows, classifier.min_gap_factor, include_detections=True)
        probs = classifier.proba(candidate_features(cands, rows, stats))
        chosen = select_gaps(cands, probs, classifier.threshold)
    else:
        cands = gap_candidates(boxes, rows, min_gap_factor, include_detections=False)
        chosen = []
        for c in cands:
            if stats is not None:
                row = rows[c.row]
                ref = float(np.median([stats.edge(b.x1, b.y1, b.x2, b.y2) for b in row]))
                if ref > 0 and stats.edge(c.x1, c.y1, c.x2, c.y2) >= ref * texture_ratio:
                    continue
            chosen.append(c)

    # 被判成空位的偵測框不算商品
    gap_boxes = {c.box_index for c in chosen if c.box_index is not None}
    gaps = [Gap(c.x1, c.y1, c.x2, c.y2, c.row) for c in chosen]
    shelf_x1 = min(b.x1 for r in rows for b in r)
    shelf_x2 = max(b.x2 for r in rows for b in r)
    index = {id(b): i for i, b in enumerate(boxes)}
    gap_width = row_width = covered_area = row_area = 0.0
    for idx, row in enumerate(rows):
        y1 = float(np.median([b.y1 for b in row]))
        y2 = float(np.median([b.y2 for b in row]))
        items = [b for b in row if index[id(b)] not in gap_boxes]
        occupied = _merge_intervals([(b.x1, b.x2) for b in items])
        row_gaps = _merge_intervals([(g.x1, g.x2) for g in gaps if g.row == idx])
        gap_width += sum(b - a for a, b in row_gaps)
        row_width += shelf_x2 - shelf_x1
        covered_area += sum(b - a for a, b in occupied) * (y2 - y1)
        row_area += (shelf_x2 - shelf_x1) * (y2 - y1)

    return ShelfAnalysis(
        item_count=len(boxes) - len(gap_boxes),
        row_count=len(rows),
        gaps=gaps,
        gap_ratio=round(min(1.0, gap_width / row_width), 4) if row_width > 0 else 0.0,
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
