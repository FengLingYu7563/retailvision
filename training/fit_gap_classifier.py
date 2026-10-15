"""訓練缺貨空位分類器（vision/shelf.py 的 v2 模式），**只用調參集** gap_annotations_tune.json。

流程：
1. 每張調參圖：偵測 -> 分層 -> 產生候選（空隙 + 偵測框）-> 算特徵。
   候選和任一人工標註空位 IoU >= 0.3 就標為正樣本。
2. Leave-one-image-out 交叉驗證：每次留一張圖，用其他圖訓練、預測留下的那張。
   門檻依交叉驗證的「端到端」召回率與精確率選（和正式評估同樣用一對一配對），
   不是依訓練資料上的分數選，否則會高估。
3. 用全部調參圖訓練最終模型，存成 models/gap_classifier.json（只有十幾個數字，可以 commit）。

為什麼用 logistic regression：正樣本只有幾十個，特徵只有 6 個；
參數越少越不容易對 10 張圖過擬合，係數也能直接解讀（例如越暗越像空位）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from training.common import DATA, MODELS, SKU, update_metrics
from vision.detector import OnnxDetector
from vision.io import load_image
from vision.shelf import (
    FEATURE_NAMES,
    GapCandidate,
    GapClassifier,
    RegionStats,
    analyze_shelf,
    box_iou,
    candidate_features,
    gap_candidates,
    gap_recall,
    select_gaps,
    shelf_rows,
)

TUNE = DATA / "gap_annotations_tune.json"
OUT = MODELS / "gap_classifier.json"
MIN_GAP_FACTOR = 0.4
TARGET_RECALL = 0.75  # 比驗收門檻 70% 多留一點餘裕，因為評估集和調參集不同
C = 1.0


@dataclass
class ImageData:
    name: str
    cands: list[GapCandidate]
    X: np.ndarray
    y: np.ndarray
    gt: list[tuple[float, float, float, float]]
    v1_pred: list[tuple[float, float, float, float]]


def load_images() -> list[ImageData]:
    ann = json.loads(TUNE.read_text(encoding="utf-8"))
    det = OnnxDetector(MODELS / "det-yolov8n-v1.onnx")
    out = []
    for item in ann["images"]:
        if not item.get("reviewed"):
            continue
        img = load_image(SKU / "images" / item["image"])
        boxes = det.detect(img)
        rows = shelf_rows(boxes)
        cands = gap_candidates(boxes, rows, MIN_GAP_FACTOR, include_detections=True)
        gt = [tuple(g) for g in item["gaps"]]
        X = candidate_features(cands, rows, RegionStats(img))
        y = np.array(
            [any(box_iou((c.x1, c.y1, c.x2, c.y2), g) >= 0.3 for g in gt) for c in cands],
            dtype=int,
        )
        v1 = analyze_shelf(boxes, img.shape, image=img)
        out.append(
            ImageData(
                item["image"],
                cands,
                X,
                y,
                gt,  # type: ignore[arg-type]
                [(g.x1, g.y1, g.x2, g.y2) for g in v1.gaps],
            )
        )
    return out


def fit(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X)
    # class_weight="balanced"：候選裡大部分是商品框，不平衡的話模型會傾向全部判成「不是空位」
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(scaler.transform(X), y)
    return scaler, lr


def end_to_end(
    data: list[ImageData], probs: list[np.ndarray], threshold: float
) -> tuple[float, float, int, int, int]:
    hit = total = n_pred = 0
    for d, p in zip(data, probs, strict=True):
        pred = [(c.x1, c.y1, c.x2, c.y2) for c in select_gaps(d.cands, p, threshold)]
        h, t = gap_recall(pred, d.gt)
        hit, total, n_pred = hit + h, total + t, n_pred + len(pred)
    return hit / total, (hit / n_pred if n_pred else 0.0), hit, total, n_pred


def main() -> None:
    data = load_images()
    n_pos = int(sum(d.y.sum() for d in data))
    n_gt = sum(len(d.gt) for d in data)
    print(
        f"調參集 {len(data)} 張、標註空位 {n_gt}、"
        f"候選 {sum(len(d.y) for d in data)}（正樣本 {n_pos}）"
    )
    print(
        "候選能涵蓋的空位上限（任一候選 IoU>=0.3）："
        f"{end_to_end(data, [np.ones(len(d.y)) for d in data], 0.5)[0]:.1%}"
    )

    v1_hit = sum(gap_recall(d.v1_pred, d.gt)[0] for d in data)
    v1_pred = sum(len(d.v1_pred) for d in data)
    print(f"v1 規則：召回 {v1_hit / n_gt:.1%}、精確 {v1_hit / max(v1_pred, 1):.1%}")

    # leave-one-image-out 交叉驗證
    cv_probs = []
    for i, d in enumerate(data):
        X = np.vstack([o.X for j, o in enumerate(data) if j != i])
        y = np.concatenate([o.y for j, o in enumerate(data) if j != i])
        scaler, lr = fit(X, y)
        cv_probs.append(lr.predict_proba(scaler.transform(d.X))[:, 1])

    table = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        r, p, h, tot, npred = end_to_end(data, cv_probs, float(t))
        table.append(
            {
                "threshold": float(t),
                "recall": round(r, 4),
                "precision": round(p, 4),
                "hit": h,
                "pred": npred,
            }
        )
        print(f"  門檻 {t:.2f}：CV 召回 {r:.1%}（{h}/{tot}）、精確 {p:.1%}（{h}/{npred}）")
    ok = [row for row in table if row["recall"] >= TARGET_RECALL]
    chosen = (
        max(ok, key=lambda r: (r["precision"], r["threshold"]))
        if ok
        else max(table, key=lambda r: r["recall"])
    )
    print(
        f"選定門檻 {chosen['threshold']}（CV 召回 {chosen['recall']:.1%}、"
        f"精確 {chosen['precision']:.1%}）"
    )

    scaler, lr = fit(np.vstack([d.X for d in data]), np.concatenate([d.y for d in data]))
    clf = GapClassifier(
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=lr.coef_[0].tolist(),
        intercept=float(lr.intercept_[0]),
        threshold=float(chosen["threshold"]),
        min_gap_factor=MIN_GAP_FACTOR,
    )
    clf.save(OUT)
    coef = dict(zip(FEATURE_NAMES, np.round(lr.coef_[0], 3).tolist(), strict=True))
    print(f"係數（標準化後）：{coef}")
    report: dict[str, Any] = {
        "n_images": len(data),
        "n_gt_gaps": n_gt,
        "v1_rule_recall": round(v1_hit / n_gt, 4),
        "cv_table": table,
        "chosen": chosen,
        "coef": coef,
        "C": C,
        "min_gap_factor": MIN_GAP_FACTOR,
    }
    update_metrics("gap_classifier_tuning", report)
    print(f"分類器 -> {OUT}")


if __name__ == "__main__":
    main()
