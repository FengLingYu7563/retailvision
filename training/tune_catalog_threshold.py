"""選 catalog_match 的相似度門檻（Phase 2 調整第 5 點），**只用 Grocery val**。

模擬情境：固定 seed 挑一半的細類登錄進 catalog（每類 3 張 train 實拍當參考圖），另一半沒登錄。
對每張 val 照片：
- 該類有登錄：最相似的商品是它、而且相似度 >= 門檻，才算對。
- 該類沒登錄：相似度 < 門檻（回傳 null）才算對；硬配一個別的商品算錯。
選讓整體正確率最高的門檻，寫入 models/metrics.json 的 catalog_threshold。
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from training.common import MODELS, SEED, grocery_split, update_metrics
from vision.classifier import OnnxClassifier
from vision.embedder import EmbeddingIndex
from vision.io import load_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cls-dinov2-v2")
    ap.add_argument("--refs", type=int, default=3)
    args = ap.parse_args()

    clf = OnnxClassifier(MODELS / f"{args.model}.onnx")
    train, val = grocery_split("train"), grocery_split("val")
    classes = sorted({s.fine_id for s in train})
    registered = set(random.Random(SEED).sample(classes, len(classes) // 2))

    index = EmbeddingIndex(1280)
    by_class: dict[int, list] = {}
    for s in train:
        by_class.setdefault(s.fine_id, []).append(s)
    for c in sorted(registered):
        refs = by_class[c][: args.refs]
        index.add(c, np.stack([clf.embed(load_image(s.path)) for s in refs]))

    tops = []
    for s in val:
        m = index.search(clf.embed(load_image(s.path)), k=1)[0]
        tops.append((s.fine_id, m.item_id, m.similarity))

    rows = []
    for t in np.round(np.arange(0.30, 0.96, 0.025), 3):
        correct = matched = matched_right = 0
        for true, pred, sim in tops:
            hit = sim >= t
            if true in registered:
                correct += hit and pred == true
            else:
                correct += not hit
            matched += hit
            matched_right += hit and pred == true
        rows.append(
            {
                "threshold": float(t),
                "accuracy": round(correct / len(tops), 4),
                "match_precision": round(matched_right / matched, 4) if matched else None,
                "match_rate": round(matched / len(tops), 4),
            }
        )
    best = max(rows, key=lambda r: (r["accuracy"], -r["threshold"]))
    for r in rows:
        mark = "  <-- 選定" if r is best else ""
        print(
            f"門檻 {r['threshold']:.3f}：正確率 {r['accuracy']:.1%}、"
            f"有配對時的精確率 {r['match_precision']}、配對比例 {r['match_rate']:.1%}{mark}"
        )
    update_metrics(
        "catalog_threshold",
        {
            "model_version": args.model,
            "split": "val",
            "registered_classes": len(registered),
            "refs_per_class": args.refs,
            "n_queries": len(tops),
            "chosen": best,
            "table": rows,
        },
    )


if __name__ == "__main__":
    main()
