"""離線評估：產出 models/metrics.json 與混淆矩陣圖。

所有分類、檢索、空位指標都用 ONNX 模型（也就是實際上線的檔案）計算，不用 PyTorch 權重，
這樣報告的數字就是 API 會得到的數字。YOLO 的 mAP 用 Ultralytics 標準的 val 流程計算。

用法：python -m training.evaluate --only cnn,retrieval,detector,gaps
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from training.common import (
    CHECKPOINTS,
    DATA,
    GROCERY,
    METRICS,
    MODELS,
    grocery_meta,
    grocery_split,
    update_metrics,
)
from vision.classifier import OnnxClassifier
from vision.detector import OnnxDetector
from vision.embedder import EmbeddingIndex
from vision.io import load_image
from vision.preprocess import to_classifier_input
from vision.shelf import analyze_shelf, gap_recall

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

CLS_VERSION = "cls-mnv3-v1"
DET_VERSION = "det-yolov8n-v1"
GAP_ANNOTATIONS = DATA / "gap_annotations.json"


def percentile_ms(samples: list[float]) -> dict[str, float]:
    arr = np.array(samples) * 1000
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def plot_confusion(cm: np.ndarray, names: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    size = max(6, n * 0.18)
    fig, ax = plt.subplots(figsize=(size, size))
    # 以列正規化（每個真實類別的召回分布），類別樣本數不平均時比原始次數好讀
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), names, rotation=90, fontsize=6 if n > 50 else 8)
    ax.set_yticks(range(n), names, fontsize=6 if n > 50 else 8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(cm: np.ndarray, names: list[str], k: int = 10) -> list[dict[str, Any]]:
    """把 (A 誤判成 B) 與 (B 誤判成 A) 合併成一組，依總次數排序。"""
    pairs: dict[tuple[int, int], int] = {}
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j]:
                key = (min(i, j), max(i, j))
                pairs[key] = pairs.get(key, 0) + int(cm[i, j])
    ranked = sorted(pairs.items(), key=lambda t: -t[1])[:k]
    return [
        {"a": names[a], "b": names[b], "count": c, "a_as_b": int(cm[a, b]), "b_as_a": int(cm[b, a])}
        for (a, b), c in ranked
    ]


def eval_cnn() -> None:
    meta = grocery_meta()
    clf = OnnxClassifier(MODELS / f"{CLS_VERSION}.onnx")
    test = grocery_split("test")
    yf = np.array([s.fine_id for s in test])
    yc = np.array([s.coarse_id for s in test])
    fine_logits, coarse_logits = [], []
    for i in range(0, len(test), 64):
        batch = np.concatenate([to_classifier_input(load_image(s.path)) for s in test[i : i + 64]])
        f, c, _ = clf.logits(batch)
        fine_logits.append(f)
        coarse_logits.append(c)
    F_, C_ = np.concatenate(fine_logits), np.concatenate(coarse_logits)
    top5 = np.argsort(-F_, axis=1)[:, :5]
    pf, pc = top5[:, 0], C_.argmax(1)
    f2c = np.array(meta["fine_to_coarse"])

    # CPU 延遲：單張、含前處理（解碼後），模擬 API 實際情境
    imgs = [load_image(s.path) for s in test[:50]]
    for img in imgs[:5]:
        clf.predict(img)
    times = []
    for _ in range(4):
        for img in imgs:
            t = time.perf_counter()
            clf.predict(img)
            times.append(time.perf_counter() - t)

    metrics: dict[str, Any] = {
        "model_version": CLS_VERSION,
        "fine_top1": round(float((pf == yf).mean()), 4),
        "fine_top5": round(float((top5 == yf[:, None]).any(1).mean()), 4),
        "coarse_top1": round(float((pc == yc).mean()), 4),
        "coarse_top1_from_fine": round(float((f2c[pf] == yc).mean()), 4),
        "latency_ms_cpu_onnx": percentile_ms(times),
        "n_test": len(test),
    }
    metrics["latency_ms_gpu_torch"] = gpu_latency()

    n_f, n_c = len(meta["fine_names"]), len(meta["coarse_names"])
    cm_f = np.zeros((n_f, n_f), np.int64)
    np.add.at(cm_f, (yf, pf), 1)
    cm_c = np.zeros((n_c, n_c), np.int64)
    np.add.at(cm_c, (yc, pc), 1)
    plot_confusion(cm_f, meta["fine_names"], MODELS / "confusion_fine.png", "Fine classes (test)")
    plot_confusion(
        cm_c, meta["coarse_names"], MODELS / "confusion_coarse.png", "Coarse classes (test)"
    )
    metrics["top_confusions_fine"] = top_confusions(cm_f, meta["fine_names"])

    svm = json.loads(METRICS.read_text(encoding="utf-8")).get("svm") if METRICS.exists() else None
    if svm:
        metrics["beats_svm"] = {
            "fine_top1": metrics["fine_top1"] > svm["fine_top1"],
            "coarse_top1": metrics["coarse_top1"] > svm["coarse_top1"],
        }
    metrics["acceptance_1_4"] = bool(
        metrics["coarse_top1"] >= 0.85
        and metrics["fine_top1"] >= 0.75
        and metrics["fine_top5"] >= 0.95
        and svm is not None
        and all(metrics["beats_svm"].values())
    )
    update_metrics("cnn", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def gpu_latency() -> dict[str, float] | None:
    try:
        import torch

        from training.export_onnx import load_classifier
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    model = load_classifier(CHECKPOINTS / "classifier_mnv3" / "best.pt").cuda()
    x = torch.randn(1, 3, 224, 224, device="cuda")
    times = []
    with torch.no_grad():
        for i in range(120):
            torch.cuda.synchronize()
            t = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            if i >= 20:
                times.append(time.perf_counter() - t)
    return percentile_ms(times)


def tile(img: np.ndarray, n: int) -> np.ndarray:
    """把單一商品縮小後排成 n x n。"""
    small = cv2.resize(img, (img.shape[1] // n, img.shape[0] // n), interpolation=cv2.INTER_AREA)
    return np.vstack([np.hstack([small] * n)] * n)


def eval_retrieval() -> None:
    meta = grocery_meta()
    clf = OnnxClassifier(MODELS / f"{CLS_VERSION}.onnx")
    test = grocery_split("test")
    q = np.concatenate(
        [
            clf.logits(
                np.concatenate([to_classifier_input(load_image(s.path)) for s in test[i : i + 64]])
            )[2]
            for i in range(0, len(test), 64)
        ]
    )
    y = np.array([s.fine_id for s in test])

    def accuracy(index: EmbeddingIndex) -> dict[str, float]:
        res = index.search_batch(q, k=5)
        top1 = np.mean([r[0].item_id == t for r, t in zip(res, y, strict=True)])
        top5 = np.mean([t in {m.item_id for m in r} for r, t in zip(res, y, strict=True)])
        return {"top1": round(float(top1), 4), "top5": round(float(top5), 4)}

    # gallery A（驗收 1.7）：每類 1 張官方標準圖 + 翻轉 + 排成 2x2 / 3x3。
    # 排列是為了模擬賣場裡「一整堆」的外觀；這個組合是在 val 上比較 4 種做法後選的
    # （實驗紀錄見 README），test 只跑這一次。
    iconic = EmbeddingIndex(q.shape[1])
    for fid, rel in enumerate(meta["iconic_images"]):
        img = load_image(GROCERY / rel)
        flipped = cv2.flip(img, 1)
        views = [img, flipped, tile(img, 2), tile(img, 3), tile(flipped, 2)]
        iconic.add(fid, np.stack([clf.embed(v) for v in views]))
    # gallery B（參考用）：模擬新品登錄流程，每類只拿 3 張訓練集實拍照當參考圖
    few_shot = EmbeddingIndex(q.shape[1])
    by_class: dict[int, list[Path]] = {}
    for s in grocery_split("train"):
        by_class.setdefault(s.fine_id, []).append(s.path)
    for fid, paths in by_class.items():
        few_shot.add(fid, np.stack([clf.embed(load_image(p)) for p in paths[:3]]))

    metrics = {
        "gallery_iconic": accuracy(iconic),
        "gallery_train_3shot": accuracy(few_shot),
        "n_queries": len(test),
    }
    metrics["acceptance_1_7"] = metrics["gallery_iconic"]["top1"] >= 0.70
    update_metrics("retrieval", metrics)
    print(json.dumps(metrics, indent=2))


def eval_detector() -> None:
    from ultralytics import YOLO

    yolo_yaml = DATA / "processed" / "sku110k_yolo" / "sku110k.yaml"
    weights = CHECKPOINTS / "detector_v8n" / "weights" / "best.pt"
    res = YOLO(str(weights)).val(
        data=str(yolo_yaml),
        split="test",
        imgsz=640,
        batch=8,
        max_det=1000,
        workers=2,
        plots=False,
        verbose=False,
        project=str(CHECKPOINTS),
        name="detector_v8n_test",
        exist_ok=True,
    )
    metrics: dict[str, Any] = {
        "model_version": DET_VERSION,
        "mAP50": round(float(res.box.map50), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
        "n_test": len((DATA / "splits" / "sku110k_test.txt").read_text().split()),
    }
    metrics["acceptance_1_5"] = metrics["mAP50"] >= 0.75

    det = OnnxDetector(MODELS / f"{DET_VERSION}.onnx")
    names = (DATA / "splits" / "sku110k_test.txt").read_text().split()[:20]
    imgs = [load_image(DATA / "raw" / "SKU110K_fixed" / "images" / n) for n in names]
    det.detect(imgs[0])
    times = []
    for img in imgs:
        t = time.perf_counter()
        analyze_shelf(det.detect(img), img.shape)
        times.append(time.perf_counter() - t)
    metrics["latency_ms_cpu_onnx_detect_and_analyze"] = percentile_ms(times)
    update_metrics("detector", metrics)
    print(json.dumps(metrics, indent=2))


def eval_gaps() -> None:
    if not GAP_ANNOTATIONS.exists():
        print("尚無空位人工標註，略過（驗收 1.8 記為待人工標註）")
        update_metrics("gaps", {"status": "pending_annotation"})
        return
    ann = json.loads(GAP_ANNOTATIONS.read_text(encoding="utf-8"))
    det = OnnxDetector(MODELS / f"{DET_VERSION}.onnx")
    hit = total = n_pred = 0
    per_image = []
    for item in ann["images"]:
        if not item.get("reviewed"):
            continue
        img = load_image(DATA / "raw" / "SKU110K_fixed" / "images" / item["image"])
        analysis = analyze_shelf(det.detect(img), img.shape)
        pred = [(g.x1, g.y1, g.x2, g.y2) for g in analysis.gaps]
        gt = [tuple(g) for g in item["gaps"]]
        h, t = gap_recall(pred, gt)  # type: ignore[arg-type]
        hit, total, n_pred = hit + h, total + t, n_pred + len(pred)
        per_image.append({"image": item["image"], "hit": h, "gt": t, "pred": len(pred)})
    # 驗收要求「至少 10 張人工標註空位的貨架圖」：只算確實標了空位的圖，
    # 已確認但沒有空位的圖另外列出（它們仍會貢獻誤報，但不能拿來湊張數）
    n_images = sum(1 for p in per_image if p["gt"] > 0)
    metrics = {
        "n_images": n_images,
        "n_reviewed_images": len(per_image),
        "gt_gaps": total,
        "predicted_gaps": n_pred,
        "recall": round(hit / total, 4) if total else None,
        "iou_threshold": 0.3,
        "per_image": per_image,
        "acceptance_1_8": bool(n_images >= 10 and total and hit / total >= 0.70),
    }
    update_metrics("gaps", metrics)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="cnn,retrieval,detector,gaps")
    args = ap.parse_args()
    steps = {
        "cnn": eval_cnn,
        "retrieval": eval_retrieval,
        "detector": eval_detector,
        "gaps": eval_gaps,
    }
    for name in args.only.split(","):
        steps[name.strip()]()


if __name__ == "__main__":
    main()
