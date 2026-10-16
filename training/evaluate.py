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
from vision.shelf import GapClassifier, analyze_shelf, gap_recall

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# v2：DINOv2-small + SupCon（驗收 1.7 重做後選定，比較紀錄見 docs/experiments/retrieval_1_7.md）
CLS_VERSION = "cls-dinov2-v2"
CLS_CKPT = CHECKPOINTS / "classifier_dino_metric" / "best.pt"
HOLDOUT_CKPT = CHECKPOINTS / "metric_dino_holdout20" / "best.pt"
DET_VERSION = "det-yolov8n-v1"
GAP_ANNOTATIONS = DATA / "gap_annotations.json"
GAP_CLASSIFIER = MODELS / "gap_classifier.json"


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
    model = load_classifier(CLS_CKPT).cuda()
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

    # gallery A（驗收 1.7 的正式值）：嚴格定義，每類只有 1 張官方標準圖、1 個向量
    iconic = EmbeddingIndex(q.shape[1])
    # gallery A'（實驗，不作為驗收值）：同一張標準圖加翻轉、排成 2x2 / 3x3，
    # 模擬賣場裡「一整堆」的外觀。這個組合是在 val 上比較 4 種做法後選的
    # （docs/experiments/retrieval_1_7.md），test 只跑這一次。
    iconic_aug = EmbeddingIndex(q.shape[1])
    for fid, rel in enumerate(meta["iconic_images"]):
        img = load_image(GROCERY / rel)
        flipped = cv2.flip(img, 1)
        views = [img, flipped, tile(img, 2), tile(img, 3), tile(flipped, 2)]
        embs = np.stack([clf.embed(v) for v in views])
        iconic.add(fid, embs[:1])
        iconic_aug.add(fid, embs)
    # gallery B（參考用）：模擬新品登錄流程，每類只拿 3 張訓練集實拍照當參考圖
    few_shot = EmbeddingIndex(q.shape[1])
    by_class: dict[int, list[Path]] = {}
    for s in grocery_split("train"):
        by_class.setdefault(s.fine_id, []).append(s.path)
    for fid, paths in by_class.items():
        few_shot.add(fid, np.stack([clf.embed(load_image(p)) for p in paths[:3]]))

    metrics = {
        "gallery_iconic": accuracy(iconic),
        "gallery_iconic_augmented": accuracy(iconic_aug),
        "gallery_train_3shot": accuracy(few_shot),
        "n_queries": len(test),
    }
    metrics["acceptance_1_7"] = metrics["gallery_iconic"]["top1"] >= 0.70
    update_metrics("retrieval", metrics)
    print(json.dumps(metrics, indent=2))


def eval_holdout() -> None:
    """新商品泛化：用「訓練時完全沒看過 20 個類別」的模型，在 test 上量這 20 類的檢索 top-1。

    gallery 是全部 81 類各 1 張標準圖（嚴格定義），所以沒看過的類別還要和看過的類別競爭，
    比只在 20 類之間搜尋更接近實際的新品登錄情境。這是實驗模型，不是上線模型。
    """
    import torch

    from training.export_onnx import load_classifier

    state = torch.load(HOLDOUT_CKPT, map_location="cpu", weights_only=False)
    holdout = state["holdout"]
    model = load_classifier(HOLDOUT_CKPT)
    meta = grocery_meta()
    test = grocery_split("test")

    @torch.no_grad()
    def embed(imgs: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), 64):
            x = np.concatenate([to_classifier_input(im) for im in imgs[i : i + 64]])
            out.append(model(torch.from_numpy(x))[2].numpy())
        return np.concatenate(out)

    q = embed([load_image(s.path) for s in test])
    g = embed([load_image(GROCERY / rel) for rel in meta["iconic_images"]])
    y = np.array([s.fine_id for s in test])
    sims = q @ g.T
    top1 = sims.argmax(1)
    top5 = np.argsort(-sims, axis=1)[:, :5]
    ho = np.isin(y, holdout)

    def acc(mask: np.ndarray) -> dict[str, float]:
        return {
            "top1": round(float((top1[mask] == y[mask]).mean()), 4),
            "top5": round(float((top5[mask] == y[mask, None]).any(1).mean()), 4),
            "n": int(mask.sum()),
        }

    metrics = {
        "holdout_classes": [meta["fine_names"][i] for i in holdout],
        "unseen_classes": acc(ho),
        "seen_classes": acc(~ho),
        "note": "實驗模型（20 類完全不參與訓練），gallery 為 81 類各 1 張標準圖",
    }
    update_metrics("retrieval_holdout", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


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
    # 再用匯出的 ONNX 檔（上線實際使用的模型）算一次 mAP，確認匯出沒有損失準確率。
    # ONNX 是固定輸入大小，Ultralytics 會自動改用 batch=1、正方形 letterbox。
    onnx_res = YOLO(str(MODELS / f"{DET_VERSION}.onnx"), task="detect").val(
        data=str(yolo_yaml),
        split="test",
        imgsz=640,
        batch=1,
        max_det=1000,
        workers=2,
        plots=False,
        verbose=False,
        project=str(CHECKPOINTS),
        name="detector_v8n_test_onnx",
        exist_ok=True,
    )
    metrics["onnx_mAP50"] = round(float(onnx_res.box.map50), 4)
    metrics["onnx_mAP50_95"] = round(float(onnx_res.box.map), 4)
    metrics["acceptance_1_5"] = min(metrics["mAP50"], metrics["onnx_mAP50"]) >= 0.75
    metrics["latency_ms_cpu_onnx_detect_and_analyze"] = shelf_latency()
    update_metrics("detector", metrics)
    print(json.dumps(metrics, indent=2))


def shelf_latency() -> dict[str, float]:
    """偵測 + 貨架分析（含空位分類器，也就是上線時的完整流程）的 CPU 延遲。"""
    det = OnnxDetector(MODELS / f"{DET_VERSION}.onnx")
    clf = GapClassifier.load(GAP_CLASSIFIER) if GAP_CLASSIFIER.exists() else None
    names = (DATA / "splits" / "sku110k_test.txt").read_text().split()[:20]
    imgs = [load_image(DATA / "raw" / "SKU110K_fixed" / "images" / n) for n in names]
    analyze_shelf(det.detect(imgs[0]), imgs[0].shape, image=imgs[0], classifier=clf)
    times = []
    for img in imgs:
        t = time.perf_counter()
        analyze_shelf(det.detect(img), img.shape, image=img, classifier=clf)
        times.append(time.perf_counter() - t)
    return percentile_ms(times)


def eval_shelf_latency() -> None:
    """只更新 detector 區塊的延遲欄位，不重跑 mAP。"""
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    section = data.get("detector", {})
    section["latency_ms_cpu_onnx_detect_and_analyze"] = shelf_latency()
    update_metrics("detector", section)
    print(json.dumps(section["latency_ms_cpu_onnx_detect_and_analyze"]))


def eval_gaps() -> None:
    if not GAP_ANNOTATIONS.exists():
        print("尚無空位人工標註，略過（驗收 1.8 記為待人工標註）")
        update_metrics("gaps", {"status": "pending_annotation"})
        return
    ann = json.loads(GAP_ANNOTATIONS.read_text(encoding="utf-8"))
    det = OnnxDetector(MODELS / f"{DET_VERSION}.onnx")
    # 驗收值用 v2 分類器（只用調參集訓練與選門檻）；v1 規則一起算出來當對照
    clf = GapClassifier.load(GAP_CLASSIFIER)
    hit = total = n_pred = 0
    v1_hit = v1_pred = 0
    per_image = []
    for item in ann["images"]:
        if not item.get("reviewed"):
            continue
        img = load_image(DATA / "raw" / "SKU110K_fixed" / "images" / item["image"])
        boxes = det.detect(img)
        analysis = analyze_shelf(boxes, img.shape, image=img, classifier=clf)
        pred = [(g.x1, g.y1, g.x2, g.y2) for g in analysis.gaps]
        gt = [tuple(g) for g in item["gaps"]]
        h, t = gap_recall(pred, gt)  # type: ignore[arg-type]
        hit, total, n_pred = hit + h, total + t, n_pred + len(pred)
        v1 = [(g.x1, g.y1, g.x2, g.y2) for g in analyze_shelf(boxes, img.shape, image=img).gaps]
        v1_hit += gap_recall(v1, gt)[0]  # type: ignore[arg-type]
        v1_pred += len(v1)
        per_image.append({"image": item["image"], "hit": h, "gt": t, "pred": len(pred)})
    # 驗收要求「至少 10 張人工標註空位的貨架圖」：只算確實標了空位的圖，
    # 已確認但沒有空位的圖另外列出（它們仍會貢獻誤報，但不能拿來湊張數）
    n_images = sum(1 for p in per_image if p["gt"] > 0)
    if not per_image:
        print("候選圖都還沒有人工確認，略過（驗收 1.8 記為待人工標註）")
        update_metrics("gaps", {"status": "pending_annotation"})
        return
    metrics = {
        "n_images": n_images,
        "n_reviewed_images": len(per_image),
        "gt_gaps": total,
        "predicted_gaps": n_pred,
        "recall": round(hit / total, 4) if total else None,
        "precision": round(hit / n_pred, 4) if n_pred else None,
        "method": f"v2 classifier ({clf.kind}, threshold {clf.threshold})",
        "v1_rule": {
            "recall": round(v1_hit / total, 4) if total else None,
            "precision": round(v1_hit / v1_pred, 4) if v1_pred else None,
        },
        "iou_threshold": 0.3,
        "per_image": per_image,
        "acceptance_1_8": bool(n_images >= 10 and total and hit / total >= 0.70),
    }
    update_metrics("gaps", metrics)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="cnn,retrieval,holdout,detector,gaps")
    args = ap.parse_args()
    steps = {
        "cnn": eval_cnn,
        "retrieval": eval_retrieval,
        "holdout": eval_holdout,
        "detector": eval_detector,
        "shelf_latency": eval_shelf_latency,
        "gaps": eval_gaps,
    }
    for name in args.only.split(","):
        steps[name.strip()]()


if __name__ == "__main__":
    main()
