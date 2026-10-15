"""把訓練好的模型匯出成 ONNX，並寫出旁邊的 .json 中繼資料。

為什麼轉 ONNX：API 與 worker 映像檔只裝 onnxruntime（約 50 MB），不用裝 PyTorch（GB 級）；
CPU 推論也有圖最佳化，延遲更穩定。匯出後由 tests/unit/test_onnx_parity.py 檢查數值一致。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from training.common import CHECKPOINTS, MODELS, grocery_meta
from training.models import build_model

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

OPSET = 17


def load_classifier(ckpt: Path) -> torch.nn.Module:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    # v1 的 checkpoint 沒有記錄 arch，預設是 MobileNetV3
    model = build_model(
        state.get("arch", "mnv3"), state["n_fine"], state["n_coarse"], pretrained=False
    )
    model.load_state_dict(state["model"])
    return model.eval()


def export_classifier(ckpt: Path, version: str) -> Path:
    model = load_classifier(ckpt)
    out = MODELS / f"{version}.onnx"
    MODELS.mkdir(exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["image"],
        output_names=["fine_logits", "coarse_logits", "embedding"],
        # 批次維度設成動態：API 單張推論用 1，評估與批次任務可以一次送多張
        dynamic_axes={
            "image": {0: "batch"},
            "fine_logits": {0: "batch"},
            "coarse_logits": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    meta = grocery_meta()
    (out.with_suffix(".json")).write_text(
        json.dumps(
            {
                "version": version,
                "task": "classification",
                "input_size": 224,
                "fine_names": meta["fine_names"],
                "coarse_names": meta["coarse_names"],
                "fine_to_coarse": meta["fine_to_coarse"],
                "source_checkpoint": ckpt.resolve().relative_to(CHECKPOINTS.parent).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"分類模型 -> {out}")
    return out


def export_detector(weights: Path, version: str, imgsz: int) -> Path:
    from ultralytics import YOLO

    exported = Path(
        YOLO(str(weights)).export(
            format="onnx", imgsz=imgsz, opset=OPSET, simplify=False, dynamic=False
        )
    )
    out = MODELS / f"{version}.onnx"
    shutil.copy2(exported, out)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": version,
                "task": "detection",
                "imgsz": imgsz,
                "source_checkpoint": weights.resolve().relative_to(CHECKPOINTS.parent).as_posix(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"偵測模型 -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classifier", "detector", "all"], default="all")
    ap.add_argument("--cls-ckpt", default=str(CHECKPOINTS / "classifier_mnv3" / "best.pt"))
    ap.add_argument("--cls-version", default="cls-mnv3-v1")
    ap.add_argument(
        "--det-weights", default=str(CHECKPOINTS / "detector_v8n" / "weights" / "best.pt")
    )
    ap.add_argument("--det-version", default="det-yolov8n-v1")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    if args.task in ("classifier", "all"):
        export_classifier(Path(args.cls_ckpt), args.cls_version)
    if args.task in ("detector", "all"):
        export_detector(Path(args.det_weights), args.det_version, args.imgsz)


if __name__ == "__main__":
    main()
