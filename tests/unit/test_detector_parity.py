"""自寫的 ONNX YOLO 解碼（vision/detector.py）要和 Ultralytics 官方推論結果一致。

參考值是 Ultralytics 載入「同一個 ONNX 檔」的推論結果。
（若用 .pt 當參考，Ultralytics 會改用矩形 letterbox，模型輸入就不同了，無法分辨是誰的問題。）
Ultralytics predict 縮圖用 INTER_LINEAR，這裡也指定 INTER_LINEAR，讓模型輸入完全相同，
只比較解碼、座標還原與 NMS：要求框的數量相同，而且每個框都能找到 IoU >= 0.99 的對應框。

這個測試抓到過一個真實錯誤：把 max_det 傳給 cv2.dnn.NMSBoxes 的 top_k，
密集貨架上會在 NMS 之前就丟掉候選框，一張圖少了 10% 以上的商品。
"""

from __future__ import annotations

import cv2
import pytest

from tests.conftest import MODELS_DIR, ROOT
from vision.detector import OnnxDetector
from vision.io import load_image
from vision.shelf import box_iou

pytestmark = [pytest.mark.needs_weights, pytest.mark.needs_data]

SKU_IMAGES = ROOT / "training" / "data" / "raw" / "SKU110K_fixed" / "images"


def test_onnx_detector_matches_ultralytics() -> None:
    onnx_path = MODELS_DIR / "det-yolov8n-v1.onnx"
    if not onnx_path.exists():
        pytest.skip("尚未匯出偵測模型")
    from ultralytics import YOLO

    yolo = YOLO(str(onnx_path), task="detect")
    det = OnnxDetector(onnx_path)
    names = (ROOT / "training" / "data" / "splits" / "sku110k_test.txt").read_text().split()[:5]
    for name in names:
        img = load_image(SKU_IMAGES / name)
        ref = yolo.predict(img, imgsz=640, conf=0.25, iou=0.6, max_det=1000, verbose=False)[0]
        ref_boxes = [tuple(map(float, b)) for b in ref.boxes.xyxy.cpu().numpy()]
        ours = [
            (b.x1, b.y1, b.x2, b.y2)
            for b in det.detect(img, conf=0.25, iou=0.6, interpolation=cv2.INTER_LINEAR)
        ]
        assert len(ours) == len(ref_boxes), name
        for r in ref_boxes:
            assert max(box_iou(r, o) for o in ours) >= 0.99, f"{name}: {r}"  # type: ignore[arg-type]
