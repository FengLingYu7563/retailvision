"""ONNX YOLOv8 貨架商品偵測（單一類別「商品」）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.io import Image
from vision.onnx_utils import create_session
from vision.preprocess import Letterbox, letterbox

# SKU-110K 一張貨架圖可能有 700 個商品，Ultralytics 預設上限 300 會直接截掉
DEFAULT_MAX_DET = 1000


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in asdict(self).items()}


def decode_yolo(
    output: NDArray[np.float32],
    lb: Letterbox,
    image_shape: tuple[int, ...],
    conf: float = 0.25,
    iou: float = 0.6,
    max_det: int = DEFAULT_MAX_DET,
) -> list[Box]:
    """解析 YOLOv8 輸出 (1, 4 + n_cls, N)，做 NMS 並換回原圖座標。"""
    pred = output[0].T  # (N, 4 + n_cls)
    scores = pred[:, 4:].max(axis=1)
    keep = scores >= conf
    pred, scores = pred[keep], scores[keep]
    if len(pred) == 0:
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    # 不能把 max_det 傳給 NMSBoxes 的 top_k：OpenCV 的 top_k 是在 NMS「之前」只保留前 k 個候選框，
    # 密集貨架的候選框有好幾千個（同一商品有多個重疊框），會讓後面的商品在 NMS 前就被丟掉。
    # 先完整做 NMS（結果已依分數排序），再截取前 max_det 個。
    kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    idx = np.asarray(kept, dtype=np.int64).ravel()[:max_det]
    H, W = image_shape[:2]
    boxes: list[Box] = []
    for i in idx:
        x, y, bw, bh = xywh[i]
        x1 = float(np.clip((x - lb.pad_x) / lb.scale, 0, W))
        y1 = float(np.clip((y - lb.pad_y) / lb.scale, 0, H))
        x2 = float(np.clip((x + bw - lb.pad_x) / lb.scale, 0, W))
        y2 = float(np.clip((y + bh - lb.pad_y) / lb.scale, 0, H))
        if x2 > x1 and y2 > y1:
            boxes.append(Box(x1, y1, x2, y2, round(float(scores[i]), 4)))
    return boxes


class OnnxDetector:
    def __init__(self, onnx_path: str | Path, threads: int | None = None) -> None:
        onnx_path = Path(onnx_path)
        meta_path = onnx_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.version: str = meta.get("version", onnx_path.stem)
        self.imgsz: int = int(meta.get("imgsz", 640))
        self.session = create_session(onnx_path, threads)
        self.input_name = self.session.get_inputs()[0].name

    def detect(
        self,
        img: Image,
        conf: float = 0.25,
        iou: float = 0.6,
        max_det: int = DEFAULT_MAX_DET,
        interpolation: int | None = None,
    ) -> list[Box]:
        padded, lb = letterbox(img, self.imgsz, interpolation=interpolation)
        x = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        (out,) = self.session.run(None, {self.input_name: x[None]})
        return decode_yolo(out, lb, img.shape, conf, iou, max_det)
