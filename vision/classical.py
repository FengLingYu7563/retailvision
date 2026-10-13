"""傳統機器學習 baseline：HSV 色彩直方圖 + HOG 特徵，SVM 分類。

用途是當作 CNN 的對照組：如果 CNN 沒有明顯贏過這個 baseline，就代表深度模型沒學到東西，
或是資料本身靠顏色就能分。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from numpy.typing import NDArray
from skimage.feature import hog

from vision.io import Image
from vision.preprocess import center_crop, resize_short_side

FEATURE_SIZE = 128
# H 16、S 8、V 4 個 bin 的聯合直方圖：色相切細一點，因為蔬果主要靠顏色區分；
# 亮度切粗，降低賣場燈光變化的影響。
HSV_BINS = (16, 8, 4)


def color_histogram(img: Image) -> NDArray[np.float32]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HSV_BINS), [0, 180, 0, 256, 0, 256])
    flat = np.asarray(hist, dtype=np.float32).ravel()
    flat = flat / max(float(flat.sum()), 1.0)
    # 開根號（Hellinger 轉換）：直方圖是機率分布，直接用歐氏距離會被少數大 bin 主導，
    # 開根號後 RBF/線性 SVM 的距離比較接近分布間的差異。
    return np.sqrt(flat).astype(np.float32)


def hog_features(img: Image) -> NDArray[np.float32]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(feat, dtype=np.float32)


def extract_features(img: Image) -> NDArray[np.float32]:
    """和 CNN 一樣先短邊縮放再中央裁切，兩邊模型看到的畫面範圍相同，比較才公平。"""
    patch = center_crop(resize_short_side(img, FEATURE_SIZE), FEATURE_SIZE)
    return np.concatenate([color_histogram(patch), hog_features(patch)])


@dataclass
class ClassicalPrediction:
    fine_id: int
    coarse_id: int


class ClassicalClassifier:
    """包裝兩個 sklearn pipeline（細類、粗類），用 joblib 存成單一檔案。"""

    def __init__(self, fine_model: Any, coarse_model: Any, meta: dict[str, Any]) -> None:
        self.fine_model = fine_model
        self.coarse_model = coarse_model
        self.meta = meta

    def save(self, path: str | Path) -> None:
        joblib.dump({"fine": self.fine_model, "coarse": self.coarse_model, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> ClassicalClassifier:
        obj = joblib.load(path)
        return cls(obj["fine"], obj["coarse"], obj["meta"])

    def predict_features(
        self, X: NDArray[np.float32]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        return self.fine_model.predict(X), self.coarse_model.predict(X)

    def predict(self, img: Image) -> ClassicalPrediction:
        fine, coarse = self.predict_features(extract_features(img)[None])
        return ClassicalPrediction(fine_id=int(fine[0]), coarse_id=int(coarse[0]))
