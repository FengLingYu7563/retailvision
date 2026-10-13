"""前處理：CLAHE、去雜訊、resize、透視校正，以及模型輸入張量。

分類模型訓練（training/）的驗證與測試流程也呼叫這裡的 to_classifier_input，
確保訓練時評估的數字和 ONNX 上線後看到的前處理完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.io import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def clahe(img: Image, clip_limit: float = 2.0, tile_grid: int = 8) -> Image:
    """對比受限的自適應直方圖等化。

    為什麼用 CLAHE 而不是一般直方圖等化：賣場燈光不均，貨架上層亮、下層暗，
    全域等化會把亮區推到過曝；CLAHE 分區處理並限制對比放大倍數，避免放大雜訊。
    只處理 LAB 的 L 通道，色相不變，後面的色彩直方圖特徵才不會被扭曲。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def denoise(img: Image, d: int = 7, sigma_color: float = 50, sigma_space: float = 50) -> Image:
    """雙邊濾波去雜訊。

    選雙邊濾波而不是高斯：它會保留商品輪廓與包裝上的文字邊緣，只抹平平坦區域的雜訊；
    NL-means 品質更好但慢一個數量級，不適合同步 API。
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def resize_long_side(img: Image, long_side: int) -> Image:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale == 1.0:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


def resize_short_side(img: Image, short_side: int) -> Image:
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    if scale == 1.0:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


def center_crop(img: Image, size: int) -> Image:
    h, w = img.shape[:2]
    top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return img[top : top + size, left : left + size]


def to_classifier_input(img: Image, size: int = 224, resize_to: int = 256) -> NDArray[np.float32]:
    """BGR 影像 -> (1, 3, size, size) float32，ImageNet 正規化。

    和 torchvision 標準評估流程相同：短邊縮到 256，再中央裁 224。
    """
    crop = center_crop(resize_short_side(img, resize_to), size)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(norm.transpose(2, 0, 1)[None], dtype=np.float32)


@dataclass(frozen=True)
class Letterbox:
    """letterbox 的縮放與位移，用來把偵測框換回原圖座標。"""

    scale: float
    pad_x: int
    pad_y: int


def letterbox(img: Image, size: int = 640, color: int = 114) -> tuple[Image, Letterbox]:
    """等比例縮放後補邊成正方形（YOLO 的輸入格式）。

    不直接拉伸成正方形：貨架照片多半是寬長形，拉伸會讓商品變形，降低偵測準確率。
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return out, Letterbox(scale=scale, pad_x=pad_x, pad_y=pad_y)


def order_quad(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """四個點排成 左上、右上、右下、左下。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def perspective_correct(img: Image, quad: NDArray[np.float32]) -> Image:
    """把斜拍的貨架四邊形拉正成矩形。

    斜拍時遠端的商品看起來比較小、間距比較窄，空位寬度和陳列密度都會被低估；
    先拉正再分析，同一排貨架上的距離才能互相比較。
    """
    src = order_quad(quad)
    tl, tr, br, bl = src
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if width < 2 or height < 2:
        raise ValueError("四邊形太小，無法校正")
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def find_quad(img: Image, min_area_ratio: float = 0.2) -> NDArray[np.float32] | None:
    """嘗試找出畫面中最大的四邊形（例如貨架外框）。找不到回傳 None，由呼叫端決定是否略過校正。"""
    work = resize_long_side(img, 800)
    ratio = img.shape[1] / work.shape[1]
    gray = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * work.shape[0] * work.shape[1]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(cnt) < min_area:
            break
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return (approx.reshape(4, 2).astype(np.float32) * ratio).astype(np.float32)
    return None
