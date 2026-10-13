"""影像品質檢查：模糊、亮度、曝光。

品質不佳時只回傳警告，由呼叫端決定是否擋下（API 的 strict 模式）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from vision.io import Image

# 計算前先把長邊縮到固定大小：Laplacian 變異數會隨解析度改變，
# 不正規化的話同一張圖的 4K 版本和縮圖版本分數會差好幾倍，門檻就沒有意義。
ANALYSIS_LONG_SIDE = 512

# 門檻用 Grocery Store 訓練集 2,640 張校正（不看測試集）：原圖 99.8% >= 30；
# GaussianBlur(9x9, sigma=2) 後最高 26.5、(21x21, sigma=6) 後最高 3.0。
# 網路上常見的 100 是給高解析度文件照的，蔬果表面平滑，用 100 會誤殺六成原圖。
DEFAULT_BLUR_THRESHOLD = 30.0
DEFAULT_MIN_BRIGHTNESS = 0.15
DEFAULT_MAX_BRIGHTNESS = 0.90
# 超過這個比例的像素接近全黑或全白，視為曝光不足或過曝
DEFAULT_CLIP_RATIO = 0.25


@dataclass(frozen=True)
class QualityReport:
    blur_score: float
    brightness: float
    overexposed_ratio: float
    underexposed_ratio: float
    passed: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _analysis_gray(img: Image) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    # 只縮小不放大：放大時的插值本身就是低通濾波，會把小圖的分數壓低、誤判為模糊
    scale = ANALYSIS_LONG_SIDE / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return gray


def blur_score(img: Image) -> float:
    """Laplacian 變異數：邊緣越銳利值越大，模糊圖的高頻被抹掉所以值很小。"""
    gray = _analysis_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(img: Image) -> float:
    """平均亮度，0 到 1。"""
    return float(_analysis_gray(img).mean() / 255.0)


def check_quality(
    img: Image,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
    clip_ratio: float = DEFAULT_CLIP_RATIO,
) -> QualityReport:
    gray = _analysis_gray(img)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean() / 255.0)
    over = float((gray >= 250).mean())
    under = float((gray <= 5).mean())

    warnings: list[str] = []
    if score < blur_threshold:
        warnings.append("BLURRY")
    if bright < min_brightness:
        warnings.append("TOO_DARK")
    elif bright > max_brightness:
        warnings.append("TOO_BRIGHT")
    if over > clip_ratio:
        warnings.append("OVEREXPOSED")
    if under > clip_ratio:
        warnings.append("UNDEREXPOSED")

    return QualityReport(
        blur_score=round(score, 2),
        brightness=round(bright, 4),
        overexposed_ratio=round(over, 4),
        underexposed_ratio=round(under, 4),
        passed=not warnings,
        warnings=warnings,
    )
