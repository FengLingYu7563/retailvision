"""影像讀寫。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

# OpenCV 的型別 stub 回傳 ndarray[Any, dtype[integer | floating]]，寫成 NDArray[uint8]
# 會讓每一個 cv2 呼叫都需要 cast；約定上 Image 一律是 HxWx3 的 BGR uint8。
Image: TypeAlias = NDArray[Any]


class InvalidImageError(ValueError):
    """位元組無法解碼成影像。"""


def decode_image(data: bytes) -> Image:
    """把上傳的位元組解碼成 BGR uint8 影像。

    用 cv2.imdecode 實際解碼一次，而不是只看副檔名：改副檔名的假圖片會在這裡失敗。
    """
    if not data:
        raise InvalidImageError("空檔案")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise InvalidImageError("檔案不是有效的圖片")
    return img


def load_image(path: str | Path) -> Image:
    """從路徑讀 BGR 影像。

    不直接用 cv2.imread：它在 Windows 上無法處理含非 ASCII 字元的路徑（本專案目錄就有中文），
    改成 np.fromfile 讀位元組再解碼。
    """
    return decode_image(np.fromfile(str(path), dtype=np.uint8).tobytes())


def encode_jpeg(img: Image, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise InvalidImageError("JPEG 編碼失敗")
    return bytes(buf.tobytes())
