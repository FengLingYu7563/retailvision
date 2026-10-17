"""上傳檔案驗證與儲存（驗收 2.3、2.6，以及 Phase 2 調整第 6 點）。

驗證順序，每一步失敗都回固定的錯誤碼，不會讓服務崩潰：
1. 大小：讀取時最多讀 上限 + 1 bytes，超過就停，不把整個大檔讀進記憶體。
2. 副檔名白名單：只看最後一個副檔名，大小寫不拘。
3. magic bytes：用檔頭判斷實際格式，改副檔名的假圖片（例如文字檔改成 .jpg）在這裡擋下。
4. 像素數：只讀檔頭拿到寬高（不解碼），超過上限就拒絕，防止 decompression bomb。
5. 實際解碼：OpenCV 解碼失敗代表檔案損壞。

存檔用伺服器產生的 UUID 檔名，副檔名依 magic bytes 決定，完全不使用使用者提供的檔名（防路徑穿越）。
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile
from PIL import Image as PILImage

from app.core.config import Settings
from app.core.errors import AppError
from vision.io import Image, InvalidImageError, decode_image

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    image: Image
    ext: str  # 依實際內容決定，不是使用者給的副檔名


def sniff_format(data: bytes) -> str | None:
    """依檔頭判斷影像格式。"""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


async def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AppError("FILE_TOO_LARGE", f"檔案超過 {max_bytes // (1024 * 1024)} MB 上限", 413)
    return data


def validate_image_bytes(data: bytes, filename: str | None, settings: Settings) -> ValidatedImage:
    if len(data) > settings.max_upload_bytes:
        raise AppError(
            "FILE_TOO_LARGE", f"檔案超過 {settings.max_upload_bytes // (1024 * 1024)} MB 上限", 413
        )
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise AppError("UNSUPPORTED_FORMAT", "只接受 jpg、png、webp 圖片", 415)
    real = sniff_format(data)
    if real is None:
        raise AppError("INVALID_IMAGE", "檔案不是有效的圖片", 400)
    too_large = f"圖片解析度過大，上限 {settings.max_image_pixels:,} 像素"
    try:
        with PILImage.open(io.BytesIO(data)) as im:  # 只讀檔頭，不解碼像素
            width, height = im.size
    except PILImage.DecompressionBombError as exc:
        # Pillow 自己的防護：像素數超過它內建上限的兩倍時，開檔頭就直接拒絕
        raise AppError("INVALID_IMAGE", too_large, 400) from exc
    except Exception as exc:  # Pillow 對壞檔可能丟出各種例外
        raise AppError("INVALID_IMAGE", "檔案不是有效的圖片", 400) from exc
    if width * height > settings.max_image_pixels:
        raise AppError("INVALID_IMAGE", f"{too_large}（{width}x{height}）", 400)
    try:
        image = decode_image(data)
    except InvalidImageError as exc:
        raise AppError("INVALID_IMAGE", "檔案不是有效的圖片", 400) from exc
    return ValidatedImage(data=data, image=image, ext=real)


async def read_and_validate(upload: UploadFile, settings: Settings) -> ValidatedImage:
    data = await read_limited(upload, settings.max_upload_bytes)
    return validate_image_bytes(data, upload.filename, settings)


class Storage:
    """所有檔案都存在 root 底下；路徑一律由伺服器產生，讀檔時再檢查一次不會跑出 root。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(self, data: bytes, ext: str, kind: str) -> str:
        if kind not in {"uploads", "annotated", "catalog", "batch"}:
            raise ValueError(f"未知的儲存類別：{kind}")
        now = datetime.now(UTC)
        rel = Path(kind) / f"{now:%Y}" / f"{now:%m}" / f"{uuid.uuid4().hex}{ext}"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return rel.as_posix()

    def resolve(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        # 防禦性檢查：資料庫裡的路徑理論上都是 save() 產生的，但仍然不信任它
        if not path.is_relative_to(self.root):
            raise AppError("NOT_FOUND", "找不到檔案", 404)
        return path

    def read(self, rel: str) -> bytes:
        path = self.resolve(rel)
        if not path.is_file():
            raise AppError("NOT_FOUND", "找不到檔案", 404)
        return path.read_bytes()

    def delete(self, rel: str) -> None:
        path = self.resolve(rel)
        path.unlink(missing_ok=True)
