"""驗收 2.3、2.6 與調整第 6 點：上傳檔案驗證、UUID 檔名、decompression bomb。"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import Recognition
from tests.api.conftest import RED, STORAGE, assert_error, auth, jpeg, textured, upload

PRODUCT = "/api/v1/recognitions/product"


def test_text_file_renamed_to_jpg_rejected(client: TestClient, user_token: str) -> None:
    r = client.post(
        PRODUCT, files=upload(b"just some text, not an image", "fake.jpg"), headers=auth(user_token)
    )
    assert_error(r, 400, "INVALID_IMAGE")


def test_pdf_renamed_to_png_rejected(client: TestClient, user_token: str) -> None:
    r = client.post(
        PRODUCT, files=upload(b"%PDF-1.7\n...", "doc.png", "image/png"), headers=auth(user_token)
    )
    assert_error(r, 400, "INVALID_IMAGE")


def test_truncated_jpeg_rejected(client: TestClient, user_token: str) -> None:
    data = jpeg(textured(RED))[:200]  # 檔頭正確但內容截斷
    assert_error(
        client.post(PRODUCT, files=upload(data), headers=auth(user_token)), 400, "INVALID_IMAGE"
    )


def test_unsupported_extension_415(client: TestClient, user_token: str) -> None:
    for name in ("a.gif", "a.exe", "noext", "a.jpg.php"):
        r = client.post(PRODUCT, files=upload(jpeg(textured(RED)), name), headers=auth(user_token))
        assert_error(r, 415, "UNSUPPORTED_FORMAT")


def test_too_large_413(client: TestClient, user_token: str) -> None:
    big = b"\xff\xd8\xff" + b"\0" * (10 * 1024 * 1024)
    assert_error(
        client.post(PRODUCT, files=upload(big), headers=auth(user_token)), 413, "FILE_TOO_LARGE"
    )


def _png_header_only(width: int, height: int) -> bytes:
    """檔頭宣稱超大解析度的 PNG：只有幾十 bytes，解碼的話要配置數 GB 記憶體。"""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0" * 64))


def test_decompression_bomb_rejected_without_decoding(client: TestClient, user_token: str) -> None:
    r = client.post(
        PRODUCT,
        files=upload(_png_header_only(50_000, 50_000), "bomb.png", "image/png"),
        headers=auth(user_token),
    )
    assert_error(r, 400, "INVALID_IMAGE")
    assert "解析度過大" in r.json()["error"]["message"]


def test_over_our_pixel_limit_but_under_pillow_limit(client: TestClient, user_token: str) -> None:
    """8100 萬像素：Pillow 內建防護不會擋（上限約 1.8 億），要靠我們自己的 4000 萬上限。"""
    r = client.post(
        PRODUCT,
        files=upload(_png_header_only(9000, 9000), "big.png", "image/png"),
        headers=auth(user_token),
    )
    assert_error(r, 400, "INVALID_IMAGE")
    assert "9000x9000" in r.json()["error"]["message"]


def test_service_still_healthy_after_bad_uploads(client: TestClient, user_token: str) -> None:
    for data in (b"", b"\x00" * 10, b"\xff\xd8\xff garbage"):
        r = client.post(PRODUCT, files=upload(data), headers=auth(user_token))
        assert 400 <= r.status_code < 500
    r = client.post(PRODUCT, files=upload(jpeg(textured(RED))), headers=auth(user_token))
    assert r.status_code == 201


def test_server_generated_uuid_filename(client: TestClient, user_token: str, db: Session) -> None:
    """使用者提供的檔名（含路徑穿越）完全不會被用到。"""
    evil = "../../../../etc/passwd.jpg"
    r = client.post(PRODUCT, files=upload(jpeg(textured(RED)), evil), headers=auth(user_token))
    assert r.status_code == 201
    rec = db.get(Recognition, r.json()["id"])
    assert rec is not None
    for rel in (rec.image_path, rec.annotated_path):
        assert rel is not None
        assert ".." not in rel and "passwd" not in rel
        name = Path(rel).stem
        assert len(name) == 32 and all(c in "0123456789abcdef" for c in name)
        assert (STORAGE / rel).resolve().is_relative_to(STORAGE.resolve())
        assert (STORAGE / rel).exists()
