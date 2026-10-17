"""請求大小在 multipart 解析之前就被限制（驗收 2.3，Codex 第 2 輪 L2）。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient

from app.core.bodylimit import BodySizeLimitMiddleware, limit_for
from app.core.config import get_settings
from tests.api.conftest import RED, assert_error, auth, jpeg, textured

MB = 1024 * 1024


def _run(middleware: BodySizeLimitMiddleware, scope: dict[str, Any], chunks: list[bytes]):  # type: ignore[no-untyped-def]
    sent: list[dict[str, Any]] = []
    queue = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]

    async def receive() -> dict[str, Any]:
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def test_declared_content_length_rejected_before_reading() -> None:
    """Content-Length 超過上限：endpoint 完全不會被呼叫，請求內容一個 byte 都不讀。"""
    called = {"app": False}

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        called["app"] = True

    mw = BodySizeLimitMiddleware(app, get_settings())
    scope = {
        "type": "http",
        "path": "/api/v1/recognitions/product",
        "headers": [(b"content-length", str(500 * MB).encode())],
    }
    sent = _run(mw, scope, [b"x"])
    assert not called["app"]
    assert sent[0]["status"] == 413


def test_streaming_body_cut_off_when_exceeding() -> None:
    """沒有 Content-Length（chunked）：累計超過上限就中斷，endpoint 讀不到後面的資料。"""
    consumed = {"bytes": 0}

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        while True:
            msg = await receive()
            consumed["bytes"] += len(msg.get("body", b""))
            if not msg.get("more_body"):
                break

    mw = BodySizeLimitMiddleware(app, get_settings())
    limit = limit_for("/api/v1/recognitions/product", get_settings())
    scope = {"type": "http", "path": "/api/v1/recognitions/product", "headers": []}
    sent = _run(mw, scope, [b"\0" * MB] * 20)
    assert sent[0]["status"] == 413
    assert consumed["bytes"] <= limit


def test_chunked_upload_over_limit_413(client: TestClient, user_token: str) -> None:
    def body() -> Iterator[bytes]:
        for _ in range(15):
            yield b"\0" * MB

    r = client.post(
        "/api/v1/recognitions/product",
        content=body(),
        headers={**auth(user_token), "Content-Type": "multipart/form-data; boundary=x"},
    )
    assert_error(r, 413, "FILE_TOO_LARGE")


def test_batch_total_size_limit(user_token: str, registry) -> None:  # type: ignore[no-untyped-def]
    from app.main import create_app

    settings = get_settings().model_copy(update={"max_batch_bytes": 2 * MB})
    with TestClient(create_app(registry=registry, settings=settings)) as c:
        files = [
            ("files", (f"{i}.jpg", jpeg(textured(RED, (600, 800))) + b"\0" * MB, "image/jpeg"))
            for i in range(3)
        ]
        r = c.post("/api/v1/recognitions/batch", files=files, headers=auth(user_token))
    assert_error(r, 413, "FILE_TOO_LARGE")


def test_json_endpoints_have_small_limit(client: TestClient, admin_token: str) -> None:
    r = client.put(
        "/api/v1/models/active",
        content=b'{"name": "' + b"a" * (2 * MB) + b'"}',
        headers={**auth(admin_token), "Content-Type": "application/json"},
    )
    assert_error(r, 413, "FILE_TOO_LARGE")
