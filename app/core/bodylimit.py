"""請求大小限制（ASGI middleware），在 multipart 解析之前就生效（驗收 2.3，Codex 第 2 輪 L2）。

為什麼不能只靠 endpoint 裡的 read_limited：FastAPI 在呼叫 endpoint 之前，就已經把整個 multipart
請求讀完、寫進暫存檔；超大的上傳在 413 之前就已經消耗了頻寬與硬碟。

這裡在最底層攔截：
- 有 Content-Length 且超過上限：直接回 413，一個 byte 都不讀。
- 沒有 Content-Length（chunked）或 Content-Length 造假：一邊接收一邊累計，超過就中斷並回 413。

上限依路徑決定（單張上傳、批次總量、其他 JSON / 表單）。正式部署時反向代理（nginx 的
client_max_body_size）也要設定同樣的上限，做為第二道防線。
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from app.core.config import Settings
from app.core.errors import error_body

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# multipart 的邊界字串、欄位標頭等額外開銷
MULTIPART_SLACK = 1024 * 1024
DEFAULT_LIMIT = 1024 * 1024


class BodyTooLargeError(Exception):
    pass


def limit_for(path: str, settings: Settings) -> int:
    single = settings.max_upload_bytes + MULTIPART_SLACK
    if path.endswith("/recognitions/batch"):
        return settings.max_batch_bytes
    if path.endswith("/catalog/products"):
        return 5 * settings.max_upload_bytes + MULTIPART_SLACK
    if path.endswith(("/recognitions/product", "/recognitions/shelf", "/search/similar")):
        return single
    return DEFAULT_LIMIT


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = limit_for(scope["path"], self.settings)
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > limit
            except ValueError:
                too_big = False  # 格式錯誤的 Content-Length 交給下面的串流計數處理
            if too_big:
                await _reject(send, limit)
                return

        received = 0
        exceeded = False
        rejected = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise BodyTooLargeError
            return message

        async def guarded_send(message: Message) -> None:
            # 超過上限後，app 不管怎麼處理那個例外（FastAPI 會把 multipart 解析失敗包成 400），
            # 它要送出的回應一律換成 413，保證 client 看到的是正確的錯誤碼
            nonlocal rejected
            if exceeded:
                if not rejected:
                    rejected = True
                    await _reject(send, limit)
                return
            await send(message)

        with contextlib.suppress(BodyTooLargeError):
            await self.app(scope, counting_receive, guarded_send)
        if exceeded and not rejected:
            await _reject(send, limit)


async def _reject(send: Send, limit: int) -> None:
    body = json.dumps(
        error_body("FILE_TOO_LARGE", f"請求超過 {limit // (1024 * 1024)} MB 上限"),
        ensure_ascii=False,
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                # 不再讀取剩下的請求內容，告訴 client 這條連線不再使用
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
