"""結構化日誌（JSON）與 request_id。

每個請求都有一個 request_id：從 X-Request-ID 帶入（限長度與字元），沒有就產生 UUID。
放在 contextvar 裡，log 與錯誤回應都拿得到（驗收 3.8）。
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
# 外部帶入的 request_id 只接受安全字元：避免 log injection（換行、控制字元）
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def get_request_id() -> str:
    return _request_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": get_request_id(),
        }
        for key in ("method", "path", "status", "duration_ms", "user_id"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


access_log = logging.getLogger("app.access")


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    incoming = request.headers.get("x-request-id", "")
    rid = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
    token = _request_id.set(rid)
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            # 未預期的例外在這裡就轉成 500：FastAPI 的通用 Exception handler
            # 在 middleware 外層執行，那時 request_id 已經不在 context 裡
            logging.getLogger("app.errors").exception("未預期的錯誤")
            from app.core.errors import error_body

            response = JSONResponse(error_body("INTERNAL_ERROR", "伺服器內部錯誤"), 500)
        response.headers["X-Request-ID"] = rid
        access_log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response
    finally:
        _request_id.reset(token)
