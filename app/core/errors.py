"""統一錯誤格式：所有 4xx / 5xx 都回 {"error": {"code", "message", "request_id"}}。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_request_id

log = logging.getLogger("app.errors")


class AppError(Exception):
    """業務錯誤。code 是給程式判斷用的固定字串，message 是給人看的說明。"""

    def __init__(
        self, code: str, message: str, status: int = 400, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.headers = headers


def unauthorized(message: str = "需要登入或 token 無效") -> AppError:
    return AppError("UNAUTHORIZED", message, 401, {"WWW-Authenticate": "Bearer"})


def forbidden(message: str = "權限不足") -> AppError:
    return AppError("FORBIDDEN", message, 403)


def not_found(message: str = "找不到資源") -> AppError:
    return AppError("NOT_FOUND", message, 404)


def error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "request_id": get_request_id()}}


_STATUS_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "FILE_TOO_LARGE",
    429: "RATE_LIMITED",
}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message), exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # 只回欄位位置與錯誤類型，不回輸入值：避免把使用者送的內容（可能含密碼）原樣印回去
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}" for e in exc.errors()
        )
        return JSONResponse(error_body("VALIDATION_ERROR", f"請求格式錯誤（{details}）"), 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            error_body(code, str(exc.detail)),
            exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # 內部錯誤只記在 log（含 request_id），回應不帶 stack trace 或例外訊息
        log.exception("未預期的錯誤", exc_info=exc)
        return JSONResponse(error_body("INTERNAL_ERROR", "伺服器內部錯誤"), 500)
