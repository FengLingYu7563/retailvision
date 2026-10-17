from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import DB, AppSettings, error_responses
from app.core.errors import unauthorized
from app.core.ratelimit import rate_limit_by_ip
from app.core.security import authenticate, create_access_token
from app.schemas import TokenOut

router = APIRouter(tags=["auth"])


@router.post(
    "/auth/token",
    response_model=TokenOut,
    responses=error_responses(401, 429, auth=False),
    dependencies=[Depends(rate_limit_by_ip)],
)
def issue_token(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DB, settings: AppSettings
) -> TokenOut:
    """帳號密碼換 JWT access token。帳號不存在與密碼錯誤回同一個訊息，不洩漏帳號是否存在。"""
    user = authenticate(db, form.username, form.password)
    if user is None:
        raise unauthorized("帳號或密碼錯誤")
    token, expires = create_access_token(user, settings)
    return TokenOut(access_token=token, expires_in=expires)
