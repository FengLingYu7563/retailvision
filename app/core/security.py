"""認證與授權：argon2 密碼雜湊、JWT access token、角色檢查。"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import forbidden, unauthorized
from app.db.models import User
from app.db.session import get_db

# argon2id（argon2-cffi 預設參數）：贏得 Password Hashing Competition 的演算法，
# 記憶體成本高，GPU 暴力破解比 bcrypt 更貴（驗收 2.5）
_hasher = PasswordHasher()
# 帳號不存在時也做一次雜湊比對，回應時間才不會洩漏「這個帳號存在」
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing")
# JWT sub：只接受 ASCII 數字、不可有前導 0（str.isdecimal 也會接受全形與其他文字的數字）
_USER_ID = re.compile(r"[1-9][0-9]{0,9}")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def authenticate(db: Session, username: str, password: str) -> User | None:
    user = db.query(User).filter(User.username == username).one_or_none()
    if user is None:
        verify_password(password, _DUMMY_HASH)
        return None
    return user if verify_password(password, user.password_hash) else None


def create_access_token(user: User, settings: Settings) -> tuple[str, int]:
    expires = settings.jwt_expire_minutes * 60
    now = datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "iat": now,
        "exp": now + timedelta(seconds=expires),
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm), expires


_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    if creds is None or creds.scheme.lower() != "bearer":
        raise unauthorized()
    try:
        # algorithms 明確指定，拒絕 alg=none 或換成其他演算法的偽造 token
        payload = jwt.decode(
            creds.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise unauthorized("token 已過期") from exc
    except jwt.PyJWTError as exc:
        raise unauthorized("token 無效") from exc
    if payload.get("type") != "access":
        raise unauthorized("token 類型錯誤")
    # 簽章正確但 claim 格式不對也是無效 token，回 401 而不是 500。
    # 先檢查型別與格式再轉換：int() 會把 JSON 的 1.5、true 都轉成 1，
    # 舊版 PyJWT 不強制 sub 是字串，這類 token 就會被當成使用者 1。
    sub = payload.get("sub")
    if not isinstance(sub, str) or not _USER_ID.fullmatch(sub):
        raise unauthorized("token 無效")
    user_id = int(sub)
    # 超出 PostgreSQL integer 範圍的 id 直接拒絕，否則查詢會丟資料庫錯誤（500）
    if user_id > 2**31 - 1:
        raise unauthorized("token 無效")
    # 角色以資料庫為準，不信任 token 裡的 role：使用者被降級後，舊 token 不能繼續當 admin 用
    user = db.get(User, user_id)
    if user is None:
        raise unauthorized("使用者不存在")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    if user.role != "admin":
        raise forbidden("需要 admin 權限")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
