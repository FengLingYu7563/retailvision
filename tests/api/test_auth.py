"""驗收 2.4、2.5、2.11：認證、授權、密碼雜湊、rate limit。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import User
from tests.api.conftest import assert_error, auth


def test_token_ok(client: TestClient, user_token: str) -> None:
    r = client.get("/api/v1/models", headers=auth(user_token))
    assert r.status_code == 200


def test_wrong_password_and_unknown_user_same_error(client: TestClient, user_token: str) -> None:
    for name, pw in (("alice", "wrong-password-xx"), ("nobody", "whatever-password")):
        r = client.post("/api/v1/auth/token", data={"username": name, "password": pw})
        assert_error(r, 401, "UNAUTHORIZED")
        # 不洩漏帳號是否存在
        assert r.json()["error"]["message"] == "帳號或密碼錯誤"


def test_no_token_401(client: TestClient) -> None:
    for method, path in (
        ("get", "/api/v1/recognitions"),
        ("get", "/api/v1/models"),
        ("get", "/api/v1/stats/summary"),
        ("post", "/api/v1/recognitions/product"),
        ("put", "/api/v1/models/active"),
    ):
        r = getattr(client, method)(path)
        assert_error(r, 401, "UNAUTHORIZED")


def _forge(sub: str, *, exp_delta: timedelta, secret: str | None = None, alg: str = "HS256") -> str:
    now = datetime.now(UTC)
    payload = {"sub": sub, "role": "admin", "iat": now, "exp": now + exp_delta, "type": "access"}
    return jwt.encode(payload, secret or get_settings().jwt_secret, algorithm=alg)


def test_expired_token_401(client: TestClient, user_token: str, db: Session) -> None:
    uid = str(db.query(User.id).filter(User.username == "alice").scalar())
    expired = _forge(uid, exp_delta=timedelta(seconds=-10))
    r = client.get("/api/v1/models", headers=auth(expired))
    assert_error(r, 401, "UNAUTHORIZED")
    assert "過期" in r.json()["error"]["message"]


def test_wrong_secret_and_alg_none_rejected(client: TestClient, user_token: str) -> None:
    bad = _forge("1", exp_delta=timedelta(minutes=5), secret="x" * 40)
    assert_error(client.get("/api/v1/models", headers=auth(bad)), 401, "UNAUTHORIZED")
    none = jwt.encode({"sub": "1", "exp": 9999999999, "type": "access"}, None, algorithm="none")
    assert_error(client.get("/api/v1/models", headers=auth(none)), 401, "UNAUTHORIZED")
    assert_error(client.get("/api/v1/models", headers=auth("not-a-jwt")), 401, "UNAUTHORIZED")


def test_role_comes_from_db_not_token(client: TestClient, user_token: str, db: Session) -> None:
    """token 裡寫 role=admin 也沒用：角色以資料庫為準。"""
    uid = str(db.query(User.id).filter(User.username == "alice").scalar())
    forged = _forge(uid, exp_delta=timedelta(minutes=5))  # payload 寫 admin，但簽章是真的
    r = client.put("/api/v1/models/active", json={"name": "cls-fake-b"}, headers=auth(forged))
    assert_error(r, 403, "FORBIDDEN")


def test_user_calling_admin_endpoint_403(client: TestClient, user_token: str) -> None:
    r = client.put("/api/v1/models/active", json={"name": "cls-fake-b"}, headers=auth(user_token))
    assert_error(r, 403, "FORBIDDEN")
    r = client.delete("/api/v1/catalog/products/1", headers=auth(user_token))
    assert_error(r, 403, "FORBIDDEN")


def test_password_hashed_with_argon2(user_token: str, db: Session) -> None:
    user = db.query(User).filter(User.username == "alice").one()
    assert user.password_hash.startswith("$argon2id$")
    assert "correct-horse-battery" not in user.password_hash


def test_rate_limit_429(client: TestClient, user_token: str) -> None:
    settings = get_settings().model_copy(update={"rate_limit_per_minute": 3})
    client.app.dependency_overrides[get_settings] = lambda: settings  # type: ignore[attr-defined]
    try:
        codes = [
            client.get("/api/v1/models", headers=auth(user_token)).status_code for _ in range(5)
        ]
    finally:
        client.app.dependency_overrides.clear()  # type: ignore[attr-defined]
    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]
    r = client.get("/api/v1/models", headers=auth(user_token))
    assert r.status_code == 200  # 恢復預設額度後正常


def test_rate_limit_error_format(client: TestClient, user_token: str) -> None:
    settings = get_settings().model_copy(update={"rate_limit_per_minute": 0})
    client.app.dependency_overrides[get_settings] = lambda: settings  # type: ignore[attr-defined]
    try:
        r = client.get("/api/v1/models", headers=auth(user_token))
    finally:
        client.app.dependency_overrides.clear()  # type: ignore[attr-defined]
    assert_error(r, 429, "RATE_LIMITED")
    assert r.headers["Retry-After"] == "60"
