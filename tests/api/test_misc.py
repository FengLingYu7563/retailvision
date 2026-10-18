"""統計、健康檢查、OpenAPI（2.1）、Alembic（2.10）、request_id（3.8）、靜態檢查（2.5、2.7）。"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from app.schemas import StatsOut
from tests.api.conftest import (
    GREEN,
    RED,
    ROOT,
    TEST_DB_URL,
    assert_error,
    auth,
    jpeg,
    textured,
    upload,
)

# 規格第 6 節的端點表
SPEC_ENDPOINTS = {
    ("post", "/api/v1/auth/token"),
    ("post", "/api/v1/recognitions/product"),
    ("post", "/api/v1/recognitions/shelf"),
    ("post", "/api/v1/recognitions/batch"),
    ("get", "/api/v1/jobs/{job_id}"),
    ("get", "/api/v1/recognitions"),
    ("get", "/api/v1/recognitions/{rec_id}"),
    ("get", "/api/v1/recognitions/{rec_id}/annotated"),
    ("post", "/api/v1/catalog/products"),
    ("get", "/api/v1/catalog/products"),
    ("delete", "/api/v1/catalog/products/{product_id}"),
    ("post", "/api/v1/search/similar"),
    ("get", "/api/v1/models"),
    ("put", "/api/v1/models/active"),
    ("get", "/api/v1/stats/summary"),
    ("get", "/api/v1/healthz"),
}


def test_openapi_has_all_spec_endpoints(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    found = {(m, p) for p, ops in spec["paths"].items() for m in ops}
    assert found >= SPEC_ENDPOINTS, SPEC_ENDPOINTS - found
    assert client.get("/docs").status_code == 200
    # 每個需要登入的端點都在 OpenAPI 標示 Bearer 驗證
    for method, path in SPEC_ENDPOINTS - {
        ("post", "/api/v1/auth/token"),
        ("get", "/api/v1/healthz"),
    }:
        op = spec["paths"][path][method]
        assert op.get("security"), (method, path)
        # 需要登入的端點都會回這些錯誤，文件也要列出來，而且格式是統一的 ErrorResponse
        for code in ("401", "422", "429"):
            ref = op["responses"][code]["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ErrorResponse"), (method, path, code)
    health = spec["paths"]["/api/v1/healthz"]["get"]["responses"]["503"]
    assert health["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")


def test_stats_summary(client: TestClient, user_token: str) -> None:
    h = auth(user_token)
    for color in (RED, RED, GREEN):
        client.post("/api/v1/recognitions/product", files=upload(jpeg(textured(color))), headers=h)
    client.post(
        "/api/v1/recognitions/shelf", files=upload(jpeg(textured(RED, (400, 600)))), headers=h
    )
    r = client.get("/api/v1/stats/summary?days=7", headers=h)
    assert r.status_code == 200
    s = StatsOut.model_validate(r.json())
    assert s.product.count == 3
    assert {x.label: x.count for x in s.product.by_label} == {"Red-Apple": 2, "Green-Apple": 1}
    assert s.shelf.count == 1
    assert s.shelf.stockout_rate is not None
    assert sum(d.count for d in s.daily) == 4
    assert_error(client.get("/api/v1/stats/summary?days=0", headers=h), 422, "VALIDATION_ERROR")


def test_stats_scoped_to_user(client: TestClient, user_token: str, admin_token: str) -> None:
    """一般使用者的統計只算自己的紀錄，admin 算全部（IDOR）。"""
    q = upload(jpeg(textured(RED)))
    client.post("/api/v1/recognitions/product", files=q, headers=auth(user_token))
    client.post("/api/v1/recognitions/product", files=q, headers=auth(admin_token))
    client.post("/api/v1/recognitions/product", files=q, headers=auth(admin_token))
    mine = client.get("/api/v1/stats/summary", headers=auth(user_token)).json()
    everyone = client.get("/api/v1/stats/summary", headers=auth(admin_token)).json()
    assert mine["product"]["count"] == 1
    assert everyone["product"]["count"] == 3


def test_healthz(client: TestClient) -> None:
    for path in ("/api/v1/healthz", "/healthz"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["models"]["classifier"] == "cls-fake-a"


def test_healthz_503_when_model_missing(client: TestClient) -> None:
    registry = client.app.state.registry  # type: ignore[attr-defined]
    saved = registry.detector
    registry.detector = None
    try:
        r = client.get("/api/v1/healthz")
    finally:
        registry.detector = saved
    # 503 也是統一錯誤格式，元件狀態在 details
    assert_error(r, 503, "SERVICE_UNAVAILABLE")
    details = r.json()["error"]["details"]
    assert details["status"] == "degraded" and details["models"]["detector"] is None
    # liveness 不受影響：docker healthcheck 用 /livez，模型缺少時容器仍是 healthy，nginx 才起得來
    assert client.get("/livez").status_code == 200
    registry.detector = None
    try:
        assert client.get("/livez").json() == {"status": "alive"}
    finally:
        registry.detector = saved


def test_request_id_propagated_and_sanitized(client: TestClient) -> None:
    r = client.get("/api/v1/healthz", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    # 含換行等不安全字元的 request_id 不接受（避免 log injection），改用伺服器產生的
    r = client.get("/api/v1/healthz", headers={"X-Request-ID": "bad id\nINJECT"})
    assert r.headers["X-Request-ID"] != "bad id\nINJECT"
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["X-Request-ID"])


def test_unknown_route_uses_error_format(client: TestClient) -> None:
    assert_error(client.get("/api/v1/nope"), 404, "NOT_FOUND")


def test_unhandled_exception_returns_500_with_request_id(client: TestClient) -> None:
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    client.app.include_router(router)  # type: ignore[attr-defined]
    r = client.get("/boom", headers={"X-Request-ID": "trace-500"})
    assert_error(r, 500, "INTERNAL_ERROR")
    assert r.json()["error"]["request_id"] == "trace-500"
    assert "secret internal detail" not in r.text  # 內部錯誤訊息不外洩


def test_alembic_downgrade_and_upgrade() -> None:
    """驗收 2.10：從最新版 downgrade 到空資料庫，再 upgrade 回最新版。"""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.cmd_opts = type("Opts", (), {"x": [f"url={TEST_DB_URL}"]})()  # type: ignore[assignment]
    eng = create_engine(TEST_DB_URL)
    command.downgrade(cfg, "base")
    assert set(inspect(eng).get_table_names()) <= {"alembic_version"}
    command.upgrade(cfg, "head")
    tables = set(inspect(eng).get_table_names())
    expected = {
        "users",
        "recognitions",
        "jobs",
        "catalog_products",
        "catalog_embeddings",
        "model_versions",
    }
    assert expected <= tables
    idx = {i["name"] for i in inspect(eng).get_indexes("recognitions")}
    assert {"ix_recognitions_type_created_at", "ix_recognitions_top_label"} <= idx
    eng.dispose()


def test_only_one_active_model_per_task(db) -> None:  # type: ignore[no-untyped-def]
    """資料庫層級保證每個任務最多一個使用中的模型（Codex 第 9 輪 L1）。"""
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    # 兩個分類模型同時 active：commit 時被拒絕
    db.execute(text("UPDATE model_versions SET is_active = true WHERE task = 'classification'"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    # 同一個交易裡先設新的、再取消舊的（切換時的順序）：commit 時只有一個 active，允許
    db.execute(text("UPDATE model_versions SET is_active = true WHERE name = 'cls-fake-b'"))
    db.execute(text("UPDATE model_versions SET is_active = false WHERE name = 'cls-fake-a'"))
    db.commit()
    active = (
        db.execute(
            text("SELECT name FROM model_versions WHERE task = 'classification' AND is_active")
        )
        .scalars()
        .all()
    )
    assert active == ["cls-fake-b"]


# ---------- 靜態檢查 ----------


def _py_files(*dirs: str) -> list[Path]:
    return [p for d in dirs for p in (ROOT / d).rglob("*.py")]


def test_no_string_built_sql() -> None:
    """驗收 2.7：程式裡沒有用 f-string / format / + 組 SQL 再交給 text() 或 execute()。"""
    offenders = []
    for path in _py_files("app", "migrations"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name not in {"text", "execute", "exec_driver_sql"} or not node.args:
                continue
            arg = node.args[0]
            dynamic = isinstance(arg, ast.JoinedStr | ast.BinOp) or (
                isinstance(arg, ast.Call) and getattr(arg.func, "attr", "") == "format"
            )
            if dynamic:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


# ci-only-password：CI 的 service container 專用、只在 runner 內有效（.github/workflows/ci.yml）
SECRET_PATTERNS = [
    re.compile(r"JWT_SECRET\s*=\s*(?!replace-with)[A-Za-z0-9_\-]{20,}"),
    re.compile(r"POSTGRES_PASSWORD\s*[=:]\s*(?!change-me|ci-only-password)\S{8,}"),
    re.compile(r"postgresql\+psycopg://[^:\s]+:(?!change-me@|ci-only-password@)[^@\s]{8,}@"),
    re.compile(r"ADMIN_PASSWORD\s*=\s*\S{8,}"),
]


def test_no_real_secrets_tracked() -> None:
    """驗收 2.5：git 追蹤的檔案裡沒有真實 secret（.env.example 只有範例值，.env 不追蹤）。"""
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert ".env" not in files
    hits = []
    for f in files:
        if Path(f).suffix in {".png", ".jpg", ".onnx"} or f.startswith("training/data/splits"):
            continue
        try:
            content = (ROOT / f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits += [f"{f}: {p.pattern}" for p in SECRET_PATTERNS if p.search(content)]
    assert hits == []


def test_jwt_secret_required() -> None:
    """沒有設定 JWT_SECRET 時啟動就失敗，不會默默用預設值。"""
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(database_url="postgresql+psycopg://x/y", _env_file=None, jwt_secret="short")  # type: ignore[call-arg]
    # 文件裡的範例值即使夠長也拒絕
    with pytest.raises(ValidationError, match="範例值"):
        Settings(  # type: ignore[call-arg]
            database_url="postgresql+psycopg://x/y",
            _env_file=None,
            jwt_secret="replace-with-a-long-random-string-at-least-32-chars",
        )
