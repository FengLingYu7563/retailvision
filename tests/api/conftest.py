"""API 測試共用設定。

- 資料庫與 Redis 用真的（PostgreSQL 16、Redis 7，docker compose 啟動）：JSONB 與 Alembic 必須在
  PostgreSQL 上驗證，不用 SQLite 代替（Phase 2 調整第 2 點）。連不上時整組 API 測試 skip。
- vision 模型用假的：CI 沒有 ONNX 權重也能測所有端點（Phase 2 調整第 1 點）。
- 測試資料庫是 retailvision_test、Redis 用第 15 號 db，不會動到開發資料。

環境變數必須在 import app 之前設定：app 的設定（get_settings）會被快取。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
_env = {**dotenv_values(ROOT / ".env"), **os.environ}
_base_url = _env.get("TEST_DATABASE_URL") or (_env.get("DATABASE_URL") or "").rsplit("/", 1)[0]
TEST_DB_URL = _env.get("TEST_DATABASE_URL") or f"{_base_url}/retailvision_test"
TEST_REDIS_URL = (_env.get("REDIS_URL") or "redis://127.0.0.1:6379/0").rsplit("/", 1)[0] + "/15"
STORAGE = Path(tempfile.mkdtemp(prefix="rv-storage-"))

os.environ.update(
    {
        "DATABASE_URL": TEST_DB_URL,
        "REDIS_URL": TEST_REDIS_URL,
        "JWT_SECRET": "test-secret-only-for-pytest-not-used-anywhere-else",
        "STORAGE_DIR": str(STORAGE),
        "CELERY_EAGER": "true",
        "RATE_LIMIT_PER_MINUTE": "10000",
        "MODELS_DIR": str(ROOT / "models"),
    }
)

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.registry import ModelRegistry  # noqa: E402
from vision.classifier import Classification, LabelScore  # noqa: E402
from vision.detector import Box  # noqa: E402
from vision.embedder import l2_normalize  # noqa: E402

LABELS = ["Red-Apple", "Green-Apple", "Banana"]


class FakeClassifier:
    """依畫面主色決定標籤；embedding 是色彩直方圖，同一種顏色的圖會很像。

    不同版本的 embedding 乘上不同的置換矩陣：模擬「不同模型的向量不在同一個空間」，
    沒有重新計算 catalog embedding 就切換模型的話，搜尋結果會錯（用來測 3.6）。
    """

    def __init__(self, version: str, onnx_path: str | None = None) -> None:
        self.version = version
        self.onnx_path = onnx_path
        # crc32 而不是 hash()：hash 每個 process 的種子不同，測試要可重現
        self._perm = np.random.default_rng(zlib.crc32(version.encode())).permutation(48)

    def embed(self, img: np.ndarray) -> np.ndarray:
        hist = [np.histogram(img[:, :, c], bins=16, range=(0, 256))[0] for c in range(3)]
        vec = np.concatenate(hist).astype(np.float32)[self._perm]
        return l2_normalize(vec)

    def predict(self, img: np.ndarray, k: int = 5) -> Classification:
        b, g, r = (float(img[:, :, c].mean()) for c in range(3))
        idx = int(np.argmax([r - g, g - r, (r + g) / 2 - b - 60]))
        top = [LabelScore(LABELS[idx], 0.9)] + [
            LabelScore(lbl, 0.05) for i, lbl in enumerate(LABELS) if i != idx
        ]
        return Classification(
            fine=top[0], coarse=LabelScore("Fruit", 0.95), top_k=top[:k], embedding=self.embed(img)
        )


class FakeDetector:
    version = "det-fake"

    def detect(self, img: np.ndarray) -> list[Box]:
        h, w = img.shape[:2]
        bw, bh = w / 8, h / 3
        boxes = []
        for row in range(2):
            for col in range(8):
                if row == 1 and col in (3, 4):  # 第二層中間留兩格空位
                    continue
                x1, y1 = col * bw + 2, row * bh * 1.2 + 2
                boxes.append(Box(x1, y1, x1 + bw - 4, y1 + bh - 4, 0.9))
        return boxes


def fake_registry() -> ModelRegistry:
    reg = ModelRegistry(FakeClassifier, FakeDetector(), None)
    reg.load_classifier("cls-fake-a")
    return reg


def textured(color: tuple[int, int, int], size: tuple[int, int] = (240, 320)) -> np.ndarray:
    """有紋理的單色圖：模糊檢查會通過。"""
    rng = np.random.default_rng(sum(color))
    img = np.full((*size, 3), color, np.uint8)
    noise = rng.integers(-40, 40, img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    for i in range(0, size[1], 24):
        cv2.line(img, (i, 0), (i, size[0]), (0, 0, 0), 2)
    return img


def jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


RED, GREEN, YELLOW = (40, 40, 220), (40, 200, 40), (40, 220, 230)


def _db_available() -> bool:
    try:
        eng = create_engine(TEST_DB_URL)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        eng.dispose()
        import redis

        redis.Redis.from_url(TEST_REDIS_URL, socket_timeout=1).ping()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _db_available():
        return
    skip = pytest.mark.skip(reason="PostgreSQL / Redis 未啟動（docker compose up -d db redis）")
    for item in items:
        if "tests/api" in item.nodeid.replace("\\", "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _migrate() -> Iterator[None]:
    """整個測試開始前，把測試資料庫升級到最新版（同時也驗證 migration 能跑）。"""
    if not _db_available():
        yield
        return
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.cmd_opts = type("Opts", (), {"x": [f"url={TEST_DB_URL}"]})()  # type: ignore[assignment]
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield
    shutil.rmtree(STORAGE, ignore_errors=True)


@pytest.fixture
def db() -> Iterator[Session]:
    from app.db.session import session_factory

    s = session_factory()()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _clean(request: pytest.FixtureRequest) -> Iterator[None]:
    """每個測試開始前清空資料表與 Redis，測試之間互不影響。"""
    if "tests/api" not in request.node.nodeid.replace("\\", "/") or not _db_available():
        yield
        return
    from app.core.ratelimit import get_redis
    from app.db.session import get_engine

    with get_engine().begin() as c:
        c.execute(
            text(
                "TRUNCATE recognitions, jobs, catalog_embeddings, catalog_products, "
                "model_versions, users RESTART IDENTITY CASCADE"
            )
        )
        c.execute(
            text(
                "INSERT INTO model_versions (name, task, onnx_path, metrics, is_active, created_at)"
                " VALUES ('cls-fake-a','classification','a.onnx','{}',true,now()),"
                " ('cls-fake-b','classification','b.onnx','{}',false,now()),"
                " ('det-fake','detection','d.onnx','{}',true,now())"
            )
        )
    get_redis().flushdb()
    yield


@pytest.fixture
def registry() -> ModelRegistry:
    from app.worker import set_worker_registry

    reg = fake_registry()
    set_worker_registry(reg)  # Celery eager 模式在同一個 process 執行，共用假模型
    return reg


@pytest.fixture
def client(registry: ModelRegistry) -> Iterator[TestClient]:
    from app.main import create_app

    app = create_app(registry=registry, settings=get_settings())
    with TestClient(app) as c:
        yield c


def _make_user(db: Session, name: str, role: str) -> None:
    from app.cli import create_user

    create_user(db, name, "correct-horse-battery", role)


@pytest.fixture
def user_token(client: TestClient, db: Session) -> str:
    _make_user(db, "alice", "user")
    return token(client, "alice")


@pytest.fixture
def admin_token(client: TestClient, db: Session) -> str:
    _make_user(db, "root", "admin")
    return token(client, "root")


def token(client: TestClient, username: str, password: str = "correct-horse-battery") -> str:
    r = client.post("/api/v1/auth/token", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return str(r.json()["access_token"])


def auth(tok: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {tok}"}


def upload(data: bytes, name: str = "a.jpg", ctype: str = "image/jpeg") -> dict[str, tuple]:
    return {"file": (name, data, ctype)}


def assert_error(r, status: int, code: str) -> None:  # type: ignore[no-untyped-def]
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == r.headers["X-Request-ID"]
