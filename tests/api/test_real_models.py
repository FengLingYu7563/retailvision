"""真實模型整合測試（只在本機跑：需要 ONNX 權重與資料集，CI 自動 skip）。

- 2.9：用 Grocery 某一類的 3 張 train 實拍照登錄成新商品，查詢用 test 集同類照片（登錄時沒用過），
  必須出現在 /search/similar 的前 5 名（Phase 2 調整第 3 點）。上線模型看過全部 81 類，
  所以這裡驗證的是「不重新訓練就能登錄並搜到」的流程；新商品的辨識率見 Phase 1 保留類別實驗。
- 3.6：在兩個真實分類模型之間切換，回應的 model_version 跟著改變，catalog 重新計算 embedding。
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.api.conftest import ROOT, auth

MODELS = ROOT / "models"
GROCERY = ROOT / "training" / "data" / "raw" / "GroceryStoreDataset" / "dataset"
SKU = ROOT / "training" / "data" / "raw" / "SKU110K_fixed" / "images"

pytestmark = pytest.mark.skipif(
    not (
        (MODELS / "cls-dinov2-v2.onnx").exists()
        and (MODELS / "cls-mnv3-v1.onnx").exists()
        and (MODELS / "det-yolov8n-v1.onnx").exists()
        and GROCERY.exists()
        and SKU.exists()
    ),
    reason="需要本機的 ONNX 權重與資料集",
)


@pytest.fixture
def real_client(db) -> TestClient:  # type: ignore[no-untyped-def]
    from app.core.config import get_settings
    from app.main import create_app
    from app.services.registry import ModelRegistry
    from app.worker import set_worker_registry

    with db.bind.begin() as c:  # 換成真實模型的版本列
        c.execute(text("TRUNCATE model_versions RESTART IDENTITY CASCADE"))
        c.execute(
            text(
                "INSERT INTO model_versions (name, task, onnx_path, metrics, is_active, created_at)"
                " VALUES ('cls-dinov2-v2','classification','cls-dinov2-v2.onnx','{}',true,now()),"
                " ('cls-mnv3-v1','classification','cls-mnv3-v1.onnx','{}',false,now()),"
                " ('det-yolov8n-v1','detection','det-yolov8n-v1.onnx','{}',true,now())"
            )
        )
    settings = get_settings()
    registry = ModelRegistry.from_settings(settings)
    set_worker_registry(registry)
    with TestClient(create_app(registry=registry, settings=settings)) as c:
        yield c


def _split(name: str) -> dict[int, list[Path]]:
    by_class: dict[int, list[Path]] = {}
    for line in (
        (ROOT / "training" / "data" / "splits" / f"grocery_{name}.csv")
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    ):
        path, fine, _ = line.split(",")
        by_class.setdefault(int(fine), []).append(GROCERY / path)
    return by_class


def _files(paths: list[Path]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths]


def test_register_new_product_and_find_it(real_client: TestClient, admin_token: str) -> None:
    train, test = _split("train"), _split("test")
    classes = random.Random(42).sample(sorted(train), 10)  # 固定 seed 挑 10 類登錄
    ids = {}
    for c in classes:
        r = real_client.post(
            "/api/v1/catalog/products",
            data={"name": f"class-{c}"},
            files=_files(train[c][:3]),
            headers=auth(admin_token),
        )
        assert r.status_code == 201, r.text
        ids[c] = r.json()["id"]

    top1 = top5 = 0
    for c in classes:
        query = test[c][0]  # test 集的照片，登錄時沒用過
        r = real_client.post(
            "/api/v1/search/similar?k=5",
            files={"file": (query.name, query.read_bytes(), "image/jpeg")},
            headers=auth(admin_token),
        )
        found = [x["product_id"] for x in r.json()["results"]]
        top1 += found[0] == ids[c]
        top5 += ids[c] in found
    print(f"\n2.9 真實模型：10 類中 top-1 {top1}/10、top-5 {top5}/10")
    assert top5 == 10


def test_switch_between_real_models(real_client: TestClient, admin_token: str) -> None:
    train, test = _split("train"), _split("test")
    real_client.post(
        "/api/v1/catalog/products",
        data={"name": "p"},
        files=_files(train[1][:2]),
        headers=auth(admin_token),
    )
    q = test[1][0]
    query = {"file": (q.name, q.read_bytes(), "image/jpeg")}
    r = real_client.post("/api/v1/recognitions/product", files=query, headers=auth(admin_token))
    assert r.json()["model_version"] == "cls-dinov2-v2"

    r = real_client.put(
        "/api/v1/models/active", json={"name": "cls-mnv3-v1"}, headers=auth(admin_token)
    )
    assert r.json() == {"active": "cls-mnv3-v1", "reembedded": 2}
    r = real_client.post("/api/v1/recognitions/product", files=query, headers=auth(admin_token))
    assert r.json()["model_version"] == "cls-mnv3-v1"


def test_real_shelf_analysis(real_client: TestClient, admin_token: str) -> None:
    name = (ROOT / "training" / "data" / "splits" / "sku110k_test.txt").read_text().split()[0]
    img = SKU / name
    r = real_client.post(
        "/api/v1/recognitions/shelf",
        files={"file": (img.name, img.read_bytes(), "image/jpeg")},
        headers=auth(admin_token),
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["item_count"] > 50  # SKU-110K 的貨架照通常有上百個商品
    assert len(body["boxes"]) == body["item_count"]
    assert body["model_version"] == "det-yolov8n-v1"
