"""單品辨識、貨架分析、歷史查詢、單筆、標註圖（驗收 2.1、2.2）。"""

from __future__ import annotations

import uuid

import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.cli import create_user
from app.schemas import ProductRecognition, ShelfRecognition
from tests.api.conftest import (
    GREEN,
    RED,
    assert_error,
    auth,
    jpeg,
    textured,
    token,
    upload,
)

PRODUCT = "/api/v1/recognitions/product"
SHELF = "/api/v1/recognitions/shelf"


def test_product_recognition(client: TestClient, user_token: str) -> None:
    r = client.post(PRODUCT, files=upload(jpeg(textured(RED))), headers=auth(user_token))
    assert r.status_code == 201, r.text
    body = ProductRecognition.model_validate(r.json())  # 回應和 OpenAPI schema 一致
    assert body.type == "product"
    assert body.prediction.fine.label == "Red-Apple"
    assert body.model_version == "cls-fake-a"
    assert body.catalog_match is None  # catalog 是空的
    assert body.quality.passed
    assert r.headers["X-Request-ID"]


def test_png_and_webp_accepted(client: TestClient, user_token: str) -> None:
    img = textured(GREEN)
    for ext, ctype in ((".png", "image/png"), (".webp", "image/webp")):
        data = cv2.imencode(ext, img)[1].tobytes()
        r = client.post(PRODUCT, files=upload(data, f"a{ext}", ctype), headers=auth(user_token))
        assert r.status_code == 201, (ext, r.text)


def test_low_quality_warns_by_default_and_blocks_when_strict(
    client: TestClient, user_token: str
) -> None:
    blurry = jpeg(cv2.GaussianBlur(textured(RED), (31, 31), 10))
    r = client.post(PRODUCT, files=upload(blurry), headers=auth(user_token))
    assert r.status_code == 201
    assert "BLURRY" in r.json()["quality"]["warnings"]
    r = client.post(f"{PRODUCT}?strict=true", files=upload(blurry), headers=auth(user_token))
    assert_error(r, 422, "LOW_QUALITY_IMAGE")


def test_shelf_recognition(client: TestClient, user_token: str) -> None:
    img = np.full((600, 800, 3), 200, np.uint8)
    for x in range(0, 800, 16):
        cv2.line(img, (x, 0), (x, 600), (30, 30, 30), 3)
    # 假偵測器在第二層第 4、5 格（x 300 到 500）沒有框；那裡畫成乾淨的背板，
    # 否則紋理驗證會把有條紋的空隙當成「偵測器漏掉的商品」
    img[240:440, 300:500] = 90
    r = client.post(SHELF, files=upload(jpeg(img)), headers=auth(user_token))
    assert r.status_code == 201, r.text
    body = ShelfRecognition.model_validate(r.json())
    assert body.item_count == 14
    assert body.row_count == 2
    assert len(body.boxes) == 14
    assert len(body.gaps) == 1  # 假偵測器在第二層留了兩格（相鄰，合成一個空位）
    assert 0 < body.gap_ratio < 1
    assert body.model_version == "det-fake"


def test_get_and_annotated(client: TestClient, user_token: str) -> None:
    rid = client.post(PRODUCT, files=upload(jpeg(textured(RED))), headers=auth(user_token)).json()[
        "id"
    ]
    r = client.get(f"/api/v1/recognitions/{rid}", headers=auth(user_token))
    assert r.status_code == 200
    assert r.json()["id"] == rid
    r = client.get(f"/api/v1/recognitions/{rid}/annotated", headers=auth(user_token))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:3] == b"\xff\xd8\xff"


def test_not_found(client: TestClient, user_token: str) -> None:
    missing = uuid.uuid4()
    assert_error(
        client.get(f"/api/v1/recognitions/{missing}", headers=auth(user_token)), 404, "NOT_FOUND"
    )
    assert_error(
        client.get(f"/api/v1/recognitions/{missing}/annotated", headers=auth(user_token)),
        404,
        "NOT_FOUND",
    )
    r = client.get("/api/v1/recognitions/not-a-uuid", headers=auth(user_token))
    assert_error(r, 422, "VALIDATION_ERROR")


def test_cannot_see_other_users_records(
    client: TestClient, user_token: str, admin_token: str, db: Session
) -> None:
    rid = client.post(PRODUCT, files=upload(jpeg(textured(RED))), headers=auth(user_token)).json()[
        "id"
    ]
    create_user(db, "bob", "correct-horse-battery", "user")
    bob = token(client, "bob")
    # 別人的紀錄回 404（不是 403），不讓人用猜 ID 的方式確認紀錄存在
    assert_error(client.get(f"/api/v1/recognitions/{rid}", headers=auth(bob)), 404, "NOT_FOUND")
    assert_error(
        client.get(f"/api/v1/recognitions/{rid}/annotated", headers=auth(bob)), 404, "NOT_FOUND"
    )
    assert client.get("/api/v1/recognitions", headers=auth(bob)).json()["total"] == 0
    # admin 看得到全部
    assert client.get(f"/api/v1/recognitions/{rid}", headers=auth(admin_token)).status_code == 200


def test_history_filters_and_pagination(client: TestClient, user_token: str) -> None:
    h = auth(user_token)
    for color in (RED, RED, GREEN):
        client.post(PRODUCT, files=upload(jpeg(textured(color))), headers=h)
    client.post(SHELF, files=upload(jpeg(textured(RED, (400, 600)))), headers=h)

    r = client.get("/api/v1/recognitions", headers=h).json()
    assert r["total"] == 4
    assert client.get("/api/v1/recognitions?type=shelf", headers=h).json()["total"] == 1
    assert client.get("/api/v1/recognitions?label=Red-Apple", headers=h).json()["total"] == 2
    p1 = client.get("/api/v1/recognitions?size=3&page=1", headers=h).json()
    p2 = client.get("/api/v1/recognitions?size=3&page=2", headers=h).json()
    assert len(p1["items"]) == 3 and len(p2["items"]) == 1
    assert {i["id"] for i in p1["items"]}.isdisjoint({i["id"] for i in p2["items"]})
    future = "2999-01-01T00:00:00Z"
    assert client.get(f"/api/v1/recognitions?from={future}", headers=h).json()["total"] == 0
    assert client.get(f"/api/v1/recognitions?to={future}", headers=h).json()["total"] == 4
    assert_error(client.get("/api/v1/recognitions?size=101", headers=h), 422, "VALIDATION_ERROR")
    assert_error(client.get("/api/v1/recognitions?page=0", headers=h), 422, "VALIDATION_ERROR")


def test_model_not_ready_503(client: TestClient, user_token: str) -> None:
    registry = client.app.state.registry  # type: ignore[attr-defined]
    saved = registry.detector
    registry.detector = None
    try:
        r = client.post(SHELF, files=upload(jpeg(textured(RED))), headers=auth(user_token))
    finally:
        registry.detector = saved
    assert_error(r, 503, "MODEL_NOT_READY")
