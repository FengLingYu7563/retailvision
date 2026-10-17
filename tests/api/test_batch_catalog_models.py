"""批次（2.8）、新品登錄與以圖搜圖（2.9）、模型切換（3.6）。

Celery 在測試裡用 eager 模式（同一個 process 同步執行），用的是和 worker 相同的 run_batch。
真正的 worker process 另外在整合測試跑（見 tests/api/test_real_models.py 與 docs）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CatalogEmbedding
from app.services.uploads import Storage
from tests.api.conftest import (
    GREEN,
    RED,
    STORAGE,
    YELLOW,
    assert_error,
    auth,
    jpeg,
    textured,
    upload,
)


def _files(blobs: list[tuple[str, bytes]]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, data, "image/jpeg")) for name, data in blobs]


# ---------- 2.8 批次 ----------


def test_batch_20_images_one_bad(client: TestClient, user_token: str) -> None:
    blobs = [(f"img{i}.jpg", jpeg(textured((40 + i * 5, 60, 200)))) for i in range(19)]
    blobs.insert(7, ("broken.jpg", b"this is not an image"))
    r = client.post(
        "/api/v1/recognitions/batch",
        files=_files(blobs),
        data={"type": "product"},
        headers=auth(user_token),
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert r.json()["total"] == 20

    job = client.get(f"/api/v1/jobs/{job_id}", headers=auth(user_token)).json()
    assert job["status"] == "done"
    assert job["done_count"] == 20
    assert job["failed_count"] == 1
    assert len(job["recognition_ids"]) == 19
    assert job["item_errors"] == [
        {
            "index": 7,
            "filename": "broken.jpg",
            "code": "INVALID_IMAGE",
            "message": "檔案不是有效的圖片",
        }
    ]
    # 批次產生的紀錄也查得到，而且有 job_id
    rec = client.get(f"/api/v1/recognitions/{job['recognition_ids'][0]}", headers=auth(user_token))
    assert rec.json()["job_id"] == job_id


def test_batch_shelf_type(client: TestClient, user_token: str) -> None:
    r = client.post(
        "/api/v1/recognitions/batch",
        files=_files([("s.jpg", jpeg(textured(RED, (400, 600))))]),
        data={"type": "shelf"},
        headers=auth(user_token),
    )
    job = client.get(f"/api/v1/jobs/{r.json()['job_id']}", headers=auth(user_token)).json()
    assert job["type"] == "shelf" and job["status"] == "done" and job["failed_count"] == 0


def test_batch_limits_and_job_ownership(
    client: TestClient, user_token: str, admin_token: str
) -> None:
    too_many = _files([(f"{i}.jpg", jpeg(textured(RED))) for i in range(51)])
    assert_error(
        client.post("/api/v1/recognitions/batch", files=too_many, headers=auth(user_token)),
        422,
        "VALIDATION_ERROR",
    )
    r = client.post(
        "/api/v1/recognitions/batch",
        files=_files([("a.jpg", jpeg(textured(RED)))]),
        headers=auth(admin_token),
    )
    # 一般使用者看不到 admin 的批次
    assert_error(
        client.get(f"/api/v1/jobs/{r.json()['job_id']}", headers=auth(user_token)), 404, "NOT_FOUND"
    )


def _batch_files_on_disk() -> list[Path]:
    root = STORAGE / "batch"
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def test_invalid_batch_items_are_never_written_to_disk(
    client: TestClient, user_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不合格的檔案（壞圖、超過大小、不支援的格式）在上傳時就記成錯誤，不會寫進硬碟。"""
    saved: list[str] = []
    real_save = Storage.save

    def spy(self: Storage, data: bytes, ext: str, kind: str) -> str:
        path = real_save(self, data, ext, kind)
        if kind == "batch":
            saved.append(path)
        return path

    monkeypatch.setattr(Storage, "save", spy)
    blobs = [
        ("ok.jpg", jpeg(textured(RED))),
        ("text.jpg", b"not an image"),
        ("big.jpg", b"\xff\xd8\xff" + b"\0" * (10 * 1024 * 1024)),
        ("pic.gif", jpeg(textured(RED))),
    ]
    r = client.post("/api/v1/recognitions/batch", files=_files(blobs), headers=auth(user_token))
    assert r.status_code == 202
    job = client.get(f"/api/v1/jobs/{r.json()['job_id']}", headers=auth(user_token)).json()
    assert job["status"] == "done" and job["done_count"] == 4 and job["failed_count"] == 3
    assert [(e["index"], e["code"]) for e in job["item_errors"]] == [
        (1, "INVALID_IMAGE"),
        (2, "FILE_TOO_LARGE"),
        (3, "UNSUPPORTED_FORMAT"),
    ]
    assert len(saved) == 1  # 只有通過驗證的那張被寫進硬碟
    assert _batch_files_on_disk() == []  # worker 處理完就刪掉暫存檔


def test_batch_all_invalid_finishes_without_queue(client: TestClient, user_token: str) -> None:
    r = client.post(
        "/api/v1/recognitions/batch",
        files=_files([("a.jpg", b"nope"), ("b.jpg", b"nope")]),
        headers=auth(user_token),
    )
    assert r.status_code == 202 and r.json()["status"] == "done"


def test_batch_queue_unavailable(
    client: TestClient, user_token: str, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """佇列送不出去：回 503、job 標成 failed、已存的暫存檔刪掉，不留下永遠 pending 的 job。"""
    from app import worker
    from app.db.models import Job

    def broken_delay(*_a: object, **_k: object) -> None:
        raise ConnectionError("broker down")

    monkeypatch.setattr(worker.process_batch, "delay", broken_delay)
    r = client.post(
        "/api/v1/recognitions/batch",
        files=_files([("a.jpg", jpeg(textured(RED)))]),
        headers=auth(user_token),
    )
    assert_error(r, 503, "QUEUE_UNAVAILABLE")
    job = db.scalars(select(Job)).one()
    assert job.status == "failed" and job.error == "QUEUE_UNAVAILABLE"
    assert _batch_files_on_disk() == []


# ---------- 2.9 新品登錄與以圖搜圖 ----------


def _register(
    client: TestClient,
    tok: str,
    name: str,
    color: tuple[int, int, int],
    n: int = 3,
    sku: str | None = None,
) -> dict:
    data = {"name": name} | ({"sku": sku} if sku else {})
    files = _files(
        [(f"{name}{i}.jpg", jpeg(textured(color, (200 + i * 10, 300)))) for i in range(n)]
    )
    r = client.post("/api/v1/catalog/products", data=data, files=files, headers=auth(tok))
    assert r.status_code == 201, r.text
    return dict(r.json())


def test_register_then_search_without_retraining(client: TestClient, user_token: str) -> None:
    red = _register(client, user_token, "紅蘋果", RED)
    _register(client, user_token, "綠蘋果", GREEN)
    _register(client, user_token, "香蕉", YELLOW)
    assert red["reference_images"] == 3

    # 查詢用登錄時沒用過的另一張同類照片（不同尺寸與雜訊）
    query = jpeg(textured((45, 35, 215), (260, 360)))
    r = client.post("/api/v1/search/similar?k=5", files=upload(query), headers=auth(user_token))
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["product_id"] == red["id"]
    assert results[0]["name"] == "紅蘋果"
    assert [x["similarity"] for x in results] == sorted(
        [x["similarity"] for x in results], reverse=True
    )

    # 單品辨識也會帶出 catalog_match
    r = client.post("/api/v1/recognitions/product", files=upload(query), headers=auth(user_token))
    assert r.json()["catalog_match"]["product_id"] == red["id"]


def test_search_empty_catalog_and_k_bounds(client: TestClient, user_token: str) -> None:
    q = upload(jpeg(textured(RED)))
    r = client.post("/api/v1/search/similar", files=q, headers=auth(user_token))
    assert r.json()["results"] == []
    assert_error(
        client.post("/api/v1/search/similar?k=21", files=q, headers=auth(user_token)),
        422,
        "VALIDATION_ERROR",
    )


def test_catalog_validation(client: TestClient, user_token: str) -> None:
    _register(client, user_token, "A", RED, sku="SKU-1")
    files = _files([("x.jpg", jpeg(textured(RED)))])
    r = client.post(
        "/api/v1/catalog/products",
        data={"name": "B", "sku": "SKU-1"},
        files=files,
        headers=auth(user_token),
    )
    assert_error(r, 409, "CONFLICT")
    six = _files([(f"{i}.jpg", jpeg(textured(RED))) for i in range(6)])
    r = client.post(
        "/api/v1/catalog/products", data={"name": "C"}, files=six, headers=auth(user_token)
    )
    assert_error(r, 422, "VALIDATION_ERROR")
    r = client.post(
        "/api/v1/catalog/products",
        data={"name": "D"},
        files=_files([("bad.jpg", b"nope")]),
        headers=auth(user_token),
    )
    assert_error(r, 400, "INVALID_IMAGE")


def test_list_and_delete(
    client: TestClient, user_token: str, admin_token: str, db: Session
) -> None:
    p = _register(client, user_token, "A", RED)
    _register(client, user_token, "B", GREEN)
    page = client.get("/api/v1/catalog/products?size=1", headers=auth(user_token)).json()
    assert page["total"] == 2 and len(page["items"]) == 1

    assert_error(
        client.delete(f"/api/v1/catalog/products/{p['id']}", headers=auth(user_token)),
        403,
        "FORBIDDEN",
    )
    r = client.delete(f"/api/v1/catalog/products/{p['id']}", headers=auth(admin_token))
    assert r.status_code == 204
    assert (
        db.scalar(select(CatalogEmbedding.id).where(CatalogEmbedding.product_id == p["id"])) is None
    )
    assert_error(
        client.delete(f"/api/v1/catalog/products/{p['id']}", headers=auth(admin_token)),
        404,
        "NOT_FOUND",
    )


def test_catalog_create_failure_cleans_up_files(
    client: TestClient,
    user_token: str,
    registry,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    """登錄到一半失敗（第二張參考圖算 embedding 時出錯）：資料庫回滾，已存的參考圖也刪掉。"""
    clf = registry.classifier
    calls = {"n": 0}
    real_embed = clf.embed

    def flaky(img):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("embedding failed")
        return real_embed(img)

    def catalog_files() -> set[Path]:
        root = STORAGE / "catalog"  # 整輪測試共用 storage，所以比對請求前後的差異
        return {p for p in root.rglob("*") if p.is_file()} if root.exists() else set()

    monkeypatch.setattr(clf, "embed", flaky)
    before = catalog_files()
    files = _files([(f"{i}.jpg", jpeg(textured(RED))) for i in range(3)])
    r = client.post(
        "/api/v1/catalog/products", data={"name": "X"}, files=files, headers=auth(user_token)
    )
    assert_error(r, 500, "INTERNAL_ERROR")
    assert catalog_files() == before
    page = client.get("/api/v1/catalog/products", headers=auth(user_token)).json()
    assert page["total"] == 0


# ---------- 3.6 切換模型 ----------


def test_switch_model_changes_version_and_reembeds(
    client: TestClient, user_token: str, admin_token: str, db: Session
) -> None:
    red = _register(client, user_token, "紅蘋果", RED)
    _register(client, user_token, "綠蘋果", GREEN)
    query = upload(jpeg(textured((45, 35, 215), (260, 360))))

    r = client.post("/api/v1/recognitions/product", files=query, headers=auth(user_token))
    assert r.json()["model_version"] == "cls-fake-a"

    r = client.put("/api/v1/models/active", json={"name": "cls-fake-b"}, headers=auth(admin_token))
    assert r.status_code == 200, r.text
    assert r.json() == {"active": "cls-fake-b", "reembedded": 6}

    # 不需重啟，下一個請求就用新版本
    r = client.post("/api/v1/recognitions/product", files=query, headers=auth(user_token))
    assert r.json()["model_version"] == "cls-fake-b"
    # catalog 已用新模型重算：仍然找得到正確商品（假模型不同版本的向量空間不同）
    r = client.post("/api/v1/search/similar", files=query, headers=auth(user_token))
    assert r.json()["model_version"] == "cls-fake-b"
    assert r.json()["results"][0]["product_id"] == red["id"]

    models = {
        m["name"]: m["is_active"]
        for m in client.get("/api/v1/models", headers=auth(user_token)).json()
    }
    assert models == {"cls-fake-a": False, "cls-fake-b": True, "det-fake": True}

    # 切回舊版本：舊向量還在，不用重算
    r = client.put("/api/v1/models/active", json={"name": "cls-fake-a"}, headers=auth(admin_token))
    assert r.json() == {"active": "cls-fake-a", "reembedded": 0}


def test_other_process_switch_is_picked_up(
    client: TestClient, user_token: str, db: Session
) -> None:
    """另一個 process（例如另一台 API 或 worker）在資料庫切換了模型，這邊下一個請求就跟上。"""
    from app.db.models import ModelVersion

    for m in db.scalars(select(ModelVersion).where(ModelVersion.task == "classification")):
        m.is_active = m.name == "cls-fake-b"
    db.commit()
    r = client.post(
        "/api/v1/recognitions/product", files=upload(jpeg(textured(RED))), headers=auth(user_token)
    )
    assert r.json()["model_version"] == "cls-fake-b"


def test_switch_errors(client: TestClient, admin_token: str) -> None:
    h = auth(admin_token)
    assert_error(
        client.put("/api/v1/models/active", json={"name": "nope"}, headers=h), 404, "NOT_FOUND"
    )
    assert_error(
        client.put("/api/v1/models/active", json={"name": "det-fake"}, headers=h),
        422,
        "VALIDATION_ERROR",
    )
    assert_error(client.put("/api/v1/models/active", json={}, headers=h), 422, "VALIDATION_ERROR")
