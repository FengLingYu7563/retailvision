"""端到端冒煙測試：對真的 API + 真的 Celery worker 送 20 張圖的批次（驗收 2.8）。

前提：API 與 worker 已啟動、資料庫已 migrate 並 seed 模型。
帳號：腳本自己建立一個隨機密碼的一次性帳號（不印出密碼），測完不保留。

  python -m tools.smoke_batch --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
GROCERY = ROOT / "training" / "data" / "raw" / "GroceryStoreDataset" / "dataset"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    from sqlalchemy import delete

    from app.cli import create_user
    from app.db.models import User
    from app.db.session import session_factory

    username, password = f"smoke-{secrets.token_hex(4)}", secrets.token_urlsafe(24)
    db = session_factory()()
    create_user(db, username, password, "user")
    try:
        api = f"{args.base}/api/v1"
        tok = httpx.post(f"{api}/auth/token", data={"username": username, "password": password})
        tok.raise_for_status()
        h = {"Authorization": f"Bearer {tok.json()['access_token']}"}

        lines = (ROOT / "training/data/splits/grocery_test.csv").read_text().splitlines()[1:]
        paths = [GROCERY / ln.split(",")[0] for ln in lines[:: len(lines) // args.n]][: args.n - 1]
        files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths]
        files.insert(5, ("files", ("broken.jpg", b"definitely not an image", "image/jpeg")))

        t0 = time.perf_counter()
        r = httpx.post(
            f"{api}/recognitions/batch",
            files=files,
            data={"type": "product"},
            headers=h,
            timeout=60,
        )
        print(f"POST /recognitions/batch -> {r.status_code}（{len(files)} 張）")
        if r.status_code != 202:
            print(r.text)
            return 1
        job_id = r.json()["job_id"]
        while time.perf_counter() - t0 < args.timeout:
            job = httpx.get(f"{api}/jobs/{job_id}", headers=h).json()
            print(
                f"  {time.perf_counter() - t0:5.1f}s  {job['status']:8s} "
                f"{job['done_count']}/{job['total']}（失敗 {job['failed_count']}）"
            )
            if job["status"] in ("done", "failed"):
                break
            time.sleep(1)
        ok = (
            job["status"] == "done"
            and job["done_count"] == len(files)
            and job["failed_count"] == 1
            and len(job["recognition_ids"]) == len(files) - 1
            and job["item_errors"][0]["filename"] == "broken.jpg"
        )
        print("結果：" + ("通過" if ok else f"失敗 {job}"))
        return 0 if ok else 1
    finally:
        db.execute(delete(User).where(User.username == username))  # 連帶刪除這個帳號的紀錄
        db.commit()
        db.close()


if __name__ == "__main__":
    sys.exit(main())
