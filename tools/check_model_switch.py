"""實測 3.6：切換使用中的分類模型後，API 與 Celery worker 的新請求都改用新版本，不需重啟。

帳密從 .env 的 ADMIN_USERNAME / ADMIN_PASSWORD 讀（不印出）。測完切回原本的模型。

  python -m tools.check_model_switch --base http://localhost:8080 --to cls-mnv3-v1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
IMG = (
    ROOT
    / "training/data/raw/GroceryStoreDataset/dataset/test/Fruit/Apple/Granny-Smith"
    / "Granny-Smith_001.jpg"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--to", default="cls-mnv3-v1")
    args = ap.parse_args()
    env = dotenv_values(ROOT / ".env")
    api = f"{args.base}/api/v1"
    tok = httpx.post(
        f"{api}/auth/token",
        data={"username": env["ADMIN_USERNAME"], "password": env["ADMIN_PASSWORD"]},
    )
    tok.raise_for_status()
    h = {"Authorization": f"Bearer {tok.json()['access_token']}"}
    img = IMG.read_bytes()

    def product_version() -> str:
        r = httpx.post(f"{api}/recognitions/product", files={"file": (IMG.name, img)}, headers=h)
        r.raise_for_status()
        return str(r.json()["model_version"])

    def worker_version() -> str:
        r = httpx.post(f"{api}/recognitions/batch", files=[("files", (IMG.name, img))], headers=h)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        for _ in range(60):
            job = httpx.get(f"{api}/jobs/{job_id}", headers=h).json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(1)
        rid = job["recognition_ids"][0]
        return str(httpx.get(f"{api}/recognitions/{rid}", headers=h).json()["model_version"])

    before_api, before_worker = product_version(), worker_version()
    print(f"切換前：API {before_api}、worker {before_worker}")
    r = httpx.put(f"{api}/models/active", json={"name": args.to}, headers=h, timeout=120)
    print(f"PUT /models/active -> {r.status_code} {r.json()}")
    after_api, after_worker = product_version(), worker_version()
    print(f"切換後：API {after_api}、worker {after_worker}")
    httpx.put(f"{api}/models/active", json={"name": before_api}, headers=h, timeout=120)
    restored = product_version()
    print(f"切回：API {restored}")
    ok = after_api == after_worker == args.to and restored == before_api != args.to
    print("結果：" + ("通過（兩個 process 都改用新版本，沒有重啟）" if ok else "失敗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
