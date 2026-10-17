"""API 延遲壓測（驗收 2.12）：單品辨識 p95 < 300 ms、貨架分析 p95 < 1.5 秒（CPU）。

量的是 client 端看到的完整延遲（HTTP、上傳、驗證、推論、存檔、寫資料庫），依序送出、不並行。
結果與機器規格寫入 models/metrics.json 的 api_latency。

  python -m tools.bench_latency --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import os
import platform
import secrets
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROCERY = ROOT / "training" / "data" / "raw" / "GroceryStoreDataset" / "dataset"
SKU = ROOT / "training" / "data" / "raw" / "SKU110K_fixed" / "images"


def cpu_name() -> str:
    if platform.system() == "Windows":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    return platform.processor() or "unknown"


def run(client: httpx.Client, url: str, paths: list[Path], warmup: int = 3) -> dict[str, float]:
    times = []
    for i, p in enumerate(paths):
        data = p.read_bytes()
        t = time.perf_counter()
        r = client.post(url, files={"file": (p.name, data, "image/jpeg")})
        elapsed = (time.perf_counter() - t) * 1000
        if r.status_code != 201:
            raise RuntimeError(f"{url} {r.status_code} {r.text}")
        if i >= warmup:  # 前幾個請求包含 ONNX session 第一次執行的初始化，不計入
            times.append(elapsed)
    arr = np.array(times)
    return {
        "n": len(times),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "mean": round(statistics.fmean(times), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--product-n", type=int, default=100)
    ap.add_argument("--shelf-n", type=int, default=30)
    args = ap.parse_args()

    from sqlalchemy import delete

    from app.cli import create_user
    from app.db.models import User
    from app.db.session import session_factory
    from training.common import update_metrics

    username, password = f"bench-{secrets.token_hex(4)}", secrets.token_urlsafe(24)
    db = session_factory()()
    create_user(db, username, password, "user")
    try:
        api = f"{args.base}/api/v1"
        tok = httpx.post(f"{api}/auth/token", data={"username": username, "password": password})
        tok.raise_for_status()
        headers = {"Authorization": f"Bearer {tok.json()['access_token']}"}
        lines = (ROOT / "training/data/splits/grocery_test.csv").read_text().splitlines()[1:]
        products = [GROCERY / ln.split(",")[0] for ln in lines[:: len(lines) // args.product_n]]
        shelves = [
            SKU / n for n in (ROOT / "training/data/splits/sku110k_test.txt").read_text().split()
        ][: args.shelf_n + 3]
        with httpx.Client(headers=headers, timeout=60) as client:
            product = run(client, f"{api}/recognitions/product", products[: args.product_n + 3])
            shelf = run(client, f"{api}/recognitions/shelf", shelves)
        result = {
            "product": product,
            "shelf": shelf,
            "pass_product_p95_lt_300ms": product["p95"] < 300,
            "pass_shelf_p95_lt_1500ms": shelf["p95"] < 1500,
            "machine": {
                "cpu": cpu_name(),
                "logical_cpus": os.cpu_count(),
                "os": f"{platform.system()} {platform.release()}",
                "python": platform.python_version(),
            },
            "method": (
                "client 端端到端延遲，依序送出、不並行；API 與資料庫在同一台機器；"
                "ONNX Runtime CPU；壓測時 RATE_LIMIT_PER_MINUTE 調高"
            ),
        }
        update_metrics("api_latency", result)
        print(result)
        return (
            0 if result["pass_product_p95_lt_300ms"] and result["pass_shelf_p95_lt_1500ms"] else 1
        )
    finally:
        db.execute(delete(User).where(User.username == username))
        db.commit()
        db.close()


if __name__ == "__main__":
    sys.exit(main())
