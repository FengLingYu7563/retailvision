"""模擬相機（驗收 3.4）：監看資料夾，有新圖片就送去辨識。

用公開資料集的照片代替真的相機（規格第 2 節）。為什麼用輪詢而不用 watchdog：
少一個依賴，而且 Docker 在 Windows / macOS 的 bind mount 上，檔案系統事件常常收不到。

- 新檔案要「寫完」才送出：連續兩次輪詢檔案大小相同才算寫完，避免送出寫到一半的圖。
- 送出後移到 processed/ 或 failed/（失敗時另外寫一個 .error.txt 說明原因），不會重複送。
- 帳密從環境變數讀（CAMERA_USERNAME / CAMERA_PASSWORD），不寫在程式或命令列；
  token 過期會自動重新登入。

  python -m tools.camera_sim --watch camera_inbox --type product
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

log = logging.getLogger("camera_sim")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class Camera:
    def __init__(self, api: str, username: str, password: str, kind: str) -> None:
        self.api = api.rstrip("/") + "/api/v1"
        self.username, self.password, self.kind = username, password, kind
        self.client = httpx.Client(timeout=60)
        self.token: str | None = None

    def login(self) -> None:
        r = self.client.post(
            f"{self.api}/auth/token", data={"username": self.username, "password": self.password}
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]

    def send(self, path: Path) -> httpx.Response:
        if self.token is None:
            self.login()
        for attempt in range(2):
            with path.open("rb") as f:
                r = self.client.post(
                    f"{self.api}/recognitions/{self.kind}",
                    files={"file": (path.name, f, "application/octet-stream")},
                    headers={"Authorization": f"Bearer {self.token}"},
                )
            if r.status_code == 401 and attempt == 0:  # token 過期：重新登入再送一次
                self.login()
                continue
            return r
        return r


def stable_files(folder: Path, sizes: dict[Path, int]) -> list[Path]:
    """回傳已經寫完的新檔案（大小和上一次輪詢相同）。"""
    ready = []
    current = {
        p: p.stat().st_size for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS
    }
    for p, size in current.items():
        if size > 0 and sizes.get(p) == size:
            ready.append(p)
    sizes.clear()
    sizes.update({p: s for p, s in current.items() if p not in ready})
    return sorted(ready, key=lambda p: p.stat().st_mtime)


def move(path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / path.name
    if target.exists():
        target = dest / f"{path.stem}-{int(time.time() * 1000)}{path.suffix}"
    shutil.move(str(path), target)
    return target


def run(cam: Camera, folder: Path, interval: float, once: bool = False) -> None:
    processed, failed = folder / "processed", folder / "failed"
    folder.mkdir(parents=True, exist_ok=True)
    sizes: dict[Path, int] = {}
    log.info("監看 %s，每 %.1f 秒檢查一次，送到 /recognitions/%s", folder, interval, cam.kind)
    while True:
        for path in stable_files(folder, sizes):
            try:
                r = cam.send(path)
            except httpx.HTTPError as exc:
                log.warning("送出 %s 失敗（%s），下次再試", path.name, exc)
                continue
            if r.status_code == 201:
                body = r.json()
                if cam.kind == "product":
                    fine = body["prediction"]["fine"]
                    summary = f"{fine['label']} {fine['confidence']:.2f}"
                else:
                    summary = f"{body['item_count']} 個商品、{len(body['gaps'])} 個空位"
                log.info(
                    "%s -> %s（%s，%.0f ms）", path.name, summary, body["id"], body["latency_ms"]
                )
                move(path, processed)
            else:
                log.warning("%s 被拒絕：%s %s", path.name, r.status_code, r.text[:200])
                target = move(path, failed)
                target.with_suffix(target.suffix + ".error.txt").write_text(
                    f"{r.status_code}\n{r.text}\n", encoding="utf-8"
                )
        if once and not sizes:
            return
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", default=os.environ.get("CAMERA_WATCH_DIR", "camera_inbox"))
    ap.add_argument("--api", default=os.environ.get("API_BASE", "http://127.0.0.1:8000"))
    ap.add_argument(
        "--type", choices=["product", "shelf"], default=os.environ.get("CAMERA_TYPE", "product")
    )
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    username, password = os.environ.get("CAMERA_USERNAME"), os.environ.get("CAMERA_PASSWORD")
    if not username or not password:
        print("請設定 CAMERA_USERNAME 與 CAMERA_PASSWORD 環境變數", file=sys.stderr)
        return 1
    cam = Camera(args.api, username, password, args.type)
    # 啟動時 API 可能還沒好：重試登入
    for _ in range(30):
        try:
            cam.login()
            break
        except httpx.HTTPError:
            time.sleep(2)
    else:
        print(f"無法登入 {args.api}", file=sys.stderr)
        return 1
    run(cam, Path(args.watch), args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
