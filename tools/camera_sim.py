"""模擬相機（驗收 3.4）：監看資料夾，有新圖片就送去辨識。

用公開資料集的照片代替真的相機（規格第 2 節）。為什麼用輪詢而不用 watchdog：
少一個依賴，而且 Docker 在 Windows / macOS 的 bind mount 上，檔案系統事件常常收不到。

- 新檔案要「寫完」才送出：連續兩次輪詢檔案大小相同才算寫完，避免送出寫到一半的圖。
- 送出後移到 processed/ 或 failed/（失敗時另外寫一個 .error.txt 說明原因），不會重複送。
- 帳密從環境變數讀（CAMERA_USERNAME / CAMERA_PASSWORD），不寫在程式或命令列；
  token 過期會自動重新登入。請用一般使用者帳號（最小權限），不要用 admin：
  相機是長時間執行、監看可寫入資料夾的服務，token 外洩時影響範圍要越小越好。
- 只處理監看資料夾「直接底下」的一般檔案：符號連結、特殊檔案、跑出資料夾的路徑一律拒絕，
  否則能寫入資料夾的人可以放一個連結，讓相機把資料夾外的檔案上傳出去。

  python -m tools.camera_sim --watch camera_inbox --type product
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

import httpx

log = logging.getLogger("camera_sim")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# 和 API 的單張上限一致（MAX_UPLOAD_BYTES 預設 10 MB）：超過的檔案送了也會被 API 拒絕，
# 在相機端就擋下，避免有人丟一個超大檔案讓長時間執行的相機吃光記憶體
MAX_BYTES = int(os.environ.get("CAMERA_MAX_BYTES", 10 * 1024 * 1024))


class NotRegularFileError(Exception):
    """不能處理的檔案：符號連結、特殊檔案、路徑跑出資料夾、超過大小上限。"""


def read_regular_file(path: Path, max_bytes: int = MAX_BYTES) -> bytes:
    """讀檔，但不跟隨符號連結，也不讀超過 max_bytes 的內容。

    O_NOFOLLOW（Linux / macOS）讓 open 遇到符號連結直接失敗，避免「檢查時是一般檔案、
    開啟前被換成連結」的競爭；開啟後再用 fstat 確認是一般檔案。Windows 沒有 O_NOFOLLOW，
    靠 scan() 的 lstat 檢查。大小在開檔後用 fstat 檢查，讀取時最多讀 max_bytes + 1，
    檢查之後檔案才變大也不會整個讀進記憶體。
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise  # 輪詢之間被移走或刪掉：不是錯誤，呼叫端略過
    except OSError as exc:
        raise NotRegularFileError(f"無法以不跟隨連結的方式開啟：{exc}") from exc
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise NotRegularFileError("不是一般檔案")
        if st.st_size > max_bytes:
            raise NotRegularFileError(f"檔案超過 {max_bytes} bytes 上限")
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise NotRegularFileError(f"檔案超過 {max_bytes} bytes 上限")
    return data


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
        data = read_regular_file(path)
        for attempt in range(2):
            r = self.client.post(
                f"{self.api}/recognitions/{self.kind}",
                files={"file": (path.name, data, "application/octet-stream")},
                headers={"Authorization": f"Bearer {self.token}"},
            )
            if r.status_code == 401 and attempt == 0:  # token 過期：重新登入再送一次
                self.login()
                continue
            return r
        return r


def scan(
    folder: Path, max_bytes: int = MAX_BYTES
) -> tuple[dict[Path, int], list[tuple[Path, str]]]:
    """列出資料夾直接底下的圖片：回傳（一般檔案 -> 大小、要拒絕的項目與原因）。

    用 lstat（不跟隨連結）判斷；子資料夾（processed/、failed/）與非圖片副檔名直接略過。
    列出之後才被刪掉或改名的項目直接略過，下次輪詢再看，不讓相機中止。
    """
    root = folder.resolve()
    files: dict[Path, int] = {}
    rejected: list[tuple[Path, str]] = []
    for p in folder.iterdir():
        if p.suffix.lower() not in EXTS:
            continue
        try:
            st = os.lstat(p)
            if stat.S_ISDIR(st.st_mode):
                continue
            if not stat.S_ISREG(st.st_mode) or p.resolve().parent != root:
                rejected.append((p, "不是一般檔案（符號連結或特殊檔案），不處理"))
                continue
        except FileNotFoundError:
            continue
        if st.st_size > max_bytes:
            rejected.append((p, f"檔案超過 {max_bytes} bytes 上限"))
            continue
        files[p] = st.st_size
    return files, rejected


def _mtime(p: Path) -> float:
    try:
        return os.lstat(p).st_mtime
    except FileNotFoundError:
        return 0.0


def stable_files(folder: Path, sizes: dict[Path, int]) -> list[Path]:
    """回傳已經寫完的新檔案（大小和上一次輪詢相同）。"""
    ready = []
    current, _ = scan(folder)
    for p, size in current.items():
        if size > 0 and sizes.get(p) == size:
            ready.append(p)
    sizes.clear()
    sizes.update({p: s for p, s in current.items() if p not in ready})
    return sorted(ready, key=_mtime)


def move(path: Path, dest: Path) -> Path:
    """移到 processed/ 或 failed/。符號連結會移動連結本身（rename），不會碰到它指向的檔案。"""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / path.name
    if target.exists() or target.is_symlink():
        target = dest / f"{path.stem}-{int(time.time() * 1000)}{path.suffix}"
    shutil.move(str(path), target)
    return target


def reject(path: Path, failed: Path, reason: str) -> None:
    log.warning("%s 被拒絕：%s", path.name, reason[:200])
    target = move(path, failed)
    target.with_name(target.name + ".error.txt").write_text(reason + "\n", encoding="utf-8")


def prepare(folder: Path) -> None:
    """建立監看資料夾與 processed/、failed/，並確認可以寫入；不行就丟出說明清楚的錯誤。"""
    try:
        for d in (folder, folder / "processed", folder / "failed"):
            d.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(f"無法在 {folder} 建立資料夾：{exc}") from exc
    # 實際試寫，不用 os.access：Docker Desktop（Windows / macOS）的 bind mount 權限位元
    # 和實際能不能寫不一致，os.access 會誤判
    for d in (folder, folder / "processed", folder / "failed"):
        try:
            with tempfile.NamedTemporaryFile(dir=d, prefix=".write-test-"):
                pass
        except OSError as exc:
            raise PermissionError(f"{d} 沒有寫入權限：{exc}") from exc


def run(cam: Camera, folder: Path, interval: float, once: bool = False) -> None:
    processed, failed = folder / "processed", folder / "failed"
    sizes: dict[Path, int] = {}
    log.info("監看 %s，每 %.1f 秒檢查一次，送到 /recognitions/%s", folder, interval, cam.kind)
    while True:
        for path, reason in scan(folder)[1]:
            with contextlib.suppress(FileNotFoundError):
                reject(path, failed, reason)
        for path in stable_files(folder, sizes):
            try:
                r = cam.send(path)
            except FileNotFoundError:
                continue  # 送出前被移走：略過
            except NotRegularFileError as exc:
                with contextlib.suppress(FileNotFoundError):
                    reject(path, failed, str(exc))
                continue
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
                reject(path, failed, f"{r.status_code}\n{r.text}")
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
        print(
            "請設定 CAMERA_USERNAME 與 CAMERA_PASSWORD（建議另建一般使用者帳號，不要用 admin）",
            file=sys.stderr,
        )
        return 1
    try:
        prepare(Path(args.watch))
    except PermissionError as exc:
        uid = os.getuid() if hasattr(os, "getuid") else "?"
        print(
            f"{exc}。目前 uid {uid}；docker compose 請把 .env 的 CAMERA_UID / CAMERA_GID "
            "設成監看資料夾擁有者的 id -u / id -g",
            file=sys.stderr,
        )
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
