"""下載並解壓 SKU-110K（Goldman et al., CVPR 2019）。

來源是論文作者提供的 S3 bucket（Ultralytics SKU-110K.yaml 也用同一個網址），約 12.2 GB。

為什麼用多連線分段下載：實測這個 bucket 每條連線限速約 250 KB/s，單線要 13 小時以上；
多條 Range 請求可以疊加頻寬。每段寫成獨立的 part 檔，中斷後重跑只會續傳沒完成的部分。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
EXPECTED_SIZE = 12_196_152_718
RAW = Path(__file__).resolve().parent / "raw"
ARCHIVE = RAW / "SKU110K_fixed.tar.gz"
PARTS_DIR = RAW / "SKU110K_parts"
DEST = RAW / "SKU110K_fixed"
CHUNK = 1 << 20


def fetch_part(index: int, start: int, end: int, retries: int = 20) -> None:
    """下載 [start, end] 位元組到 part 檔，已存在的部分直接續傳。"""
    part = PARTS_DIR / f"part_{index:03d}"
    want = end - start + 1
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            part.unlink()
            have = 0
        req = urllib.request.Request(URL, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, part.open("ab") as f:
                if resp.status != 206:
                    raise RuntimeError(f"伺服器不支援 Range（HTTP {resp.status}）")
                while block := resp.read(CHUNK):
                    f.write(block)
        except (OSError, RuntimeError) as exc:
            print(f"part {index} 第 {attempt + 1} 次失敗：{exc}，重試", flush=True)
            time.sleep(min(60, 2**attempt))
    if part.stat().st_size != want:
        raise RuntimeError(f"part {index} 重試 {retries} 次仍未完成")


def progress() -> int:
    return sum(p.stat().st_size for p in PARTS_DIR.glob("part_*"))


def download(connections: int) -> None:
    if ARCHIVE.exists() and ARCHIVE.stat().st_size == EXPECTED_SIZE:
        return
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    size = -(-EXPECTED_SIZE // connections)
    ranges = [(i, i * size, min(EXPECTED_SIZE, (i + 1) * size) - 1) for i in range(connections)]
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(fetch_part, *r) for r in ranges]
        while not all(f.done() for f in futures):
            time.sleep(30)
            print(f"進度 {progress() / EXPECTED_SIZE:6.1%}", flush=True)
        for f in futures:
            f.result()
    with ARCHIVE.open("wb") as out:
        for i, _, _ in ranges:
            with (PARTS_DIR / f"part_{i:03d}").open("rb") as src:
                shutil.copyfileobj(src, out, 16 * CHUNK)
    if ARCHIVE.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError("合併後大小不符")
    shutil.rmtree(PARTS_DIR)


def extract() -> None:
    if (DEST / "annotations").exists() and (DEST / "images").exists():
        return
    print(f"解壓 {ARCHIVE} ...", flush=True)
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        # filter="data" 拒絕絕對路徑與 ../ 之類的成員，防止解壓時路徑穿越
        tar.extractall(RAW, filter="data")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connections", type=int, default=16)
    args = parser.parse_args()
    download(args.connections)
    extract()
    print(f"SKU-110K 就緒：{DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
