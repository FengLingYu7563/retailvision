"""下載 Grocery Store Dataset（Klasson et al., WACV 2019）。

資料集授權為研究用途，不可重新散布，因此只放在 training/data/raw/（已 gitignore）。
固定到特定 commit：上游 repo 若更新，切分清單與指標才不會悄悄改變。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/marcusklasson/GroceryStoreDataset.git"
PINNED_COMMIT = "fc80ba90f803d79d0383df52c5a4ac5de99ff6fc"
DEST = Path(__file__).resolve().parent / "raw" / "GroceryStoreDataset"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    if not DEST.exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        print(f"clone {REPO_URL} -> {DEST}")
        run(["git", "clone", REPO_URL, str(DEST)])
    head = run(["git", "rev-parse", "HEAD"], cwd=DEST)
    if head != PINNED_COMMIT:
        # 淺層 clone 可能沒有該 commit，先補抓再切換
        run(["git", "fetch", "--depth", "1", "origin", PINNED_COMMIT], cwd=DEST)
        run(["git", "checkout", PINNED_COMMIT], cwd=DEST)
    print(f"Grocery Store Dataset 就緒：{DEST}（commit {PINNED_COMMIT[:10]}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
