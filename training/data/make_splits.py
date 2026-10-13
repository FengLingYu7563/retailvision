"""產生可重現的資料切分檔，輸出到 training/data/splits/（會 commit）。

- Grocery Store：沿用官方 train/val/test 清單，只做正規化（統一分隔符、排序），
  讓結果能和論文比較。類別表另存 grocery_classes.json。
- SKU-110K：完整訓練集 8,000+ 張對 4 GB VRAM 太重，用固定 seed 抽樣子集。
  test 用完整官方測試集，指標才有代表性。

重跑必須產生位元組完全相同的檔案（驗收 1.1），所以：排序後再抽樣、固定 seed、
固定換行字元、不寫入時間戳。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SPLITS = HERE / "splits"
GROCERY = RAW / "GroceryStoreDataset" / "dataset"
SKU = RAW / "SKU110K_fixed"

SEED = 42
SKU_SUBSET = {"train": 3000, "val": 500}  # test 用完整集合


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n"：Windows 與 Linux 產出相同位元組
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def grocery_splits() -> None:
    classes: list[dict[str, object]] = []
    with (GROCERY / "classes.csv").open(encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            classes.append(
                {
                    "fine_id": int(row[1]),
                    "fine": row[0].strip(),
                    "coarse_id": int(row[3]),
                    "coarse": row[2].strip(),
                    "iconic": row[4].strip().lstrip("/"),
                }
            )
    classes.sort(key=lambda c: int(c["fine_id"]))  # type: ignore[call-overload]
    coarse_names: dict[int, str] = {}
    for c in classes:
        coarse_names[int(c["coarse_id"])] = str(c["coarse"])  # type: ignore[call-overload]
    meta = {
        "fine_names": [c["fine"] for c in classes],
        "coarse_names": [coarse_names[i] for i in sorted(coarse_names)],
        "fine_to_coarse": [c["coarse_id"] for c in classes],
        "iconic_images": [c["iconic"] for c in classes],
    }
    SPLITS.mkdir(parents=True, exist_ok=True)
    with (SPLITS / "grocery_classes.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    for split in ("train", "val", "test"):
        rows: list[str] = []
        with (GROCERY / f"{split}.txt").open(encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3 or not parts[0]:
                    continue
                path, fine_id, coarse_id = parts[0].replace("\\", "/"), int(parts[1]), int(parts[2])
                if meta["fine_to_coarse"][fine_id] != coarse_id:
                    raise ValueError(f"官方清單的粗類與 classes.csv 不一致：{line!r}")
                if not (GROCERY / path).exists():
                    raise FileNotFoundError(GROCERY / path)
                rows.append(f"{path},{fine_id},{coarse_id}")
        rows.sort()
        write_lines(SPLITS / f"grocery_{split}.csv", ["path,fine_id,coarse_id", *rows])
        print(f"grocery {split}: {len(rows)} 張")


def sku_images(split: str) -> list[str]:
    names: set[str] = set()
    with (SKU / "annotations" / f"annotations_{split}.csv").open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                names.add(row[0].strip())
    # 只保留實際存在的圖片（官方資料有少數標註指向缺漏的檔案）
    return sorted(n for n in names if (SKU / "images" / n).exists())


def sku_splits() -> None:
    # 每個 split 用獨立的 Random 實例：改動某一個 split 的數量不會影響其他 split
    for split in ("train", "val", "test"):
        images = sku_images(split)
        n = SKU_SUBSET.get(split)
        if n is not None and n < len(images):
            images = sorted(random.Random(f"{SEED}-{split}").sample(images, n))
        write_lines(SPLITS / f"sku110k_{split}.txt", images)
        print(f"sku110k {split}: {len(images)} 張")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["grocery", "sku110k", "all"], default="all")
    args = parser.parse_args()
    if args.dataset in ("grocery", "all"):
        grocery_splits()
    if args.dataset in ("sku110k", "all"):
        if not SKU.exists():
            print("找不到 SKU-110K，請先執行 download_sku110k.py", file=sys.stderr)
            return 1
        sku_splits()
    return 0


if __name__ == "__main__":
    sys.exit(main())
