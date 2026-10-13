"""YOLOv8n 貨架偵測訓練（SKU-110K 子集）。

步驟：
1. 依 splits/sku110k_*.txt 建立 Ultralytics 格式資料夾（圖片用硬連結，不多佔 GB 級空間）。
2. 從 COCO 預訓練的 yolov8n.pt 開始訓練。每個 epoch 存 checkpoint，可用 --resume 接續。

為什麼選 YOLOv8n：4 GB VRAM 在 640 輸入下 batch 8 可以訓練，CPU 推論也在 1 秒內；
v8s 準確率略高但 VRAM 只能用 batch 4，訓練時間約兩倍，列為不達標時的備案。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

from training.common import CHECKPOINTS, DATA, SEED, SKU, SPLITS

# 關掉 Ultralytics 自動 pip 安裝缺少套件的行為：所有安裝都要先經過使用者同意
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

YOLO_DIR = DATA / "processed" / "sku110k_yolo"
MAX_DET = 1000


def read_annotations(split: str) -> dict[str, list[list[str]]]:
    by_image: dict[str, list[list[str]]] = defaultdict(list)
    with (SKU / "annotations" / f"annotations_{split}.csv").open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                by_image[row[0].strip()].append(row)
    return by_image


def to_yolo_line(row: list[str]) -> str | None:
    # 欄位：image_name, x1, y1, x2, y2, class, image_width, image_height
    x1, y1, x2, y2 = (float(v) for v in row[1:5])
    W, H = float(row[6]), float(row[7])
    x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
    y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (
        f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
    )


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_dataset() -> Path:
    for split in ("train", "val", "test"):
        names = (SPLITS / f"sku110k_{split}.txt").read_text(encoding="utf-8").split()
        ann = read_annotations(split)
        img_dir, lbl_dir = YOLO_DIR / "images" / split, YOLO_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            link_or_copy(SKU / "images" / name, img_dir / name)
            lines = [ln for r in ann.get(name, []) if (ln := to_yolo_line(r))]
            (lbl_dir / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        print(f"{split}: {len(names)} 張")
    yaml_path = YOLO_DIR / "sku110k.yaml"
    yaml_path.write_text(
        f"path: {YOLO_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: object\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--name", default="detector_v8n")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = prepare_dataset()
    run_dir = CHECKPOINTS / args.name
    if args.resume and (run_dir / "weights" / "last.pt").exists():
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
        return
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,  # Windows 上 worker 多了反而因為 spawn 開銷變慢，也比較吃記憶體
        amp=True,
        seed=SEED,
        deterministic=True,
        max_det=MAX_DET,
        project=str(CHECKPOINTS),
        name=args.name,
        exist_ok=True,
        # Ultralytics 每個 epoch 都會覆寫 weights/last.pt，筆電過熱中斷時用 --resume 接續
        patience=15,
        cache=False,
        plots=True,
        # 貨架照片上下顛倒沒有意義，但左右翻轉合理；mosaic 對密集小物件有幫助，保留預設
        fliplr=0.5,
        flipud=0.0,
    )


if __name__ == "__main__":
    main()
