"""訓練腳本共用：路徑、切分檔讀取、指標檔合併寫入。"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data"
SPLITS = DATA / "splits"
GROCERY = DATA / "raw" / "GroceryStoreDataset" / "dataset"
SKU = DATA / "raw" / "SKU110K_fixed"
MODELS = ROOT / "models"
CHECKPOINTS = ROOT / "checkpoints"
METRICS = MODELS / "metrics.json"
SEED = 42


@dataclass(frozen=True)
class Sample:
    path: Path
    fine_id: int
    coarse_id: int


def grocery_meta() -> dict[str, Any]:
    with (SPLITS / "grocery_classes.json").open(encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def grocery_split(split: str) -> list[Sample]:
    with (SPLITS / f"grocery_{split}.csv").open(encoding="utf-8") as f:
        return [
            Sample(GROCERY / r["path"], int(r["fine_id"]), int(r["coarse_id"]))
            for r in csv.DictReader(f)
        ]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def update_metrics(section: str, values: dict[str, Any]) -> None:
    """合併寫入 models/metrics.json：各訓練腳本各自更新自己的區塊，不互相覆蓋。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if METRICS.exists():
        data = json.loads(METRICS.read_text(encoding="utf-8"))
    data[section] = values
    METRICS.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
