"""驗收 1.7 保留類別實驗：訓練程式完全不讀取保留類別的任何圖（Codex 第 6 輪審查補充）。

攔截 train_metric 裡的 load_image，記錄所有被讀取的路徑，
確認保留類別的 train、val、標準圖都沒有被讀到（包含當背景素材）。
只需要 commit 過的切分檔與類別表，不需要真的讀圖，所以 CI 只要有 torch 就能跑。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_prepare_data_never_reads_holdout_images(monkeypatch: pytest.MonkeyPatch) -> None:
    from training import train_metric
    from training.common import GROCERY, grocery_meta, grocery_split

    read: list[Path] = []

    def fake_load(path: object) -> np.ndarray:
        read.append(Path(str(path)))
        return np.zeros((32, 32, 3), np.uint8)

    monkeypatch.setattr(train_metric, "load_image", fake_load)
    meta = grocery_meta()
    holdout = train_metric.holdout_classes(20, len(meta["fine_names"]))
    assert len(holdout) == 20

    train, val, iconic, backgrounds = train_metric.prepare_data(holdout)

    forbidden = {
        s.path
        for split in ("train", "val", "test")
        for s in grocery_split(split)
        if s.fine_id in holdout
    }
    forbidden |= {GROCERY / meta["iconic_images"][f] for f in holdout}
    assert read, "應該要讀到訓練範圍內的圖"
    assert not set(read) & forbidden
    assert all(s.fine_id not in holdout for s in train + val)
    assert not set(iconic) & set(holdout)
    assert len(backgrounds) == 60


def test_holdout_classes_are_fixed() -> None:
    from training.train_metric import holdout_classes

    # 固定 seed：重跑實驗時保留的是同一批類別
    assert holdout_classes(20, 81) == holdout_classes(20, 81)
    assert holdout_classes(0, 81) == []
