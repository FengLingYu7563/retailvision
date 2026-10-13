"""驗收 1.6：ONNX 與 PyTorch 輸出一致（logits 最大誤差 < 1e-3、top-1 一致率 100%）。

需要 torch、本機權重與資料集，CI 自動 skip；本機必須跑過。
用真實的 val 圖片而不是亂數輸入：亂數輸入的 logits 分布和真實圖片差很多，
可能掩蓋只在真實資料上出現的數值問題。
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from tests.conftest import CHECKPOINT_DIR, GROCERY_DIR, MODELS_DIR, ROOT
from vision.io import load_image
from vision.preprocess import to_classifier_input

pytestmark = [pytest.mark.needs_weights, pytest.mark.needs_data]

VERSION = "cls-mnv3-v1"


def test_classifier_onnx_matches_torch() -> None:
    import torch

    from training.export_onnx import load_classifier
    from vision.classifier import OnnxClassifier

    onnx_path = MODELS_DIR / f"{VERSION}.onnx"
    if not onnx_path.exists():
        pytest.skip("尚未匯出 ONNX")
    model = load_classifier(CHECKPOINT_DIR / "classifier_mnv3" / "best.pt")
    clf = OnnxClassifier(onnx_path)

    with (ROOT / "training" / "data" / "splits" / "grocery_val.csv").open(encoding="utf-8") as f:
        paths = [r["path"] for r in csv.DictReader(f)]
    batch = np.concatenate([to_classifier_input(load_image(GROCERY_DIR / p)) for p in paths])

    with torch.no_grad():
        t_fine, t_coarse, t_emb = (t.numpy() for t in model(torch.from_numpy(batch)))
    o_fine, o_coarse, o_emb = clf.logits(batch)

    assert np.abs(t_fine - o_fine).max() < 1e-3
    assert np.abs(t_coarse - o_coarse).max() < 1e-3
    assert np.abs(t_emb - o_emb).max() < 1e-3
    assert (t_fine.argmax(1) == o_fine.argmax(1)).all()
    assert (t_coarse.argmax(1) == o_coarse.argmax(1)).all()
