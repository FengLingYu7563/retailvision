"""驗收 1.1：切分腳本重跑兩次，輸出位元組完全相同，也和 repo 裡 commit 的切分檔相同。"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.data import make_splits

COMMITTED = make_splits.SPLITS


def _run(monkeypatch: pytest.MonkeyPatch, out: Path, fn_name: str) -> dict[str, bytes]:
    monkeypatch.setattr(make_splits, "SPLITS", out)
    getattr(make_splits, fn_name)()
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


@pytest.mark.needs_data
def test_grocery_split_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _run(monkeypatch, tmp_path / "a", "grocery_splits")
    b = _run(monkeypatch, tmp_path / "b", "grocery_splits")
    assert a == b
    for name, content in a.items():
        assert (COMMITTED / name).read_bytes() == content, name


@pytest.mark.needs_data
def test_sku_split_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not make_splits.SKU.exists():
        pytest.skip("缺少 SKU-110K")
    a = _run(monkeypatch, tmp_path / "a", "sku_splits")
    b = _run(monkeypatch, tmp_path / "b", "sku_splits")
    assert a == b
    assert len(a["sku110k_train.txt"].split()) == make_splits.SKU_SUBSET["train"]
    for name, content in a.items():
        assert (COMMITTED / name).read_bytes() == content, name
