"""annotate_gaps 的 --set 參數：放在子指令前後都要選到正確的標註檔與切分（Codex 第 7 輪補充）。

調參集必須來自 SKU-110K val、評估集來自 test，選錯會讓調參與評估混在一起。
"""

from __future__ import annotations

import pytest

from training import annotate_gaps


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["review"], "eval"),
        (["--set", "tune", "review"], "tune"),
        (["review", "--set", "tune"], "tune"),
        (["propose", "--set", "tune", "--n", "3"], "tune"),
        (["--set", "eval", "propose"], "eval"),
    ],
)
def test_set_option_position(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: str
) -> None:
    seen: dict[str, object] = {}

    def record(*_args: object) -> None:
        seen["out"], seen["split"] = annotate_gaps.OUT, annotate_gaps.SPLIT_FILE

    monkeypatch.setattr(annotate_gaps, "propose", record)
    monkeypatch.setattr(annotate_gaps, "review", record)
    monkeypatch.setattr(annotate_gaps, "OUT", annotate_gaps.OUT)
    monkeypatch.setattr(annotate_gaps, "SPLIT_FILE", annotate_gaps.SPLIT_FILE)
    annotate_gaps.main(argv)
    out, split = annotate_gaps.SETS[expected]
    assert (seen["out"], seen["split"]) == (out, split)
    assert split == ("sku110k_val.txt" if expected == "tune" else "sku110k_test.txt")
