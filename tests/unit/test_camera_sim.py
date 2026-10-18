"""模擬相機的檔案處理（Codex Phase 3 第 2 輪）：只處理監看資料夾直接底下的一般圖片檔。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.camera_sim import NotRegularFileError, read_regular_file, reject, scan, stable_files


def _symlink(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError:  # Windows 沒有開發人員模式時不能建符號連結
        pytest.skip("這個環境不能建立符號連結")


def test_scan_accepts_regular_images_and_skips_others(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    (tmp_path / "notes.txt").write_text("略過")
    (tmp_path / "processed").mkdir()
    files, rejected = scan(tmp_path)
    assert files == {tmp_path / "a.jpg": 10}
    assert rejected == []


def test_symlink_rejected_and_target_untouched(tmp_path: Path) -> None:
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("資料夾外的檔案")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _symlink(secret, inbox / "evil.jpg")

    files, rejected = scan(inbox)
    assert files == {} and rejected == [inbox / "evil.jpg"]
    # 兩次輪詢也不會被當成寫完的檔案送出
    sizes: dict[Path, int] = {}
    assert stable_files(inbox, sizes) == [] and stable_files(inbox, sizes) == []

    reject(inbox / "evil.jpg", inbox / "failed", "不是一般檔案")
    moved = inbox / "failed" / "evil.jpg"
    assert moved.is_symlink()  # 移動的是連結本身
    assert (inbox / "failed" / "evil.jpg.error.txt").exists()
    assert secret.read_text() == "資料夾外的檔案"


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="Windows 沒有 O_NOFOLLOW")
def test_read_regular_file_does_not_follow_symlink(tmp_path: Path) -> None:
    """檢查之後才被換成連結（競爭）時，開檔本身也會拒絕。"""
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"secret")
    link = tmp_path / "swap.jpg"
    _symlink(secret, link)
    with pytest.raises(NotRegularFileError):
        read_regular_file(link)
    assert read_regular_file(secret) == b"secret"
