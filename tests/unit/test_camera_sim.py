"""模擬相機的檔案處理（Codex Phase 3 第 2 輪）：只處理監看資料夾直接底下的一般圖片檔。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.camera_sim import (
    NotRegularFileError,
    prepare,
    read_regular_file,
    reject,
    scan,
    stable_files,
)


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
    assert files == {} and [p for p, _ in rejected] == [inbox / "evil.jpg"]
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


def test_oversized_file_rejected_without_reading(tmp_path: Path) -> None:
    (tmp_path / "big.jpg").write_bytes(b"x" * 101)
    (tmp_path / "ok.jpg").write_bytes(b"x" * 100)
    files, rejected = scan(tmp_path, max_bytes=100)
    assert files == {tmp_path / "ok.jpg": 100}
    assert [(p.name, "上限" in reason) for p, reason in rejected] == [("big.jpg", True)]
    # 檢查之後才變大：讀取時也有上限
    with pytest.raises(NotRegularFileError, match="上限"):
        read_regular_file(tmp_path / "big.jpg", max_bytes=100)
    assert read_regular_file(tmp_path / "ok.jpg", max_bytes=100) == b"x" * 100


def test_scan_skips_files_removed_between_listing_and_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "gone.jpg").write_bytes(b"x")
    (tmp_path / "ok.jpg").write_bytes(b"x")
    real_lstat = os.lstat

    def flaky_lstat(p: os.PathLike[str] | str) -> os.stat_result:
        if Path(p).name == "gone.jpg":
            raise FileNotFoundError(p)
        return real_lstat(p)

    monkeypatch.setattr("tools.camera_sim.os.lstat", flaky_lstat)
    files, rejected = scan(tmp_path)
    assert list(files) == [tmp_path / "ok.jpg"] and rejected == []


def test_prepare_creates_subfolders(tmp_path: Path) -> None:
    prepare(tmp_path / "inbox")
    assert (tmp_path / "inbox" / "processed").is_dir() and (tmp_path / "inbox" / "failed").is_dir()


@pytest.mark.skipif(
    not hasattr(os, "getuid") or os.getuid() == 0, reason="需要 POSIX 權限且不是 root"
)
def test_prepare_reports_unwritable_folder(tmp_path: Path) -> None:
    """乾淨的 Linux clone：資料夾不屬於容器使用者時，啟動就說明原因，不要跑到一半才失敗。"""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    inbox.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            prepare(inbox)
    finally:
        inbox.chmod(0o755)
