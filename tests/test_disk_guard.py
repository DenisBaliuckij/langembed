"""Tests for the disk-space guard used by long-running write loops."""

from __future__ import annotations

import shutil

import pytest

from langembed.disk_guard import (
    DiskSpaceError,
    check_free_space,
    estimate_jsonl_embedding_bytes,
)


def test_check_free_space_passes_when_above_minimum(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(100, 50, 50 * 1024**3)
    )
    check_free_space(tmp_path, min_free_gb=10)


def test_check_free_space_raises_when_below_minimum(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(100, 99, 1 * 1024**3)
    )
    with pytest.raises(DiskSpaceError, match="only 1.00GB free"):
        check_free_space(tmp_path, min_free_gb=5)


def test_check_free_space_accounts_for_reserve_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(100, 10, 10 * 1024**3)
    )
    # 10GB free, reserving 8GB for the estimated output leaves 2GB < 5GB minimum.
    with pytest.raises(DiskSpaceError):
        check_free_space(tmp_path, min_free_gb=5, reserve_bytes=8 * 1024**3)


def test_estimate_jsonl_embedding_bytes_scales_with_rows_and_dim():
    small = estimate_jsonl_embedding_bytes(n_rows=100, dim=256)
    large = estimate_jsonl_embedding_bytes(n_rows=1000, dim=256)
    assert large == small * 10
    assert small > 0
