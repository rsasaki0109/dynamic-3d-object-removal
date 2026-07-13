from __future__ import annotations

import zipfile

import numpy as np
import pytest

from scripts import run_dynamicmap_benchmark as dynamicmap


def test_archive_uncompressed_bytes_counts_members(tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.bin", b"a" * 10)
        zf.writestr("nested/b.bin", b"b" * 7)
    assert dynamicmap._archive_uncompressed_bytes(archive) == 17


def test_extract_capacity_fails_before_unpacking(tmp_path, monkeypatch):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("large.bin", b"x" * 32)
    monkeypatch.setattr(
        dynamicmap.shutil,
        "disk_usage",
        lambda _: dynamicmap.shutil._ntuple_diskusage(total=64, used=63, free=1),
    )
    with pytest.raises(SystemExit, match="Refusing to extract"):
        dynamicmap._require_extract_capacity(archive, tmp_path)


def test_run_methods_can_select_fusion_only(monkeypatch):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    keep = np.array([True, False])
    monkeypatch.setattr(
        dynamicmap.core,
        "clean_map_by_fusion",
        lambda map_points, scans, workers: (map_points[keep], keep),
    )
    for name in ("clean_map_by_visibility", "clean_map_by_scan_ratio"):
        monkeypatch.setattr(
            dynamicmap.core,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(f"unexpected {_name}"),
        )

    result = dynamicmap._run_methods(
        points,
        [(points, np.zeros(3))],
        [(0, 2)],
        h_res=1.0,
        v_res=1.0,
        range_margin=0.5,
        min_see_through=3,
        max_surface_hits=3,
        resolutions=None,
        voxel_size=0.1,
        temporal_min_hits=2,
        sr_min_votes=None,
        fusion_workers=1,
        height_candidate_ablation=False,
        height_xy_cell=2.0,
        height_coarse_z_bin=0.5,
        height_fine_z_bin=0.25,
        height_min_cell_height=0.5,
        height_ground_margin=0.2,
        height_min_visits=3,
        height_max_persistence=1.0,
        methods=["fusion"],
    )
    assert list(result) == ["fusion"]
    np.testing.assert_array_equal(result["fusion"], points[keep])
