"""Comprehensive tests for dynamic_object_removal module."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import struct

from dynamic_object_removal import (
    DetectionBox,
    TemporalConsistencyFilter,
    load_boxes,
    load_points,
    load_pcd_scan,
    remove_points_in_boxes,
    save_points,
    main,
    _parse_kitti_calib,
)
import bench


# ---------------------------------------------------------------------------
# DetectionBox
# ---------------------------------------------------------------------------

class TestDetectionBox:
    def test_construction_defaults(self):
        box = DetectionBox(
            center=np.array([1.0, 2.0, 3.0]),
            size=np.array([0.5, 0.5, 0.5]),
        )
        assert box.yaw == 0.0
        assert box.label is None
        np.testing.assert_array_equal(box.center, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(box.size, [0.5, 0.5, 0.5])

    def test_construction_with_yaw_and_label(self):
        box = DetectionBox(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([1.0, 2.0, 3.0]),
            yaw=1.57,
            label="car",
        )
        assert box.yaw == pytest.approx(1.57)
        assert box.label == "car"

    def test_frozen(self):
        box = DetectionBox(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([1.0, 1.0, 1.0]),
        )
        with pytest.raises(AttributeError):
            box.yaw = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_points
# ---------------------------------------------------------------------------

class TestLoadPoints:
    def test_load_pcd_ascii(self, demo_pcd_path: Path):
        pts = load_points(demo_pcd_path, fmt="auto")
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.dtype == np.float64
        assert pts.shape[0] > 0

    def test_load_pcd_explicit_fmt(self, demo_pcd_path: Path):
        pts = load_points(demo_pcd_path, fmt="pcd")
        assert pts.shape[0] > 0

    def test_load_csv(self, tmp_path: Path):
        csv_file = tmp_path / "cloud.csv"
        csv_file.write_text("x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
        pts = load_points(csv_file, fmt="auto")
        assert pts.shape == (2, 3)
        np.testing.assert_allclose(pts[0], [1.0, 2.0, 3.0])

    def test_load_text_space_delimited(self, tmp_path: Path):
        txt_file = tmp_path / "cloud.txt"
        txt_file.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")
        pts = load_points(txt_file, fmt="text")
        assert pts.shape == (2, 3)

    def test_load_text_with_header(self, tmp_path: Path):
        txt_file = tmp_path / "cloud.xyz"
        txt_file.write_text("x y z intensity\n1.0 2.0 3.0 0.5\n4.0 5.0 6.0 0.8\n")
        pts = load_points(txt_file, fmt="auto")
        assert pts.shape == (2, 3)
        np.testing.assert_allclose(pts[1], [4.0, 5.0, 6.0])

    def test_load_npy(self, tmp_path: Path):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        npy_file = tmp_path / "cloud.npy"
        np.save(npy_file, arr)
        pts = load_points(npy_file, fmt="auto")
        assert pts.shape == (2, 3)
        np.testing.assert_allclose(pts, arr)

    def test_load_npy_extra_columns(self, tmp_path: Path):
        arr = np.array([[1.0, 2.0, 3.0, 0.5, 0.8], [4.0, 5.0, 6.0, 0.1, 0.2]])
        npy_file = tmp_path / "cloud.npy"
        np.save(npy_file, arr)
        pts = load_points(npy_file, fmt="npy")
        assert pts.shape == (2, 3)
        np.testing.assert_allclose(pts, arr[:, :3])

    def test_load_empty_file(self, tmp_path: Path):
        """A completely empty file (no header) returns 0 points."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        pts = load_points(empty_file, fmt="text")
        assert pts.shape == (0, 3)

    def test_load_header_only_csv_raises(self, tmp_path: Path):
        """A CSV with header but no data rows raises ValueError."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("x,y,z\n")
        with pytest.raises(ValueError, match="not enough numeric columns"):
            load_points(csv_file, fmt="auto")

    def test_load_kitti_bin(self, tmp_path: Path):
        """Load a KITTI-format .bin file (float32 x4 per point)."""
        points_expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        bin_file = tmp_path / "test.bin"
        with bin_file.open("wb") as f:
            for row in points_expected:
                f.write(struct.pack("ffff", row[0], row[1], row[2], 0.5))
        pts = load_points(bin_file, fmt="auto")
        assert pts.shape == (2, 3)
        assert pts.dtype == np.float64
        np.testing.assert_allclose(pts, points_expected, atol=1e-6)

    def test_load_kitti_bin_explicit_fmt(self, tmp_path: Path):
        bin_file = tmp_path / "cloud.dat"
        with bin_file.open("wb") as f:
            f.write(struct.pack("ffff", 1.0, 2.0, 3.0, 0.5))
        pts = load_points(bin_file, fmt="bin")
        assert pts.shape == (1, 3)

    def test_load_kitti_bin_bad_size(self, tmp_path: Path):
        bin_file = tmp_path / "bad.bin"
        bin_file.write_bytes(b"\x00" * 5)  # not divisible by 4*4=16
        with pytest.raises(ValueError, match="not divisible by 4"):
            load_points(bin_file, fmt="bin")

    def test_unsupported_format(self, tmp_path: Path):
        f = tmp_path / "cloud.bin"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="unsupported cloud format"):
            load_points(f, fmt="parquet")


# ---------------------------------------------------------------------------
# load_boxes
# ---------------------------------------------------------------------------

class TestLoadBoxes:
    def test_load_json_demo(self, demo_objects_path: Path):
        boxes = load_boxes(demo_objects_path, fmt="auto", skip_invalid=False)
        assert len(boxes) > 0
        for box in boxes:
            assert isinstance(box, DetectionBox)
            assert box.size.shape == (3,)
            assert np.all(box.size > 0)

    def test_load_json_explicit(self, demo_objects_path: Path):
        boxes = load_boxes(demo_objects_path, fmt="json", skip_invalid=False)
        assert len(boxes) > 0

    def test_empty_list(self, tmp_path: Path):
        f = tmp_path / "empty.json"
        f.write_text("[]")
        boxes = load_boxes(f, fmt="json", skip_invalid=False)
        assert boxes == []

    def test_skip_invalid(self, tmp_path: Path):
        data = [
            {"center": [1, 2, 3], "size": [1, 1, 1]},
            {"bad_key": "no_center"},
        ]
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data))
        boxes = load_boxes(f, fmt="json", skip_invalid=True)
        assert len(boxes) == 1

    def test_no_skip_invalid_raises(self, tmp_path: Path):
        data = [{"bad_key": "no_center"}]
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="invalid box entry"):
            load_boxes(f, fmt="json", skip_invalid=False)

    def test_objects_wrapper(self, tmp_path: Path):
        data = {"objects": [{"center": [1, 2, 3], "size": [1, 1, 1]}]}
        f = tmp_path / "wrapped.json"
        f.write_text(json.dumps(data))
        boxes = load_boxes(f, fmt="json", skip_invalid=False)
        assert len(boxes) == 1

    def test_unsupported_format(self, tmp_path: Path):
        f = tmp_path / "boxes.xml"
        f.write_text("<boxes/>")
        with pytest.raises(ValueError, match="unsupported box format"):
            load_boxes(f, fmt="xml", skip_invalid=False)


# ---------------------------------------------------------------------------
# KITTI format
# ---------------------------------------------------------------------------

class TestKITTI:
    def _write_calib(self, path: Path) -> None:
        """Write a standard KITTI-like calibration file."""
        calib_text = (
            "P0: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            "P1: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            "P2: 7.215377e+02 0 6.095593e+02 4.485728e+01 0 7.215377e+02 1.728540e+02 2.163791e-01 0 0 1 2.745884e-03\n"
            "P3: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            "R0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
            "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        )
        path.write_text(calib_text, encoding="utf-8")

    def test_parse_kitti_calib(self, tmp_path: Path):
        calib_file = tmp_path / "calib.txt"
        self._write_calib(calib_file)
        cam_to_velo = _parse_kitti_calib(calib_file)
        assert cam_to_velo.shape == (4, 4)
        # velo(1,0,0) -> cam(0,0,1) so inverse should map cam(0,0,1) -> velo(1,0,0)
        result = cam_to_velo @ np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(result[:3], [1.0, 0.0, 0.0], atol=1e-10)

    def test_parse_kitti_calib_missing(self, tmp_path: Path):
        calib_file = tmp_path / "bad_calib.txt"
        calib_file.write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        with pytest.raises(ValueError, match="Tr_velo_to_cam not found"):
            _parse_kitti_calib(calib_file)

    def test_load_kitti_labels_with_calib(self, tmp_path: Path):
        calib_file = tmp_path / "calib.txt"
        self._write_calib(calib_file)
        # Car at velo(10, -2, -0.5): cam_x=2, cam_y=0.5, cam_z=10
        # bottom center: cam_y_bottom = 0.5 + 0.75 = 1.25
        label_file = tmp_path / "label.txt"
        label_file.write_text("Car 0.00 0 0.00 100 100 300 250 1.50 1.80 4.50 2.00 1.25 10.00 0.00\n")

        boxes = load_boxes(label_file, fmt="kitti", skip_invalid=False, calib_path=calib_file)
        assert len(boxes) == 1
        box = boxes[0]
        assert box.label == "Car"
        np.testing.assert_allclose(box.center[0], 10.0, atol=0.01)
        np.testing.assert_allclose(box.center[1], -2.0, atol=0.01)
        np.testing.assert_allclose(box.center[2], -0.5, atol=0.01)
        np.testing.assert_allclose(box.size, [4.5, 1.8, 1.5], atol=0.01)

    def test_load_kitti_filters_dontcare(self, tmp_path: Path):
        calib_file = tmp_path / "calib.txt"
        self._write_calib(calib_file)
        label_file = tmp_path / "label.txt"
        label_file.write_text(
            "Car 0.00 0 0.00 100 100 300 250 1.50 1.80 4.50 2.00 1.25 10.00 0.00\n"
            "DontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10\n"
            "Misc 0.00 0 0.00 100 100 300 250 1.00 1.00 1.00 0.00 0.00 5.00 0.00\n"
        )
        boxes = load_boxes(label_file, fmt="kitti", skip_invalid=True, calib_path=calib_file)
        assert len(boxes) == 1
        assert boxes[0].label == "Car"

    def test_load_kitti_without_calib(self, tmp_path: Path):
        """Without calib file, uses approximate transform."""
        label_file = tmp_path / "label.txt"
        label_file.write_text("Pedestrian 0.00 0 0.00 100 100 200 300 1.70 0.60 0.80 1.00 1.50 8.00 0.00\n")
        boxes = load_boxes(label_file, fmt="kitti", skip_invalid=False)
        assert len(boxes) == 1
        assert boxes[0].label == "Pedestrian"

    def test_kitti_end_to_end(self, tmp_path: Path):
        """Full pipeline: bin -> load_boxes(kitti) -> remove -> verify removal."""
        calib_file = tmp_path / "calib.txt"
        self._write_calib(calib_file)

        # Create points: ground + car cluster at velo(10, -2, -0.5)
        rng = np.random.default_rng(42)
        ground = np.column_stack([
            rng.uniform(0, 40, 1000),
            rng.uniform(-10, 10, 1000),
            np.full(1000, -1.7) + rng.normal(0, 0.02, 1000),
        ])
        car = np.column_stack([
            10.0 + rng.normal(0, 0.5, 200),
            -2.0 + rng.normal(0, 0.2, 200),
            -0.5 + rng.normal(0, 0.2, 200),
        ])
        all_pts = np.vstack([ground, car]).astype(np.float32)

        # Write .bin
        bin_file = tmp_path / "test.bin"
        with bin_file.open("wb") as f:
            for row in all_pts:
                f.write(struct.pack("ffff", row[0], row[1], row[2], 0.5))

        # Write label: Car at cam(2, 0.5, 10), bottom center cam_y=1.25
        label_file = tmp_path / "label.txt"
        label_file.write_text("Car 0.00 0 0.00 100 100 300 250 1.50 1.80 4.50 2.00 1.25 10.00 0.00\n")

        pts = load_points(bin_file, fmt="auto")
        boxes = load_boxes(label_file, fmt="kitti", skip_invalid=False, calib_path=calib_file)
        kept, mask = remove_points_in_boxes(pts, boxes)

        removed = pts.shape[0] - kept.shape[0]
        assert removed > 50, f"Expected significant removal, got {removed}"
        assert kept.shape[0] > 800, f"Expected most ground points kept, got {kept.shape[0]}"

    def test_kitti_sample_data(self):
        """Test with generated sample data if available."""
        kitti_dir = Path(__file__).resolve().parent.parent / "data" / "kitti_sample"
        velodyne = kitti_dir / "velodyne" / "000000.bin"
        label = kitti_dir / "label_2" / "000000.txt"
        calib = kitti_dir / "calib" / "000000.txt"
        if not velodyne.exists():
            pytest.skip("KITTI sample data not generated yet")
        pts = load_points(velodyne, fmt="auto")
        boxes = load_boxes(label, fmt="kitti", skip_invalid=False, calib_path=calib)
        kept, _ = remove_points_in_boxes(pts, boxes)
        assert pts.shape[0] > kept.shape[0]


# ---------------------------------------------------------------------------
# remove_points_in_boxes
# ---------------------------------------------------------------------------

class TestRemovePointsInBoxes:
    def test_demo_data_removes_315_points(self, demo_pcd_path: Path, demo_objects_path: Path):
        pts = load_points(demo_pcd_path, fmt="auto")
        boxes = load_boxes(demo_objects_path, fmt="auto", skip_invalid=False)
        filtered, mask = remove_points_in_boxes(pts, boxes)
        removed = pts.shape[0] - filtered.shape[0]
        assert removed == 315, f"Expected 315 removed, got {removed}"

    def test_empty_points(self):
        empty = np.zeros((0, 3), dtype=np.float64)
        box = DetectionBox(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([1.0, 1.0, 1.0]),
        )
        result, mask = remove_points_in_boxes(empty, [box])
        assert result.shape[0] == 0
        assert mask.shape[0] == 0

    def test_empty_boxes(self, sample_points: np.ndarray):
        result, mask = remove_points_in_boxes(sample_points, [])
        assert result.shape[0] == sample_points.shape[0]
        assert np.all(mask)

    def test_all_points_inside_box(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]])
        box = DetectionBox(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([10.0, 10.0, 10.0]),
        )
        result, mask = remove_points_in_boxes(pts, [box])
        assert result.shape[0] == 0
        assert not np.any(mask)

    def test_no_points_inside_box(self):
        pts = np.array([[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]])
        box = DetectionBox(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([1.0, 1.0, 1.0]),
        )
        result, mask = remove_points_in_boxes(pts, [box])
        assert result.shape[0] == 2
        assert np.all(mask)

    def test_margin_zero_removes_fewer(self, demo_pcd_path: Path, demo_objects_path: Path):
        pts = load_points(demo_pcd_path, fmt="auto")
        boxes = load_boxes(demo_objects_path, fmt="auto", skip_invalid=False)
        filtered_default, _ = remove_points_in_boxes(pts, boxes)
        filtered_no_margin, _ = remove_points_in_boxes(pts, boxes, margin=(0.0, 0.0, 0.0))
        # Zero margin should remove fewer (or equal) points than default margin
        assert filtered_no_margin.shape[0] >= filtered_default.shape[0]

    def test_default_margin_parameter(self, sample_points: np.ndarray, sample_box):
        """Verify the default margin parameter works (regression test for the bug fix)."""
        # Call without explicit margin - should use default (0.05, 0.05, 0.05)
        result_default, _ = remove_points_in_boxes(sample_points, [sample_box])
        # Call with explicit default margin
        result_explicit, _ = remove_points_in_boxes(
            sample_points, [sample_box], margin=(0.05, 0.05, 0.05)
        )
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_mask_consistency(self, sample_points: np.ndarray, sample_box):
        result, mask = remove_points_in_boxes(sample_points, [sample_box])
        np.testing.assert_array_equal(result, sample_points[mask])

    def test_yaw_rotation(self):
        """Box rotated 90 degrees should remove different points."""
        pts = np.array([
            [0.6, 0.0, 0.0],   # outside unrotated 1x0.2 box, inside if rotated 90 deg
            [0.0, 0.6, 0.0],   # inside unrotated 1x0.2 box (length axis), outside if rotated
        ])
        box_no_yaw = DetectionBox(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([2.0, 0.2, 2.0]),
            yaw=0.0,
        )
        box_yaw_90 = DetectionBox(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([2.0, 0.2, 2.0]),
            yaw=np.pi / 2,
        )
        result_no_yaw, _ = remove_points_in_boxes(pts, [box_no_yaw], margin=(0, 0, 0))
        result_yaw_90, _ = remove_points_in_boxes(pts, [box_yaw_90], margin=(0, 0, 0))
        # Results should differ
        assert result_no_yaw.shape[0] != result_yaw_90.shape[0] or not np.allclose(
            result_no_yaw, result_yaw_90
        )


# ---------------------------------------------------------------------------
# TemporalConsistencyFilter
# ---------------------------------------------------------------------------

class TestTemporalConsistencyFilter:
    def test_basic_filtering(self):
        """Points appearing in fewer frames than min_hits get removed."""
        tcf = TemporalConsistencyFilter(voxel_size=1.0, window_size=3, min_hits=2)
        static_pt = np.array([[0.0, 0.0, 0.0]])
        transient_pt = np.array([[100.0, 100.0, 100.0]])

        # Frame 1: both points
        combined = np.vstack([static_pt, transient_pt])
        result1, mask1 = tcf.filter(combined)
        # After 1 frame, nothing meets min_hits=2 yet
        assert result1.shape[0] == 0

        # Frame 2: only static point
        result2, mask2 = tcf.filter(static_pt)
        # Static point appeared in 2 frames now -> passes
        assert result2.shape[0] == 1

        # Frame 3: only transient point (appeared in frame 1 and 3)
        result3, mask3 = tcf.filter(transient_pt)
        # Transient point has 2 hits -> passes
        assert result3.shape[0] == 1

    def test_all_static_after_warmup(self):
        """Static points survive after warmup period."""
        tcf = TemporalConsistencyFilter(voxel_size=0.5, window_size=3, min_hits=3)
        pts = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        for _ in range(3):
            result, mask = tcf.filter(pts)

        # After 3 identical frames, all static points should survive
        assert result.shape[0] == pts.shape[0]
        assert np.all(mask)

    def test_empty_input(self):
        tcf = TemporalConsistencyFilter()
        empty = np.zeros((0, 3), dtype=np.float64)
        result, mask = tcf.filter(empty)
        assert result.shape[0] == 0
        assert mask.shape[0] == 0

    def test_negative_voxel_size_raises(self):
        with pytest.raises(ValueError, match="voxel_size must be positive"):
            TemporalConsistencyFilter(voxel_size=-1.0)

    def test_zero_voxel_size_raises(self):
        with pytest.raises(ValueError, match="voxel_size must be positive"):
            TemporalConsistencyFilter(voxel_size=0.0)

    def test_negative_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            TemporalConsistencyFilter(window_size=-1)

    def test_negative_min_hits_raises(self):
        with pytest.raises(ValueError, match="min_hits must be positive"):
            TemporalConsistencyFilter(min_hits=0)

    def test_window_eviction(self):
        """Old frames get evicted when window is full."""
        tcf = TemporalConsistencyFilter(voxel_size=1.0, window_size=2, min_hits=2)
        pt_a = np.array([[0.0, 0.0, 0.0]])
        pt_b = np.array([[50.0, 50.0, 50.0]])

        tcf.filter(pt_a)  # frame 1: pt_a (hits: a=1)
        tcf.filter(pt_a)  # frame 2: pt_a (hits: a=2) -> pt_a passes

        # frame 3: pt_b only. Window evicts frame 1 (a drops to 1)
        result, _ = tcf.filter(pt_b)
        assert result.shape[0] == 0  # pt_b only has 1 hit


# ---------------------------------------------------------------------------
# save_points (round-trip)
# ---------------------------------------------------------------------------

class TestSavePoints:
    def test_roundtrip_pcd(self, tmp_path: Path):
        pts = np.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]], dtype=np.float64)
        out = tmp_path / "out.pcd"
        save_points(out, pts, fmt="pcd")
        loaded = load_points(out, fmt="pcd")
        np.testing.assert_allclose(loaded, pts, atol=1e-6)

    def test_roundtrip_text(self, tmp_path: Path):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        out = tmp_path / "out.xyz"
        save_points(out, pts, fmt="text")
        loaded = load_points(out, fmt="text")
        np.testing.assert_allclose(loaded, pts, atol=1e-8)

    def test_roundtrip_csv(self, tmp_path: Path):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        out = tmp_path / "out.csv"
        save_points(out, pts, fmt="csv")
        loaded = load_points(out, fmt="auto")
        np.testing.assert_allclose(loaded, pts, atol=1e-8)

    def test_roundtrip_npy(self, tmp_path: Path):
        pts = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], dtype=np.float64)
        out = tmp_path / "out.npy"
        save_points(out, pts, fmt="npy")
        loaded = load_points(out, fmt="npy")
        np.testing.assert_allclose(loaded, pts)

    def test_auto_format_by_extension(self, tmp_path: Path):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        out = tmp_path / "out.pcd"
        save_points(out, pts, fmt="auto")
        loaded = load_points(out, fmt="auto")
        np.testing.assert_allclose(loaded, pts, atol=1e-6)

    def test_save_empty(self, tmp_path: Path):
        pts = np.zeros((0, 3), dtype=np.float64)
        out = tmp_path / "empty.pcd"
        save_points(out, pts, fmt="pcd")
        loaded = load_points(out, fmt="pcd")
        assert loaded.shape == (0, 3)


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

class TestCLI:
    def test_help_does_not_crash(self):
        result = subprocess.run(
            [sys.executable, "-m", "dynamic_object_removal", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "Remove dynamic points" in result.stdout
        assert "--algorithm" in result.stdout

    def test_main_with_demo_data(self, tmp_path: Path, demo_pcd_path: Path, demo_objects_path: Path):
        out_file = tmp_path / "output.pcd"
        summary_file = tmp_path / "summary.json"
        ret = main([
            "--input-cloud", str(demo_pcd_path),
            "--input-objects", str(demo_objects_path),
            "--output-cloud", str(out_file),
            "--summary-json", str(summary_file),
            "--quiet",
        ])
        assert ret == 0
        assert out_file.exists()
        assert summary_file.exists()
        summary = json.loads(summary_file.read_text())
        assert summary["removed_points"] == 315
        assert summary["total_points"] > 0

    def test_main_missing_input(self, tmp_path: Path):
        ret = main([
            "--input-cloud", str(tmp_path / "nonexistent.pcd"),
            "--input-objects", str(tmp_path / "nonexistent.json"),
            "--output-cloud", str(tmp_path / "out.pcd"),
        ])
        assert ret == 1

    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "dynamic_object_removal", "--version"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        import dynamic_object_removal
        assert dynamic_object_removal.__version__ in result.stdout


# ---------------------------------------------------------------------------
# Range-image visibility removal
# ---------------------------------------------------------------------------

from dynamic_object_removal import (
    remove_ghost_by_range_image,
    RangeImageGhostFilter,
    clean_map_by_visibility,
    remove_dynamic_by_scan_ratio,
    clean_map_by_scan_ratio,
)


def _dense_wall(x=10.0, n=121, half=3.0):
    ys, zs = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
    return np.column_stack([np.full(ys.size, x), ys.ravel(), zs.ravel()])


class TestRangeImageRemoval:
    def test_removes_seen_through_point(self):
        wall = _dense_wall()
        ghost = np.array([[3.0, 0.0, 0.0], [3.0, 0.1, 0.05]])  # in front of the wall
        mp = np.vstack([wall, ghost])
        kept, mask = remove_ghost_by_range_image(mp, wall, (0, 0, 0), range_margin=0.5)
        assert mask[:-2].all()          # wall kept
        assert not mask[-1] and not mask[-2]  # both ghosts removed
        assert len(kept) == len(wall)

    def test_keeps_coincident_and_occluded(self):
        wall = _dense_wall()
        coincident = np.array([[10.0, 0.0, 0.0]])
        occluded = np.array([[15.0, 0.0, 0.0]])  # behind the wall
        for extra in (coincident, occluded):
            _, mask = remove_ghost_by_range_image(np.vstack([wall, extra]), wall, (0, 0, 0))
            assert mask[-1]  # kept

    def test_self_comparison_removes_nothing(self):
        wall = _dense_wall()
        _, mask = remove_ghost_by_range_image(wall, wall, (0, 0, 0), range_margin=0.5)
        assert mask.all()

    def test_empty_inputs(self):
        wall = _dense_wall()
        empty = np.zeros((0, 3))
        assert remove_ghost_by_range_image(empty, wall)[1].shape == (0,)
        # No query -> keep everything.
        assert remove_ghost_by_range_image(wall, empty)[1].all()

    def test_streaming_filter(self):
        wall = _dense_wall()
        ghost = np.array([[3.0, 0.0, 0.0]])
        f = RangeImageGhostFilter(window_size=3, range_margin=0.5)
        _, m0 = f.filter(wall, (0, 0, 0))
        assert m0.all()  # first scan: no history -> unchanged
        _, m1 = f.filter(np.vstack([wall, ghost]), (0, 0, 0))
        assert m1[:-1].all() and not m1[-1]  # the freshly-appeared point is removed


class TestCleanMapByVisibility:
    def test_revert_keeps_repeatedly_observed_surface(self):
        wall = _dense_wall()
        ghost = np.array([[3.0, 0.0, 0.0]])
        mp = np.vstack([wall, ghost])
        # Two scans both see the wall (confirm surface) and see past the ghost.
        scans = [(wall, (0, 0, 0)), (wall, (0, 0, 0))]
        _, keep = clean_map_by_visibility(
            mp, scans, range_margin=0.5, min_see_through=1, max_surface_hits=1
        )
        assert keep[:-1].all()   # wall confirmed as surface -> kept
        assert not keep[-1]      # ghost seen-through, never a surface -> removed

    def test_ground_protected(self):
        wall = _dense_wall()
        ghost = np.array([[3.0, 0.0, -2.0]])  # below ground_z
        mp = np.vstack([wall, ghost])
        scans = [(wall, (0, 0, 0)), (wall, (0, 0, 0))]
        _, keep = clean_map_by_visibility(
            mp, scans, range_margin=0.5, min_see_through=1, max_surface_hits=0, ground_z=-1.4
        )
        assert keep[-1]  # protected by ground_z despite being seen-through

    def test_empty(self):
        assert clean_map_by_visibility(np.zeros((0, 3)), [(_dense_wall(), (0, 0, 0))])[1].shape == (0,)
        wall = _dense_wall()
        assert clean_map_by_visibility(wall, [])[1].all()  # no scans -> keep all

    def test_multi_resolution_consensus(self):
        wall = _dense_wall()
        ghost = np.array([[3.0, 0.0, 0.0]])
        mp = np.vstack([wall, ghost])
        scans = [(wall, (0, 0, 0)), (wall, (0, 0, 0))]
        # A single-entry resolutions list is identical to the scalar h/v args.
        _, keep_scalar = clean_map_by_visibility(
            mp, scans, h_res_deg=2.0, v_res_deg=2.0, range_margin=0.5,
            min_see_through=1, max_surface_hits=1)
        _, keep_list = clean_map_by_visibility(
            mp, scans, range_margin=0.5, min_see_through=1, max_surface_hits=1,
            resolutions=[2.0])
        assert np.array_equal(keep_scalar, keep_list)
        # Tuple (h, v) form is accepted and equivalent.
        _, keep_tuple = clean_map_by_visibility(
            mp, scans, range_margin=0.5, min_see_through=1, max_surface_hits=1,
            resolutions=[(2.0, 2.0)])
        assert np.array_equal(keep_tuple, keep_list)
        # Consensus (surface guard off) removes only points seen through at EVERY
        # resolution -> a subset of either single resolution, never more.
        kw = dict(range_margin=0.5, min_see_through=1, max_surface_hits=10_000)
        _, keep_fine = clean_map_by_visibility(mp, scans, h_res_deg=2.0, v_res_deg=2.0, **kw)
        _, keep_coarse = clean_map_by_visibility(mp, scans, h_res_deg=8.0, v_res_deg=8.0, **kw)
        _, keep_consensus = clean_map_by_visibility(mp, scans, resolutions=[2.0, 8.0], **kw)
        removed_consensus = ~keep_consensus
        assert not keep_fine[removed_consensus].any()    # also removed at the fine image
        assert not keep_coarse[removed_consensus].any()  # and at the coarse image
        assert keep_consensus.sum() >= keep_fine.sum()
        assert keep_consensus.sum() >= keep_coarse.sum()
        assert not keep_consensus[-1]                    # the clear ghost is still removed
        with pytest.raises(ValueError):
            clean_map_by_visibility(mp, scans, resolutions=[])


def _flat_ground(half=15.0, n=60):
    gx, gy = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])


def _tall_box(cx, cy, top=2.0):
    ox, oy, oz = np.meshgrid(
        np.linspace(cx - 0.5, cx + 0.5, 6),
        np.linspace(cy - 0.5, cy + 0.5, 6),
        np.linspace(0.1, top, 12),
    )
    return np.column_stack([ox.ravel(), oy.ravel(), oz.ravel()])


class TestScanRatioRemoval:
    def test_removes_tall_dynamic_keeps_static(self):
        ground = _flat_ground()
        obj = _tall_box(8.0, 0.0)        # dynamic: present in map, gone in query
        wall = _tall_box(-8.0, 0.0, 3.0)  # static: present in both -> ratio ~1, kept
        mp = np.vstack([ground, obj, wall])
        query = np.vstack([ground, wall])
        _, keep = remove_dynamic_by_scan_ratio(
            mp, query, (0, 0, 0), max_range=30.0, min_map_height=0.5, ground_margin=0.25
        )
        ng, no = len(ground), len(obj)
        obj_keep = keep[ng:ng + no]
        wall_keep = keep[ng + no:]
        ground_keep = keep[:ng]
        assert (~obj_keep).sum() > no // 2   # most of the dynamic object body removed
        assert wall_keep.all()               # static tall structure untouched
        assert ground_keep.all()             # ground reverted, never removed

    def test_no_query_keeps_all(self):
        mp = np.vstack([_flat_ground(), _tall_box(8.0, 0.0)])
        _, keep = remove_dynamic_by_scan_ratio(mp, np.zeros((0, 3)), (0, 0, 0))
        assert keep.all()

    def test_empty_map(self):
        kept, keep = remove_dynamic_by_scan_ratio(np.zeros((0, 3)), _flat_ground())
        assert keep.shape == (0,) and kept.shape == (0, 3)

    def test_bad_origin_raises(self):
        with pytest.raises(ValueError):
            remove_dynamic_by_scan_ratio(_flat_ground(), _flat_ground(), (0, 0))

    def test_voting_threshold(self):
        ground = _flat_ground()
        obj = _tall_box(8.0, 0.0)
        mp = np.vstack([ground, obj])
        query_gone = np.vstack([ground])           # flags the object
        query_present = np.vstack([ground, obj])    # does not flag it
        scans = [(query_gone, (0, 0, 0)), (query_present, (0, 0, 0))]
        kw = dict(max_range=30.0, min_map_height=0.5, ground_margin=0.25)
        _, keep1 = clean_map_by_scan_ratio(mp, scans, min_votes=1, **kw)
        _, keep2 = clean_map_by_scan_ratio(mp, scans, min_votes=2, **kw)
        ng = len(ground)
        # 1 vote removes the object (one scan saw it gone); 2 votes spares it (only one did).
        assert (~keep1[ng:]).sum() > 0
        assert keep2[ng:].all()
        assert clean_map_by_scan_ratio(mp, [], **kw)[1].all()  # no scans -> keep all
        # Default min_votes=None scales with the scan count (30% of 2 scans -> 1 vote).
        _, keep_auto = clean_map_by_scan_ratio(mp, scans, **kw)
        assert np.array_equal(keep_auto, keep1)


class TestRangeCLI:
    def test_cli_range_algorithm(self, tmp_path: Path):
        wall = _dense_wall()
        ghost = np.array([[3.0, 0.0, 0.0], [3.0, 0.1, 0.05]])
        map_path = tmp_path / "map.npy"
        query_path = tmp_path / "q.npy"
        out_path = tmp_path / "out.npy"
        np.save(map_path, np.vstack([wall, ghost]).astype(np.float32))
        np.save(query_path, wall.astype(np.float32))
        ret = main([
            "--algorithm", "range",
            "--input-map", str(map_path),
            "--input-cloud", str(query_path),
            "--sensor-origin", "0", "0", "0",
            "--output-cloud", str(out_path),
            "--cloud-format", "npy",
        ])
        assert ret == 0
        assert len(np.load(out_path)) == len(wall)  # both ghosts removed

    def test_cli_range_requires_map(self, tmp_path: Path):
        q = tmp_path / "q.npy"
        np.save(q, _dense_wall().astype(np.float32))
        ret = main([
            "--algorithm", "range",
            "--input-cloud", str(q),
            "--output-cloud", str(tmp_path / "out.npy"),
            "--cloud-format", "npy",
        ])
        assert ret == 1

    def test_cli_scan_ratio_algorithm(self, tmp_path: Path):
        ground = _flat_ground()
        obj = _tall_box(8.0, 0.0)
        map_path = tmp_path / "map.npy"
        query_path = tmp_path / "q.npy"
        out_path = tmp_path / "out.npy"
        np.save(map_path, np.vstack([ground, obj]).astype(np.float32))
        np.save(query_path, ground.astype(np.float32))  # object gone
        ret = main([
            "--algorithm", "scan_ratio",
            "--input-map", str(map_path),
            "--input-cloud", str(query_path),
            "--sensor-origin", "0", "0", "0",
            "--output-cloud", str(out_path),
            "--cloud-format", "npy",
            "--scan-ratio-ground-margin", "0.25",
        ])
        assert ret == 0
        kept = len(np.load(out_path))
        assert kept < len(ground) + len(obj)  # object body removed
        assert kept >= len(ground)            # ground reverted, kept

    def test_cli_scan_ratio_requires_map(self, tmp_path: Path):
        q = tmp_path / "q.npy"
        np.save(q, _flat_ground().astype(np.float32))
        ret = main([
            "--algorithm", "scan_ratio",
            "--input-cloud", str(q),
            "--output-cloud", str(tmp_path / "out.npy"),
            "--cloud-format", "npy",
        ])
        assert ret == 1


# ---------------------------------------------------------------------------
# Accuracy metrics (bench)
# ---------------------------------------------------------------------------

def test_accuracy_metrics():
    import bench
    removed = np.array([1, 1, 0, 0, 1], dtype=bool)
    gt = np.array([1, 0, 0, 1, 1], dtype=bool)
    m = bench.compute_accuracy_metrics(removed, gt)
    assert m["true_positive"] == 2 and m["false_positive"] == 1
    assert m["false_negative"] == 1 and m["true_negative"] == 1
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["iou"] - 0.5) < 1e-9


def test_dynamic_gt_mask():
    import bench
    pts = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    box = DetectionBox(center=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0]), label="vehicle")
    mask = bench.dynamic_gt_mask(pts, [box], dynamic_labels={"vehicle"})
    assert mask[0] and not mask[1]
    # Non-dynamic label is filtered out -> no GT.
    box2 = DetectionBox(center=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0]), label="BOLLARD")
    assert not bench.dynamic_gt_mask(pts, [box2], dynamic_labels={"vehicle"}).any()


def _load_nuscenes_script():
    """Import the nuScenes benchmark script as a module (no network / argparse)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_nuscenes_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_nuscenes_benchmark", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nuscenes_quaternion_helpers():
    """A wrong quaternion convention would silently corrupt every benchmark number."""
    nb = _load_nuscenes_script()
    # Identity quaternion -> identity rotation.
    assert np.allclose(nb._quat_to_rot(1.0, 0.0, 0.0, 0.0), np.eye(3))
    # 90 deg about +z (nuScenes/pyquaternion order w, x, y, z): x-axis maps to +y.
    s = math.sqrt(0.5)
    rot = nb._quat_to_rot(s, 0.0, 0.0, s)
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)
    assert rot.shape == (3, 3) and np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    # Yaw extraction matches the rotation about z.
    assert abs(nb._yaw_from_quat(s, 0.0, 0.0, s) - math.pi / 2) < 1e-9
    assert abs(nb._yaw_from_quat(1.0, 0.0, 0.0, 0.0)) < 1e-9


def _write_binary_pcd(path: Path, xyz: np.ndarray, intensity: np.ndarray, viewpoint: list[float]) -> None:
    n = len(xyz)
    header = (
        "VERSION .7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        f"VIEWPOINT {' '.join(str(v) for v in viewpoint)}\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    payload = bytearray()
    for (x, y, z), i in zip(xyz, intensity):
        payload.extend(struct.pack("<ffff", float(x), float(y), float(z), float(i)))
    path.write_bytes(header.encode("ascii") + bytes(payload))


def test_load_pcd_scan_binary_viewpoint_and_intensity(tmp_path: Path):
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    intensity = np.array([0.0, 1.0], dtype=np.float64)
    vp = [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]
    pcd = tmp_path / "scan.pcd"
    _write_binary_pcd(pcd, xyz, intensity, vp)

    scan = load_pcd_scan(pcd)
    assert np.allclose(scan.points, xyz)
    assert scan.intensity is not None and np.allclose(scan.intensity, intensity)
    assert scan.viewpoint is not None and np.allclose(scan.viewpoint, vp)


def test_dynamicmap_eval_perfect_and_removed(tmp_path: Path):
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    gt_labels = np.array([0, 1, 0], dtype=np.int64)
    cleaned = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    est = bench.export_dynamicmap_eval_labels(gt, cleaned, max_dist=0.05)
    metrics = bench.compute_dynamicmap_metrics(est, gt_labels)
    assert est[0] == 0.0
    assert est[1] == 1.0  # dynamic point removed
    assert est[2] == 0.0
    assert metrics["SA"] == pytest.approx(100.0)
    assert metrics["DA"] == pytest.approx(100.0)
