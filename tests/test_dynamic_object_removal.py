"""Comprehensive tests for dynamic_object_removal module."""

from __future__ import annotations

import json
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
    remove_points_in_boxes,
    save_points,
    main,
    _parse_kitti_calib,
)


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
        assert "Remove points" in result.stdout

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
        assert "0.1.0" in result.stdout
