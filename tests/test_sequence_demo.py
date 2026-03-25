"""Regression tests for the sequence demo generator."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from dynamic_object_removal import DetectionBox

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQUENCE_DEMO_PATH = PROJECT_ROOT / "demo" / "run_scan_sequence_demo.py"

_spec = importlib.util.spec_from_file_location("run_scan_sequence_demo", SEQUENCE_DEMO_PATH)
assert _spec is not None and _spec.loader is not None
sequence_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sequence_demo)


def _write_feather(path: Path, data: dict[str, list[object]]) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    feather = pytest.importorskip("pyarrow.feather")
    table = pyarrow.table(data)
    feather.write_feather(table, path)


class TestSequenceDemoHelpers:
    def test_frame_timestamp_ns_uses_numeric_stem(self):
        assert sequence_demo._frame_timestamp_ns(Path("/tmp/315969904359876000.feather")) == 315969904359876000
        assert sequence_demo._frame_timestamp_ns(Path("/tmp/cloud.pcd")) is None

    def test_quat_wxyz_to_rotation_matrix_rotates_x_to_y(self):
        half = math.sqrt(0.5)
        rotation = sequence_demo._quat_wxyz_to_rotation_matrix(half, 0.0, 0.0, half)
        rotated = rotation @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-6)

    def test_transform_points_with_pose_applies_rotation_and_translation(self):
        half = math.sqrt(0.5)
        rotation = sequence_demo._quat_wxyz_to_rotation_matrix(half, 0.0, 0.0, half)
        translation = np.array([10.0, 20.0, 30.0])
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        transformed = sequence_demo._transform_points_with_pose(points, rotation, translation)

        np.testing.assert_allclose(
            transformed,
            [[10.0, 21.0, 30.0], [9.0, 20.0, 30.0]],
            atol=1e-6,
        )

    def test_transform_boxes_with_pose_rotates_center_and_yaw(self):
        half = math.sqrt(0.5)
        rotation = sequence_demo._quat_wxyz_to_rotation_matrix(half, 0.0, 0.0, half)
        translation = np.array([10.0, 20.0, 30.0])
        boxes = [
            DetectionBox(
                center=np.array([1.0, 0.0, 0.0]),
                size=np.array([4.0, 2.0, 1.5]),
                yaw=0.0,
                label="car",
            )
        ]

        transformed = sequence_demo._transform_boxes_with_pose(boxes, rotation, translation)

        assert len(transformed) == 1
        np.testing.assert_allclose(transformed[0].center, [10.0, 21.0, 30.0], atol=1e-6)
        assert transformed[0].yaw == pytest.approx(math.pi / 2.0, abs=1e-6)
        assert transformed[0].label == "car"


class TestSequenceDemoAV2Inputs:
    def test_load_pose_map_from_feather(self, tmp_path: Path):
        pose_file = tmp_path / "city_SE3_egovehicle.feather"
        _write_feather(
            pose_file,
            {
                "timestamp_ns": [100, 200],
                "qw": [1.0, math.sqrt(0.5)],
                "qx": [0.0, 0.0],
                "qy": [0.0, 0.0],
                "qz": [0.0, math.sqrt(0.5)],
                "tx_m": [1.0, 10.0],
                "ty_m": [2.0, 20.0],
                "tz_m": [3.0, 30.0],
            },
        )

        pose_map = sequence_demo._load_pose_map(pose_file)
        rotation, translation = pose_map[200]

        np.testing.assert_allclose(translation, [10.0, 20.0, 30.0], atol=1e-6)
        rotated = rotation @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-6)

    def test_resolve_boxes_from_av2_feather_filters_by_timestamp(self, tmp_path: Path):
        ann_file = tmp_path / "annotations.feather"
        _write_feather(
            ann_file,
            {
                "timestamp_ns": [100, 200],
                "category": ["REGULAR_VEHICLE", "REGULAR_VEHICLE"],
                "length_m": [4.5, 4.5],
                "width_m": [1.8, 1.8],
                "height_m": [1.5, 1.5],
                "qw": [1.0, 1.0],
                "qx": [0.0, 0.0],
                "qy": [0.0, 0.0],
                "qz": [0.0, 0.0],
                "tx_m": [1.0, 9.0],
                "ty_m": [2.0, 8.0],
                "tz_m": [3.0, 7.0],
                "num_interior_pts": [25, 30],
            },
        )

        boxes = sequence_demo._resolve_boxes(ann_file, tmp_path / "200.feather")

        assert len(boxes) == 1
        np.testing.assert_allclose(boxes[0].center, [9.0, 8.0, 7.0], atol=1e-6)
        np.testing.assert_allclose(boxes[0].size, [4.5, 1.8, 1.5], atol=1e-6)

    def test_lookup_pose_requires_numeric_frame_name(self):
        with pytest.raises(ValueError, match="cannot derive timestamp_ns"):
            sequence_demo._lookup_pose({100: (np.eye(3), np.zeros(3))}, Path("/tmp/cloud.pcd"))
