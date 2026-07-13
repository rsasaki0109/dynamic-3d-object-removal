from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import compare_downstream_gt_maps as downstream

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "examples" / "lidarslam_ros2" / "av2_downstream_gt_map_proof.json"


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="optional scipy integration")
def test_evaluate_maps_separates_dynamic_reduction_and_static_preservation():
    gt = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    gt_dynamic = np.array([False, False, True, True])
    baseline = gt.copy()
    cleaned = gt[[0, 2]]

    metrics, labels = downstream.evaluate_maps(
        baseline, cleaned, gt, gt_dynamic, match_tolerance_m=1e-6
    )

    assert metrics["baseline"]["matched_ratio"] == 1.0
    assert metrics["cleaned"]["matched_ratio"] == 1.0
    assert metrics["removal"]["total_points"] == 2
    assert metrics["removal"]["dynamic_points"] == 1
    assert metrics["removal"]["static_points"] == 1
    assert metrics["removal"]["dynamic_gt_reduction"] == 0.5
    assert metrics["removal"]["static_gt_preservation"] == 0.5
    assert metrics["removal"]["removed_point_precision"] == 0.5
    np.testing.assert_array_equal(labels["baseline_labels"], gt_dynamic)


def test_proof_contract_rejects_different_pose_artifacts(tmp_path: Path):
    baseline = tmp_path / "baseline.tum"
    cleaned = tmp_path / "cleaned.tum"
    baseline.write_text("same\n", encoding="utf-8")
    cleaned.write_text("same\n", encoding="utf-8")
    assert downstream._require_identical(baseline, cleaned, "trajectory") == downstream._sha256(baseline)

    cleaned.write_text("different\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory differs"):
        downstream._require_identical(baseline, cleaned, "trajectory")


def test_gt_map_uses_backend_trajectory_and_exact_manifest_timestamp(tmp_path: Path):
    np.save(tmp_path / "cloud.npy", np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    np.save(tmp_path / "labels.npy", np.array([0, 1], dtype=np.uint8))
    timestamp_ns = 315968121560163000
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "frames": [{
            "cloud": "cloud.npy",
            "point_labels": "labels.npy",
            "timestamp_ns": timestamp_ns,
            "timestamp_sec": timestamp_ns / 1e9,
        }]
    }), encoding="utf-8")
    trajectory = [{
        "timestamp_sec": timestamp_ns / 1e9,
        "translation": np.array([10.0, 20.0, 0.0]),
        "rotation": downstream._rotation_from_quaternion_xyzw(
            np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
        ),
    }]

    points, labels, timestamps = downstream._build_gt_map(
        manifest, trajectory, timestamp_tolerance_sec=0.001
    )

    np.testing.assert_allclose(points, [[10.0, 21.0, 0.0], [8.0, 20.0, 0.0]], atol=1e-12)
    np.testing.assert_array_equal(labels, [False, True])
    assert timestamps == [timestamp_ns]


def test_checked_in_downstream_proof_keeps_strict_contract_and_honest_metrics():
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    contract = proof["proof_contract"]
    removal = proof["metrics"]["removal"]

    assert contract["frames"] == 11
    assert contract["baseline_and_cleaned_raw_trajectory_byte_identical"] is True
    assert contract["baseline_and_cleaned_optimized_trajectory_byte_identical"] is True
    assert contract["baseline_and_cleaned_loop_edges_byte_identical"] is True
    assert contract["labels_available_to_filter"] is False
    assert proof["metrics"]["baseline"]["matched_ratio"] == 1.0
    assert proof["metrics"]["cleaned"]["matched_ratio"] == 1.0
    assert removal["dynamic_gt_reduction"] == pytest.approx(0.14128018397853584)
    assert removal["static_gt_preservation"] == pytest.approx(0.9622763354913104)
    assert removal["removed_point_precision"] == pytest.approx(0.2175101791931391)
