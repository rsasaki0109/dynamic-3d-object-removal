from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_online_benchmark as online
from scripts import run_av2_benchmark as av2


def _build_manifest(tmp_path: Path, *, missing_pose_index: int | None = None) -> Path:
    ys = np.linspace(-1.0, 1.0, 21)
    zs = np.linspace(-0.5, 0.5, 11)
    wall = np.array([[10.0, y, z] for y in ys for z in zs], dtype=np.float64)
    origins = [0.0, 1.0, 2.0, 3.0]
    dynamic_x = [None, 8.0, 7.0, 6.0]
    frames = []
    for index, (origin_x, object_x) in enumerate(zip(origins, dynamic_x)):
        world = wall.copy()
        labels = np.zeros(len(world), dtype=np.uint8)
        if object_x is not None:
            world = np.vstack([world, [[object_x, 0.0, 0.0]]])
            labels = np.append(labels, 1)
        local = world - np.array([origin_x, 0.0, 0.0])
        cloud_path = tmp_path / f"cloud_{index}.npy"
        label_path = tmp_path / f"labels_{index}.npy"
        np.save(cloud_path, local)
        np.save(label_path, labels)
        frame = {
            "cloud": cloud_path.name,
            "point_labels": label_path.name,
            "timestamp_sec": index * 0.1,
        }
        if index != missing_pose_index:
            frame["pose"] = {
                "translation": [origin_x, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        frames.append(frame)
    manifest = {
        "sensor_profile": {
            "name": "synthetic-16",
            "beams": 16,
            "rate_hz": 10,
            "deskewed": True,
        },
        "frames": frames,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.mark.parametrize("algorithm", ["temporal", "range"])
def test_online_replay_writes_metrics_latency_and_pose_noise(tmp_path, algorithm):
    manifest = _build_manifest(tmp_path)
    output = tmp_path / f"{algorithm}.json"
    argv = [
        "--manifest",
        str(manifest),
        "--summary-json",
        str(output),
        "--algorithm",
        algorithm,
        "--pose-noise-translation",
        "0.05",
        "--pose-noise-yaw",
        "0.5",
    ]
    if algorithm == "temporal":
        argv += ["--voxel-size", "0.1", "--temporal-window", "3", "--temporal-min-hits", "3"]
    else:
        argv += ["--range-window", "3", "--range-h-res", "0.4", "--range-v-res", "1.0"]

    assert online.main(argv) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["task"] == "online_moving_object_segmentation"
    assert payload["sensor_profile"]["beams"] == 16
    assert [s["name"] for s in payload["scenarios"]] == [
        "baseline",
        "translation_sigma_0.05m",
        "yaw_sigma_0.5deg",
    ]
    baseline = payload["scenarios"][0]
    assert baseline["frames"] == 4
    assert baseline["fail_open_frames"] == 0
    assert baseline["dropped_frames"] == 0
    assert baseline["deadline_misses"] == 0
    assert baseline["filter_latency"]["p95_ms"] >= 0.0
    assert baseline["metrics"]["recall"] == pytest.approx(1.0)
    assert baseline["time_to_confirm_frame"] == (2 if algorithm == "temporal" else 1)


def test_missing_pose_fail_open_is_counted(tmp_path):
    manifest = _build_manifest(tmp_path, missing_pose_index=1)
    output = tmp_path / "fail_open.json"
    assert online.main(
        [
            "--manifest",
            str(manifest),
            "--summary-json",
            str(output),
            "--algorithm",
            "temporal",
            "--missing-pose",
            "fail-open",
        ]
    ) == 0
    baseline = json.loads(output.read_text(encoding="utf-8"))["scenarios"][0]
    assert baseline["fail_open_frames"] == 1
    assert baseline["per_frame"][1]["fail_open"] is True
    assert baseline["per_frame"][1]["removed_points"] == 0


def test_missing_pose_errors_by_default(tmp_path):
    manifest = _build_manifest(tmp_path, missing_pose_index=0)
    args = online.build_parser().parse_args(
        ["--manifest", str(manifest), "--summary-json", str(tmp_path / "out.json")]
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="has no pose"):
        online.replay_scenario(payload, tmp_path, args)


def test_box_ground_truth_is_supported(tmp_path):
    points = np.array([[2.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    mask = online._load_gt_mask(
        {
            "dynamic_boxes": [
                {"center": [2.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "yaw": 0.0}
            ]
        },
        points,
        tmp_path,
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_av2_exporter_writes_replayable_relative_manifest(tmp_path):
    output = tmp_path / "nested" / "av2_online.json"
    timestamps = [100, 200]
    scans = [np.array([[10.0, 0.0, 0.0]]), np.array([[9.0, 0.0, 0.0]])]
    masks = [np.array([False]), np.array([True])]
    poses = {
        100: (np.eye(3), np.array([0.0, 0.0, 0.0])),
        200: (np.eye(3), np.array([1.0, 0.0, 0.0])),
    }
    av2._export_online_manifest(
        output,
        scene="test-scene",
        stride=2,
        timestamps=timestamps,
        local_scans=scans,
        gt_masks=masks,
        poses=poses,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["sensor_profile"] == {
        "name": "dual VLP-32C",
        "beams": 64,
        "rate_hz": 5.0,
        "deskewed": True,
        "source": "https://argoverse.github.io/user-guide/datasets/sensor.html",
    }
    assert manifest["frames"][1]["pose"]["translation"] == [1.0, 0.0, 0.0]
    assert [frame["timestamp_ns"] for frame in manifest["frames"]] == timestamps
    for frame in manifest["frames"]:
        assert not Path(frame["cloud"]).is_absolute()
        assert (output.parent / frame["cloud"]).exists()
        assert (output.parent / frame["point_labels"]).exists()


def test_av2_proof_sampling_is_deterministic_and_mask_scoped():
    mask = np.array([False, True, True, False, True, False, True])
    first = av2._sample_indices(mask, 2, np.random.default_rng(12))
    second = av2._sample_indices(mask, 2, np.random.default_rng(12))

    np.testing.assert_array_equal(first, second)
    assert len(first) == 2
    assert np.all(mask[first])


def test_av2_gt_proof_renders_same_pose_audit(tmp_path):
    pytest.importorskip("matplotlib")
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    gt = np.array([False, True, False, True])
    keep = np.array([True, False, False, True])
    metrics = {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "static_preservation": 0.5,
    }
    output = tmp_path / "proof.png"

    av2._render_gt_proof(
        output,
        acc_map=points,
        gt_dynamic=gt,
        keep_mask=keep,
        metrics=metrics,
        scene="test-scene",
        frames=2,
        moving_tracks=1,
        moving_thresh=2.0,
        max_points_per_layer=10,
        seed=7,
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
