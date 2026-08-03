from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import run_online_benchmark as online
from scripts.online_mapping_backend import (
    _BackendConfig,
    _BoundedFreeSpaceBackend,
    _BoundedVoxelEvidenceStore,
)


def test_evidence_store_is_bounded_and_counters_saturate() -> None:
    store = _BoundedVoxelEvidenceStore(max_voxels=2, counter_max=3)
    first = np.array([[0, 0, 0]], dtype=np.int64)
    second = np.array([[1, 0, 0]], dtype=np.int64)
    third = np.array([[2, 0, 0]], dtype=np.int64)

    for frame in range(10):
        store.update(first, np.empty((0, 3), dtype=np.int64), frame_index=frame)
    store.update(second, np.empty((0, 3), dtype=np.int64), frame_index=10)
    store.update(third, np.empty((0, 3), dtype=np.int64), frame_index=11)

    assert len(store) <= 2
    assert store.peak_size <= 2
    assert store.evictions >= 1
    assert store.counter_saturations > 0
    evidence = store.get((0, 0, 0))
    if evidence is not None:
        assert evidence[0] <= 3
        assert evidence[1] <= 3
    assert store.memory_bound_bytes() > 0


def test_rejudgment_removes_accepted_points_and_honors_slice_budget() -> None:
    points = np.array([[5.1 + 0.3 * i, 0.0, 1.0] for i in range(7)], dtype=np.float64)
    config = _BackendConfig(
        free_floor=1,
        slice_budget_points=3,
        max_slices_per_frame=1,
        max_recent_points=32,
        max_queue_points=32,
        max_pending_voxels=32,
        max_voxels=32,
    )
    backend = _BoundedFreeSpaceBackend(config)
    active = np.ones(len(points), dtype=bool)
    indices = np.arange(len(points), dtype=np.int64)
    backend.add_map_points(points, indices, frame_index=0)
    voxel_rows = np.floor(points / config.voxel_size).astype(np.int64)
    touched = backend.evidence.update(
        voxel_rows,
        np.empty((0, 3), dtype=np.int64),
        frame_index=0,
    )
    backend._mark_pending(touched)

    first = backend.service(points, active, frame_index=0, force=True)
    assert first["processed_points"] <= config.slice_budget_points
    assert first["slices"] == 1
    assert int(active.sum()) == len(points) - first["removed_points"]
    assert backend.summary()["amortized_cost"]["max_slice_points"] <= config.slice_budget_points

    backend.drain(points, active)
    assert not active.any()


def _write_deterministic_manifest(root: Path) -> Path:
    wall = np.array(
        [[10.0, y, z] for y in np.linspace(-1.0, 1.0, 21) for z in np.linspace(-0.5, 0.5, 11)],
        dtype=np.float64,
    )
    frames = []
    for index, origin_x in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]):
        world = wall.copy()
        labels = np.zeros(len(world), dtype=np.uint8)
        if index == 0:
            world = np.vstack([world, [[5.0, 0.0, 0.0]]])
            labels = np.append(labels, 1)
        local = world - np.array([origin_x, 0.0, 0.0])
        cloud = root / f"cloud_{index}.npy"
        label = root / f"label_{index}.npy"
        np.save(cloud, local)
        np.save(label, labels)
        frames.append(
            {
                "cloud": cloud.name,
                "point_labels": label.name,
                "timestamp_sec": index * 0.1,
                "pose": {
                    "translation": [origin_x, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sensor_profile": {"name": "synthetic-64", "beams": 64, "rate_hz": 10.0, "deskewed": True},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_deterministic_mapping_replay_reports_frontend_backend_and_gated_row(tmp_path: Path) -> None:
    manifest = _write_deterministic_manifest(tmp_path)
    output = tmp_path / "mapping.json"
    assert online.main(
        [
            "--manifest",
            str(manifest),
            "--summary-json",
            str(output),
            "--algorithm",
            "range",
            "--backend",
            "bounded",
            "--range-window",
            "3",
            "--backend-slice-budget",
            "100",
            "--backend-max-voxels",
            "5000",
            "--backend-max-recent-points",
            "10000",
            "--backend-max-queue-points",
            "10000",
            "--backend-max-pending-voxels",
            "5000",
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["task"] == "online_static_mapping"
    scenario = payload["scenarios"][0]
    assert set(scenario) >= {
        "front_end_only",
        "front_end_plus_backend",
        "gated_temporal_front_end_only",
    }
    frontend = scenario["front_end_only"]["metrics"]
    backend = scenario["front_end_plus_backend"]["metrics"]
    assert frontend["map_dynamic_retention"] == 1.0
    assert backend["map_dynamic_retention"] == 0.0
    assert backend["map_static_completeness"] == 1.0
    assert backend["backend_removed_points"] == 1
    assert scenario["front_end_plus_backend"]["backend"]["memory"]["evidence_peak_voxels"] <= 5000
    assert all(
        frame["backend"]["processed_points"] <= 100
        for frame in scenario["front_end_plus_backend"]["per_frame"]
    )
