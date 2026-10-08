import json

import numpy as np
import pytest

import bench
import dynamic_object_removal as core
from scripts import validate_multiscan_cli as validation


def manifest_fixture(root):
    frames = []
    for index in range(4):
        points = [[5, 0, 0], [10, 0, 2]]
        labels = [0, 0]
        if index == 0:
            points.append([5, 0, 1])
            labels.append(1)
        np.save(root / f"scan{index}.npy", np.array(points, float))
        np.save(root / f"labels{index}.npy", np.array(labels, np.uint8))
        frames.append({"cloud": f"scan{index}.npy", "point_labels": f"labels{index}.npy",
                       "pose": {"translation": [0, 0, 0],
                                "quaternion_xyzw": [0, 0, 0, 1]}})
    path = root / "manifest.json"
    path.write_text(json.dumps({"scene": "synthetic-cli-validation",
                                "sensor_profile": {"deskewed": True}, "frames": frames}))
    return path


def test_validation_executes_both_clis_and_saves_reusable_results(tmp_path):
    manifest = manifest_fixture(tmp_path)
    report = validation.validate(manifest, tmp_path / "run", workers=1)
    assert report["status"] == "passed"
    assert report["point_masks_identical"]
    assert not report["reference_metrics_matched"]
    assert report["gt_dynamic_points"] == 1
    assert report["metrics"]["fusion"]["recall"] == 1
    baseline = json.loads((tmp_path / "run" / "baseline_api.json").read_text())
    candidate = json.loads((tmp_path / "run" / "candidate_cli.json").read_text())
    assert baseline["metrics"] == candidate["metrics"]
    assert len(baseline["config"]["map_sha256"]) == 64
    for method in ("fusion", "range_scan_ratio"):
        assert (tmp_path / "run" / f"{method}_keep.npy").exists()
    with pytest.raises(ValueError, match="new or empty"):
        validation.validate(manifest, tmp_path / "run", workers=1)


def test_reference_metrics_checked_before_cli_validation(tmp_path):
    manifest = manifest_fixture(tmp_path)
    scans, _ = core._load_scan_manifest(manifest)
    points = np.concatenate([scan for scan, _ in scans])
    gt = np.array([False, False, True] + [False] * 6)
    masks = {
        "fusion": core.clean_map_by_fusion(points, scans, free_votes_fraction=0.7,
                                         free_votes_floor=3, void_min_scans=4)[1],
        "range": core.clean_map_by_visibility(points, scans, h_res_deg=1, v_res_deg=1,
                                             min_see_through=3, max_surface_hits=3, ground_z=-1.4)[1],
        "scan_ratio": core.clean_map_by_scan_ratio(points, scans)[1],
    }
    reference = {
        "dataset": "argoverse-2-sensor-val", "scene": "synthetic-cli-validation",
        "frames": 4, "stride": 3, "map_points": 9, "gt_dynamic_points": 1, "config": {},
        "metrics": {name: bench.compute_accuracy_metrics(~mask, gt) for name, mask in masks.items()},
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    report = validation.validate(manifest, tmp_path / "matched", workers=1, reference_path=path)
    assert report["reference_metrics_matched"]
    reference["metrics"]["fusion"]["f1"] = 0
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="reference metric mismatch"):
        validation.validate(manifest, tmp_path / "mismatch", workers=1, reference_path=path)


@pytest.mark.parametrize("mask_name", ["mask.txt", "out.npy"])
def test_mask_output_validation(tmp_path, mask_name):
    manifest = manifest_fixture(tmp_path)
    points = np.concatenate([scan for scan, _ in core._load_scan_manifest(manifest)[0]])
    np.save(tmp_path / "map.npy", points)
    assert core.main(["--algorithm", "fusion", "--input-map", str(tmp_path / "map.npy"),
                      "--input-manifest", str(manifest),
                      "--output-cloud", str(tmp_path / "out.npy"),
                      "--output-mask", str(tmp_path / mask_name)]) == 1
    assert not (tmp_path / "out.npy").exists()


def test_default_validation_is_invariant_to_map_vertical_translation(tmp_path):
    manifest = manifest_fixture(tmp_path)
    # Tall accumulated column, low query column, and a farther return along the ghost direction.
    np.save(tmp_path / 'scan0.npy', np.array([[5, 0, 0], [5, 0, 10], [5, 0, 1]], float))
    for i in range(1, 4):
        np.save(tmp_path / f'scan{i}.npy', np.array([[7, 0, 0], [7, 0, 1.4]], float))
    original = validation.validate(manifest, tmp_path / 'original', workers=1)
    content = json.loads(manifest.read_text())
    for frame in content['frames']:
        frame['pose']['translation'][2] = -20
    manifest.write_text(json.dumps(content))
    shifted = validation.validate(manifest, tmp_path / 'shifted', workers=1)
    assert original['summaries']['range_scan_ratio']['parameters']['range']['ground_z'] is None
    assert shifted['summaries']['range_scan_ratio']['parameters']['range']['ground_z'] is None
    np.testing.assert_array_equal(np.load(tmp_path / 'original' / 'range_scan_ratio_keep.npy'),
                                  np.load(tmp_path / 'shifted' / 'range_scan_ratio_keep.npy'))
    assert original['metrics']['range_scan_ratio']['true_positive'] == 1
    assert original['metrics']['range_scan_ratio'] == shifted['metrics']['range_scan_ratio']
