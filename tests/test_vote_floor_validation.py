import hashlib
import json

import numpy as np
import pytest

import bench
import dynamic_object_removal as core
from scripts.validate_vote_floor import validate
from scripts.diagnose_common_misses import diagnose


def fixture(root):
    points = np.array([[5., 0., 0.], [5., 0., 1.], [10., 0., 2.]])
    labels = np.array([0, 1, 0], dtype=np.uint8)
    np.save(root / "scan.npy", points)
    np.save(root / "labels.npy", labels)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"scene": "test", "sensor_profile": {"deskewed": True},
        "frames": [{"cloud": "scan.npy", "point_labels": "labels.npy", "sensor_origin": [0, 0, 0]}] * 3}))
    scans, _ = core._load_scan_manifest(manifest)
    points = np.concatenate([p for p, _ in scans])
    gt = np.tile(labels.astype(bool), 3)
    args = core._build_parser().parse_args(["--algorithm", "range_scan_ratio", "--output-cloud", "unused"])
    args.range_h_res = args.range_v_res = 2.5
    params = core._range_scan_ratio_cli_parameters(args)
    _, kr = core.clean_map_by_visibility(points, scans, **params["range"])
    _, ks = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    baseline = root / "baseline.json"
    baseline.write_text(json.dumps({"dataset": "synthetic", "scene": "test", "frames": 3,
        "stride": 1, "map_points": len(points), "gt_dynamic_points": int(gt.sum()),
        "config": {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
                   "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest(),
                   "parameters": {"range_scan_ratio": params}},
        "metrics": {"range_scan_ratio": bench.compute_accuracy_metrics(~kr & ~ks, gt)}}))
    return manifest, baseline


def test_frozen_vote_floor_validator_executes_both_clis(tmp_path):
    manifest, baseline = fixture(tmp_path)
    result = validate(manifest, baseline, tmp_path / "out")
    assert result["results"]["baseline"]["parameters"]["scan_ratio"]["votes_floor"] == 3
    assert result["results"]["candidate"]["parameters"]["scan_ratio"]["votes_floor"] == 2
    for record in result["results"].values():
        assert record["api_cli_masks_equal"] and record["api_cli_points_equal"]
    assert json.loads((tmp_path / "out/validation.json").read_text()) == result


def test_frozen_vote_floor_validator_rejects_changed_input_before_output(tmp_path):
    manifest, baseline = fixture(tmp_path)
    np.save(tmp_path / "scan.npy", np.ones((3, 3)))
    with pytest.raises(ValueError, match="fingerprints"):
        validate(manifest, baseline, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_common_misses_replay_evidence_and_partition_observed_opportunities(tmp_path):
    manifest, baseline = fixture(tmp_path)
    result = diagnose(manifest, json.loads(baseline.read_text()))
    assert result["evidence_api_masks_equal"]
    assert result["common_misses"] == 3
    assert result["range"]["insufficient_see_through"] == 3
    gates = result["scan_ratio"]["observed_point_scan_opportunities"]
    assert gates["total"] == 9
    assert gates["query_map_height_ratio_too_large"] == 9
    assert sum(value for key, value in gates.items() if key != "total") == gates["total"]
