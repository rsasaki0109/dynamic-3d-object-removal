import hashlib
import json

import numpy as np
import pytest

import bench
import dynamic_object_removal as core
from scripts import diagnose_frozen_nuscenes_misses as module


def test_range_trace_distinguishes_empty_nearer_surface_and_strict_boundaries():
    points = np.array([[0., 0, 0], [1., 0, 0], [2.5, 0, 0], [3., 0, 0],
                       [3.5, 0, 0], [4., 0, 0], [-1., 0, 0]])
    scan = np.array([[3., 0, 0], [10., 0, 0]])  # nearest return must win
    states = module.range_states(points, scan, np.zeros(3),
                                 dict(h_res_deg=2.5, v_res_deg=2.5, range_margin=.5))
    assert states.tolist() == [0, 5, 3, 4, 3, 2, 1]


def test_scan_ratio_partition_rejects_overlaps_and_preserves_unobserved_columns():
    valid = np.array([0, 1, 1, 1, 1, 1], bool)
    observed = np.array([0, 0, 1, 1, 1, 1], bool)
    stages = [np.arange(6) == i for i in range(2, 6)]
    assert module.encode_scan_ratio(valid, observed, *stages).tolist() == list(range(6))
    stages[0][3] = True
    with pytest.raises(ValueError, match="partition"):
        module.encode_scan_ratio(valid, observed, *stages)


def test_empty_acquisition_frame_is_undefined_and_query_rows_stay_aligned():
    states = np.array([[5, 2, 5], [2, 5, 2]], np.uint8)
    matrix = module.heatmap(states, 5, np.array([1, 1, 0], bool), np.array([0, 0, 1]), 2)
    np.testing.assert_array_equal(matrix[:, 0], [.5, .5])
    assert np.isnan(matrix[:, 1]).all()


def fixture(tmp_path):
    points = np.array([[5., 0, 0], [5., 0, 1], [10., 0, 2]])
    np.save(tmp_path / "scan.npy", points)
    np.save(tmp_path / "gt.npy", np.array([0, 1, 0], np.uint8))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scene": "test", "sensor_profile": {"deskewed": True},
        "frames": [{"cloud": "scan.npy", "point_labels": "gt.npy", "sensor_origin": [0, 0, 0],
                    "timestamp_sec": i * .5} for i in range(3)]}))
    scans, _ = core._load_scan_manifest(manifest)
    accumulated = np.concatenate([p for p, _ in scans])
    gt = np.tile([False, True, False], 3)
    args = core._build_parser().parse_args(["--algorithm", "range_scan_ratio", "--output-cloud", "unused"])
    args.range_h_res = args.range_v_res = 2.5
    params = core._range_scan_ratio_cli_parameters(args)
    _, kr = core.clean_map_by_visibility(accumulated, scans, **params["range"])
    _, ks = core.clean_map_by_scan_ratio(accumulated, scans, **params["scan_ratio"])
    record = {"scene": "test", "parameters": params, "map_points": 9, "gt_dynamic_points": 3,
              "input_hashes": {"manifest_sha256": module.sha256(manifest),
                               "map_sha256": hashlib.sha256(accumulated.tobytes()).hexdigest(),
                               "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()},
              "results": {"none": {"metrics": bench.compute_accuracy_metrics(~kr & ~ks, gt)}}}
    return manifest, record


def test_trace_replays_full_map_and_keeps_point_source_identity(tmp_path):
    path, record = fixture(tmp_path)
    output = tmp_path / "trace"
    report, _ = module.analyze_scene(path, record, output)
    assert report["api_masks_equal"]
    common = report["groups"]["moving_common_miss"]
    assert common["points"] == 3
    assert common["point_scan_opportunities"] == 9
    assert common["scan_ratio_opportunities"]["height_ratio_too_large"] == 9
    trace = np.load(output / "point_scan_evidence.npz")
    np.testing.assert_array_equal(trace["source_frame"], np.repeat([0, 1, 2], 3))
    np.testing.assert_array_equal(trace["source_row"], np.tile([0, 1, 2], 3))
    assert trace["range_state"].shape == (3, 9)
    np.testing.assert_array_equal(np.flatnonzero(trace["group_id"] == 5), [1, 4, 7])
    assert report["groups"]["static_removed"]["points"] == 0
    assert report["groups"]["static_removed"]["scan_ratio"]["vote_quantiles"] is None


def test_changed_fingerprint_or_baseline_metrics_fail_before_writing(tmp_path):
    path, record = fixture(tmp_path)
    record["results"]["none"]["metrics"]["f1"] = .99
    with pytest.raises(ValueError, match="baseline metrics"):
        module.analyze_scene(path, record, tmp_path / "wrong-metric")
    assert not (tmp_path / "wrong-metric").exists()
    np.save(tmp_path / "scan.npy", np.ones((3, 3)))
    with pytest.raises(ValueError, match="fingerprints"):
        module.analyze_scene(path, record, tmp_path / "wrong-input")
    assert not (tmp_path / "wrong-input").exists()


def test_complete_scene_set_is_required_before_any_output(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"plan": {"dataset": "nuscenes-trainval", "scenes": ["a", "b"]},
                                 "scene_results": [{"scene": "a"}]}))
    with pytest.raises(SystemExit) as exc:
        module.main(["--frozen-results", str(frozen), "--manifest-root", str(tmp_path),
                     "--output", str(tmp_path / "out"), "--report-json", str(tmp_path / "report.json")])
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()
