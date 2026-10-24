import hashlib
import json

import numpy as np
import pytest

import dynamic_object_removal as core
from scripts import attribute_nuscenes_nearer_returns as module
from scripts.diagnose_frozen_nuscenes_misses import replay, groups


def test_nearest_row_is_stable_but_equal_ranges_remain_explicit_ties():
    points = np.array([[5., 0, 0], [-5., 0, 0], [0., 0, 0]])
    query = np.array([[3., 0, 0], [8., 0, 0], [3., 0, 0]])
    row, ties, gap = module.nearest_rows(points, query, np.zeros(3), dict(h_res_deg=2.5, v_res_deg=2.5))
    assert row.tolist() == [0, -1, -1]
    assert ties.tolist() == [2, 0, 0]
    assert gap[0] == 2


def annotation(token, sample, center):
    return dict(instance_token=token, sample_token=sample, translation=[center, 0, 0],
                rotation=[1, 0, 0, 0], size=[.2, .2, .2])


def test_ownership_preserves_overlap_and_does_not_call_unannotated_points_static():
    anns = [annotation("a", "sample", 0), annotation("b", "sample", .1), annotation("c", "sample", 5)]
    owner, overlap, gt = module.assign_frame_owners(np.array([[0., 0, 0], [5., 0, 0], [10., 0, 0]]),
        anns, {"a": 0, "b": 1, "c": 2}, {"b"}, .25)
    assert owner.tolist() == [0, 2, -1]
    assert overlap.tolist() == [True, False, False]
    assert gt.tolist() == [True, False, False]


def test_track_relations_partition_events_and_ambiguity_overrides_identity():
    target = np.zeros(7, int)
    nearest = np.array([0, 1, 2, -1, 0, 0, 0])
    ta = np.arange(7) == 4
    na = np.arange(7) == 5
    ties = np.array([1, 1, 1, 1, 1, 1, 2])
    assert module.relations(target, nearest, ta, na, ties, np.array([0, 1])).tolist() == list(range(7))
    ta[-1] = True
    assert module.relations(target, nearest, ta, na, ties, np.array([0, 1]))[-1] == 4


def fixture(tmp_path):
    arrays = [np.array([[5., 0, 0], [3., 0, 0]]), np.array([[2., 0, 0], [3., 0, 0]])]
    frames = []
    for i, cloud in enumerate(arrays):
        np.save(tmp_path / f"cloud{i}.npy", cloud)
        np.save(tmp_path / f"gt{i}.npy", np.array([1, 0], np.uint8))
        frames.append(dict(sample_token=f"sample{i}", cloud=f"cloud{i}.npy", point_labels=f"gt{i}.npy",
                           sensor_origin=[0, 0, 0], timestamp_sec=i * .5))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(scene="test", sensor_profile={"deskewed": True}, frames=frames)))
    scans, _ = core._load_scan_manifest(path)
    points = np.concatenate(arrays)
    gt = np.array([1, 0, 1, 0], bool)
    args = core._build_parser().parse_args(["--algorithm", "range_scan_ratio", "--output-cloud", "unused"])
    args.range_h_res = args.range_v_res = 2.5
    params = core._range_scan_ratio_cli_parameters(args)
    evidence = replay(points, scans, params)
    group_id = groups(gt, evidence["range_dynamic"], evidence["scan_ratio_dynamic"])
    trace = tmp_path / "trace.npz"
    np.savez_compressed(trace, xyz=points, gt_dynamic=gt, source_frame=[0, 0, 1, 1],
                        group_id=group_id, range_state=evidence["range_state"])
    hashes = dict(manifest_sha256=module.sha256(path), map_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                  gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest())
    baseline = dict(scene="test", input_hashes=hashes)
    record = dict(input_hashes=hashes, trace_sha256=module.sha256(trace),
                  groups={"moving_common_miss": {"range_opportunities": {"nearer_return": 2}}})
    tables = {"ann": [annotation("moving", "sample0", 5), annotation("moving", "sample1", 2),
                      annotation("stationary", "sample0", 3), annotation("stationary", "sample1", 3)]}
    plan = dict(moving_threshold_m=2., box_margin_m=.25, parameters=params)
    return path, baseline, record, trace, tables, ["sample0", "sample1"], plan


def test_event_indices_use_query_time_owner_and_correct_source_frame(tmp_path):
    args = fixture(tmp_path)
    report = module.analyze_scene(*args, tmp_path / "out")
    assert report["moving_gt_reconstructed_exactly"]
    assert report["populations"]["all_queries"]["counts"]["same_moving_track"] == 1
    assert report["populations"]["same_acquisition_scan"]["counts"]["track_not_moving_by_window_rule"] == 1
    with np.load(tmp_path / "out/nearer_return_events.npz") as events:
        assert events["target_map_index"].tolist() == [0, 0]
        assert events["nearest_map_index"].tolist() == [1, 2]
        assert events["query_frame"].tolist() == [0, 1]
        assert events["relation"].tolist() == [2, 0]


def test_changed_trace_or_unreproduced_annotation_gt_rejected_before_output(tmp_path):
    args = fixture(tmp_path)
    args[2]["trace_sha256"] = "wrong"
    with pytest.raises(ValueError, match="trace fingerprint"):
        module.analyze_scene(*args, tmp_path / "bad-trace")
    assert not (tmp_path / "bad-trace").exists()
    args[2]["trace_sha256"] = module.sha256(args[3])
    args[4]["ann"][0]["translation"] = [100., 0, 0]
    with pytest.raises(AssertionError):
        module.analyze_scene(*args, tmp_path / "bad-annotations")
    assert not (tmp_path / "bad-annotations").exists()


def test_empty_event_summary_reports_no_quantiles():
    report = module.summarize_events(np.array([], np.uint8), np.array([], int), np.array([]), np.array([]))
    assert report["events"] == report["distinct_target_points"] == 0
    assert all(v["angular_separation_quantiles_deg"] is None for v in report["details"].values())
