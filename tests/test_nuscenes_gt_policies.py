import hashlib
import json

import numpy as np
import pytest

import bench
import dynamic_object_removal as core
from scripts.compare_nuscenes_gt_policies import compare_scene, export_manifest, labels_from_tracks
from scripts.experiment_neighbor_columns import clean
from scripts.nuscenes_gt_policies import POLICIES, definition, select_tracks
from scripts.validate_frozen_nuscenes_suite import sha256


ATTR = {"moving": "pedestrian.moving", "parked": "vehicle.parked", "rider": "cycle.with_rider"}


def ann(token, xyz, attributes=()):
    return dict(instance_token=token, translation=xyz, attribute_tokens=list(attributes), size=[.2, .2, .2],
                rotation=[1, 0, 0, 0])


def test_vertical_shift_retains_legacy_default_and_horizontal_policy_is_explicit():
    rows = {"a": [ann("barrier", [0, 0, 0])], "b": [ann("barrier", [0, 0, 3])]}
    assert select_tracks(rows, ["a", "b"])["moving"] == {"barrier"}
    assert definition()["motion_policy"] == "xyz"
    for policy in ("xy", "xy_or_moving_attribute"):
        assert select_tracks(rows, ["a", "b"], policy=policy, attribute_names=ATTR)["moving"] == set()


def test_attributes_admit_slow_and_single_observation_movers_but_not_rider_or_parked():
    rows = {"a": [ann("slow", [0, 0, 0], ["moving"]), ann("single", [10, 0, 0], ["moving"]),
                   ann("parked", [20, 0, 0], ["parked"]), ann("bike", [30, 0, 0], ["rider"])],
            "b": [ann("slow", [.1, 0, 0], ["parked"])]}
    assert select_tracks(rows, ["a", "b"], policy="xy")["moving"] == set()
    result = select_tracks(rows, ["a", "b"], policy="xy_or_moving_attribute", attribute_names=ATTR)
    assert result["moving"] == {"slow", "single"}
    assert result["tracks"]["single"]["qualifies_by_attribute"]
    assert not result["tracks"]["slow"]["qualifies_by_displacement"]


def test_anchor_is_selected_order_and_cutoff_is_strict():
    rows = {"middle": [ann("car", [1.1, 0, 0])], "last": [ann("car", [2.1, 0, 0])],
            "first": [ann("car", [0, 0, 0])]}
    for policy in POLICIES:
        assert select_tracks(rows, ["first", "middle", "last"], policy=policy, attribute_names=ATTR)["moving"] == {"car"}
        assert not select_tracks(rows, ["middle", "last", "first"], policy=policy, attribute_names=ATTR)["moving"]
    boundary = {"a": [ann("car", [0, 0, 0])], "b": [ann("car", [2, 0, 0])]}
    assert not select_tracks(boundary, ["a", "b"], policy="xy")["moving"]


def test_missing_attributes_duplicates_and_nonfinite_thresholds_are_rejected():
    rows = {"a": [ann("person", [0, 0, 0], ["unknown"])]}
    with pytest.raises(ValueError, match="official attribute mapping"):
        select_tracks(rows, ["a"], policy="xy_or_moving_attribute")
    with pytest.raises(ValueError, match="missing from official mapping"):
        select_tracks(rows, ["a"], policy="xy_or_moving_attribute", attribute_names=ATTR)
    with pytest.raises(ValueError, match="ordered unique"):
        select_tracks(rows, ["a", "a"])
    with pytest.raises(ValueError, match="duplicate instance"):
        select_tracks({"a": rows["a"] * 2}, ["a"])
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            definition("xy", value)


def test_export_keeps_coordinates_but_names_the_new_gt_and_parent_lineage(tmp_path):
    np.save(tmp_path / "cloud.npy", np.array([[5., 0, 0], [6., 0, 0]]))
    np.save(tmp_path / "old.npy", np.array([1, 0], np.uint8))
    path = tmp_path / "original.json"
    original = {"scene": "test", "sensor_profile": {"deskewed": True}, "frozen_plan_sha256": "old",
                "frames": [{"cloud": "cloud.npy", "point_labels": "old.npy", "sensor_origin": [0, 0, 0]}]}
    path.write_text(json.dumps(original))
    export_manifest(path, original, np.array([False, True]), [2], tmp_path / "export", definition("xy"), "old")
    clone = json.loads((tmp_path / "export/manifest.json").read_text())
    assert "frozen_plan_sha256" not in clone and clone["source_frozen_plan_sha256"] == "old"
    assert clone["gt_definition"]["motion_policy"] == "xy"
    before, _ = core._load_scan_manifest(path)
    after, _ = core._load_scan_manifest(tmp_path / "export/manifest.json")
    np.testing.assert_array_equal(before[0][0], after[0][0])
    np.testing.assert_array_equal(np.load(tmp_path / "old.npy"), [1, 0])
    np.testing.assert_array_equal(np.load(tmp_path / "export/labels_0000.npy"), [0, 1])


def test_comparison_changes_gt_only_and_replays_frozen_predictions(tmp_path):
    selected = ["a", "b"]
    rows = {"a": [ann("barrier", [5, 0, 0]), ann("person", [10, 0, 0], ["moving"])],
            "b": [ann("barrier", [5, 0, 3]), ann("person", [10.1, 0, 0], ["moving"])]}
    clouds = [np.array([[5., 0, 0], [10., 0, 0], [40., 40, 0]]),
              np.array([[5., 0, 3], [10.1, 0, 0], [40., 40, 0]])]
    frames = []
    for i, cloud in enumerate(clouds):
        np.save(tmp_path / f"cloud{i}.npy", cloud)
        np.save(tmp_path / f"gt{i}.npy", np.array([1, 0, 0], np.uint8))
        frames.append(dict(sample_token=selected[i], cloud=f"cloud{i}.npy", point_labels=f"gt{i}.npy", sensor_origin=[0, 0, 0]))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(scene="test", sensor_profile={"deskewed": True}, frames=frames)))
    scans, _ = core._load_scan_manifest(path); points = np.concatenate(clouds)
    gt = np.array([1, 0, 0, 1, 0, 0], bool)
    args = core._build_parser().parse_args(["--algorithm", "range_scan_ratio", "--output-cloud", "unused"])
    args.range_h_res = args.range_v_res = 2.5
    params = core._range_scan_ratio_cli_parameters(args)
    _, kr = core.clean_map_by_visibility(points, scans, **params["range"])
    frozen = {}
    for mode in ("none", "either", "visibility_supported"):
        sr, inferred = clean(points, scans, params["scan_ratio"], mode, params["range"])
        frozen[mode] = {"metrics": bench.compute_accuracy_metrics(~kr & sr, gt), "inferred_point_scan_observations": inferred}
    reference = {"scene": "test", "parameters": params, "results": frozen,
        "input_hashes": {"manifest_sha256": sha256(path), "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
                         "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}}
    categories = {"barrier": {"name": "movable_object.barrier"}, "person": {"name": "human.pedestrian.adult"}}
    result = compare_scene(path, reference, {"ann": [dict(a, sample_token=sample) for sample in selected for a in rows[sample]]}, selected,
        ATTR, categories, {key: {"category_token": key} for key in categories}, list(POLICIES),
        {"moving_threshold_m": 2., "box_margin_m": .25, "parent_plan_sha256": "old"}, tmp_path / "out")
    assert result["legacy_gt_reproduced_exactly"] and result["frozen_metrics_and_evidence_counts_reproduced"]
    assert result["prediction_masks_shared_across_gt_policies"]
    assert result["policies"]["xyz"]["moving_gt_points"] == 2
    assert result["policies"]["xy"]["moving_gt_points"] == 0
    hybrid = result["policies"]["xy_or_moving_attribute"]
    assert hybrid["moving_gt_points"] == 2
    assert hybrid["label_changes_from_legacy"] == {"moving_to_static": 2, "static_to_moving": 2}
    np.testing.assert_array_equal(np.load(tmp_path / "gt0.npy"), [1, 0, 0])


def test_export_rejects_partial_frame_or_label_lengths_before_writing(tmp_path):
    with pytest.raises(ValueError, match="frame sizes differ"):
        export_manifest(tmp_path / "missing", {"frames": [{}, {}]}, np.array([True]), [1], tmp_path / "out", definition(), "old")
    assert not (tmp_path / "out").exists()
