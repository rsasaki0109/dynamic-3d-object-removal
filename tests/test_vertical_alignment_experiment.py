import copy
import json

import numpy as np
import pytest

import dynamic_object_removal as core
from scripts.experiment_nuscenes_vertical_alignment import DEFAULTS, estimate, export_corrected, holdout_cells, validate_parameters


def surface():
    x, y = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    cells = np.column_stack((x.ravel(), y.ravel()))
    xy = np.repeat(cells, 4, axis=0) + np.tile([[.1, .1], [.2, .1], [.1, .2], [.2, .2]], (len(cells), 1))
    z = .03 * xy[:, 0] + .02 * xy[:, 1]
    return np.column_stack((xy, z)), cells


def test_known_offsets_recovered_with_fixed_anchor_and_unseen_spatial_cells():
    points, _ = surface()
    frames = [points + [0, 0, z] for z in [0, .4, 1.2, 1.8]]
    result = estimate(frames, copy.deepcopy(DEFAULTS))
    assert result["applied"]
    np.testing.assert_allclose(result["delta_z_m"], [0, -.4, -1.2, -1.8], atol=1e-12)
    assert result["holdout_before"]["cells"] > 0
    assert result["holdout_after"]["p90_abs_residual_m"] < 1e-12


def test_holdout_height_corruption_cannot_change_fit_and_is_visible_in_validation():
    points, cells = surface()
    clean = [points.copy(), points + [0, 0, .7]]
    expected = estimate(clean, copy.deepcopy(DEFAULTS))
    corrupted = [a.copy() for a in clean]
    mask = np.repeat(holdout_cells(cells, DEFAULTS), 4)
    corrupted[1][mask, 2] += 20
    result = estimate(corrupted, copy.deepcopy(DEFAULTS))
    np.testing.assert_allclose(result["delta_z_m"], expected["delta_z_m"], atol=1e-12)
    assert result["holdout_after"]["median_abs_residual_m"] > 19


def test_outlying_training_cells_are_not_treated_as_vertical_motion():
    points, cells = surface()
    moved = points + [0, 0, .6]
    outliers = ~holdout_cells(cells, DEFAULTS) & ((cells[:, 0] + cells[:, 1]) % 7 == 0)
    moved[np.repeat(outliers, 4), 2] += 10
    result = estimate([points, moved], copy.deepcopy(DEFAULTS))
    assert result["applied"]
    np.testing.assert_allclose(result["delta_z_m"], [0, -.6], atol=1e-12)


def test_disconnected_graph_falls_back_for_entire_scene_without_invented_offsets():
    points, _ = surface()
    result = estimate([points, points + [0, 0, 1], points + [100, 100, 10]], copy.deepcopy(DEFAULTS))
    assert not result["applied"]
    assert result["delta_z_m"] == [0., 0., 0.]
    assert result["frames_connected_to_anchor"] == [0, 1]


def test_insufficient_spatial_span_is_rejected():
    points, _ = surface(); points = points[points[:, 1] < 2]
    p = copy.deepcopy(DEFAULTS); p["min_training_cells"] = 5; p["min_holdout_cells"] = 1
    result = estimate([points, points + [0, 0, 1]], p)
    assert not result["applied"]
    assert result["edges"][0]["reason"] == "insufficient_spatial_span"


def test_manifest_export_changes_only_vertical_pose_and_retains_source_labels(tmp_path):
    cloud = np.array([[1., 2, 3], [4., 5, 6]])
    np.save(tmp_path / "cloud.npy", cloud); np.save(tmp_path / "gt.npy", np.array([1, 0], np.uint8))
    source = tmp_path / "source.json"
    manifest = {"sensor_profile": {"deskewed": True}, "gt_definition": {"motion_policy": "xy_or_moving_attribute"},
                "frames": [{"cloud": "cloud.npy", "point_labels": "gt.npy", "pose": {
                    "translation": [1., 2, 3], "rotation": np.eye(3).tolist()}}]}
    source.write_text(json.dumps(manifest))
    export_corrected(source, manifest, [2.], tmp_path / "corrected", "digest")
    a, _ = core._load_scan_manifest(source)
    b, _ = core._load_scan_manifest(tmp_path / "corrected/manifest.json")
    np.testing.assert_array_equal(a[0][0] + [0, 0, 2], b[0][0])
    np.testing.assert_array_equal(a[0][1] + [0, 0, 2], b[0][1])
    corrected = json.loads((tmp_path / "corrected/manifest.json").read_text())
    assert corrected["gt_definition"] == manifest["gt_definition"]
    np.testing.assert_array_equal(np.load(tmp_path / "corrected" / corrected["frames"][0]["point_labels"]), [1, 0])
    assert json.loads(source.read_text())["frames"][0]["pose"]["translation"] == [1, 2, 3]


@pytest.mark.parametrize("key,value", [("cell_size_m", 0), ("height_quantile", 2), ("holdout_modulus", 1), ("pair_lags", [0])])
def test_invalid_estimator_parameters_rejected(key, value):
    p = copy.deepcopy(DEFAULTS); p[key] = value
    with pytest.raises(ValueError):
        validate_parameters(p)
