"""Exercise the sparse validator with exported poses and known point labels."""
import json

import numpy as np
import pytest

from scripts.run_nuscenes_benchmark import _export_online_manifest
from scripts.validate_nuscenes_cli import validate


def test_sparse_validator_preserves_profile_and_matches_cli(tmp_path):
    cloud = np.array([[5., 0., 0.], [5., 0., 1.], [10., 0., 2.]])
    manifest = tmp_path / "manifest.json"
    _export_online_manifest(manifest, scene="scene-test", stride=3,
        local_scans=[cloud] * 3,
        gt_masks=[np.array([False, True, False])] * 3,
        poses=[(np.eye(3), np.zeros(3))] * 3, timestamps_sec=[0., 1.5, 3.])
    output = tmp_path / "results"
    report = validate(manifest, output)
    assert report["sensor_profile"]["deskewed"] is False
    assert report["gt_dynamic_points"] == 3
    assert report["map_points"] == 9
    for result in report["results"].values():
        assert result["api_cli_masks_equal"]
        assert result["api_cli_points_equal"]
        assert result["parameters"]["range"]["h_res_deg"] == 2.5
    assert report["results"]["defaults"]["parameters"]["range"]["ground_z"] is None
    assert report["results"]["ground_protected"]["parameters"]["range"]["ground_z"] == 0
    assert json.loads((output / "validation.json").read_text()) == report
    with pytest.raises(ValueError, match="new or empty"):
        validate(manifest, output)
