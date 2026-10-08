"""Check dynamic-mask intersection, including disagreements between channels."""

import json
import subprocess
import sys

import numpy as np
import pytest

import dynamic_object_removal as core


def make_scene(root):
    points = np.array([
        [5, 0, 0], [5, 0, 1], [10, 0, 2],
        [0, 5, 0], [0, 5, 1],
        [-5, 0, 0], [-5, 0, 1], [-10, 0, 2],
    ], dtype=float)
    scan = np.array([[5, 0, 0], [10, 0, 2], [0, 5, 0], [-10, 0, 2]], dtype=float)
    np.save(root / "map.npy", points)
    np.save(root / "scan.npy", scan)
    manifest = {
        "sensor_profile": {"beams": 32, "deskewed": True},
        "frames": [{"cloud": "scan.npy", "sensor_origin": [0, 0, 0]} for _ in range(3)],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return points, [(scan, np.zeros(3)) for _ in range(3)]


def cli_args(root):
    return ["--algorithm", "range_scan_ratio",
            "--input-map", str(root / "map.npy"),
            "--input-manifest", str(root / "manifest.json"),
            "--output-cloud", str(root / "out.npy"),
            "--summary-json", str(root / "summary.json"), "--quiet"]


def test_intersection_keeps_single_channel_candidates(tmp_path):
    points, scans = make_scene(tmp_path)
    result = subprocess.run([
        sys.executable, "-m", "dynamic_object_removal", *cli_args(tmp_path),
        "--scan-ratio-rings", "1", "--scan-ratio-max-range", "6",
    ], cwd=tmp_path.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    _, keep_range = core.clean_map_by_visibility(
        points, scans, h_res_deg=2.5, v_res_deg=2.5, min_see_through=3, max_surface_hits=5,
    )
    _, keep_sr = core.clean_map_by_scan_ratio(points, scans, n_rings=1, max_range=6)
    assert np.flatnonzero(~keep_range).tolist() == [1, 6]
    assert np.flatnonzero(~keep_sr).tolist() == [1, 4]
    expected = points[keep_range | keep_sr]
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), expected)
    assert len(expected) == 7
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["algorithm"] == "range_scan_ratio"
    assert summary["channel_removed_points"] == {"range": 2, "scan_ratio": 2, "intersection": 1}
    assert summary["removed_points"] == 1
    assert summary["scan_count"] == 3
    assert summary["preset"] is None
    assert summary["parameters"]["range"]["h_res_deg"] == 2.5
    assert summary["parameters"]["scan_ratio"]["max_range"] == 6


@pytest.mark.parametrize("options,range_params,sr_params", [
    ([], {}, {}),
    (["--range-ground-z", "1"], {"ground_z": 1}, {}),
    (["--range-h-res", "1", "--range-v-res", "2",
      "--range-resolutions", "2.5", "4"], {"h_res_deg": 1, "v_res_deg": 2,
                                        "resolutions": [2.5, 4]}, {}),
    (["--scan-ratio-min-votes", "2", "--scan-ratio-votes-fraction", "0.9",
      "--scan-ratio-votes-floor", "1"], {}, {"min_votes": 2, "votes_fraction": 0.9, "votes_floor": 1}),
])
def test_intersection_options_match_apis(tmp_path, options, range_params, sr_params):
    points, scans = make_scene(tmp_path)
    assert core.main(cli_args(tmp_path) + options) == 0
    _, kr = core.clean_map_by_visibility(
        points, scans, **{"h_res_deg": 2.5, "v_res_deg": 2.5,
                         "min_see_through": 3, "max_surface_hits": 5, **range_params},
    )
    _, ks = core.clean_map_by_scan_ratio(points, scans, **sr_params)
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), points[kr | ks])
    summary = json.loads((tmp_path / "summary.json").read_text())
    for name, value in range_params.items():
        assert summary["parameters"]["range"][name] == value
    for name, value in sr_params.items():
        assert summary["parameters"]["scan_ratio"][name] == value


@pytest.mark.parametrize("option,value", [
    ("--range-h-res", "0"), ("--range-v-res", "nan"),
    ("--range-min-see-through", "0"), ("--range-max-surface-hits", "-1"),
    ("--range-ground-z", "inf"), ("--range-resolutions", "-2.5"),
    ("--scan-ratio-rings", "0"), ("--scan-ratio-sectors", "0"),
    ("--scan-ratio-max-range", "0"), ("--scan-ratio-min-votes", "0"),
    ("--scan-ratio-votes-fraction", "1.1"), ("--scan-ratio-votes-floor", "0"),
    ("--scan-ratio-threshold", "-0.1"), ("--scan-ratio-ground-margin", "-1"),
])
def test_invalid_parameters_fail_before_output(tmp_path, capsys, option, value):
    make_scene(tmp_path)
    assert core.main(cli_args(tmp_path) + [option, value]) == 1
    assert "range_scan_ratio:" in capsys.readouterr().err
    assert not (tmp_path / "out.npy").exists()


@pytest.mark.parametrize("options", [
    ["--preset", "short-window"], ["--fusion-workers", "2"],
])
def test_reject_fusion_options(tmp_path, options):
    make_scene(tmp_path)
    with pytest.raises(SystemExit) as exc:
        core.main(cli_args(tmp_path) + options)
    assert exc.value.code == 2
    assert not (tmp_path / "out.npy").exists()


def test_required_manifest(tmp_path, capsys):
    assert core.main(["--algorithm", "range_scan_ratio",
                      "--output-cloud", str(tmp_path / "out.npy")]) == 1
    assert "requires --input-map and --input-manifest" in capsys.readouterr().err


def test_reject_simultaneous_input_cloud(tmp_path, capsys):
    make_scene(tmp_path)
    assert core.main(cli_args(tmp_path) + ["--input-cloud", str(tmp_path / "scan.npy")]) == 1
    assert "instead of --input-cloud" in capsys.readouterr().err


def test_shared_manifest_validation(tmp_path, capsys):
    make_scene(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["sensor_profile"]["deskewed"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert core.main(cli_args(tmp_path)) == 1
    assert "deskewed" in capsys.readouterr().err


def test_single_query_range_retains_original_defaults(tmp_path):
    points, scans = make_scene(tmp_path)
    assert core.main([
        "--algorithm", "range", "--input-map", str(tmp_path / "map.npy"),
        "--input-cloud", str(tmp_path / "scan.npy"),
        "--output-cloud", str(tmp_path / "out.npy"), "--quiet",
    ]) == 0
    expected, _ = core.remove_ghost_by_range_image(points, scans[0][0], scans[0][1])
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), expected)
