"""End-to-end manifest CLI checks, including sensor-to-map geometry."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import dynamic_object_removal as core


def make_scene(root):
    rng = np.random.default_rng(14)
    points = rng.uniform([-8, -8, -1], [8, 8, 3], size=(220, 3))
    np.save(root / "map.npy", points)
    scans = []
    frames = []
    for index in range(4):
        origin = np.array([index * 0.25, -0.5, 0.1])
        scan = np.vstack([points[rng.choice(len(points), 100, replace=False)],
                          rng.uniform([-10, -10, -2], [10, 10, 4], size=(50, 3))])
        # A 90-degree yaw checks rotation direction, not only translation.
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.save(root / f"scan{index}.npy", (scan - origin) @ rotation)
        frames.append({"cloud": f"scan{index}.npy", "pose": {
            "translation": origin.tolist(),
            "quaternion_xyzw": [0, 0, 2**-0.5, 2**-0.5],
        }})
        scans.append((scan, origin))
    manifest = {"sensor_profile": {"name": "test", "beams": 64, "deskewed": True},
                "frames": frames}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return points, scans, manifest, path


def cli_args(root, path):
    return ["--algorithm", "fusion", "--input-map", str(root / "map.npy"),
            "--input-manifest", str(path), "--output-cloud", str(root / "out.npy"),
            "--summary-json", str(root / "reports" / "summary.json"), "--quiet"]


@pytest.mark.parametrize("workers", [1, 2])
def test_fusion_cli_matches_api_from_other_directory(tmp_path, workers):
    points, scans, _, path = make_scene(tmp_path)
    args = cli_args(tmp_path, path) + [
        "--output-mask", str(tmp_path / "keep.npy"),
        "--fusion-workers", str(workers), "--fusion-max-range", "12",
        "--fusion-free-votes-fraction", "0.7", "--fusion-free-votes-floor", "1",
        "--fusion-void-min-scans", "4",
    ]
    result = subprocess.run([sys.executable, "-m", "dynamic_object_removal", *args],
                            cwd=tmp_path.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    expected, mask = core.clean_map_by_fusion(
        points, scans, max_range=12, free_votes_fraction=0.7,
        free_votes_floor=1, void_min_scans=4, workers=workers,
    )
    assert (~mask).any(), "fixture must exercise actual removal"
    assert mask.any(), "fixture must also preserve points"
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), expected)
    np.testing.assert_array_equal(np.load(tmp_path / "keep.npy"), mask)
    summary = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert summary["removed_points"] == int((~mask).sum())
    assert summary["scan_count"] == 4
    assert summary["parameters"]["free_votes_fraction"] == 0.7
    assert summary["parameters"]["workers"] == workers
    assert summary["filter_seconds"] >= 0


def test_manifest_map_frame_and_matrix_pose(tmp_path):
    _, scans, manifest, path = make_scene(tmp_path)
    manifest["frames"][0] = {"cloud": "aligned.npy", "sensor_origin": scans[0][1].tolist()}
    np.save(tmp_path / "aligned.npy", scans[0][0])
    manifest["frames"][1]["pose"].pop("quaternion_xyzw")
    manifest["frames"][1]["pose"]["rotation"] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    path.write_text(json.dumps(manifest))
    loaded, _ = core._load_scan_manifest(path)
    for (points, origin), (expected, expected_origin) in zip(loaded, scans):
        np.testing.assert_allclose(points, expected, atol=1e-12)
        np.testing.assert_array_equal(origin, expected_origin)


@pytest.mark.parametrize("change,message", [
    (lambda m: m["sensor_profile"].update(deskewed=False), "deskewed"),
    (lambda m: m.update(frames=[]), "nonempty"),
    (lambda m: m["frames"][0].pop("pose"), "exactly one"),
    (lambda m: m["frames"][0].update(sensor_origin=[0, 0, 0]), "exactly one"),
    (lambda m: m["frames"][0]["pose"].update(quaternion_xyzw=[0, 0, 0, 0]), "nonzero"),
    (lambda m: m["frames"][0]["pose"].update(translation=[float("nan"), 0, 0]), "finite"),
    (lambda m: m["frames"][0].update(cloud="missing.npy"), "frame 0"),
    (lambda m: m["frames"][0].update(pose={"translation": [0, 0, 0],
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]}), "orthonormal"),
])
def test_invalid_manifest_fails_without_output(tmp_path, capsys, change, message):
    _, _, manifest, path = make_scene(tmp_path)
    change(manifest)
    path.write_text(json.dumps(manifest))
    assert core.main(cli_args(tmp_path, path)) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.npy").exists()


@pytest.mark.parametrize("option,value", [
    ("--fusion-free-step", "0"), ("--fusion-workers", "0"),
    ("--fusion-free-votes-fraction", "1.1"), ("--fusion-max-range", "nan"),
    ("--fusion-max-range", "0.5"),
])
def test_invalid_fusion_parameters(tmp_path, option, value):
    _, _, _, path = make_scene(tmp_path)
    assert core.main(cli_args(tmp_path, path) + [option, value]) == 1
    assert not (tmp_path / "out.npy").exists()


def test_fusion_required_inputs(tmp_path):
    assert core.main(["--algorithm", "fusion", "--output-cloud", str(tmp_path / "out.npy")]) == 1


def test_cli_defaults_match_python_api():
    import inspect
    signature = inspect.signature(core.clean_map_by_fusion)
    assert core._FUSION_CLI_DEFAULTS == {
        name: param.default for name, param in signature.parameters.items()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    }


@pytest.mark.parametrize("preset,options,expected", [
    (None, [], {}),
    ("long-map", ["--preset", "long-map"], {}),
    ("short-window", ["--preset", "short-window"],
     {"free_votes_fraction": 0.7, "free_votes_floor": 3, "void_min_scans": 4}),
    # Explicit values equal to API defaults must still override the preset.
    ("short-window", ["--preset", "short-window", "--fusion-free-votes-fraction", "0.9",
                      "--fusion-free-votes-floor", "2"],
     {"free_votes_fraction": 0.9, "free_votes_floor": 2, "void_min_scans": 4}),
    ("short-window", ["--fusion-free-votes-fraction", "0.9",
                      "--fusion-free-votes-floor", "2", "--preset", "short-window"],
     {"free_votes_fraction": 0.9, "free_votes_floor": 2, "void_min_scans": 4}),
])
def test_presets_match_api_and_record_effective_settings(tmp_path, preset, options, expected):
    points, scans, _, path = make_scene(tmp_path)
    assert core.main(cli_args(tmp_path, path) + options) == 0
    filtered, mask = core.clean_map_by_fusion(points, scans, **expected)
    assert mask.any() and (~mask).any()
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), filtered)
    summary = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert summary["preset"] == preset
    assert summary["parameters"] == dict(core._FUSION_CLI_DEFAULTS, **expected)


def test_invalid_override_is_not_replaced_by_preset(tmp_path, capsys):
    _, _, _, path = make_scene(tmp_path)
    assert core.main(cli_args(tmp_path, path) + [
        "--fusion-free-votes-floor", "0", "--preset", "short-window",
    ]) == 1
    assert "free_votes_floor must be positive" in capsys.readouterr().err
    assert not (tmp_path / "out.npy").exists()


@pytest.mark.parametrize("options,message", [
    (["--algorithm", "fusion", "--preset", "unknown"], "invalid choice"),
    (["--algorithm", "box", "--preset", "short-window"], "requires --algorithm fusion"),
    (["--algorithm", "range", "--preset", "long-map"], "requires --algorithm fusion"),
])
def test_invalid_preset_usage(tmp_path, capsys, options, message):
    with pytest.raises(SystemExit) as exc:
        core.main(options + ["--output-cloud", str(tmp_path / "out.npy")])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.npy").exists()


@pytest.mark.parametrize("target", ["map.npy", "manifest.json", "scan0.npy", "reports/summary.json"])
def test_outputs_cannot_overwrite_inputs_or_each_other(tmp_path, capsys, target):
    _, _, _, path = make_scene(tmp_path)
    destination = tmp_path / target
    original = destination.read_bytes() if destination.exists() else None
    assert core.main(cli_args(tmp_path, path) + ["--output-cloud", str(destination)]) == 1
    assert "paths" in capsys.readouterr().err
    if original is not None:
        assert destination.read_bytes() == original


def test_mask_cannot_overwrite_scan(tmp_path):
    _, _, _, path = make_scene(tmp_path)
    scan_path = tmp_path / "scan0.npy"
    original = scan_path.read_bytes()
    assert core.main(cli_args(tmp_path, path) + ["--output-mask", str(scan_path)]) == 1
    assert scan_path.read_bytes() == original


def test_fusion_parameters_rejected_by_single_scan_cli(tmp_path):
    with pytest.raises(SystemExit) as exc:
        core.main(["--algorithm", "range", "--fusion-workers", "2",
                   "--output-cloud", str(tmp_path / "out.npy")])
    assert exc.value.code == 2
