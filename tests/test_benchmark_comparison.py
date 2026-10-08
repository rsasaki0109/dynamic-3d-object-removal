from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import compare_benchmark_results as comparison
from scripts import run_online_benchmark as online
from scripts import run_av2_benchmark as av2
from scripts import run_nuscenes_benchmark as nuscenes


def single_scene(dataset="argoverse-2-sensor-val", scene="a"):
    return {
        "dataset": dataset, "scene": scene, "frames": 12, "stride": 3,
        "map_points": 10000, "gt_dynamic_points": 6000,
        "config": {"h_res": 1.0}, "method_keys": ["range", "fusion"],
        "metrics": {method: {"precision": 0.7, "recall": 0.6, "f1": 0.64,
                             "static_preservation": 0.97}
                    for method in ("range", "fusion")},
        "runtime_seconds": 10.0,
    }


def multiscene(dataset):
    scenes = [single_scene(dataset, name) for name in ("a", "b")]
    runner = av2 if dataset == "argoverse-2-sensor-val" else nuscenes
    aggregate = runner._aggregate_scene_results(scenes, scenes[0]["method_keys"])
    return {"dataset": dataset, "scenes": ["a", "b"], "frames": 12, "stride": 3,
            "config": {"h_res": 1.0}, "scene_results": scenes, "aggregate": aggregate}


@pytest.mark.parametrize("dataset", ["argoverse-2-sensor-val", "nuscenes-mini"])
def test_existing_single_and_aggregate_schemas(dataset):
    for payload in (single_scene(dataset), multiscene(dataset)):
        report = comparison.compare(payload, deepcopy(payload))
        assert report["status"] == "passed"
        assert report["measurement_count"] > 0
        assert report["timing_measured"]


def test_regression_and_threshold_boundary():
    old = single_scene()
    new = deepcopy(old)
    new["metrics"]["fusion"]["f1"] = 0.63
    new["runtime_seconds"] = 12
    report = comparison.compare(old, new)
    assert report["regression_count"] == 1
    assert comparison.compare(old, new, max_metric_drop=0.01)["status"] == "passed"
    new["runtime_seconds"] = 12.01
    assert comparison.compare(old, new, max_metric_drop=0.01)["regression_count"] == 1


def test_zero_runtime_baseline_no_division_by_zero():
    old = single_scene()
    old["runtime_seconds"] = 0
    new = deepcopy(old)
    new["runtime_seconds"] = 1
    report = comparison.compare(old, new)
    timing = next(m for m in report["measurements"] if m["metric"] == "runtime_seconds")
    assert timing["regressed"]
    assert timing["relative_change"] is None


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(dataset="nuscenes-mini"),
    lambda p: p.update(scene="other"),
    lambda p: p.update(frames=11),
    lambda p: p["config"].update(h_res=2.5),
    lambda p: p.update(gt_dynamic_points=5000),
    lambda p: p["metrics"].pop("fusion"),
    lambda p: p.pop("runtime_seconds"),
])
def test_reject_incompatible_comparisons(mutate):
    old = single_scene()
    new = deepcopy(old)
    mutate(new)
    with pytest.raises(ValueError):
        comparison.compare(old, new)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1, True, "0.7"])
def test_missing_or_invalid_metrics_cannot_pass(value):
    payload = single_scene()
    payload["metrics"]["fusion"]["f1"] = value
    with pytest.raises(ValueError):
        comparison.compare(payload, payload)


def test_missing_metric_cannot_pass():
    payload = single_scene()
    payload["metrics"]["fusion"].pop("f1")
    with pytest.raises(ValueError, match="missing"):
        comparison.compare(payload, payload)


def test_per_scene_regression_not_hidden_by_mean():
    old = multiscene("argoverse-2-sensor-val")
    new = deepcopy(old)
    new["scene_results"][0]["metrics"]["fusion"]["f1"] -= 0.1
    new["scene_results"][1]["metrics"]["fusion"]["f1"] += 0.1
    new["aggregate"] = av2._aggregate_scene_results(new["scene_results"], ["range", "fusion"])
    report = comparison.compare(old, new)
    assert report["regression_count"] == 1
    assert next(m for m in report["measurements"] if m["regressed"])["case"] == "a"


def test_reject_stale_aggregate():
    payload = multiscene("nuscenes-mini")
    payload["aggregate"]["methods"]["fusion"]["f1"] += 0.1
    with pytest.raises(ValueError, match="means"):
        comparison.compare(payload, payload)


def test_reject_empty_eligible_mean():
    payload = multiscene("nuscenes-mini")
    for scene in payload["scene_results"]:
        scene["gt_dynamic_points"] = 1
    payload["aggregate"] = nuscenes._aggregate_scene_results(payload["scene_results"], ["range", "fusion"])
    with pytest.raises(ValueError, match="at least one"):
        comparison.compare(payload, payload)


def test_dynamicmap_percent_units():
    old = {"dataset": "DynamicMap_Benchmark/Semantic-KITTI", "sequences": ["00"],
           "config": {"eval_max_dist": 100},
           "results": {"00": {"fusion": {"SA": 98.0, "DA": 97.0, "AA": 97.5, "HA": 95.0}}}}
    new = deepcopy(old)
    new["results"]["00"]["fusion"]["AA"] = 96.5
    assert comparison.compare(old, new)["regression_count"] == 1
    report = comparison.compare(old, new, max_metric_drop=0.01)
    assert report["status"] == "passed"
    assert not report["timing_measured"]


def test_cli_exit_codes_and_json_report(tmp_path, capsys):
    old = single_scene()
    new = deepcopy(old)
    before, after, report = [tmp_path / name for name in ("old.json", "new.json", "report.json")]
    before.write_text(json.dumps(old))
    args = ["--baseline", str(before), "--candidate", str(after), "--report-json", str(report)]
    after.write_text(json.dumps(new))
    assert comparison.main(args) == 0
    new["metrics"]["range"]["static_preservation"] -= 0.01
    after.write_text(json.dumps(new))
    assert comparison.main(args) == 1
    assert json.loads(report.read_text())["status"] == "regression"
    new["config"]["h_res"] = 2.5
    after.write_text(json.dumps(new))
    assert comparison.main(args) == 2
    assert json.loads(report.read_text())["status"] == "comparison_error"
    assert "comparison error" in capsys.readouterr().err


def test_report_cannot_overwrite_baseline(tmp_path):
    path = tmp_path / "old.json"
    original = json.dumps(single_scene())
    path.write_text(original)
    assert comparison.main(["--baseline", str(path), "--candidate", str(path),
                            "--report-json", str(path)]) == 2
    assert path.read_text() == original


@pytest.mark.parametrize("name", ["frames", "stride", "map_points"])
def test_empty_runs_cannot_pass(name):
    payload = single_scene()
    payload[name] = 0
    with pytest.raises(ValueError, match="empty runs"):
        comparison.compare(payload, payload)


def test_actual_online_runner_output(tmp_path):
    frames = []
    for index in range(3):
        cloud = tmp_path / f"scan{index}.npy"
        labels = tmp_path / f"labels{index}.npy"
        np.save(cloud, np.array([[10, 0, 0], [10, 0.2, 0], [5 + index, 1, 0]], float))
        np.save(labels, np.array([0, 0, 1], dtype=np.uint8))
        frames.append({"cloud": cloud.name, "point_labels": labels.name,
                       "timestamp_sec": index * 0.1,
                       "pose": {"translation": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sensor_profile": {"deskewed": True, "rate_hz": 10},
                                    "frames": frames}))
    output = tmp_path / "online.json"
    assert online.main(["--manifest", str(manifest), "--algorithm", "range",
                        "--summary-json", str(output), "--pose-noise-translation", "0.05"]) == 0
    old = json.loads(output.read_text())
    assert comparison.compare(old, deepcopy(old))["status"] == "passed"
    new = deepcopy(old)
    new["scenarios"][0]["filter_latency"]["p95_ms"] *= 2
    new["scenarios"][0]["fail_open_frames"] += 1
    assert comparison.compare(old, new)["regression_count"] == 2
    new["scenarios"].pop()
    with pytest.raises(ValueError, match="metadata"):
        comparison.compare(old, new)
