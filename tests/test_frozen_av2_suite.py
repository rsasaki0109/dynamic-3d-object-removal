import copy
import json
from pathlib import Path

import pytest

from scripts import validate_frozen_av2_suite as suite


SCENES = [f"00000000-0000-0000-0000-{i:012d}" for i in range(4)]


def test_selection_is_sorted_excludes_seen_and_never_backfills_by_metrics():
    assert suite.select_scenes(list(reversed(SCENES)), [SCENES[0]], 2) == SCENES[1:3]
    with pytest.raises(ValueError, match="not enough"):
        suite.select_scenes(SCENES, SCENES[:3], 2)
    with pytest.raises(ValueError, match="unique"):
        suite.select_scenes([SCENES[0], SCENES[0]], [], 1)


def frozen_plan(tmp_path):
    catalogue, seen = tmp_path / "catalogue.json", tmp_path / "seen.json"
    catalogue.write_text(json.dumps({"source_url": "public catalogue", "scenes": SCENES}))
    seen.write_text(json.dumps([SCENES[0]]))
    return suite.freeze(catalogue, seen, suite.ROOT / "examples/cli_validation/av2_coordinate_ground_fix.json", 3)


def test_plan_rejects_changed_code_selection_and_absolute_ground(tmp_path, monkeypatch):
    plan = frozen_plan(tmp_path)
    suite.verify_plan(plan)
    changed = copy.deepcopy(plan)
    changed["scenes"].reverse()
    with pytest.raises(ValueError, match="scene selection"):
        suite.verify_plan(changed)
    changed = copy.deepcopy(plan)
    changed["parameters"]["range"]["ground_z"] = -1.4
    with pytest.raises(ValueError, match="map-Z"):
        suite.verify_plan(changed)
    real_hash = suite.sha256
    monkeypatch.setattr(suite, "sha256", lambda path: "changed" if Path(path).name == "bench.py" else real_hash(path))
    with pytest.raises(ValueError, match="code changed"):
        suite.verify_plan(plan)


def test_aggregate_retains_zero_gt_and_exposes_regression_behind_better_mean():
    records = []
    for i, (baseline, candidate) in enumerate(((.5, .4), (.1, .5), (0, 0))):
        results = {}
        for mode in suite.MODES:
            f1 = baseline if mode == "none" else candidate
            static = 1.0 if mode == "none" or i != 0 else .99
            results[mode] = {"metrics": {k: static if k == "static_preservation" else f1 for k in suite.METRICS}}
        records.append({"scene": SCENES[i], "gt_dynamic_points": 0 if i == 2 else 100, "results": results})
    report = suite.aggregate(records)
    assert report["included_scenes"] == SCENES[:3]
    assert report["excluded_scenes"] == []
    assert report["zero_moving_gt_scenes"] == [SCENES[2]]
    candidate = report["methods"]["either"]
    assert candidate["mean"]["f1"] > report["methods"]["none"]["mean"]["f1"]
    assert candidate["f1_regression_scenes"] == [SCENES[0]]
    assert candidate["static_regression_scenes"] == [SCENES[0]]
    assert candidate["worst_scene_static_preservation"] == .99


def test_evaluation_rejects_mutated_plan_before_reading_inputs(tmp_path):
    plan = tmp_path / "plan.json"
    suite.write_new(plan, frozen_plan(tmp_path))
    digest = suite.sha256(plan)
    plan.write_text(plan.read_text() + "\n")
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        suite.main(["evaluate", "--plan-json", str(plan), "--plan-sha256", digest,
                    "--manifest-root", str(tmp_path / "missing"), "--data-root", str(tmp_path / "missing"),
                    "--report-json", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_existing_plan_or_report_is_never_overwritten(tmp_path):
    path = tmp_path / "record.json"
    suite.write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        suite.write_new(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}
