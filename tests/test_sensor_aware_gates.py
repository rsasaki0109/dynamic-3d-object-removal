from __future__ import annotations

import pytest

from scripts.check_sensor_aware_gates import evaluate_gates


def _summaries():
    av2 = {
        "sensor_aware": {
            "baseline_strategy": "fusion",
            "selected_metrics": {"f1": 0.657091, "static_preservation": 0.973881},
        }
    }
    nuscenes = {
        "sensor_aware": {
            "baseline_strategy": "range_and_scan_ratio",
            "selected_metrics": {"f1": 0.641749, "static_preservation": 0.841699},
            "best_normalized_range_and_scan_ratio_candidate": {
                "metrics": {"f1": 0.642004, "static_preservation": 0.842618}
            },
        }
    }
    dynamicmap = {
        "results": {
            "00": {"fusion": {"AA": 98.6}},
            "05": {"fusion": {"AA": 98.0}},
        }
    }
    return av2, nuscenes, dynamicmap


def test_all_baseline_gates_pass_at_public_precision():
    report = evaluate_gates(*_summaries())
    assert report["all_non_regression_gates_pass"]
    assert all(report["checks"].values())
    assert report["normalized_candidate"]["same_scene_nuscenes_improvement"]
    assert not report["normalized_candidate"]["promotion_ready"]


def test_dynamicmap_regression_fails_gate():
    av2, nuscenes, dynamicmap = _summaries()
    dynamicmap["results"]["05"]["fusion"]["AA"] = 97.79
    report = evaluate_gates(av2, nuscenes, dynamicmap)
    assert not report["checks"]["dynamicmap_05_aa"]
    assert not report["all_non_regression_gates_pass"]


def test_missing_sensor_ablation_is_rejected():
    _, nuscenes, dynamicmap = _summaries()
    with pytest.raises(ValueError, match="sensor-aware-ablation"):
        evaluate_gates({}, nuscenes, dynamicmap)
