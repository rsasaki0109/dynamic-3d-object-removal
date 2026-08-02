from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import run_range_prob_ablation as rp


def _frame(index, points, origin=(0.0, 0.0, 0.0)):
    points = np.asarray(points, dtype=np.float64)
    return rp._Frame(index, float(index), points, points, np.asarray(origin), np.zeros(len(points), bool))


def test_beta_residual_flags_transient_cluster():
    target_points = np.array([[5.0, y, z] for y in (-0.05, 0.0, 0.05) for z in (-0.05, 0.0, 0.05)])
    background = np.array([[10.0, y * 2, z * 2] for _, y, z in target_points])
    target = _frame(1, target_points)
    previous = _frame(0, background)
    following = _frame(2, background)

    dynamic, probability = rp.range_prob_mask(
        previous, target, following,
        h_res_deg=1.0, v_res_deg=1.0, range_margin=0.5,
        alpha_prior=1.0, beta_prior=1.0, probability_threshold=0.7,
        min_dynamic_votes=2, min_cluster_points=3,
    )

    assert dynamic.all()
    np.testing.assert_allclose(probability, 0.75)


def test_surface_confirmation_rejects_static_points():
    points = np.array([[5.0, y, 0.0] for y in (-0.05, 0.0, 0.05)])
    target = _frame(1, points)
    dynamic, probability = rp.range_prob_mask(
        _frame(0, points), target, _frame(2, points),
        h_res_deg=1.0, v_res_deg=1.0, range_margin=0.5,
        alpha_prior=1.0, beta_prior=1.0, probability_threshold=0.7,
        min_dynamic_votes=2, min_cluster_points=1,
    )
    assert not dynamic.any()
    np.testing.assert_allclose(probability, 0.25)


def test_range_cluster_removes_isolated_candidate():
    points = np.array([[5.0, 0.0, 0.0], [5.0, 2.0, 1.0]])
    mask = rp._cluster_range_candidates(
        points, np.zeros(3), np.ones(2, bool),
        h_res_deg=1.0, v_res_deg=1.0, min_cluster_points=2,
    )
    assert not mask.any()


def test_parser_records_one_frame_delay_configuration():
    args = rp.build_parser().parse_args(["--manifest", "m.json", "--summary-json", "s.json"])
    assert args.min_dynamic_votes == 2
    assert args.probability_threshold == 0.7


def test_baseline_comparison_uses_only_interior_frames():
    frame = {
        "true_positive": 2, "false_positive": 1,
        "false_negative": 2, "true_negative": 5,
    }
    baseline = {"algorithm": "range", "scenarios": [{"per_frame": [frame, frame, frame, frame]}]}
    result = {
        "evaluated_metrics": {"f1": 0.7, "static_preservation": 0.9},
        "filter_latency": {"p95_ms": 20.0},
        "period_ms": 100.0,
        "deskew_input_contract_satisfied": True,
    }
    comparison = rp.compare_with_range_baseline(result, baseline)
    assert comparison["baseline_interior_metrics"]["true_positive"] == 4
    assert comparison["delta_f1"] > 0.0
    assert comparison["single_dataset_gate_pass"]
    assert not comparison["promotion_ready"]


def test_baseline_comparison_rejects_tradeoff_with_large_regression():
    frame = {
        "true_positive": 2, "false_positive": 1,
        "false_negative": 2, "true_negative": 5,
    }
    baseline = {"algorithm": "range", "scenarios": [{"per_frame": [frame] * 4}]}
    result = {
        "evaluated_metrics": {"f1": 0.2, "static_preservation": 0.99},
        "filter_latency": {"p95_ms": 20.0},
        "period_ms": 100.0,
        "deskew_input_contract_satisfied": True,
    }
    comparison = rp.compare_with_range_baseline(result, baseline)
    assert comparison["delta_static_preservation"] > 0.0
    assert not comparison["accuracy_gate_pass"]
    assert not comparison["single_dataset_gate_pass"]
