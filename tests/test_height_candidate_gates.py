from __future__ import annotations

import pytest

from scripts.check_height_candidate_gates import evaluate_gates


def _ablation(f1: float, static: float, *, deskewed: bool = True):
    return {
        "algorithm": "height_candidate_experimental",
        "deskew_input_contract_satisfied": deskewed,
        "baseline": {"metrics": {"f1": f1, "static_preservation": static}},
        "best_candidate": {"metrics": {"f1": f1, "static_preservation": static}},
    }


def _dynamicmap(candidate_00: float = 98.4, candidate_05: float = 97.8):
    return {"results": {
        "00": {"fusion": {"AA": 98.6}, "fusion_height_candidate": {"AA": candidate_00}},
        "05": {"fusion": {"AA": 98.0}, "fusion_height_candidate": {"AA": candidate_05}},
    }}


def test_height_candidate_all_gates_pass_at_public_precision():
    report = evaluate_gates(
        _ablation(0.6566, 0.9736),
        _ablation(0.6416, 0.8416),
        _ablation(0.20, 0.90),
        _dynamicmap(),
    )
    assert report["all_gates_pass"]
    assert report["promotion_ready"]


def test_height_candidate_requires_deskewed_sparse_input():
    report = evaluate_gates(
        _ablation(0.712, 0.988),
        _ablation(0.846, 0.966, deskewed=False),
        _ablation(0.20, 0.90),
        _dynamicmap(),
    )
    assert not report["all_gates_pass"]
    assert not report["checks"]["nuscenes_deskewed"]


def test_height_candidate_requires_dynamicmap_candidate_results():
    with pytest.raises(ValueError, match="fusion_height_candidate"):
        evaluate_gates(
            _ablation(0.712, 0.988),
            _ablation(0.846, 0.966),
            _ablation(0.20, 0.90),
            {"results": {"00": {"fusion": {"AA": 98.6}}, "05": {}}},
        )


def test_height_candidate_rejects_heldout_f1_regression():
    heldout = _ablation(0.20, 0.90)
    heldout["best_candidate"]["metrics"]["f1"] = 0.18
    report = evaluate_gates(
        _ablation(0.712, 0.988),
        _ablation(0.846, 0.966),
        heldout,
        _dynamicmap(),
    )
    assert not report["all_gates_pass"]
    assert not report["checks"]["heldout_nuscenes_f1_non_regression"]
