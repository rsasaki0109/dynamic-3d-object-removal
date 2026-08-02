#!/usr/bin/env python3
"""Validate O1 sensor-aware selector gates from reproducible benchmark summaries.

This script never substitutes README numbers for a missing run. It requires the
actual JSON emitted by each benchmark and writes a machine-readable gate report.
Metrics are compared at the same three-decimal precision used by the public tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _reported(value: Any) -> float:
    return round(float(value), 3)


def evaluate_gates(av2: dict[str, Any], nuscenes: dict[str, Any], dynamicmap: dict[str, Any]) -> dict[str, Any]:
    av2_sa = av2.get("sensor_aware")
    ns_sa = nuscenes.get("sensor_aware")
    if not isinstance(av2_sa, dict) or not isinstance(ns_sa, dict):
        raise ValueError("AV2 and nuScenes summaries must include --sensor-aware-ablation output")
    if av2_sa.get("baseline_strategy") != "fusion":
        raise ValueError("AV2 selector must route the 64-beam profile to fusion")
    if ns_sa.get("baseline_strategy") != "range_and_scan_ratio":
        raise ValueError("nuScenes selector must route the 32-beam profile to range_and_scan_ratio")

    av2_metrics = av2_sa["selected_metrics"]
    ns_metrics = ns_sa["selected_metrics"]
    dm_results = dynamicmap.get("results")
    if not isinstance(dm_results, dict):
        raise ValueError("DynamicMap summary must contain results")

    checks = {
        "av2_f1": _reported(av2_metrics["f1"]) >= 0.657,
        "av2_static": _reported(av2_metrics["static_preservation"]) >= 0.974,
        "nuscenes_f1": _reported(ns_metrics["f1"]) >= 0.642,
        "nuscenes_static": _reported(ns_metrics["static_preservation"]) >= 0.842,
    }
    dynamicmap_metrics = {}
    for sequence, floor in (("00", 98.4), ("05", 97.8)):
        try:
            metrics = dm_results[sequence]["fusion"]
            aa = float(metrics["AA"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"DynamicMap summary lacks fusion AA for sequence {sequence}") from exc
        dynamicmap_metrics[sequence] = metrics
        checks[f"dynamicmap_{sequence}_aa"] = aa >= floor

    best_sparse = ns_sa.get("best_normalized_range_and_scan_ratio_candidate")
    candidate_metrics = best_sparse.get("metrics") if isinstance(best_sparse, dict) else None
    candidate_improves_same_scene = False
    if isinstance(candidate_metrics, dict):
        candidate_improves_same_scene = (
            float(candidate_metrics["f1"]) >= float(ns_metrics["f1"])
            and float(candidate_metrics["static_preservation"])
            >= float(ns_metrics["static_preservation"])
        )

    return {
        "task": "offline_map_cleaning",
        "selector_status": "experimental_not_promoted",
        "all_non_regression_gates_pass": all(checks.values()),
        "checks": checks,
        "selected": {
            "av2": {"strategy": "fusion", "metrics": av2_metrics},
            "nuscenes": {"strategy": "range_and_scan_ratio", "metrics": ns_metrics},
            "dynamicmap": {"strategy": "fusion", "metrics": dynamicmap_metrics},
        },
        "normalized_candidate": {
            "same_scene_nuscenes_improvement": candidate_improves_same_scene,
            "promotion_ready": False,
            "reason": (
                "The normalized candidate was tuned on one sparse scene; a second "
                "held-out sparse or heterogeneous sensor is required before promotion."
            ),
        },
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av2-summary", required=True, type=Path)
    parser.add_argument("--nuscenes-summary", required=True, type=Path)
    parser.add_argument("--dynamicmap-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_gates(
            _read(args.av2_summary),
            _read(args.nuscenes_summary),
            _read(args.dynamicmap_summary),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))
    print(f"Saved: {args.output}")
    return 0 if report["all_non_regression_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
