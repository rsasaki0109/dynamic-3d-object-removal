#!/usr/bin/env python3
"""Validate O2 height-candidate gates from actual benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _metrics(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    try:
        metrics = summary["best_candidate"]["metrics"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{dataset} summary lacks best_candidate.metrics") from exc
    if summary.get("algorithm") != "height_candidate_experimental":
        raise ValueError(f"{dataset} summary is not a height-candidate ablation")
    return metrics


def evaluate_gates(
    av2: dict[str, Any],
    nuscenes: dict[str, Any],
    heldout_nuscenes: dict[str, Any],
    dynamicmap: dict[str, Any],
) -> dict[str, Any]:
    av2_metrics = _metrics(av2, "AV2")
    nuscenes_metrics = _metrics(nuscenes, "nuScenes")
    heldout_metrics = _metrics(heldout_nuscenes, "held-out nuScenes")
    try:
        heldout_baseline = heldout_nuscenes["baseline"]["metrics"]
    except (KeyError, TypeError) as exc:
        raise ValueError("held-out nuScenes summary lacks baseline.metrics") from exc
    dm_results = dynamicmap.get("results")
    if not isinstance(dm_results, dict):
        raise ValueError("DynamicMap summary must contain results")

    checks = {
        "av2_f1": round(float(av2_metrics["f1"]), 3) >= 0.657,
        "av2_static": round(float(av2_metrics["static_preservation"]), 3) >= 0.974,
        "nuscenes_f1": round(float(nuscenes_metrics["f1"]), 3) >= 0.642,
        "nuscenes_static": round(float(nuscenes_metrics["static_preservation"]), 3) >= 0.842,
        "av2_deskewed": bool(av2.get("deskew_input_contract_satisfied")),
        "nuscenes_deskewed": bool(nuscenes.get("deskew_input_contract_satisfied")),
        "heldout_nuscenes_f1_non_regression": (
            float(heldout_metrics["f1"]) >= float(heldout_baseline["f1"]) - 0.01
        ),
        "heldout_nuscenes_static_non_regression": (
            float(heldout_metrics["static_preservation"])
            >= float(heldout_baseline["static_preservation"]) - 0.01
        ),
        "heldout_nuscenes_deskewed": bool(
            heldout_nuscenes.get("deskew_input_contract_satisfied")
        ),
    }
    dynamicmap_metrics: dict[str, Any] = {}
    for sequence in ("00", "05"):
        try:
            baseline = dm_results[sequence]["fusion"]
            candidate = dm_results[sequence]["fusion_height_candidate"]
            baseline_aa = float(baseline["AA"])
            candidate_aa = float(candidate["AA"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"DynamicMap summary lacks fusion and fusion_height_candidate AA for sequence {sequence}"
            ) from exc
        checks[f"dynamicmap_{sequence}_aa"] = candidate_aa >= baseline_aa - 0.2
        dynamicmap_metrics[sequence] = {"baseline": baseline, "candidate": candidate}

    all_pass = all(checks.values())
    return {
        "task": "offline_map_cleaning",
        "algorithm": "height_candidate_experimental",
        "checks": checks,
        "all_gates_pass": all_pass,
        "promotion_ready": all_pass,
        "selected": {
            "av2": av2_metrics,
            "nuscenes": nuscenes_metrics,
            "heldout_nuscenes": {
                "baseline": heldout_baseline,
                "candidate": heldout_metrics,
            },
            "dynamicmap": dynamicmap_metrics,
        },
        "reason": (
            "All reproducible cross-dataset and deskew input gates pass."
            if all_pass
            else "Keep private until every dataset and deskew input gate passes."
        ),
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
    parser.add_argument("--heldout-nuscenes-summary", required=True, type=Path)
    parser.add_argument("--dynamicmap-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_gates(
            _read(args.av2_summary),
            _read(args.nuscenes_summary),
            _read(args.heldout_nuscenes_summary),
            _read(args.dynamicmap_summary),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))
    print(f"Saved: {args.output}")
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
