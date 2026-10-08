#!/usr/bin/env python3
"""Compare recorded benchmark results; exit 0/pass, 1/regression, 2/invalid input."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ACCURACY = ("precision", "recall", "f1", "static_preservation")


def required(value: dict, *names: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    missing = [name for name in names if name not in value]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    return {name: value[name] for name in names}


def number(value: Any, label: str, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, not missing/null")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is outside its finite valid range")
    return value


def normalize(payload: dict) -> tuple[dict, dict]:
    """Return comparable metadata and {(scope, method, metric): (value, unit)}."""
    rows = {}
    context = {}

    def validate_context(identity, counts):
        for name in counts:
            value = number(identity[name], name)
            if value <= 0 or value != int(value):
                raise ValueError(f"{name} must be a positive integer; empty runs cannot pass")
        if "config" in identity and not isinstance(identity["config"], dict):
            raise ValueError("config must be an object")

    def put(scope, method, metric, value, unit, maximum=None, scale=1):
        key = (scope, method, metric)
        if key in rows:
            raise ValueError(f"duplicate result: {key}")
        rows[key] = (number(value, str(key), maximum) / scale, unit)

    def metrics(scope, method, values, dynamicmap=False):
        names = ("SA", "DA", "AA", "HA") if dynamicmap else ACCURACY
        required(values, *names)
        for name in names:
            put(scope, method, name, values[name], "fraction",
                100 if dynamicmap else 1, 100 if dynamicmap else 1)
        if not dynamicmap and "iou" in values:
            put(scope, method, "iou", values["iou"], "fraction", 1)

    def scene(item):
        identity = required(item, "scene", "frames", "stride", "map_points",
                            "gt_dynamic_points", "config", "method_keys", "metrics")
        scope = identity.pop("scene")
        methods = identity.pop("method_keys")
        values = identity.pop("metrics")
        validate_context(identity, ("frames", "stride", "map_points"))
        number(identity["gt_dynamic_points"], "gt_dynamic_points")
        if not isinstance(scope, str) or scope in context:
            raise ValueError("scene names must be unique strings")
        if (not isinstance(methods, list) or not methods
                or not all(isinstance(method, str) and method for method in methods)
                or len(set(methods)) != len(methods)):
            raise ValueError("method_keys must be a nonempty unique list")
        if set(methods) != set(values):
            raise ValueError("method_keys and metrics disagree")
        identity["methods"] = sorted(methods)
        context[scope] = identity
        for method in methods:
            metrics(scope, method, values[method])
        if "runtime_seconds" in item:
            put(scope, "all_methods", "runtime_seconds", item["runtime_seconds"], "seconds")

    if payload.get("task") == "online_moving_object_segmentation":
        metadata = required(payload, "task", "manifest", "algorithm", "sensor_profile", "config")
        validate_context(metadata, ())
        scenarios = required(payload, "scenarios")["scenarios"]
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("scenarios must be a nonempty list")
        for item in scenarios:
            identity = required(item, "name", "pose_noise", "frames", "points", "warmup_frames",
                                "rate_hz", "period_ms", "confirmation_static_keep_threshold")
            scope = identity.pop("name")
            validate_context(identity, ("frames", "points"))
            if not isinstance(scope, str) or scope in context:
                raise ValueError("scenario names must be unique strings")
            context[scope] = identity
            method = metadata["algorithm"]
            metrics(scope, method, required(item, "metrics")["metrics"])
            latency = required(item, "filter_latency")["filter_latency"]
            for name in ("mean_ms", "p50_ms", "p95_ms", "max_ms"):
                put(scope, method, name, required(latency, name)[name], "ms")
            for name in ("dropped_frames", "fail_open_frames"):
                put(scope, method, name, required(item, name)[name], "frames")
            deadline = required(item, "deadline_misses")["deadline_misses"]
            if deadline is not None:
                put(scope, method, "deadline_misses", deadline, "frames")
            context[scope]["deadline_measured"] = deadline is not None
    elif payload.get("dataset") == "DynamicMap_Benchmark/Semantic-KITTI":
        metadata = required(payload, "dataset", "sequences", "config")
        validate_context(metadata, ())
        results = required(payload, "results")["results"]
        if (not isinstance(results, dict) or not results
                or len(metadata["sequences"]) != len(set(metadata["sequences"]))
                or set(results) != set(metadata["sequences"])):
            raise ValueError("sequence list and results must match and be nonempty")
        metadata["sequences"] = sorted(metadata["sequences"])
        for sequence, methods in results.items():
            if not isinstance(methods, dict) or not methods:
                raise ValueError("sequence methods must be nonempty")
            for method, values in methods.items():
                metrics(sequence, method, values, dynamicmap=True)
    elif payload.get("dataset") in {"argoverse-2-sensor-val", "nuscenes-mini"}:
        metadata = required(payload, "dataset")
        if "scene_results" in payload:
            metadata.update(required(payload, "frames", "stride", "config"))
            validate_context(metadata, ("frames", "stride"))
            scenes = payload["scene_results"]
            if not isinstance(scenes, list) or not scenes:
                raise ValueError("scene_results must be a nonempty list")
            for item in scenes:
                if item.get("dataset") != payload["dataset"]:
                    raise ValueError("per-scene dataset disagrees with aggregate")
                scene(item)
            names = required(payload, "scenes")["scenes"]
            if len(names) != len(context) or set(names) != set(context):
                raise ValueError("scenes and scene_results disagree")
            aggregate = required(payload, "aggregate")["aggregate"]
            selection = required(aggregate, "min_gt_dynamic_points", "included_scenes", "excluded_scenes")
            included, excluded = selection["included_scenes"], selection["excluded_scenes"]
            if (set(included) & set(excluded) or len(included) + len(excluded) != len(context)
                    or set(included + excluded) != set(context) or not included):
                raise ValueError("aggregate eligibility must partition scenes and include at least one")
            threshold = number(selection["min_gt_dynamic_points"], "min_gt_dynamic_points")
            if set(included) != {name for name, item in context.items()
                                 if item["gt_dynamic_points"] >= threshold}:
                raise ValueError("aggregate eligibility disagrees with GT threshold")
            metadata["aggregate_selection"] = {
                "min_gt_dynamic_points": threshold, "included_scenes": sorted(included),
                "excluded_scenes": sorted(excluded),
            }
            means = required(aggregate, "methods")["methods"]
            if any(item["methods"] != context[included[0]]["methods"] for item in context.values()):
                raise ValueError("per-scene methods disagree")
            if not isinstance(means, dict) or set(means) != set(context[included[0]]["methods"]):
                raise ValueError("aggregate methods disagree with scene methods")
            for method, values in means.items():
                metrics("aggregate", method, values)
                for metric in ACCURACY:
                    expected = sum(rows[(name, method, metric)][0] for name in included) / len(included)
                    if not math.isclose(values[metric], expected, abs_tol=1e-10):
                        raise ValueError("aggregate means disagree with per-scene results")
        else:
            scene(payload)
    else:
        raise ValueError("unsupported benchmark schema; CLI removal summaries are not accuracy benchmarks")
    if not rows:
        raise ValueError("benchmark has no measurements")
    metadata["cases"] = context
    json.dumps(metadata, allow_nan=False)  # Reject non-finite configuration metadata too.
    return metadata, rows


def compare(baseline: dict, candidate: dict, *, max_metric_drop=0.0, max_slowdown=0.2) -> dict:
    number(max_metric_drop, "max_metric_drop", 1)
    number(max_slowdown, "max_slowdown")
    before_meta, before = normalize(baseline)
    after_meta, after = normalize(candidate)
    if before_meta != after_meta:
        changed = sorted(key for key in before_meta.keys() | after_meta.keys()
                         if before_meta.get(key) != after_meta.get(key))
        raise ValueError(f"incompatible evaluation metadata: {', '.join(changed)}")
    if before.keys() != after.keys():
        raise ValueError("methods or measured metrics differ between runs")
    measurements = []
    for (scope, method, metric), (old, unit) in sorted(before.items()):
        new = after[(scope, method, metric)][0]
        delta = new - old
        if unit == "fraction":
            regressed = delta < -max_metric_drop and not math.isclose(delta, -max_metric_drop, abs_tol=1e-12)
            allowed = max_metric_drop
        elif unit == "frames":
            regressed = new > old
            allowed = 0
        else:
            limit = old * (1 + max_slowdown)
            regressed = new > limit and not math.isclose(new, limit, abs_tol=1e-12)
            allowed = max_slowdown
        measurements.append({
            "case": scope, "method": method, "metric": metric, "unit": unit,
            "baseline": old, "candidate": new, "delta": delta,
            "relative_change": delta / old if old else None,
            "allowed_regression": allowed, "regressed": regressed,
        })
    regressions = sum(item["regressed"] for item in measurements)
    return {
        "status": "regression" if regressions else "passed",
        "regression_count": regressions, "measurement_count": len(measurements),
        "thresholds": {"max_metric_drop": max_metric_drop, "max_slowdown": max_slowdown},
        "metadata": before_meta, "measurements": measurements,
        "timing_measured": any(item["unit"] in {"seconds", "ms"} for item in measurements),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--max-metric-drop", type=float, default=0,
                        help="Allowed absolute drop on the 0–1 scale (0.01 = one percentage point).")
    parser.add_argument("--max-slowdown", type=float, default=0.2,
                        help="Allowed relative timing increase (0.2 = 20%%).")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        report = compare(baseline, candidate, max_metric_drop=args.max_metric_drop,
                         max_slowdown=args.max_slowdown)
        report.update(baseline_file=str(args.baseline), candidate_file=str(args.candidate))
        if args.report_json:
            if args.report_json.resolve() in {args.baseline.resolve(), args.candidate.resolve()}:
                raise ValueError("report path must not overwrite an input result")
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"{report['status']}: {report['regression_count']} regressions / {report['measurement_count']} measurements")
        for item in report["measurements"]:
            print(f"{'FAIL' if item['regressed'] else 'PASS'} {item['case']} / {item['method']} / "
                  f"{item['metric']}: {item['baseline']:.6g} -> {item['candidate']:.6g} "
                  f"(delta {item['delta']:+.6g} {item['unit']})")
        if not report["timing_measured"]:
            print("Timing not measured in these inputs.")
        return 1 if report["regression_count"] else 0
    except (ValueError, TypeError, KeyError, AttributeError, OSError, OverflowError) as exc:
        print(f"comparison error: {exc}", file=sys.stderr)
        # Replace a previous report with this invocation's failure, rather than
        # leaving stale passing results at the requested output path.
        if args.report_json and args.report_json.resolve() not in {
                args.baseline.resolve(), args.candidate.resolve()}:
            try:
                args.report_json.parent.mkdir(parents=True, exist_ok=True)
                args.report_json.write_text(json.dumps({
                    "status": "comparison_error", "error": str(exc),
                    "baseline_file": str(args.baseline), "candidate_file": str(args.candidate),
                }, indent=2), encoding="utf-8")
            except OSError as report_exc:
                print(f"cannot write error report: {report_exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
