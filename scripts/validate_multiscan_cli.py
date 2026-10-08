#!/usr/bin/env python3
"""Validate multi-scan CLIs against the API on a pose/GT manifest exported from AV2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402
import dynamic_object_removal as core  # noqa: E402
from scripts.compare_benchmark_results import compare  # noqa: E402


def validate(manifest_path: Path, output: Path, *, workers=2, stride=3, reference_path=None) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    scans, _ = core._load_scan_manifest(manifest_path)
    points = np.concatenate([scan for scan, _ in scans])
    labels = []
    for frame, (scan, _) in zip(manifest["frames"], scans):
        path = manifest_path.parent / frame["point_labels"]
        values = np.load(path)
        if (values.shape != (len(scan),) or not np.isin(values, [0, 1]).all()):
            raise ValueError(f"invalid point GT labels: {path}")
        labels.append(values.astype(bool))
    gt = np.concatenate(labels)
    if not gt.any() or gt.all():
        raise ValueError("evaluation requires both moving-GT and static points")
    reference = json.loads(Path(reference_path).read_text()) if reference_path else None
    config = reference["config"] if reference else {}
    if reference:
        expected = {"dataset": "argoverse-2-sensor-val", "scene": manifest["scene"],
                    "frames": len(scans), "stride": stride,
                    "map_points": len(points), "gt_dynamic_points": int(gt.sum())}
        if any(reference.get(key) != value for key, value in expected.items()):
            raise ValueError("reference benchmark selection/counts do not match manifest")
    visibility = {
        "h_res_deg": config.get("h_res", 1.0), "v_res_deg": config.get("v_res", 1.0),
        "range_margin": config.get("range_margin", core.DEFAULT_RANGE_MARGIN),
        "min_see_through": config.get("min_see_through", 3),
        "max_surface_hits": config.get("max_surface_hits", 3),
        "ground_z": config.get("ground_z", -1.4),
        "resolutions": config.get("resolutions"),
    }
    sr = {
        "n_rings": config.get("sr_rings", core.DEFAULT_SR_RINGS),
        "n_sectors": config.get("sr_sectors", core.DEFAULT_SR_SECTORS),
        "max_range": config.get("sr_max_range", core.DEFAULT_SR_MAX_RANGE),
        "scan_ratio_threshold": config.get("sr_ratio", core.DEFAULT_SR_RATIO),
        "min_map_height": config.get("sr_min_map_height", core.DEFAULT_SR_MIN_MAP_HEIGHT),
        "ground_margin": config.get("sr_ground_margin", core.DEFAULT_SR_GROUND_MARGIN),
        "min_votes": config.get("sr_min_votes"),
    }
    fusion = {
        "free_votes_fraction": config.get("fusion_free_fraction", 0.7),
        "free_votes_floor": config.get("fusion_free_floor", 3),
        "void_min_scans": config.get("fusion_void_min_scans", 4), "workers": workers,
    }
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty; preserve previous runs")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "map.npy", points)
    np.save(output / "gt.npy", gt)
    api_started = time.perf_counter()
    fusion_points, fusion_keep = core.clean_map_by_fusion(points, scans, **fusion)
    _, range_keep = core.clean_map_by_visibility(points, scans, **visibility)
    _, sr_keep = core.clean_map_by_scan_ratio(points, scans, **sr)
    api_seconds = time.perf_counter() - api_started
    masks = {"fusion": fusion_keep, "range_scan_ratio": range_keep | sr_keep}
    metrics = {method: bench.compute_accuracy_metrics(~mask, gt) for method, mask in masks.items()}
    if reference:
        checks = {
            "fusion": metrics["fusion"],
            "range": bench.compute_accuracy_metrics(~range_keep, gt),
            "scan_ratio": bench.compute_accuracy_metrics(~sr_keep, gt),
        }
        for method, values in checks.items():
            for name in ("precision", "recall", "f1", "static_preservation"):
                if not np.isclose(values[name], reference["metrics"][method][name], atol=1e-12, rtol=0):
                    raise ValueError(f"reference metric mismatch: {method}/{name}")
    summaries = {}
    cli_metrics = {}
    for method, expected_mask in masks.items():
        cloud_path, mask_path = output / f"{method}.npy", output / f"{method}_keep.npy"
        summary_path = output / f"{method}_summary.json"
        command = [sys.executable, "-m", "dynamic_object_removal", "--algorithm", method,
                   "--input-map", str((output / "map.npy").resolve()),
                   "--input-manifest", str(manifest_path),
                   "--output-cloud", str(cloud_path.resolve()),
                   "--output-mask", str(mask_path.resolve()),
                   "--summary-json", str(summary_path.resolve()), "--quiet"]
        if method == "fusion":
            command += ["--preset", "short-window"]
            for name, value in fusion.items():
                command += ["--fusion-" + name.replace("_", "-"), str(value)]
        else:
            flags = {
                "h_res_deg": "--range-h-res", "v_res_deg": "--range-v-res",
                "range_margin": "--range-margin", "min_see_through": "--range-min-see-through",
                "max_surface_hits": "--range-max-surface-hits", "ground_z": "--range-ground-z",
                "resolutions": "--range-resolutions",
            }
            for name, value in visibility.items():
                if value is not None:
                    command += [flags[name]] + [str(v) for v in (value if isinstance(value, list) else [value])]
            flags = {"n_rings": "rings", "n_sectors": "sectors", "max_range": "max-range",
                     "scan_ratio_threshold": "threshold", "min_map_height": "min-map-height",
                     "ground_margin": "ground-margin", "min_votes": "min-votes"}
            for name, value in sr.items():
                if value is not None:
                    command += ["--scan-ratio-" + flags[name], str(value)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (output / f"{method}_command.json").write_text(json.dumps(command, indent=2))
        (output / f"{method}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError(f"{method} CLI failed ({result.returncode}): {result.stderr}")
        actual_mask = np.load(mask_path)
        if not np.array_equal(actual_mask, expected_mask):
            raise ValueError(f"{method} CLI/API keep masks differ")
        expected_points = fusion_points if method == "fusion" else points[expected_mask]
        if not np.array_equal(np.load(cloud_path), expected_points):
            raise ValueError(f"{method} CLI/API point clouds differ")
        cli_metrics[method] = bench.compute_accuracy_metrics(~actual_mask, gt)
        summaries[method] = json.loads(summary_path.read_text())
    identity = {
        "dataset": "argoverse-2-sensor-val", "scene": manifest["scene"],
        "frames": len(scans), "stride": stride, "map_points": len(points),
        "gt_dynamic_points": int(gt.sum()), "method_keys": list(masks),
        "config": {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
            "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest(),
            "parameters": {method: value["parameters"] for method, value in summaries.items()},
        },
    }
    baseline = dict(identity, metrics=metrics, runtime_seconds=api_seconds)
    candidate = dict(identity, metrics=cli_metrics,
                     runtime_seconds=sum(value["filter_seconds"] for value in summaries.values()))
    # This validates semantic equivalence. API/CLI timings are recorded but not
    # judged here: independent repeat runs are needed for a performance baseline.
    semantic_report = compare(
        {key: value for key, value in baseline.items() if key != "runtime_seconds"},
        {key: value for key, value in candidate.items() if key != "runtime_seconds"},
    )
    for name, value in (("baseline_api.json", baseline), ("candidate_cli.json", candidate),
                        ("equivalence_report.json", semantic_report)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False))
    report = {
        "status": "passed", "point_masks_identical": True,
        "reference_metrics_matched": reference is not None,
        "frames": len(scans), "map_points": len(points), "gt_dynamic_points": int(gt.sum()),
        "metrics": cli_metrics, "summaries": summaries,
        "timing_scope": "sum of filter calls; not end-to-end startup or ROS2 latency",
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.stride <= 0:
        parser.error("workers and stride must be positive")
    try:
        report = validate(args.manifest, args.output_dir, workers=args.workers,
                          stride=args.stride, reference_path=args.reference_summary)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
