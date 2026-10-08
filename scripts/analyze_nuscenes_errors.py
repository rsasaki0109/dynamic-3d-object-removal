#!/usr/bin/env python3
"""Diagnose sparse-sensor false removals and run exploratory parameter ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
import dynamic_object_removal as core
from scripts.run_nuscenes_benchmark import MIN_GT_DYNAMIC_POINTS_FOR_MEAN

VARIANTS = {
    "baseline": {},
    "surface_guard_2": {"max_surface_hits": 2},
    "see_through_5": {"min_see_through": 5},
    "consensus_1_2_5": {"resolutions": [1.0, 2.5]},
}


def bins(values, edges, gt, dynamic):
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (values >= low) & (values < high)
        static = selected & ~gt
        moving = selected & gt
        fp = int((static & dynamic).sum())
        result.append({"low": low, "high": high, "points": int(selected.sum()),
                       "static_points": int(static.sum()), "false_positive": fp,
                       "static_removal_fraction": fp / int(static.sum()) if static.any() else None,
                       "moving_points": int(moving.sum()),
                       "true_positive": int((moving & dynamic).sum())})
    return result


def analyze(root: Path):
    records = []
    for path in sorted(root.glob("scene-*/manifest.json")):
        manifest = json.loads(path.read_text())
        scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
        points = np.concatenate([p for p, _ in scans])
        gt = np.concatenate([np.load(path.parent / f["point_labels"]).astype(bool)
                             for f in manifest["frames"]])
        # Distance/height relative to the sensor that acquired each point.
        offsets = np.concatenate([p - origin for p, origin in scans])
        distance = np.linalg.norm(offsets, axis=1)
        started = time.perf_counter()
        _, sr_keep = core.clean_map_by_scan_ratio(points, scans)
        variants = {}
        for name, overrides in VARIANTS.items():
            params = {"h_res_deg": 2.5, "v_res_deg": 2.5,
                      "min_see_through": 3, "max_surface_hits": 5, **overrides}
            _, range_keep = core.clean_map_by_visibility(points, scans, **params)
            dynamic = ~range_keep & ~sr_keep
            variants[name] = {"parameters": params,
                              "metrics": bench.compute_accuracy_metrics(dynamic, gt)}
            if name == "baseline":
                channels = {key: bench.compute_accuracy_metrics(mask, gt) for key, mask in
                            (("range", ~range_keep), ("scan_ratio", ~sr_keep))}
                diagnostics = {
                    "acquisition_distance_m": bins(distance, [0, 2, 5, 10, 20, 40, 80, 1000000], gt, dynamic),
                    "acquisition_height_m": bins(offsets[:, 2], [-1000000, -1, 0, 1, 2, 1000000], gt, dynamic),
                }
                # Replay the original validation to detect changed inputs/settings.
                reference = json.loads((path.parent / "validation/validation.json").read_text())
                if variants[name]["metrics"] != reference["results"]["defaults"]["metrics"]:
                    raise ValueError(f"baseline mismatch: {manifest['scene']}")
                variants["acquisition_range_guard_2m"] = {
                    "parameters": {"minimum_acquisition_range_m": 2.0},
                    "metrics": bench.compute_accuracy_metrics(dynamic & (distance >= 2), gt),
                }
        records.append({"scene": manifest["scene"], "map_points": len(points),
                        "gt_dynamic_points": int(gt.sum()), "channels": channels,
                        "diagnostics": diagnostics, "variants": variants,
                        "analysis_seconds": time.perf_counter() - started})
        print(f"{manifest['scene']}: baseline replay matched", flush=True)
    if not records:
        raise ValueError("no scene manifests")
    eligible = [r for r in records if r["gt_dynamic_points"] >= MIN_GT_DYNAMIC_POINTS_FOR_MEAN]
    metrics = ["precision", "recall", "f1", "static_preservation"]
    return {"dataset": "nuscenes-mini", "scene_results": records,
            "aggregate": {"min_gt_dynamic_points": MIN_GT_DYNAMIC_POINTS_FOR_MEAN,
                "included_scenes": [r["scene"] for r in eligible],
                "methods": {name: {metric: float(np.mean([
                    r["variants"][name]["metrics"][metric] for r in eligible])) if eligible else None
                    for metric in metrics} for name in [*VARIANTS, "acquisition_range_guard_2m"]}},
            "limitations": "Exploratory ablations on the same ten mini scenes, without a held-out test set. Rigid poses without intra-sweep deskew. GT uses moving-instance boxes; static labels mean outside those boxes and may include ego-vehicle returns. The acquisition-range guard requires per-point source-scan provenance and is diagnostic only, not a map-only CLI option. No defaults changed."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report_json.exists():
        parser.error("report already exists; choose a new path")
    report = analyze(args.validation_root)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
