#!/usr/bin/env python3
"""Diagnose missed moving GT with devkit close-point preprocessing fixed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
import dynamic_object_removal as core
from scripts.run_nuscenes_benchmark import MIN_GT_DYNAMIC_POINTS_FOR_MEAN

VARIANTS = {
    "baseline": {},
    "ratio_0_3": {"scan_ratio_threshold": 0.3},
    "ratio_0_4": {"scan_ratio_threshold": 0.4},
    "votes_fraction_0_35": {"votes_fraction": 0.35},
    "votes_floor_2": {"votes_floor": 2},
    "sectors_216": {"n_sectors": 216},
}


def analyze(root: Path):
    records = []
    for path in sorted(root.glob("scene-*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("preprocessing", {}).get("min_distance") != 1.0:
            raise ValueError("requires devkit min-distance=1 selection")
        scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
        points = np.concatenate([p for p, _ in scans])
        labels = []
        for frame, (scan, _) in zip(manifest["frames"], scans):
            label = np.load(path.parent / frame["point_labels"])
            if label.shape != (len(scan),) or not np.isin(label, [0, 1]).all():
                raise ValueError("invalid GT labels")
            labels.append(label.astype(bool))
        gt = np.concatenate(labels)
        reference = json.loads((path.parent / "validation/validation.json").read_text())
        for name, value in (("map", points), ("gt", gt)):
            if hashlib.sha256(value.tobytes()).hexdigest() != reference[name + "_sha256"]:
                raise ValueError(f"{name} fingerprint changed")
        _, range_keep = core.clean_map_by_visibility(points, scans, h_res_deg=2.5,
                            v_res_deg=2.5, min_see_through=3, max_surface_hits=5)
        variants = {}
        for name, params in VARIANTS.items():
            _, sr_keep = core.clean_map_by_scan_ratio(points, scans, **params)
            variants[name] = {"scan_ratio_parameters": params,
                             "metrics": bench.compute_accuracy_metrics(~range_keep & ~sr_keep, gt)}
            if name == "baseline":
                if variants[name]["metrics"] != reference["results"]["defaults"]["metrics"]:
                    raise ValueError("baseline metrics changed")
                channels = {name: bench.compute_accuracy_metrics(mask, gt) for name, mask in
                            (("range_only", ~range_keep), ("scan_ratio_only", ~sr_keep))}
                gt_decisions = {
                    "both_dynamic": int((gt & ~range_keep & ~sr_keep).sum()),
                    "range_dynamic_only": int((gt & ~range_keep & sr_keep).sum()),
                    "scan_ratio_dynamic_only": int((gt & range_keep & ~sr_keep).sum()),
                    "neither_dynamic": int((gt & range_keep & sr_keep).sum()),
                }
        records.append({"scene": manifest["scene"], "map_points": len(points),
                        "gt_dynamic_points": int(gt.sum()), "preprocessing": manifest["preprocessing"],
                        "manifest_sha256": reference["manifest_sha256"],
                        "map_sha256": reference["map_sha256"], "gt_sha256": reference["gt_sha256"],
                        "gt_decisions": gt_decisions, "channels": channels, "variants": variants})
        print(f"{manifest['scene']}: fingerprints and baseline matched", flush=True)
    if not records:
        raise ValueError("no scene manifests")
    eligible = [r for r in records if r["gt_dynamic_points"] >= MIN_GT_DYNAMIC_POINTS_FOR_MEAN]
    metrics = ["precision", "recall", "f1", "static_preservation"]
    methods = {name: {key: float(np.mean([r["variants"][name]["metrics"][key] for r in eligible]))
                     if eligible else None for key in metrics} for name in VARIANTS}
    channels = {name: {key: float(np.mean([r["channels"][name][key] for r in eligible]))
                      if eligible else None for key in metrics}
                for name in ("range_only", "scan_ratio_only")}
    return {"dataset": "nuscenes-mini", "min_distance": 1.0,
            "scene_results": records,
            "aggregate": {"min_gt_dynamic_points": MIN_GT_DYNAMIC_POINTS_FOR_MEAN,
                          "included_scenes": [r["scene"] for r in eligible],
                          "excluded_scenes": [r["scene"] for r in records if r not in eligible],
                          "methods": methods, "channels": channels},
            "limitations": "One-factor exploratory ablations on the same mini scenes, without held-out validation. Range parameters and devkit input selection fixed. Rigid poses without deskew; GT uses moving-instance boxes. No production defaults changed."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report_json.exists():
        parser.error("report exists; choose a new path")
    report = analyze(args.validation_root)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
