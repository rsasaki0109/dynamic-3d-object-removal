#!/usr/bin/env python3
"""Validate a frozen scan-ratio vote-floor candidate against a recorded baseline."""
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
import bench
import dynamic_object_removal as core

RANGE_FLAGS = {"h_res_deg": "range-h-res", "v_res_deg": "range-v-res",
    "range_margin": "range-margin", "min_see_through": "range-min-see-through",
    "max_surface_hits": "range-max-surface-hits", "ground_z": "range-ground-z",
    "resolutions": "range-resolutions"}
SR_FLAGS = {"n_rings": "scan-ratio-rings", "n_sectors": "scan-ratio-sectors",
    "max_range": "scan-ratio-max-range", "scan_ratio_threshold": "scan-ratio-threshold",
    "min_map_height": "scan-ratio-min-map-height", "ground_margin": "scan-ratio-ground-margin",
    "min_votes": "scan-ratio-min-votes", "votes_fraction": "scan-ratio-votes-fraction",
    "votes_floor": "scan-ratio-votes-floor"}


def validate(manifest_path: Path, baseline_path: Path, output: Path, *, candidate_floor=2):
    if not isinstance(candidate_floor, int) or candidate_floor < 1:
        raise ValueError("candidate floor must be a positive integer")
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    scans, profile = core._load_scan_manifest(manifest_path)
    points = np.concatenate([p for p, _ in scans])
    labels = []
    for frame, (scan, _) in zip(manifest["frames"], scans):
        label = np.load(manifest_path.parent / frame["point_labels"])
        if label.shape != (len(scan),) or not np.isin(label, [0, 1]).all():
            raise ValueError("invalid GT labels")
        labels.append(label.astype(bool))
    gt = np.concatenate(labels)
    hashes = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if any(baseline["config"].get(k) != v for k, v in hashes.items()):
        raise ValueError("input fingerprints differ from baseline")
    if (baseline["scene"] != manifest["scene"] or baseline["frames"] != len(scans)
            or baseline["map_points"] != len(points) or baseline["gt_dynamic_points"] != int(gt.sum())):
        raise ValueError("baseline selection mismatch")
    params = baseline["config"]["parameters"]["range_scan_ratio"]
    if params["scan_ratio"]["min_votes"] is not None:
        raise ValueError("vote floor requires normalized voting, not fixed min_votes")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "map.npy", points)
    np.save(output / "gt.npy", gt)
    _, range_keep = core.clean_map_by_visibility(points, scans, **params["range"])
    results = {}
    for name, floor in (("baseline", params["scan_ratio"]["votes_floor"]),
                        ("candidate", candidate_floor)):
        sr = {**params["scan_ratio"], "votes_floor": floor}
        started = time.perf_counter()
        _, sr_keep = core.clean_map_by_scan_ratio(points, scans, **sr)
        keep = range_keep | sr_keep
        sr_seconds = time.perf_counter() - started
        metrics = bench.compute_accuracy_metrics(~keep, gt)
        if name == "baseline" and metrics != baseline["metrics"]["range_scan_ratio"]:
            raise ValueError("baseline metrics mismatch")
        target = output / name
        target.mkdir()
        command = [sys.executable, "-m", "dynamic_object_removal", "--algorithm", "range_scan_ratio",
                   "--input-manifest", str(manifest_path), "--input-map", str((output / "map.npy").resolve()),
                   "--output-cloud", str((target / "points.npy").resolve()),
                   "--output-mask", str((target / "keep.npy").resolve()),
                   "--summary-json", str((target / "summary.json").resolve()), "--quiet"]
        for values, flags in ((params["range"], RANGE_FLAGS), (sr, SR_FLAGS)):
            for key, value in values.items():
                if value is not None:
                    command += ["--" + flags[key]] + [str(v) for v in (value if isinstance(value, list) else [value])]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (target / "command.json").write_text(json.dumps(command, indent=2))
        (target / "stderr.txt").write_text(run.stderr)
        if run.returncode:
            raise ValueError(f"CLI failed: {run.stderr}")
        np.testing.assert_array_equal(np.load(target / "keep.npy"), keep)
        np.testing.assert_array_equal(np.load(target / "points.npy"), points[keep])
        summary = json.loads((target / "summary.json").read_text())
        if summary["parameters"] != {"range": params["range"], "scan_ratio": sr}:
            raise ValueError("CLI parameters mismatch")
        results[name] = {"parameters": summary["parameters"], "metrics": metrics,
                         "api_cli_masks_equal": True, "api_cli_points_equal": True,
                         "api_scan_ratio_seconds": sr_seconds,
                         "cli_filter_seconds": summary["filter_seconds"]}
    payload = {"dataset": baseline["dataset"], "scene": baseline["scene"],
               "frames": len(scans), "stride": baseline["stride"],
               "map_points": len(points), "gt_dynamic_points": int(gt.sum()),
               "sensor_profile": profile, "input_hashes": hashes, "results": results,
               "limitations": "Frozen candidate from nuScenes mini, tested on one previously benchmarked AV2 scene without retuning. Single scene is not broad held-out validation. API timing covers scan-ratio only; CLI timing covers both filters."}
    (output / "validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-floor", type=int, default=2)
    args = parser.parse_args(argv)
    result = validate(args.manifest, args.baseline_summary, args.output, candidate_floor=args.candidate_floor)
    print(json.dumps({k: v["metrics"] for k, v in result["results"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
