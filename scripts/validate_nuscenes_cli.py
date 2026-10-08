#!/usr/bin/env python3
"""Evaluate sparse-sensor CLI defaults and require exact API/CLI equivalence."""
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
from scripts import run_nuscenes_benchmark as nuscenes


def validate(manifest_path: Path, output: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset") != "nuscenes-mini":
        raise ValueError("expected a nuScenes mini manifest")
    scans, profile = core._load_scan_manifest(manifest_path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    labels = []
    for frame, (scan, _) in zip(manifest["frames"], scans):
        label = np.load(manifest_path.parent / frame["point_labels"])
        if label.shape != (len(scan),) or not np.isin(label, [0, 1]).all():
            raise ValueError("invalid point GT labels")
        labels.append(label.astype(bool))
    gt = np.concatenate(labels)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "map.npy", points)
    np.save(output / "gt.npy", gt)
    results = {}
    for name, ground_z in (("defaults", None),
                           ("ground_protected", float(np.percentile(points[:, 2], 2)))):
        started = time.perf_counter()
        _, kr = core.clean_map_by_visibility(points, scans, h_res_deg=2.5,
                    v_res_deg=2.5, min_see_through=3, max_surface_hits=5, ground_z=ground_z)
        _, ks = core.clean_map_by_scan_ratio(points, scans)
        keep = kr | ks
        api_seconds = time.perf_counter() - started
        target = output / name
        target.mkdir()
        command = [sys.executable, "-m", "dynamic_object_removal",
                   "--algorithm", "range_scan_ratio", "--allow-undeskewed",
                   "--input-manifest", str(manifest_path),
                   "--input-map", str((output / "map.npy").resolve()),
                   "--output-cloud", str((target / "cleaned.npy").resolve()),
                   "--output-mask", str((target / "keep.npy").resolve()),
                   "--summary-json", str((target / "summary.json").resolve()), "--quiet"]
        if ground_z is not None:
            command += ["--range-ground-z", repr(ground_z)]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (target / "command.json").write_text(json.dumps(command, indent=2))
        (target / "stderr.txt").write_text(run.stderr)
        if run.returncode:
            raise ValueError(f"CLI failed: {run.stderr}")
        np.testing.assert_array_equal(np.load(target / "keep.npy"), keep)
        np.testing.assert_array_equal(np.load(target / "cleaned.npy"), points[keep])
        summary = json.loads((target / "summary.json").read_text())
        results[name] = {
            "metrics": bench.compute_accuracy_metrics(~keep, gt),
            "api_cli_masks_equal": True, "api_cli_points_equal": True,
            "parameters": summary["parameters"],
            "api_filter_seconds": api_seconds,
            "cli_filter_seconds": summary["filter_seconds"],
        }
    payload = {"dataset": "nuscenes-mini", "scene": manifest["scene"],
               "frames": len(scans), "map_points": len(points),
               "gt_dynamic_points": int(gt.sum()), "sensor_profile": profile,
               "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
               "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest(), "results": results}
    (output / "validation.json").write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=nuscenes.ROOT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=[nuscenes.DEFAULT_SCENE])
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.stride < 1:
        parser.error("frames and stride must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    nuscenes._ensure_data(args.root)
    tables = nuscenes._load_tables(args.root)
    scenes = nuscenes._resolve_scenes(None, args.scenes, tables["scene"])
    records = []
    for scene in scenes:
        manifest = args.output / scene / "manifest.json"
        status = nuscenes.main(["--root", str(args.root), "--scene", scene,
                    "--frames", str(args.frames), "--stride", str(args.stride),
                    "--online-only", "--online-manifest", str(manifest)])
        if status:
            raise ValueError(f"manifest export failed: {scene}")
        record = validate(manifest, args.output / scene / "validation")
        records.append(record)
        print(f"{scene}: API/CLI masks and points match for both configurations", flush=True)
    eligible = [r for r in records if r["gt_dynamic_points"] >= nuscenes.MIN_GT_DYNAMIC_POINTS_FOR_MEAN]
    means = {name: {metric: float(np.mean([r["results"][name]["metrics"][metric] for r in eligible]))
                   if eligible else None for metric in nuscenes._METRIC_KEYS}
             for name in ("defaults", "ground_protected")}
    report = {"dataset": "nuscenes-mini", "frames_requested": args.frames,
              "stride": args.stride, "scene_results": records,
              "aggregate": {"min_gt_dynamic_points": nuscenes.MIN_GT_DYNAMIC_POINTS_FOR_MEAN,
                            "included_scenes": [r["scene"] for r in eligible],
                            "excluded_scenes": [r["scene"] for r in records if r not in eligible],
                            "methods": means},
              "limitations": "Rigid keyframe poses without intra-sweep deskew. Offline filter timing only. Ground-protected configuration uses map Z second percentile; defaults have no global ground protection."}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
