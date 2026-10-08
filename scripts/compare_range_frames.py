#!/usr/bin/env python3
"""Compare map-axis and sensor-axis range images on identical recorded inputs."""
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


def pose_rotation(frame):
    """Read a pose already validated by the manifest loader."""
    pose = frame["pose"]
    if "rotation" in pose:
        return np.asarray(pose["rotation"], dtype=float)
    q = np.asarray(pose["quaternion_xyzw"], dtype=float)
    q = q / np.max(np.abs(q))
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def sensor_frame_keep(points, local_scans, params):
    """Aggregate visibility in each native sensor orientation, guard ground in map Z."""
    if params["resolutions"] is not None:
        raise ValueError("experiment requires single-resolution visibility")
    see = np.zeros(len(points), dtype=np.int64)
    surface = np.zeros_like(see)
    for scan, rotation, origin in local_scans:
        local_map = (points - origin) @ rotation
        st, sf = core._visibility_votes(local_map, scan, np.zeros(3),
                    params["h_res_deg"], params["v_res_deg"], params["range_margin"])
        see += st
        surface += sf
    dynamic = (see >= params["min_see_through"]) & (surface <= params["max_surface_hits"])
    if params["ground_z"] is not None:
        dynamic &= points[:, 2] > params["ground_z"]
    return ~dynamic


def compare_scene(path, reference):
    manifest = json.loads(path.read_text())
    scans, profile = core._load_scan_manifest(path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    local_scans, labels = [], []
    for frame, (scan, origin) in zip(manifest["frames"], scans):
        if "pose" not in frame:
            raise ValueError("sensor orientation requires poses for every frame")
        rotation = pose_rotation(frame)
        local = core.load_points(path.parent / frame["cloud"], fmt="auto")
        label = np.load(path.parent / frame["point_labels"])
        if label.shape != (len(scan),) or not np.isin(label, [0, 1]).all():
            raise ValueError("invalid GT labels")
        local_scans.append((local, rotation, origin))
        labels.append(label.astype(bool))
    gt = np.concatenate(labels)
    if "results" in reference:
        params = reference["results"]["defaults"]["parameters"]
        expected = reference["results"]["defaults"]["metrics"]
        fingerprints = reference
    else:
        params = reference["config"]["parameters"]["range_scan_ratio"]
        expected = reference["metrics"]["range_scan_ratio"]
        fingerprints = reference["config"]
    hashes = {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if any(fingerprints.get(k) != v for k, v in hashes.items()):
        raise ValueError("input fingerprints changed")
    _, sr_keep = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    _, map_keep = core.clean_map_by_visibility(points, scans, **params["range"])
    native_keep = sensor_frame_keep(points, local_scans, params["range"])
    variants = {}
    for name, keep in (("map_axes", map_keep), ("sensor_axes", native_keep)):
        variants[name] = {
            "range_only_metrics": bench.compute_accuracy_metrics(~keep, gt),
            "intersection_metrics": bench.compute_accuracy_metrics(~keep & ~sr_keep, gt)}
    if variants["map_axes"]["intersection_metrics"] != expected:
        raise ValueError("baseline metrics mismatch")
    baseline_dynamic = ~map_keep & ~sr_keep
    candidate_dynamic = ~native_keep & ~sr_keep
    return {"scene": manifest["scene"], "map_points": len(points),
            "gt_dynamic_points": int(gt.sum()), "input_hashes": hashes,
            "sensor_profile": profile, "parameters": params, "variants": variants,
            "additional_moving_removed": int((candidate_dynamic & ~baseline_dynamic & gt).sum()),
            "additional_static_removed": int((candidate_dynamic & ~baseline_dynamic & ~gt).sum()),
            "lost_moving_removed": int((~candidate_dynamic & baseline_dynamic & gt).sum()),
            "recovered_static_points": int((~candidate_dynamic & baseline_dynamic & ~gt).sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--av2-manifest", type=Path, required=True)
    parser.add_argument("--av2-baseline", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report_json.exists():
        parser.error("report exists; choose a new path")
    records = []
    for path in sorted(args.validation_root.glob("scene-*/manifest.json")):
        if json.loads(path.read_text()).get("preprocessing", {}).get("min_distance") != 1:
            raise ValueError("requires devkit min-distance 1 selection")
        reference = json.loads((path.parent / "validation/validation.json").read_text())
        records.append(compare_scene(path, reference))
        print(f"{records[-1]['scene']}: fingerprints and map-axis baseline match", flush=True)
    if not records:
        raise ValueError("no scene manifests")
    av2 = compare_scene(args.av2_manifest, json.loads(args.av2_baseline.read_text()))
    eligible = [r for r in records if r["gt_dynamic_points"] >= MIN_GT_DYNAMIC_POINTS_FOR_MEAN]
    keys = ["precision", "recall", "f1", "static_preservation"]
    summary = {name: {method: {key: float(np.mean([r["variants"][name][method][key] for r in eligible]))
                  if eligible else None for key in keys}
                  for method in ("range_only_metrics", "intersection_metrics")}
                  for name in ("map_axes", "sensor_axes")}
    report = {"nuscenes_scene_results": records, "av2_scene_result": av2,
              "aggregate": {"included_scenes": [r["scene"] for r in eligible],
                            "min_gt_dynamic_points": MIN_GT_DYNAMIC_POINTS_FOR_MEAN,
                            "methods": summary},
              "limitations": "Experimental sensor-oriented range images; no deskew added. Uses native local query points and inverse posed map points. Ground guard remains in map Z; scan-ratio unchanged. Fixed inputs and thresholds. Ten previously examined mini scenes and one AV2 scene, not held-out validation. No production defaults or CLI behavior changed; sensor-axis variant is a research helper."}
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"nuscenes": summary, "av2": av2["variants"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
