#!/usr/bin/env python3
"""Freeze a scene selection before acquisition, then evaluate every selected log."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import dynamic_object_removal as core
from scripts.experiment_neighbor_columns import compare_scene

MODES = ("none", "either", "visibility_supported")
CODE_FILES = (
    "dynamic_object_removal.py", "bench.py", "scripts/run_av2_benchmark.py",
    "scripts/experiment_neighbor_columns.py", "scripts/validate_frozen_av2_suite.py",
)
METRICS = ("precision", "recall", "f1", "iou", "static_preservation")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_scenes(catalogue, previously_examined, count):
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if not catalogue or len(catalogue) != len(set(catalogue)):
        raise ValueError("catalogue must be nonempty and contain unique scene IDs")
    for scene in catalogue:
        if str(uuid.UUID(scene)) != scene:
            raise ValueError("scene IDs must be canonical UUIDs")
    available = sorted(set(catalogue) - set(previously_examined))
    if len(available) < count:
        raise ValueError("not enough previously unexamined scenes")
    return available[:count]


def freeze(catalogue_path, seen_path, reference_path, count):
    catalogue = json.loads(catalogue_path.read_text())
    seen = json.loads(seen_path.read_text())
    reference = json.loads(reference_path.read_text())
    params = reference["validation_config"]["parameters"]["range_scan_ratio"]
    if params["range"]["ground_z"] is not None:
        raise ValueError("requires coordinate-corrected reference with ground_z=null")
    if params["scan_ratio"]["min_votes"] is not None:
        raise ValueError("requires normalized scan-ratio voting")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "argoverse-2-sensor-val",
        "catalogue_source": catalogue["source_url"],
        "catalogue_sha256": sha256(catalogue_path),
        "catalogue_scenes": catalogue["scenes"],
        "previously_examined_scenes": sorted(set(seen)),
        "selection_evidence_sha256": sha256(seen_path),
        "scenes": select_scenes(catalogue["scenes"], seen, count),
        "selection": "First lexicographic unexamined logs; no moving-content screening or result-based reselection.",
        "frames": 12, "stride": 3, "moving_threshold_m": 2.0,
        "box_margin_m": 0.2, "sensor_ground_z": -1.4,
        "ground_preprocessing_frame": "ego_before_pose_transform",
        "parameters": params,
        "parameters_source_sha256": sha256(reference_path),
        "modes": list(MODES),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "aggregation": "Unweighted mean over every selected scene, including zero-moving-GT scenes; also report worst-scene static preservation and every per-scene delta. No minimum-GT exclusion.",
    }


def verify_plan(plan):
    if plan.get("schema_version") != 1 or plan.get("modes") != list(MODES):
        raise ValueError("unsupported frozen plan")
    selected = select_scenes(plan["catalogue_scenes"], plan["previously_examined_scenes"], len(plan["scenes"]))
    if plan["scenes"] != selected:
        raise ValueError("scene selection differs from frozen lexicographic rule")
    if (plan["frames"], plan["stride"], plan["moving_threshold_m"], plan["box_margin_m"],
            plan["sensor_ground_z"], plan["ground_preprocessing_frame"]) != (
            12, 3, 2.0, 0.2, -1.4, "ego_before_pose_transform"):
        raise ValueError("unsupported input/GT contract")
    if plan["parameters"]["range"]["ground_z"] is not None:
        raise ValueError("absolute map-Z ground guard is forbidden for this evaluation")
    if set(plan["code_sha256"]) != set(CODE_FILES):
        raise ValueError("incomplete code fingerprints")
    for name, expected in plan["code_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"code changed after freeze: {name}")


def evaluate_scene(manifest_path, data_root, plan, scene):
    import pyarrow.feather as feather

    manifest = json.loads(manifest_path.read_text())
    if manifest["scene"] != scene or len(manifest["frames"]) != plan["frames"]:
        raise ValueError("manifest scene/frame selection mismatch")
    annotation = data_root / scene / "annotations.feather"
    poses = data_root / scene / "city_SE3_egovehicle.feather"
    timestamps = lambda p: set(map(int, feather.read_table(p, columns=["timestamp_ns"])["timestamp_ns"].to_pylist()))
    expected = sorted(timestamps(annotation) & timestamps(poses))[:plan["frames"] * plan["stride"]:plan["stride"]]
    if [f["timestamp_ns"] for f in manifest["frames"]] != expected:
        raise ValueError("manifest timestamps differ from frozen first-12 stride-3 selection")
    scans, _ = core._load_scan_manifest(manifest_path)
    points = np.concatenate([p for p, _ in scans])
    labels = []
    assets = {}
    for frame, (scan, _) in zip(manifest["frames"], scans):
        label = np.load(manifest_path.parent / frame["point_labels"])
        if label.shape != (len(scan),) or not np.isin(label, [0, 1]).all():
            raise ValueError("invalid row-aligned GT")
        labels.append(label.astype(bool))
        for key in ("cloud", "point_labels"):
            assets[frame[key]] = sha256(manifest_path.parent / frame[key])
    gt = np.concatenate(labels)
    params = plan["parameters"]
    _, kr = core.clean_map_by_visibility(points, scans, **params["range"])
    _, ks = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    hashes = {"manifest_sha256": sha256(manifest_path),
              "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    reference = {"config": dict(hashes, parameters={"range_scan_ratio": params}),
                 "metrics": {"range_scan_ratio": bench.compute_accuracy_metrics(~kr & ~ks, gt)}}
    result = compare_scene(manifest_path, reference, MODES)
    result.update(map_points=len(points), timestamps_ns=expected,
                  native_api_mask_verified=True, asset_sha256=assets,
                  annotation_sha256=sha256(annotation), poses_sha256=sha256(poses))
    return result


def aggregate(records):
    if not records or len({r["scene"] for r in records}) != len(records):
        raise ValueError("requires nonempty unique scene records")
    output = {}
    for mode in MODES:
        deltas = {r["scene"]: {k: r["results"][mode]["metrics"][k] - r["results"]["none"]["metrics"][k]
                              for k in METRICS} for r in records}
        output[mode] = {
            "mean": {k: float(np.mean([r["results"][mode]["metrics"][k] for r in records])) for k in METRICS},
            "worst_scene_static_preservation": min(r["results"][mode]["metrics"]["static_preservation"] for r in records),
            "per_scene_delta_from_native": deltas,
            "f1_regression_scenes": [s for s, d in deltas.items() if d["f1"] < 0],
            "static_regression_scenes": [s for s, d in deltas.items() if d["static_preservation"] < 0],
        }
    return {"included_scenes": [r["scene"] for r in records], "excluded_scenes": [],
            "zero_moving_gt_scenes": [r["scene"] for r in records if r["gt_dynamic_points"] == 0],
            "methods": output}


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    for name in ("catalogue-json", "seen-json", "reference-json", "plan-json"):
        f.add_argument("--" + name, type=Path, required=True)
    f.add_argument("--count", type=int, default=3)
    e = sub.add_parser("evaluate")
    for name in ("plan-json", "manifest-root", "data-root", "report-json"):
        e.add_argument("--" + name, type=Path, required=True)
    e.add_argument("--plan-sha256", required=True, help="Digest recorded when freezing, before acquiring new scenes")
    args = parser.parse_args(argv)
    try:
        if args.action == "freeze":
            plan = freeze(args.catalogue_json, args.seen_json, args.reference_json, args.count)
            write_new(args.plan_json, plan)
            print(json.dumps({"scenes": plan["scenes"], "plan_sha256": sha256(args.plan_json)}), flush=True)
        else:
            if args.report_json.exists():
                raise ValueError("report exists; choose a new path")
            if sha256(args.plan_json) != args.plan_sha256:
                raise ValueError("plan digest differs from pre-acquisition freeze")
            plan = json.loads(args.plan_json.read_text())
            verify_plan(plan)
            records = []
            for scene in plan["scenes"]:
                print(f"Evaluating {scene}", flush=True)
                records.append(evaluate_scene(args.manifest_root / scene / "manifest.json", args.data_root, plan, scene))
                print(json.dumps({"scene": scene, "results": records[-1]["results"]}), flush=True)
            report = {"plan_sha256": sha256(args.plan_json), "plan": plan, "scene_results": records,
                      "aggregate": aggregate(records),
                      "limitations": "Three new AV2 val logs selected without content screening; not an official held-out test split or sparse-sensor validation. Moving annotation-box GT is not per-point motion. Offline maps include future points. Visibility support is an angular-bin approximation and reuses range evidence. Zero-GT scenes remain included with the benchmark's zero precision/recall/F1 convention. No production defaults changed; per-scene regressions are reported without a promotion claim."}
            write_new(args.report_json, report)
            print(json.dumps(report["aggregate"], indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
