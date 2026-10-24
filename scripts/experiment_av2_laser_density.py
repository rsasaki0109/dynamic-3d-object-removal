#!/usr/bin/env python3
"""Controlled AV2 laser-ID thinning, with matched-population dense predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import dynamic_object_removal as core
from scripts.validate_frozen_av2_suite import write_new
from scripts.validate_frozen_nuscenes_suite import sha256

VARIANTS = [{"name": "full64", "modulus": 1, "remainder": 0, "nominal_ids": 64}] + [
    {"name": f"mod{m}_{r}", "modulus": m, "remainder": r, "nominal_ids": 64 // m}
    for m in (2, 4) for r in range(m)]
RESOLUTIONS = [1., 2.5]
METHODS = ("range", "scan_ratio", "intersection")
METRICS = ("precision", "recall", "f1", "static_preservation")


def select_rows(laser_ids, variant):
    ids = np.asarray(laser_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= 64):
        raise ValueError("requires row-aligned integer AV2 laser IDs in 0..63")
    modulus, remainder = variant["modulus"], variant["remainder"]
    if modulus not in (1, 2, 4) or not 0 <= remainder < modulus:
        raise ValueError("unsupported thinning phase")
    return ids % modulus == remainder


def subset(scans, gt, ids_by_frame, variant):
    if len(scans) != len(ids_by_frame) or sum(len(p) for p, _ in scans) != len(gt):
        raise ValueError("point/GT/laser row counts differ")
    result, indices, cursor = [], [], 0
    for (points, origin), ids in zip(scans, ids_by_frame):
        if len(ids) != len(points):
            raise ValueError("laser IDs do not align with scan rows")
        mask = select_rows(ids, variant)
        if not mask.any():
            raise ValueError("a selected laser-ID phase has an empty frame; do not replace it")
        result.append((points[mask], origin.copy()))
        indices.append(cursor + np.flatnonzero(mask)); cursor += len(points)
    indices = np.concatenate(indices)
    return result, gt[indices], indices


def range_parameters(params, resolution):
    result = dict(params["range"])
    result.update(h_res_deg=resolution, v_res_deg=resolution, resolutions=None)
    return result


def predictions(points, scans, params, resolution, sr_keep=None):
    _, kr = core.clean_map_by_visibility(points, scans, **range_parameters(params, resolution))
    if sr_keep is None:
        _, sr_keep = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    return {"range": ~kr, "scan_ratio": ~sr_keep, "intersection": ~kr & ~sr_keep}


def score_case(actual, dense, indices, gt):
    result = {}
    for method in METHODS:
        candidate, matched = actual[method], dense[method][indices]
        if candidate.shape != gt.shape or matched.shape != gt.shape:
            raise ValueError("matched evaluation populations differ")
        a = bench.compute_accuracy_metrics(candidate, gt)
        b = bench.compute_accuracy_metrics(matched, gt)
        result[method] = {"thinned_input_metrics": a, "dense_prediction_on_same_points": b,
            "metric_delta": {key: a[key] - b[key] for key in METRICS},
            "decision_changes": {"added_moving_removed": int((candidate & ~matched & gt).sum()),
                                 "lost_moving_removed": int((~candidate & matched & gt).sum()),
                                 "added_static_removed": int((candidate & ~matched & ~gt).sum()),
                                 "recovered_static_points": int((~candidate & matched & ~gt).sum())}}
    return result


def read_inputs(path, reference, data_root):
    import pyarrow.feather as feather
    manifest = json.loads(path.read_text())
    if manifest["scene"] != reference["scene"]:
        raise ValueError("scene identity differs")
    scans, _ = core._load_scan_manifest(path)
    points = np.concatenate([p for p, _ in scans])
    gt_chunks, ids_by_frame, raw = [], [], []
    for frame, (scan, _) in zip(manifest["frames"], scans):
        labels = np.load(path.parent / frame["point_labels"])
        if labels.shape != (len(scan),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("invalid source GT")
        gt_chunks.append(labels.astype(bool))
        source = data_root / manifest["scene"] / "lidar" / f"{frame['timestamp_ns']}.feather"
        table = feather.read_table(source, columns=["x", "y", "z", "laser_number"])
        local = np.column_stack([table[c].to_numpy() for c in ("x", "y", "z")]).astype(np.float64)
        keep = local[:, 2] > -1.4
        np.testing.assert_array_equal(local[keep], np.load(path.parent / frame["cloud"]))
        ids = table["laser_number"].to_numpy()[keep]
        select_rows(ids, VARIANTS[0])
        ids_by_frame.append(ids)
        raw.append({"timestamp_ns": frame["timestamp_ns"], "file_sha256": sha256(source),
                    "retained_laser_ids": np.unique(ids).tolist()})
    gt = np.concatenate(gt_chunks)
    hashes = {"manifest_sha256": sha256(path), "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if hashes != reference["input_hashes"]:
        raise ValueError("frozen input fingerprints differ")
    return scans, gt, ids_by_frame, raw


def evaluate_scene(path, reference, data_root, plan, expected_raw, output):
    scans, gt, ids, raw = read_inputs(path, reference, data_root)
    if raw != expected_raw:
        raise ValueError("raw laser metadata changed after freeze")
    points = np.concatenate([p for p, _ in scans]); params = reference["parameters"]
    _, full_sr = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    dense = {str(res): predictions(points, scans, params, res, full_sr) for res in plan["resolutions_deg"]}
    if bench.compute_accuracy_metrics(dense["1.0"]["intersection"], gt) != reference["results"]["none"]["metrics"]:
        raise ValueError("dense native baseline does not match frozen result")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for variant in plan["variants"]:
        chosen_scans, labels, indices = subset(scans, gt, ids, variant)
        chosen_points = np.concatenate([p for p, _ in chosen_scans])
        _, sr_keep = core.clean_map_by_scan_ratio(chosen_points, chosen_scans, **params["scan_ratio"])
        target = output / variant["name"]; target.mkdir()
        np.save(target / "source_map_indices.npy", indices); np.save(target / "gt.npy", labels)
        conditions = {}
        for res in plan["resolutions_deg"]:
            actual = dense[str(res)] if variant["modulus"] == 1 else predictions(chosen_points, chosen_scans, params, res, sr_keep)
            conditions[str(res)] = score_case(actual, dense[str(res)], indices, labels)
            for method, mask in actual.items():
                np.save(target / f"removed_{res}_{method}.npy", mask)
        records.append({"variant": variant, "points": len(indices), "gt_dynamic_points": int(labels.sum()),
                        "retained_fraction": len(indices) / len(points),
                        "source_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                        "gt_sha256": hashlib.sha256(labels.tobytes()).hexdigest(), "conditions": conditions})
        print(reference["scene"][:8] + " " + variant["name"] + ": measured both resolutions", flush=True)
    translations = np.asarray([origin for _, origin in scans])
    return {"scene": reference["scene"], "input_hashes": reference["input_hashes"],
            "raw_files": raw, "source_pose_z_span_m": float(np.ptp(translations[:, 2])),
            "dense_native_baseline_reproduced": True, "parameters_source": params, "variants": records}


def aggregate(records, resolutions):
    result = {}
    for res in resolutions:
        levels = {}
        for nominal in (64, 32, 16):
            cases = [v for r in records for v in r["variants"] if v["variant"]["nominal_ids"] == nominal]
            levels[str(nominal)] = {"scene_phase_cases": len(cases), "methods": {method: {
                field: {metric: float(np.mean([v["conditions"][str(res)][method][field][metric] for v in cases]))
                        for metric in METRICS} for field in ("thinned_input_metrics", "dense_prediction_on_same_points", "metric_delta")}
                for method in METHODS},
                "zero_gt_cases": sum(v["gt_dynamic_points"] == 0 for v in cases)}
        result[str(res)] = levels
    return result


def render(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ("f1", "static_preservation"), ("Intersection F1", "Static-labelled points kept")):
        for res in report["plan"]["resolutions_deg"]:
            levels = report["aggregate"][str(res)]
            for field, style, label in (("thinned_input_metrics", "o-", "Rerun on retained lasers"),
                                        ("dense_prediction_on_same_points", "x--", "Dense prediction on same retained points")):
                ax.plot([64, 32, 16], [levels[str(n)]["methods"]["intersection"][field][metric] for n in (64, 32, 16)],
                        style, label=f"{res} deg | {label}")
        ax.set_xticks([64, 32, 16]); ax.invert_xaxis(); ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal retained laser-ID count; all phases averaged")
        ax.set_ylabel(title); ax.legend(fontsize=7)
    fig.suptitle("Controlled AV2 density ablation: source poses/GT identities retained\n"
                 "Complementary laser-ID subsets; not a simulation of a different LiDAR device")
    fig.savefig(output / "laser_density.png", dpi=160); fig.savefig(output / "laser_density.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    for name in ("frozen-results", "manifest-root", "data-root", "plan-json"):
        f.add_argument("--" + name, type=Path, required=True)
    e = sub.add_parser("evaluate")
    for name in ("frozen-results", "manifest-root", "data-root", "plan-json", "output", "report-json"):
        e.add_argument("--" + name, type=Path, required=True)
    e.add_argument("--plan-sha256", required=True); e.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.frozen_results.read_text())
        if [r["scene"] for r in source["scene_results"]] != source["plan"]["scenes"]:
            raise ValueError("requires the complete frozen AV2 scene set")
        for name, digest in source["plan"]["code_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError("frozen source code changed")
        if args.action == "freeze":
            raw = {}
            for record in source["scene_results"]:
                path = args.manifest_root / record["scene"] / "manifest.json"
                _, _, _, raw[record["scene"]] = read_inputs(path, record, args.data_root)
            plan = {"schema_version": 1, "status": "Post-audit controlled thinning; previously examined scenes.",
                    "frozen_results_sha256": sha256(args.frozen_results), "variants": VARIANTS, "resolutions_deg": RESOLUTIONS,
                    "scene_order": source["plan"]["scenes"], "raw_files": raw,
                    "code_sha256": {n: sha256(ROOT / n) for n in ("scripts/experiment_av2_laser_density.py", "dynamic_object_removal.py", "bench.py")},
                    "selection": "All parity phases for modulus 2 and all four phases for modulus 4; no phase selection by metrics.",
                    "evaluation": "For each retained population, compare rerun sparse predictions to the dense predictions restricted to those same point rows, under the same range resolution.",
                    "sensor_contract": "AV2 source laser IDs 0..63; thinning is not another physical sensor, nuScenes FOV, timing or calibration. Source poses and point-attached GT are fixed."}
            write_new(args.plan_json, plan)
            print(json.dumps({"plan_sha256": sha256(args.plan_json), "scenes": plan["scene_order"]}), flush=True)
            return 0
        if args.output.exists() or args.report_json.exists():
            raise ValueError("output/report exists; choose new paths")
        if sha256(args.plan_json) != args.plan_sha256:
            raise ValueError("density plan changed after freeze")
        plan = json.loads(args.plan_json.read_text())
        if plan["frozen_results_sha256"] != sha256(args.frozen_results) or plan["scene_order"] != source["plan"]["scenes"]:
            raise ValueError("source comparison differs")
        for name, digest in plan["code_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError("ablation/filter code changed after freeze")
        records = [evaluate_scene(args.manifest_root / r["scene"] / "manifest.json", r, args.data_root, plan,
                                  plan["raw_files"][r["scene"]], args.output / r["scene"]) for r in source["scene_results"]]
        result = {"plan_sha256": args.plan_sha256, "plan": plan, "scene_results": records,
                  "aggregate": aggregate(records, plan["resolutions_deg"]),
                  "limitations": "Previously examined AV2 logs; no new held-out acquisition. Nominal ID subsets do not reproduce a 32/16-beam device or nuScenes geometry, range distribution, timing or poses. Complementary phases have different retained populations; matched dense predictions control row selection within each case. Means average all scene/phase cases, with no GT-content exclusions. AV2 poses and moving-box GT are inherited, not perfect physical truth. Comparisons isolate controlled thinning and angular resolution within AV2; they cannot assign all cross-dataset differences to sparsity or ego Z. No production defaults change."}
        if args.render:
            render(result, args.output)
        write_new(args.report_json, result)
        print(json.dumps(result["aggregate"], indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
