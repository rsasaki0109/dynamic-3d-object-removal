#!/usr/bin/env python3
"""GT-free, offline vertical-translation experiment with spatial holdout cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import dynamic_object_removal as core
from scripts.validate_frozen_av2_suite import write_new
from scripts.validate_frozen_nuscenes_suite import sha256

DEFAULTS = {"cell_size_m": 1., "height_quantile": .1, "min_cell_points": 3,
            "holdout_block_cells": 5, "holdout_modulus": 5, "pair_lags": [1, 2],
            "min_training_cells": 30, "min_holdout_cells": 5, "min_inlier_fraction": .6, "min_xy_span_m": 5.,
            "mad_multiplier": 3., "residual_floor_m": .1, "max_scaled_mad_m": .5}


def validate_parameters(p):
    if set(p) != set(DEFAULTS):
        raise ValueError("alignment parameters must be complete and explicit")
    for key in ("cell_size_m", "min_xy_span_m", "mad_multiplier", "residual_floor_m", "max_scaled_mad_m"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError("positive finite alignment parameters required")
    if not 0 <= p["height_quantile"] <= 1 or not 0 < p["min_inlier_fraction"] <= 1:
        raise ValueError("invalid quantile or inlier fraction")
    for key in ("min_cell_points", "holdout_block_cells", "holdout_modulus", "min_training_cells", "min_holdout_cells"):
        if type(p[key]) is not int or p[key] < (2 if key == "holdout_modulus" else 1):
            raise ValueError("invalid integer alignment parameter")
    if not p["pair_lags"] or len(set(p["pair_lags"])) != len(p["pair_lags"]) or any(type(x) is not int or x <= 0 for x in p["pair_lags"]):
        raise ValueError("pair lags must be unique positive integers")


def holdout_cells(cells, p):
    blocks = np.floor_divide(cells, p["holdout_block_cells"])
    return (blocks[:, 0] + 3 * blocks[:, 1]) % p["holdout_modulus"] == 0


def height_profile(points, p):
    cells, inverse, counts = np.unique(np.floor(points[:, :2] / p["cell_size_m"]).astype(np.int64),
                                       axis=0, return_inverse=True, return_counts=True)
    eligible = np.flatnonzero(counts >= p["min_cell_points"])
    order = np.argsort(inverse, kind="stable")
    ends = np.cumsum(counts); starts = ends - counts
    heights = np.array([np.quantile(points[order[starts[i]:ends[i]], 2], p["height_quantile"]) for i in eligible])
    return cells[eligible], heights


def common_profiles(a, b):
    def keys(cells):
        return np.ascontiguousarray(cells).view(np.dtype([("x", "<i8"), ("y", "<i8")])).reshape(-1)
    _, ia, ib = np.intersect1d(keys(a[0]), keys(b[0]), return_indices=True)
    return a[0][ia], a[1][ia], b[1][ib]


def estimate(points_by_frame, p):
    """Fit only unlabeled XYZ. Holdout cells never enter an offset estimate."""
    validate_parameters(p)
    if len(points_by_frame) < 2 or any(a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all() for a in points_by_frame):
        raise ValueError("requires at least two finite nonempty XYZ frames")
    profiles = [height_profile(a, p) for a in points_by_frame]
    edges, pair_data = [], []
    for i in range(len(profiles)):
        for lag in p["pair_lags"]:
            j = i + lag
            if j >= len(profiles):
                continue
            cells, zi, zj = common_profiles(profiles[i], profiles[j])
            holdout = holdout_cells(cells, p)
            train = ~holdout
            residual = zi - zj  # required delta_j - delta_i
            values = residual[train]
            edge = {"i": i, "j": j, "shared_cells": len(cells), "training_cells": int(train.sum()),
                    "holdout_cells": int(holdout.sum()), "accepted": False}
            if holdout.sum() < p["min_holdout_cells"]:
                edge["reason"] = "insufficient_holdout_cells"
            elif len(values) < p["min_training_cells"]:
                edge["reason"] = "insufficient_training_cells"
            else:
                median = float(np.median(values))
                sigma = float(1.4826 * np.median(np.abs(values - median)))
                inliers = np.abs(values - median) <= max(p["residual_floor_m"], p["mad_multiplier"] * sigma)
                support = cells[train][inliers]
                span = np.ptp(support, axis=0) * p["cell_size_m"] if len(support) else np.zeros(2)
                edge.update(scaled_mad_m=sigma, inlier_cells=int(inliers.sum()), xy_span_m=span.tolist())
                if inliers.sum() < p["min_training_cells"] or inliers.mean() < p["min_inlier_fraction"]:
                    edge["reason"] = "insufficient_robust_support"
                elif sigma > p["max_scaled_mad_m"]:
                    edge["reason"] = "inconsistent_height_profiles"
                elif np.any(span < p["min_xy_span_m"]):
                    edge["reason"] = "insufficient_spatial_span"
                else:
                    edge.update(accepted=True, reason="supported", relative_delta_z_m=float(np.median(values[inliers])))
            edges.append(edge); pair_data.append((edge, residual, holdout))
    connected = {0}
    while True:
        previous = set(connected)
        for edge in edges:
            if edge["accepted"] and ({edge["i"], edge["j"]} & connected):
                connected.update((edge["i"], edge["j"]))
        if connected == previous:
            break
    applied = len(connected) == len(profiles)
    delta = np.zeros(len(profiles))
    if applied:
        design, targets = [], []
        for edge in edges:
            if not edge["accepted"]:
                continue
            row = np.zeros(len(profiles)); row[edge["j"]] = 1; row[edge["i"]] = -1
            weight = np.sqrt(edge["inlier_cells"])
            design.append(row[1:] * weight); targets.append(edge["relative_delta_z_m"] * weight)
        delta[1:] = np.linalg.lstsq(np.asarray(design), np.asarray(targets), rcond=None)[0]
    def stats(values):
        return {"cells": len(values), "median_abs_residual_m": float(np.median(np.abs(values))) if len(values) else None,
                "p90_abs_residual_m": float(np.quantile(np.abs(values), .9)) if len(values) else None}
    before, after = [], []
    for edge, residual, holdout in pair_data:
        raw = residual[holdout]
        corrected = raw + delta[edge["i"]] - delta[edge["j"]]
        edge["holdout_before"] = stats(raw); edge["holdout_after"] = stats(corrected)
        before.extend(raw.tolist()); after.extend(corrected.tolist())
    return {"applied": applied, "status": "connected_supported_graph" if applied else "fallback_zero_offsets",
            "anchor_frame": 0, "delta_z_m": delta.tolist(), "frames_connected_to_anchor": sorted(connected),
            "profile_cells_per_frame": [len(x[0]) for x in profiles], "edges": edges,
            "holdout_before": stats(np.asarray(before)), "holdout_after": stats(np.asarray(after))}


def export_corrected(source_path, manifest, offsets, output, plan_digest):
    if len(offsets) != len(manifest["frames"]) or not np.isfinite(offsets).all():
        raise ValueError("one finite Z offset per frame is required")
    clone = json.loads(json.dumps(manifest))
    for frame, dz in zip(clone["frames"], offsets):
        if "pose" not in frame:
            raise ValueError("vertical experiment requires posed sensor-frame clouds")
        for key in ("cloud", "point_labels"):
            frame[key] = os.path.relpath((source_path.parent / frame[key]).resolve(), output.resolve())
        frame["pose"]["translation"][2] += dz
    clone.update(parent_manifest_sha256=sha256(source_path),
                 experimental_alignment={"kind": "offline_vertical_translation", "plan_sha256": plan_digest,
                                         "delta_z_m": offsets, "anchor_frame": 0, "uses_gt": False,
                                         "gt_labels": "Point-row identities retained from parent; no label regeneration."})
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "manifest.json", clone)


def native_prediction(points, scans, params):
    _, kr = core.clean_map_by_visibility(points, scans, **params["range"])
    _, ks = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    return ~kr & ~ks


def run_scene(source_path, source, policy, plan, plan_digest, output):
    manifest = json.loads(source_path.read_text())
    if manifest["gt_definition"] != source["policies"][policy]["gt_definition"]:
        raise ValueError("GT definition differs from source comparison")
    scans, _ = core._load_scan_manifest(source_path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    if hashlib.sha256(points.tobytes()).hexdigest() != source["input_hashes"]["map_sha256"]:
        raise ValueError("source point geometry changed")
    # Offset fitting has no label, box, semantic, prediction or metric input.
    alignment = estimate([p for p, _ in scans], plan["parameters"])
    labels = [np.load(source_path.parent / f["point_labels"]) for f in manifest["frames"]]
    if any(g.shape != (len(p),) or not np.isin(g, [0, 1]).all() for g, (p, _) in zip(labels, scans)):
        raise ValueError("invalid source GT")
    gt = np.concatenate(labels).astype(bool)
    if hashlib.sha256(gt.tobytes()).hexdigest() != source["policies"][policy]["gt_sha256"]:
        raise ValueError("source GT changed")
    raw = native_prediction(points, scans, source["parameters"])
    if hashlib.sha256(raw.tobytes()).hexdigest() != source["prediction_sha256"]["none"]:
        raise ValueError("source native prediction mask differs")
    before = bench.compute_accuracy_metrics(raw, gt)
    if before != source["policies"][policy]["metrics_with_identical_predictions"]["none"]:
        raise ValueError("source native metrics differ")
    export_corrected(source_path, manifest, alignment["delta_z_m"], output, plan_digest)
    corrected_scans, _ = core._load_scan_manifest(output / "manifest.json", allow_undeskewed=True)
    corrected = np.concatenate([p for p, _ in corrected_scans])
    for (a, oa), (b, ob), dz in zip(scans, corrected_scans, alignment["delta_z_m"]):
        np.testing.assert_array_equal(a[:, :2], b[:, :2])
        np.testing.assert_allclose(a[:, 2] + dz, b[:, 2], rtol=0, atol=1e-12)
        np.testing.assert_allclose(oa + np.array([0, 0, dz]), ob, rtol=0, atol=1e-12)
    candidate = native_prediction(corrected, corrected_scans, source["parameters"])
    after = bench.compute_accuracy_metrics(candidate, gt)
    np.save(output / "source_prediction.npy", raw); np.save(output / "corrected_prediction.npy", candidate)
    write_new(output / "alignment.json", alignment)
    return {"scene": source["scene"], "source_manifest_sha256": sha256(source_path),
            "corrected_manifest_sha256": sha256(output / "manifest.json"), "gt_definition": manifest["gt_definition"],
            "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest(), "points": len(points), "alignment": alignment,
            "source_metrics": before, "corrected_metrics": after,
            "prediction_changes": {"added_moving_removals": int((candidate & ~raw & gt).sum()),
                                   "lost_moving_removals": int((~candidate & raw & gt).sum()),
                                   "added_static_removals": int((candidate & ~raw & ~gt).sum()),
                                   "recovered_static_points": int((~candidate & raw & ~gt).sum())},
            "corrected_points_sha256": hashlib.sha256(corrected.tobytes()).hexdigest(),
            "corrected_prediction_sha256": hashlib.sha256(candidate.tobytes()).hexdigest()}


def render(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for r in records:
        axes[0].plot(r["alignment"]["delta_z_m"], "o-", label=r["scene"])
    axes[0].set_title("Estimated corrections; no GT used"); axes[0].set_xlabel("Frame"); axes[0].set_ylabel("Z correction relative to frame 0 (m)"); axes[0].legend()
    x = np.arange(len(records))
    for dx, key, label in ((-.18, "before", "Source"), (.18, "after", "Corrected")):
        values = [r["alignment"]["holdout_" + key]["median_abs_residual_m"] for r in records]
        axes[1].bar(x + dx, [np.nan if v is None else v for v in values], width=.35, label=label)
        for i, value in enumerate(values):
            if value is None:
                axes[1].text(i + dx, 0, "NA", ha="center")
    axes[1].set_xticks(x, [r["scene"] for r in records]); axes[1].set_ylabel("Median absolute height-profile residual (m)")
    axes[1].set_title("Spatial holdout; no fit on these cells"); axes[1].legend()
    for dx, key, label in ((-.18, "source_metrics", "Source"), (.18, "corrected_metrics", "Corrected")):
        axes[2].bar(x + dx, [r[key]["static_preservation"] * 100 for r in records], width=.35, label=label)
    axes[2].set_xticks(x, [r["scene"] for r in records]); axes[2].set_ylabel("Static-labelled points kept (%)")
    axes[2].set_title("Same attribute-aware GT; still a motion proxy"); axes[2].legend()
    fig.suptitle("Exploratory relative vertical alignment: fixed configuration, one drive\n"
                 "Unsupported graph -> zero-offset fallback; no physical-registration or accuracy promotion")
    fig.savefig(output / "vertical_alignment.png", dpi=160); fig.savefig(output / "vertical_alignment.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--policy-results", type=Path, required=True)
    freeze.add_argument("--source-root", type=Path, required=True)
    freeze.add_argument("--plan-json", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    for name in ("policy-results", "source-root", "plan-json", "output", "report-json"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    evaluate.add_argument("--plan-sha256", required=True)
    evaluate.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.policy_results.read_text())
        policy = "xy_or_moving_attribute"
        if source.get("task") != "nuscenes_gt_policy_sensitivity" or policy not in source["policy_order"]:
            raise ValueError("requires the named attribute-aware GT-policy comparison")
        if args.action == "freeze":
            parameters = json.loads(json.dumps(DEFAULTS)); validate_parameters(parameters)
            plan = {"schema_version": 1, "status": "Post-audit fixed-configuration experiment; previously examined scenes.",
                    "source_policy_results_sha256": sha256(args.policy_results), "gt_policy": policy,
                    "parameters": parameters, "scene_order": [r["scene"] for r in source["scene_results"]],
                    "source_manifest_sha256": {r["scene"]: sha256(args.source_root / r["scene"] / policy / "manifest.json") for r in source["scene_results"]},
                    "code_sha256": {n: sha256(ROOT / n) for n in
                        ("scripts/experiment_nuscenes_vertical_alignment.py", "dynamic_object_removal.py", "bench.py")},
                    "holdout_rule": "One in five 5x5 XY-cell blocks, via (block_x + 3*block_y) mod 5 == 0; never used in fitting.",
                    "fallback": "If accepted pair edges do not connect every frame to frame 0, apply no corrections to the entire scene.",
                    "promotion": "Research only; no production adoption inferred from these examined scenes."}
            write_new(args.plan_json, plan)
            print(json.dumps({"plan_sha256": sha256(args.plan_json), "parameters": parameters}), flush=True)
            return 0
        if args.output.exists() or args.report_json.exists():
            raise ValueError("output/report exists; choose new paths")
        if sha256(args.plan_json) != args.plan_sha256:
            raise ValueError("alignment plan changed after freeze")
        plan = json.loads(args.plan_json.read_text())
        if plan["source_policy_results_sha256"] != sha256(args.policy_results):
            raise ValueError("source comparison changed")
        if plan["gt_policy"] != policy or plan["scene_order"] != [r["scene"] for r in source["scene_results"]]:
            raise ValueError("source scene or GT policy differs")
        for name, digest in plan["code_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError("estimator/filter code changed after freeze")
        records = []
        for record in source["scene_results"]:
            path = args.source_root / record["scene"] / policy / "manifest.json"
            if sha256(path) != plan["source_manifest_sha256"][record["scene"]]:
                raise ValueError("source manifest changed")
            records.append(run_scene(path, record, policy, plan, args.plan_sha256, args.output / record["scene"]))
            print(record["scene"] + ": " + records[-1]["alignment"]["status"], flush=True)
        result = {"plan_sha256": args.plan_sha256, "plan": plan, "scene_results": records,
                  "aggregate_mean_metrics": {condition: {metric: float(np.mean([r[condition][metric] for r in records]))
                      for metric in ("precision", "recall", "f1", "static_preservation")} for condition in ("source_metrics", "corrected_metrics")},
                  "limitations": "A single relative Z shift per frame, not full 6-DoF registration or deskew. Lower-quantile XY-cell profiles can match different surfaces or moving objects. Spatial holdout assesses profile consistency, not physical pose truth or unseen-drive transfer. Offline pair graph uses future scans and drift remains possible. Same row labels and GT policy are retained; GT does not fit offsets. All scenes retained with conservative whole-scene fallback on unsupported graphs. Existing undeskewed/2D-pose/moving-box GT limits remain; filter defaults unchanged."}
        if args.render:
            render(records, args.output)
        write_new(args.report_json, result)
        print(json.dumps(result["aggregate_mean_metrics"], indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
