#!/usr/bin/env python3
"""Research-only empty polar-column support from adjacent angular columns."""
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


def fill_neighbors(high, low, counts, rings, sectors, support):
    """Pool adjacent sectors within the same ring; preserve occupied center bins."""
    if support not in {"none", "both", "either"}:
        raise ValueError("unknown neighbor support")
    high, low, counts = high.reshape(rings, sectors), low.reshape(rings, sectors), counts.reshape(rings, sectors)
    left, right = np.roll(counts, 1, axis=1), np.roll(counts, -1, axis=1)
    supported = ((left > 0) & (right > 0)) if support == "both" else ((left > 0) | (right > 0))
    borrowed = (counts == 0) & supported & (support != "none")
    pooled_high = np.maximum(np.roll(high, 1, axis=1), np.roll(high, -1, axis=1))
    pooled_low = np.minimum(np.roll(low, 1, axis=1), np.roll(low, -1, axis=1))
    return (np.where(borrowed, pooled_high, high).ravel(),
            np.where(borrowed, pooled_low, low).ravel(),
            np.where(borrowed, left + right, counts).ravel(), borrowed.ravel())


def ground_aligned_support(borrowed, query_low, map_low, map_counts, margin):
    """Accept inferred columns only when their lower Z agrees with map lower Z.

    This is an extrema-continuity proxy, not semantic ground or object boundaries.
    """
    finite = np.isfinite(query_low) & np.isfinite(map_low)
    aligned = np.zeros(len(borrowed), bool)
    aligned[finite] = np.abs(query_low[finite] - map_low[finite]) <= margin
    return borrowed & (map_counts > 0) & aligned


def visibility_supported_votes(dynamic, observed, inferred, seen_through, surface):
    """Gate only inferred evidence; preserve native occupied-column votes."""
    accepted = inferred & (seen_through | surface)
    return (dynamic & (~inferred | seen_through),
            observed & (~inferred | accepted), accepted)


def scan_votes(points, scan, origin, params, support, range_params=None):
    rings, sectors = params["n_rings"], params["n_sectors"]
    mf, mv = core._polar_bins(points, origin, rings, sectors, params["max_range"])
    qf, qv = core._polar_bins(scan, origin, rings, sectors, params["max_range"])
    def spread(flat, valid, z):
        high, low = np.full(rings * sectors, -np.inf), np.full(rings * sectors, np.inf)
        counts = np.zeros(rings * sectors, dtype=np.int64)
        np.maximum.at(high, flat[valid], z[valid]); np.minimum.at(low, flat[valid], z[valid])
        np.add.at(counts, flat[valid], 1)
        return high, low, counts
    mh, ml, mc = spread(mf, mv, points[:, 2])
    qh, ql, qc = spread(qf, qv, scan[:, 2])
    qh, ql, qc, borrowed_bins = fill_neighbors(
        qh, ql, qc, rings, sectors, "either" if support in {"ground_aligned", "visibility_supported"} else support)
    if support == "ground_aligned":
        accepted = ground_aligned_support(borrowed_bins, ql, ml, mc, params["ground_margin"])
        rejected = borrowed_bins & ~accepted
        qh[rejected], ql[rejected], qc[rejected] = -np.inf, np.inf, 0
        borrowed_bins = accepted
    map_height = np.where(mc > 0, mh - ml, 0)
    query_height = np.where(qc > 0, qh - ql, 0)
    observed, inferred = np.zeros(len(points), bool), np.zeros(len(points), bool)
    observed[mv] = qc[mf[mv]] > 0
    inferred[mv] = borrowed_bins[mf[mv]]
    interest = np.flatnonzero((mc > 0) & (qc > 0) & (map_height > params["min_map_height"]) &
                             (query_height / np.maximum(map_height, 1e-9) < params["scan_ratio_threshold"]))
    dynamic = np.zeros(len(points), bool)
    indices = np.flatnonzero(mv)
    order = np.argsort(mf[indices], kind="stable")
    indices = indices[order]
    sorted_bins = mf[indices]
    starts, ends = np.searchsorted(sorted_bins, interest, "left"), np.searchsorted(sorted_bins, interest, "right")
    for start, end in zip(starts, ends):
        group = indices[start:end]
        residual = core._ground_residual(points[group], max(params["ground_margin"] * 2, .3))
        dynamic[group[residual > params["ground_margin"]]] = True
    if support == "visibility_supported":
        if range_params is None or range_params.get("resolutions") is not None:
            raise ValueError("visibility support requires single-resolution range parameters")
        st, surface = core._visibility_votes(
            points, scan, origin, range_params["h_res_deg"],
            range_params["v_res_deg"], range_params["range_margin"])
        dynamic, observed, inferred = visibility_supported_votes(dynamic, observed, inferred, st, surface)
    return dynamic, observed, inferred


def clean(points, scans, params, support, range_params=None):
    if params["min_votes"] is not None:
        raise ValueError("experiment requires normalized voting")
    votes, observed = np.zeros(len(points), int), np.zeros(len(points), int)
    inferred = 0
    for scan, origin in scans:
        dyn, obs, borrowed = scan_votes(points, scan, origin, params, support, range_params)
        votes += dyn; observed += obs
        inferred += int(borrowed.sum())
    floor = max(1, min(params["votes_floor"], len(scans)))
    threshold = np.maximum(floor, np.ceil(params["votes_fraction"] * observed))
    return votes >= threshold, inferred


def compare_scene(path, reference, modes=("none", "both", "either")):
    manifest = json.loads(path.read_text())
    scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    labels = []
    for frame, (scan, _) in zip(manifest["frames"], scans):
        label = np.load(path.parent / frame["point_labels"])
        if label.shape != (len(scan),) or not np.isin(label, [0, 1]).all():
            raise ValueError("invalid GT labels")
        labels.append(label.astype(bool))
    gt = np.concatenate(labels)
    if "results" in reference:
        params = reference["results"]["defaults"]["parameters"]
        expected, fingerprints = reference["results"]["defaults"]["metrics"], reference
    else:
        params = reference["config"]["parameters"]["range_scan_ratio"]
        expected, fingerprints = reference["metrics"]["range_scan_ratio"], reference["config"]
    hashes = {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if any(fingerprints.get(k) != v for k, v in hashes.items()):
        raise ValueError("input fingerprints differ")
    _, kr = core.clean_map_by_visibility(points, scans, **params["range"])
    _, ks = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    baseline = ~kr & ~ks
    results = {}
    for support in modes:
        sr_dynamic, inferred = clean(points, scans, params["scan_ratio"], support, params["range"])
        if support == "none":
            np.testing.assert_array_equal(sr_dynamic, ~ks)
        dynamic = ~kr & sr_dynamic
        metrics = bench.compute_accuracy_metrics(dynamic, gt)
        if support == "none" and metrics != expected:
            raise ValueError("baseline metrics differ")
        results[support] = {"metrics": metrics, "inferred_point_scan_observations": inferred,
                           "additional_moving_removed": int((dynamic & ~baseline & gt).sum()),
                           "additional_static_removed": int((dynamic & ~baseline & ~gt).sum()),
                           "lost_moving_removed": int((~dynamic & baseline & gt).sum()),
                           "recovered_static_points": int((~dynamic & baseline & ~gt).sum())}
    return {"scene": manifest["scene"], "gt_dynamic_points": int(gt.sum()),
            "input_hashes": hashes, "parameters": params, "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--av2-manifest", type=Path, required=True)
    parser.add_argument("--av2-baseline", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--ground-aligned", action="store_true", help="also test lower-Z continuity using existing ground margin")
    parser.add_argument("--visibility-supported", action="store_true", help="gate inferred votes with same-scan range-image evidence")
    args = parser.parse_args(argv)
    modes = ("none", "both", "either", "ground_aligned") if args.ground_aligned else ("none", "both", "either")
    if args.visibility_supported:
        modes += ("visibility_supported",)
    if args.report_json.exists():
        parser.error("report exists; choose a new path")
    records = []
    for path in sorted(args.validation_root.glob("scene-*/manifest.json")):
        if json.loads(path.read_text()).get("preprocessing", {}).get("min_distance") != 1:
            raise ValueError("requires devkit min-distance 1 inputs")
        reference = json.loads((path.parent / "validation/validation.json").read_text())
        records.append(compare_scene(path, reference, modes))
        print(f"{records[-1]['scene']}: native mask and metrics match", flush=True)
    if not records:
        raise ValueError("no scene manifests")
    av2 = compare_scene(args.av2_manifest, json.loads(args.av2_baseline.read_text()), modes)
    eligible = [r for r in records if r["gt_dynamic_points"] >= 5000]
    keys = ["precision", "recall", "f1", "static_preservation"]
    summary = {mode: {key: float(np.mean([r["results"][mode]["metrics"][key] for r in eligible]))
                     if eligible else None for key in keys} for mode in modes}
    report = {"nuscenes_scene_results": records, "av2_scene_result": av2,
              "aggregate": {"min_gt_dynamic_points": 5000, "included_scenes": [r["scene"] for r in eligible], "methods": summary},
              "ground_aligned_rule": "Either-side support accepted only when pooled query minimum Z agrees with map-column minimum Z within existing ground_margin; lower-Z continuity proxy, not semantic ground or a proven object boundary. Native occupied columns and normalized votes unchanged.",
              "limitations": "Research-only empty-column height pooling from adjacent angular sectors within the same radial ring. Both/either support uses union height extrema, never overwrites an occupied center column. Inferred observations are not direct measurements and affect normalized vote thresholds. Fixed inputs, range and other thresholds. Same previously examined datasets; no held-out validation or production changes."}
    if not args.ground_aligned:
        report.pop("ground_aligned_rule")
    if args.visibility_supported:
        report["visibility_supported_rule"] = "Only inferred point-scan observations are gated by same-scan range-image evidence at the baseline resolution/margin. Seen-through permits inferred dynamic votes; confirmed surfaces count as inferred observations without dynamic votes. Occluded/unobserved/ambiguous points receive no inferred observation. Native column votes unchanged. Angular-bin approximation, not exact laser ray traversal. Normalized votes unchanged."
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"nuscenes": summary, "av2": av2["results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
