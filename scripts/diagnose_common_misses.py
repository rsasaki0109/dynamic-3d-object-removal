#!/usr/bin/env python3
"""Replay visibility and scan-ratio evidence for moving GT missed by both filters."""
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

EVIDENCE_VARIANTS = {
    "see_through_2": ("range", {"min_see_through": 2}),
    "range_margin_0_25": ("range", {"range_margin": .25}),
    "resolution_half": ("range", "half"),
    "resolution_double": ("range", "double"),
    "map_height_0_25": ("scan_ratio", {"min_map_height": .25}),
    "ground_margin_0_1": ("scan_ratio", {"ground_margin": .1}),
}


def ablate_evidence(points, scans, gt, params, range_keep, sr_keep):
    """One-factor ablations; preserve baseline voting and the other channel."""
    baseline_dynamic = ~range_keep & ~sr_keep
    variants = {"baseline": {"parameters": params,
                            "metrics": bench.compute_accuracy_metrics(baseline_dynamic, gt)}}
    for name, (channel, change) in EVIDENCE_VARIANTS.items():
        candidate = {k: dict(v) for k, v in params.items()}
        if isinstance(change, str):
            factor = .5 if change == "half" else 2
            change = {key: params["range"][key] * factor for key in ("h_res_deg", "v_res_deg")}
        candidate[channel].update(change)
        kr, ks = range_keep, sr_keep
        if channel == "range":
            _, kr = core.clean_map_by_visibility(points, scans, **candidate["range"])
        else:
            _, ks = core.clean_map_by_scan_ratio(points, scans, **candidate["scan_ratio"])
        dynamic = ~kr & ~ks
        variants[name] = {"parameters": candidate,
                          "metrics": bench.compute_accuracy_metrics(dynamic, gt),
                          "additional_moving_removed": int((dynamic & ~baseline_dynamic & gt).sum()),
                          "additional_static_removed": int((dynamic & ~baseline_dynamic & ~gt).sum()),
                          "lost_moving_removed": int((~dynamic & baseline_dynamic & gt).sum()),
                          "recovered_static_points": int((~dynamic & baseline_dynamic & ~gt).sum())}
    return variants


def quantiles(values):
    return np.quantile(values, [0, .25, .5, .75, 1]).tolist() if len(values) else None


def height_gates(points, scan, origin, params, observed, dynamic):
    """Classify observed point-scan opportunities; mirror the bin-height gates."""
    rings, sectors = params["n_rings"], params["n_sectors"]
    mflat, mvalid = core._polar_bins(points, origin, rings, sectors, params["max_range"])
    qflat, qvalid = core._polar_bins(scan, origin, rings, sectors, params["max_range"])
    def heights(flat, valid, z):
        high = np.full(rings * sectors, -np.inf)
        low = np.full(rings * sectors, np.inf)
        np.maximum.at(high, flat[valid], z[valid])
        np.minimum.at(low, flat[valid], z[valid])
        return np.where(np.isfinite(high), high - low, 0)
    mh = heights(mflat, mvalid, points[:, 2])
    qh = heights(qflat, qvalid, scan[:, 2])
    low_height = np.zeros(len(points), dtype=bool)
    ratio_failure = np.zeros(len(points), dtype=bool)
    idx = np.flatnonzero(observed)
    low_height[idx] = mh[mflat[idx]] <= params["min_map_height"]
    ratio_failure[idx] = (~low_height[idx] &
        (qh[mflat[idx]] / np.maximum(mh[mflat[idx]], 1e-9) >= params["scan_ratio_threshold"]))
    ground_reverted = observed & ~low_height & ~ratio_failure & ~dynamic
    np.testing.assert_array_equal(low_height | ratio_failure | ground_reverted | dynamic, observed)
    return low_height, ratio_failure, ground_reverted


def diagnose(path: Path, reference: dict, *, ablate=False):
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
        expected_metrics = reference["results"]["defaults"]["metrics"]
        fingerprints = reference
    else:
        params = reference["config"]["parameters"]["range_scan_ratio"]
        expected_metrics = reference["metrics"]["range_scan_ratio"]
        fingerprints = reference["config"]
    hashes = {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if any(fingerprints.get(k) != v for k, v in hashes.items()):
        raise ValueError("input fingerprints changed")
    rp, sp = params["range"], params["scan_ratio"]
    if rp["resolutions"] is not None or sp["min_votes"] is not None:
        raise ValueError("diagnosis supports single-resolution normalized voting")
    see = np.zeros(len(points), dtype=np.int64)
    surface = np.zeros_like(see)
    sr_votes = np.zeros_like(see)
    revisits = np.zeros_like(see)
    low_height_votes = np.zeros_like(see)
    ratio_failure_votes = np.zeros_like(see)
    ground_reverted_votes = np.zeros_like(see)
    for scan, origin in scans:
        st, sf = core._visibility_votes(points, scan, origin,
                        rp["h_res_deg"], rp["v_res_deg"], rp["range_margin"])
        see += st
        surface += sf
        dyn, obs = core._scan_ratio_dynamic(points, scan, origin,
                    sp["n_rings"], sp["n_sectors"], sp["max_range"],
                    sp["scan_ratio_threshold"], sp["min_map_height"], sp["ground_margin"])
        sr_votes += dyn
        revisits += obs
        low, ratio, ground = height_gates(points, scan, origin, sp, obs, dyn)
        low_height_votes += low
        ratio_failure_votes += ratio
        ground_reverted_votes += ground
    floor = max(1, min(sp["votes_floor"], len(scans)))
    sr_threshold = np.maximum(floor, np.ceil(sp["votes_fraction"] * revisits).astype(int))
    range_dynamic = (see >= rp["min_see_through"]) & (surface <= rp["max_surface_hits"])
    if rp["ground_z"] is not None:
        range_dynamic &= points[:, 2] > rp["ground_z"]
    sr_dynamic = sr_votes >= sr_threshold
    # Independently require agreement with the public APIs.
    _, kr = core.clean_map_by_visibility(points, scans, **rp)
    _, ks = core.clean_map_by_scan_ratio(points, scans, **sp)
    np.testing.assert_array_equal(range_dynamic, ~kr)
    np.testing.assert_array_equal(sr_dynamic, ~ks)
    if bench.compute_accuracy_metrics(range_dynamic & sr_dynamic, gt) != expected_metrics:
        raise ValueError("baseline metrics changed")
    missed = gt & ~range_dynamic & ~sr_dynamic
    count = lambda mask: int((missed & mask).sum())
    distances = np.concatenate([np.linalg.norm(p - origin, axis=1) for p, origin in scans])
    return {"scene": manifest["scene"], "input_hashes": hashes, "parameters": params,
            **({"evidence_ablations": ablate_evidence(points, scans, gt, params, kr, ks)} if ablate else {}),
            "gt_dynamic_points": int(gt.sum()), "common_misses": int(missed.sum()),
            "evidence_api_masks_equal": True,
            "range": {"insufficient_see_through": count(see < rp["min_see_through"]),
                      "surface_guard": count(surface > rp["max_surface_hits"]),
                      "no_see_through_or_surface_evidence": count((see == 0) & (surface == 0)),
                      "ground_guard": count(points[:, 2] <= rp["ground_z"]) if rp["ground_z"] is not None else 0,
                      "see_through_quantiles": quantiles(see[missed]),
                      "surface_quantiles": quantiles(surface[missed])},
            "scan_ratio": {"insufficient_revisits_for_floor": count(revisits < floor),
                           "adequate_revisits_below_threshold": count(revisits >= floor),
                           "zero_dynamic_votes": count(sr_votes == 0),
                           "observed_point_scan_opportunities": {
                               "total": int(revisits[missed].sum()),
                               "map_height_too_small": int(low_height_votes[missed].sum()),
                               "query_map_height_ratio_too_large": int(ratio_failure_votes[missed].sum()),
                               "ground_reverted": int(ground_reverted_votes[missed].sum()),
                               "dynamic_votes": int(sr_votes[missed].sum())},
                           "revisit_quantiles": quantiles(revisits[missed]),
                           "dynamic_vote_quantiles": quantiles(sr_votes[missed])},
            "acquisition_distance_quantiles_m": quantiles(distances[missed])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--av2-manifest", type=Path, required=True)
    parser.add_argument("--av2-baseline", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--ablate-evidence", action="store_true",
                        help="Also compare one-factor visibility and height-gate changes on fixed inputs.")
    args = parser.parse_args(argv)
    if args.report_json.exists():
        parser.error("report exists; choose a new path")
    records = []
    for path in sorted(args.validation_root.glob("scene-*/manifest.json")):
        if json.loads(path.read_text()).get("preprocessing", {}).get("min_distance") != 1:
            raise ValueError("requires min-distance 1 selection")
        reference = json.loads((path.parent / "validation/validation.json").read_text())
        records.append(diagnose(path, reference, ablate=args.ablate_evidence))
        print(f"{records[-1]['scene']}: evidence agrees with public APIs", flush=True)
    if not records:
        raise ValueError("no scene manifests")
    av2 = diagnose(args.av2_manifest, json.loads(args.av2_baseline.read_text()), ablate=args.ablate_evidence)
    report = {"nuscenes_scene_results": records, "av2_scene_result": av2,
              "quantile_order": [0, .25, .5, .75, 1],
              "limitations": "Counts concern moving GT missed by both baseline channels. Range reason masks overlap and must not be summed. No see-through/surface evidence also includes occlusion, not just missing pixels. Scan-ratio height gate counts classify observed point-scan opportunities, not unique points. GT is moving-instance boxes, not per-point motion. No defaults changed."}
    if args.ablate_evidence:
        eligible = [r for r in records if r["gt_dynamic_points"] >= MIN_GT_DYNAMIC_POINTS_FOR_MEAN]
        names = ["baseline", *EVIDENCE_VARIANTS]
        keys = ["precision", "recall", "f1", "static_preservation"]
        report["evidence_ablation_summary"] = {
            "included_scenes": [r["scene"] for r in eligible],
            "min_gt_dynamic_points": MIN_GT_DYNAMIC_POINTS_FOR_MEAN,
            "nuscenes_mean": {name: {key: float(np.mean([r["evidence_ablations"][name]["metrics"][key]
                                    for r in eligible])) if eligible else None for key in keys} for name in names},
            "av2": {name: av2["evidence_ablations"][name]["metrics"] for name in names},
            "limitations": "Exploratory one-factor comparisons, not held-out validation. Baseline voting remains unchanged. One AV2 scene. Half/double resolution changes both angular dimensions together. Static preservation is a scene mean, not a per-scene guarantee."}
        print(json.dumps(report["evidence_ablation_summary"], indent=2), flush=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
