#!/usr/bin/env python3
"""Trace baseline point/scan evidence on the frozen nuScenes inputs, without tuning."""
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
from scripts.diagnose_common_misses import height_gates, quantiles
from scripts.validate_frozen_av2_suite import write_new
from scripts.validate_frozen_nuscenes_suite import sha256

RANGE_STATES = ("invalid_map_range", "empty_angular_pixel", "nearer_return",
                "margin_boundary", "surface", "see_through")
SR_STATES = ("outside_polar_domain", "empty_query_column", "map_height_too_small",
             "height_ratio_too_large", "ground_reverted", "dynamic_vote")
GROUPS = ("static_kept", "static_removed", "moving_both_dynamic", "moving_range_only",
          "moving_scan_ratio_only", "moving_common_miss")


def range_states(points, scan, origin, params):
    """Classify angular-bin evidence, preserving the API's strict margin bounds."""
    h, v, margin = params["h_res_deg"], params["v_res_deg"], params["range_margin"]
    cols, rows = int(np.ceil(360 / h)), int(np.ceil(180 / v))
    qr, qc, qrow, qvalid = core._spherical_pixels(scan, origin, h, v)
    nearest = np.full(cols * rows, np.inf)
    np.minimum.at(nearest, core._pixel_indices(qc, qrow, cols, rows)[qvalid], qr[qvalid])
    mr, mc, mrow, valid = core._spherical_pixels(points, origin, h, v)
    query_range = nearest[core._pixel_indices(mc, mrow, cols, rows)]
    observed = valid & np.isfinite(query_range)
    delta = query_range - mr
    states = np.zeros(len(points), np.uint8)
    states[valid] = 1
    states[observed] = 3
    states[observed & (delta < -margin)] = 2
    states[observed & (np.abs(delta) < margin)] = 4
    states[observed & (delta > margin)] = 5
    st, sf = core._visibility_votes(points, scan, origin, h, v, margin)
    np.testing.assert_array_equal(states == 5, st)
    np.testing.assert_array_equal(states == 4, sf)
    return states


def encode_scan_ratio(valid, observed, low, ratio, ground, dynamic):
    """Encode a disjoint per-point stage partition, including unobserved points."""
    if np.any(observed & ~valid):
        raise ValueError("observed column outside polar domain")
    partition = sum(x.astype(np.uint8) for x in (low, ratio, ground, dynamic))
    if not np.array_equal(partition, observed.astype(np.uint8)):
        raise ValueError("scan-ratio stages must partition observed opportunities")
    states = np.zeros(len(valid), np.uint8)
    states[valid] = 1
    for code, mask in enumerate((low, ratio, ground, dynamic), start=2):
        states[mask] = code
    return states


def replay(points, scans, params):
    rp, sp = params["range"], params["scan_ratio"]
    if rp["resolutions"] is not None or sp["min_votes"] is not None:
        raise ValueError("requires single-resolution range and normalized scan-ratio votes")
    ranges, ratios = [], []
    # Scan-ratio and ground fitting must see the full map, including static points.
    for scan, origin in scans:
        ranges.append(range_states(points, scan, origin, rp))
        dynamic, observed = core._scan_ratio_dynamic(
            points, scan, origin, sp["n_rings"], sp["n_sectors"], sp["max_range"],
            sp["scan_ratio_threshold"], sp["min_map_height"], sp["ground_margin"])
        _, valid = core._polar_bins(points, origin, sp["n_rings"], sp["n_sectors"], sp["max_range"])
        low, ratio, ground = height_gates(points, scan, origin, sp, observed, dynamic)
        ratios.append(encode_scan_ratio(valid, observed, low, ratio, ground, dynamic))
    ranges, ratios = np.asarray(ranges), np.asarray(ratios)
    see, surface = (ranges == 5).sum(axis=0), (ranges == 4).sum(axis=0)
    votes, revisits = (ratios == 5).sum(axis=0), (ratios >= 2).sum(axis=0)
    floor = max(1, min(sp["votes_floor"], len(scans)))
    threshold = np.maximum(floor, np.ceil(sp["votes_fraction"] * revisits).astype(int))
    rd = (see >= rp["min_see_through"]) & (surface <= rp["max_surface_hits"])
    if rp["ground_z"] is not None:
        rd &= points[:, 2] > rp["ground_z"]
    sd = votes >= threshold
    _, kr = core.clean_map_by_visibility(points, scans, **rp)
    _, ks = core.clean_map_by_scan_ratio(points, scans, **sp)
    np.testing.assert_array_equal(rd, ~kr)
    np.testing.assert_array_equal(sd, ~ks)
    return {"range_state": ranges, "scan_ratio_state": ratios, "range_see_votes": see,
            "range_surface_votes": surface, "scan_ratio_votes": votes,
            "scan_ratio_revisits": revisits, "scan_ratio_threshold": threshold,
            "range_dynamic": rd, "scan_ratio_dynamic": sd}


def groups(gt, rd, sd):
    result = np.zeros(len(gt), np.uint8)
    result[~gt & rd & sd] = 1
    result[gt & rd & sd] = 2
    result[gt & rd & ~sd] = 3
    result[gt & ~rd & sd] = 4
    result[gt & ~rd & ~sd] = 5
    return result


def heatmap(states, code, selected, source_frame, nframes):
    """Rows=query frames; columns=acquisition frames; missing populations are NaN."""
    matrix = np.full((nframes, nframes), np.nan)
    for source in range(nframes):
        population = selected & (source_frame == source)
        if population.any():
            matrix[:, source] = (states[:, population] == code).mean(axis=1)
    return matrix


def summarize(mask, evidence, params):
    nframes = len(evidence["range_state"])
    n = int(mask.sum())
    ranges, ratios = evidence["range_state"][:, mask], evidence["scan_ratio_state"][:, mask]
    see, surface = evidence["range_see_votes"][mask], evidence["range_surface_votes"][mask]
    votes = evidence["scan_ratio_votes"][mask]
    revisits, threshold = evidence["scan_ratio_revisits"][mask], evidence["scan_ratio_threshold"][mask]
    range_counts = {name: int((ranges == code).sum()) for code, name in enumerate(RANGE_STATES)}
    sr_counts = {name: int((ratios == code).sum()) for code, name in enumerate(SR_STATES)}
    assert sum(range_counts.values()) == sum(sr_counts.values()) == n * nframes
    low_see, high_surface = see < params["range"]["min_see_through"], surface > params["range"]["max_surface_hits"]
    return {"points": n, "point_scan_opportunities": n * nframes,
            "range_opportunities": range_counts, "scan_ratio_opportunities": sr_counts,
            "range_blockers": {"low_see_only": int((low_see & ~high_surface).sum()),
                               "surface_only": int((~low_see & high_surface).sum()),
                               "both": int((low_see & high_surface).sum()),
                               "zero_see_through": int((see == 0).sum())},
            "scan_ratio": {"zero_dynamic_votes": int((votes == 0).sum()),
                           "positive_votes_below_threshold": int(((votes > 0) & (votes < threshold)).sum()),
                           "insufficient_revisits_for_floor": int((revisits < min(params["scan_ratio"]["votes_floor"], nframes)).sum()),
                           "vote_quantiles": quantiles(votes), "revisit_quantiles": quantiles(revisits),
                           "threshold_gap_quantiles": quantiles(threshold - votes)}}


def analyze_scene(path, record, output):
    manifest = json.loads(path.read_text())
    if manifest["scene"] != record["scene"]:
        raise ValueError("manifest scene differs from frozen result")
    scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    labels = [np.load(path.parent / f["point_labels"]) for f in manifest["frames"]]
    if any(g.shape != (len(scan),) or not np.isin(g, [0, 1]).all() for g, (scan, _) in zip(labels, scans)):
        raise ValueError("invalid row-aligned GT")
    gt = np.concatenate(labels).astype(bool)
    hashes = {"manifest_sha256": sha256(path), "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if hashes != record["input_hashes"]:
        raise ValueError("input fingerprints differ from frozen result")
    if len(points) != record["map_points"] or int(gt.sum()) != record["gt_dynamic_points"]:
        raise ValueError("frozen point/GT counts differ")
    evidence = replay(points, scans, record["parameters"])
    rd, sd = evidence["range_dynamic"], evidence["scan_ratio_dynamic"]
    if bench.compute_accuracy_metrics(rd & sd, gt) != record["results"]["none"]["metrics"]:
        raise ValueError("frozen baseline metrics differ")
    group_id = groups(gt, rd, sd)
    frame_sizes = [len(p) for p, _ in scans]
    source = np.repeat(np.arange(len(scans), dtype=np.int16), frame_sizes)
    row = np.concatenate([np.arange(n, dtype=np.int32) for n in frame_sizes])
    matrices = {"range_see_through": heatmap(evidence["range_state"], 5, group_id == 5, source, len(scans)),
                "scan_ratio_height_failure": heatmap(evidence["scan_ratio_state"], 3, group_id == 5, source, len(scans))}
    summary = {name: summarize(group_id == i, evidence, record["parameters"]) for i, name in enumerate(GROUPS)}
    if sum(g["points"] for g in summary.values()) != len(points):
        raise ValueError("point groups do not partition the full map")
    output.mkdir(parents=True, exist_ok=False)
    times = np.asarray([f.get("timestamp_sec", i) for i, f in enumerate(manifest["frames"])])
    np.savez_compressed(output / "point_scan_evidence.npz", **evidence, map_index=np.arange(len(points)),
                        source_frame=source, source_row=row, xyz=points, gt_dynamic=gt,
                        group_id=group_id, relative_timestamp_seconds=times - times[0], **matrices)
    common = group_id == 5
    report = {"scene": record["scene"], "input_hashes": hashes, "parameters": record["parameters"],
              "map_points": len(points), "gt_dynamic_points": int(gt.sum()), "api_masks_equal": True,
              "groups": summary,
              "acquisition_frames": [{"frame": i, "relative_seconds": float(times[i] - times[0]),
                                      "common_misses": int((common & (source == i)).sum())} for i in range(len(scans))],
              "query_frames": [{"frame": i, "range": {name: int((evidence["range_state"][i, common] == code).sum())
                                                         for code, name in enumerate(RANGE_STATES)},
                                "scan_ratio": {name: int((evidence["scan_ratio_state"][i, common] == code).sum())
                                               for code, name in enumerate(SR_STATES)}} for i in range(len(scans))],
              "trace_sha256": sha256(output / "point_scan_evidence.npz")}
    write_new(output / "report.json", report)
    return report, matrices


def render(records, matrices, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(records), figsize=(5 * len(records), 8), squeeze=False, constrained_layout=True)
    for j, record in enumerate(records):
        for i, (name, title) in enumerate((("range_see_through", "Range: see-through"),
                                           ("scan_ratio_height_failure", "Scan-ratio: height ratio fails"))):
            ax = axes[i, j]
            shown = ax.imshow(np.ma.masked_invalid(matrices[j][name]), vmin=0, vmax=1, cmap="viridis",
                              origin="lower", aspect="equal")
            ax.set_title(record["scene"] + "\n" + title)
            ax.set_xlabel("Point acquisition frame"); ax.set_ylabel("Voting query frame")
            fig.colorbar(shown, ax=ax, label="Fraction of common misses acquired in column frame")
    fig.suptitle("Fixed-input baseline evidence | rows: query scans; columns: point acquisition\n"
                 "All common misses retained; white columns have no target points. Offline / undeskewed / moving-box GT.")
    fig.savefig(output / "common_miss_evidence.png", dpi=160)
    fig.savefig(output / "common_miss_evidence.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-results", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.report_json.exists():
            raise ValueError("output/report exists; choose new paths")
        frozen = json.loads(args.frozen_results.read_text())
        plan = frozen["plan"]
        if plan["dataset"] != "nuscenes-trainval" or [r["scene"] for r in frozen["scene_results"]] != plan["scenes"]:
            raise ValueError("requires the complete frozen trainval scene set")
        for name, expected in plan["code_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"frozen code changed: {name}")
        records, matrices = [], []
        for record in frozen["scene_results"]:
            result, matrix = analyze_scene(args.manifest_root / record["scene"] / "manifest.json",
                                           record, args.output / record["scene"])
            records.append(result); matrices.append(matrix)
            print(record["scene"] + ": reconstructed masks match APIs and frozen metrics", flush=True)
        aggregate = {}
        for group in GROUPS:
            aggregate[group] = {"points": sum(r["groups"][group]["points"] for r in records),
                "range_opportunities": {k: sum(r["groups"][group]["range_opportunities"][k] for r in records) for k in RANGE_STATES},
                "scan_ratio_opportunities": {k: sum(r["groups"][group]["scan_ratio_opportunities"][k] for r in records) for k in SR_STATES},
                "range_blockers": {k: sum(r["groups"][group]["range_blockers"][k] for r in records) for k in records[0]["groups"][group]["range_blockers"]},
                "scan_ratio": {k: sum(r["groups"][group]["scan_ratio"][k] for r in records) for k in ("zero_dynamic_votes", "positive_votes_below_threshold", "insufficient_revisits_for_floor")}}
        report = {"frozen_results_sha256": sha256(args.frozen_results), "plan_sha256": frozen["plan_sha256"],
                  "diagnostic_code_sha256": {name: sha256(ROOT / name) for name in
                      ("scripts/diagnose_common_misses.py", "scripts/diagnose_frozen_nuscenes_misses.py")},
                  "status": "Post-result baseline diagnosis; no tuning, candidates, or defaults changed.",
                  "range_state_codes": dict(enumerate(RANGE_STATES)), "scan_ratio_state_codes": dict(enumerate(SR_STATES)),
                  "group_codes": dict(enumerate(GROUPS)), "quantile_order": [0, .25, .5, .75, 1],
                  "scene_results": records, "aggregate_counts": aggregate,
                  "limitations": "Three correlated scenes from one drive. Range state counts describe angular pixels, not exact rays or physical occlusion; strict margin equality is ambiguous. Scan-ratio gates classify repeated point/scan opportunities, not distinct points. Guard-failure counts are points; zero-vote/revisit counts overlap. Full-map ground fits retain static context. Offline maps include future points; input is undeskewed; GT is moving-instance boxes. Neither and static-removed groups are diagnostic populations, not unbiased samples for tuning."}
        if args.render:
            render(records, matrices, args.output)
        write_new(args.report_json, report)
        print(json.dumps(aggregate["moving_common_miss"], indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
