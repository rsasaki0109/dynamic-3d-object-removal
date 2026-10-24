#!/usr/bin/env python3
"""Attribute closer angular-pixel returns using frozen nuScenes annotation tracks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import dynamic_object_removal as core
from scripts.diagnose_frozen_nuscenes_misses import range_states
from scripts.run_nuscenes_benchmark import _yaw_from_quat
from scripts.validate_frozen_av2_suite import write_new
from scripts.validate_frozen_nuscenes_suite import selected_tables, sha256
from scripts.visualize_nuscenes_errors import moving_instances

RELATIONS = ("same_moving_track", "other_moving_track", "track_not_moving_by_window_rule",
             "outside_all_annotation_boxes", "ambiguous_target_boxes", "ambiguous_nearest_boxes",
             "exact_nearest_range_tie")


def nearest_rows(points, query, origin, params):
    """Select the first query row at the exact nearest range; record all ties."""
    h, v = params["h_res_deg"], params["v_res_deg"]
    cols, rows = int(np.ceil(360 / h)), int(np.ceil(180 / v))
    qr, qc, qrow, qvalid = core._spherical_pixels(query, origin, h, v)
    qpixel = core._pixel_indices(qc, qrow, cols, rows)
    nearest = np.full(cols * rows, np.inf)
    np.minimum.at(nearest, qpixel[qvalid], qr[qvalid])
    minima = qvalid & (qr == nearest[qpixel])
    query_row = np.full(cols * rows, len(query), dtype=np.int64)
    np.minimum.at(query_row, qpixel[minima], np.flatnonzero(minima))
    ties = np.zeros(cols * rows, dtype=np.int32)
    np.add.at(ties, qpixel[minima], 1)
    mr, mc, mrow, mvalid = core._spherical_pixels(points, origin, h, v)
    pixels = core._pixel_indices(mc, mrow, cols, rows)
    observed = mvalid & np.isfinite(nearest[pixels])
    chosen = np.where(observed, query_row[pixels], -1)
    return chosen, np.where(observed, ties[pixels], 0), mr - nearest[pixels]


def assign_frame_owners(points, annotations, token_index, moving, margin):
    owner = np.full(len(points), -1, dtype=np.int32)
    memberships = np.zeros(len(points), dtype=np.int32)
    union_moving = np.zeros(len(points), bool)
    for annotation in sorted(annotations, key=lambda a: a["instance_token"]):
        width, length, height = annotation["size"]
        q = annotation["rotation"]
        # Use the same yaw and box-margin convention as the frozen GT exporter.
        box = core.DetectionBox(center=np.asarray(annotation["translation"]),
            size=np.array([length, width, height]), yaw=_yaw_from_quat(*q))
        mask = bench.dynamic_gt_mask(points, [box], margin=(margin,) * 3)
        owner[mask & (memberships == 0)] = token_index[annotation["instance_token"]]
        memberships += mask
        if annotation["instance_token"] in moving:
            union_moving |= mask
    return owner, memberships > 1, union_moving


def relations(target, nearest, target_ambiguous, nearest_ambiguous, ties, moving_indices):
    """Mutually exclusive labels; ambiguity always overrides a track assignment."""
    code = np.full(len(target), 3, np.uint8)
    nearest_moving = np.isin(nearest, moving_indices)
    code[(nearest >= 0) & ~nearest_moving] = 2
    code[nearest_moving] = 1
    code[nearest_moving & (target == nearest)] = 0
    code[nearest_ambiguous] = 5
    code[ties > 1] = 6
    code[target_ambiguous] = 4
    return code


def summarize_events(codes, target, gap, angle):
    counts = {name: int((codes == i).sum()) for i, name in enumerate(RELATIONS)}
    details = {}
    for i, name in enumerate(RELATIONS):
        chosen = codes == i
        details[name] = {
            "events": counts[name], "distinct_target_points": int(np.unique(target[chosen]).size),
            "range_gap_quantiles_m": np.quantile(gap[chosen], [0, .25, .5, .75, 1]).tolist() if chosen.any() else None,
            "angular_separation_quantiles_deg": np.quantile(angle[chosen], [0, .25, .5, .75, 1]).tolist() if chosen.any() else None}
    assert sum(counts.values()) == len(codes)
    return {"events": len(codes), "distinct_target_points": int(np.unique(target).size),
            "counts": counts, "details": details}


def analyze_scene(path, frozen_record, evidence_record, trace_path, tables, selected, plan, output):
    manifest = json.loads(path.read_text())
    if manifest["scene"] != frozen_record["scene"] or [f["sample_token"] for f in manifest["frames"]] != selected:
        raise ValueError("scene/sample order differs from frozen metadata")
    if sha256(trace_path) != evidence_record["trace_sha256"]:
        raise ValueError("point/scan trace fingerprint changed")
    scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    gt = np.concatenate([np.load(path.parent / f["point_labels"]) for f in manifest["frames"]]).astype(bool)
    hashes = {"manifest_sha256": sha256(path), "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if hashes != frozen_record["input_hashes"] or hashes != evidence_record["input_hashes"]:
        raise ValueError("input fingerprints differ")
    annotations = defaultdict(list)
    for a in tables["ann"]:
        annotations[a["sample_token"]].append(a)
    moving = set(moving_instances(annotations, selected, threshold=plan["moving_threshold_m"]))
    tokens = sorted({a["instance_token"] for t in selected for a in annotations[t]})
    token_index = {token: i for i, token in enumerate(tokens)}
    moving_indices = np.array([token_index[token] for token in sorted(moving)], dtype=int)
    owners, ambiguous = [], []
    for token, (scan, _) in zip(selected, scans):
        owner, overlap, reconstructed_gt = assign_frame_owners(
            scan, annotations[token], token_index, moving, plan["box_margin_m"])
        owners.append(owner); ambiguous.append(overlap)
        frame_gt = gt[sum(len(x) for x in owners[:-1]):sum(len(x) for x in owners)]
        np.testing.assert_array_equal(reconstructed_gt, frame_gt)
    owners, ambiguous = np.concatenate(owners), np.concatenate(ambiguous)
    sizes = np.array([len(scan) for scan, _ in scans])
    offsets = np.r_[0, np.cumsum(sizes)]
    source_frame = np.repeat(np.arange(len(scans)), sizes)
    with np.load(trace_path) as trace:
        np.testing.assert_array_equal(trace["xyz"], points)
        np.testing.assert_array_equal(trace["gt_dynamic"], gt)
        np.testing.assert_array_equal(trace["source_frame"], source_frame)
        common = trace["group_id"] == 5
        arrays = defaultdict(list)
        queries = []
        for i, (scan, origin) in enumerate(scans):
            states = range_states(points, scan, origin, plan["parameters"]["range"])
            np.testing.assert_array_equal(states, trace["range_state"][i])
            chosen, ties, gap = nearest_rows(points, scan, origin, plan["parameters"]["range"])
            idx = np.flatnonzero(common & (states == 2))
            if np.any(chosen[idx] < 0) or np.any(gap[idx] <= plan["parameters"]["range"]["range_margin"]):
                raise ValueError("nearest-row reconstruction differs from nearer-return state")
            nearest_index = offsets[i] + chosen[idx]
            codes = relations(owners[idx], owners[nearest_index], ambiguous[idx], ambiguous[nearest_index],
                              ties[idx], moving_indices)
            target_direction = points[idx] - origin
            query_direction = points[nearest_index] - origin
            cosine = np.einsum("ij,ij->i", target_direction, query_direction) / (
                np.linalg.norm(target_direction, axis=1) * np.linalg.norm(query_direction, axis=1))
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            for key, value in {"target_map_index": idx, "query_frame": np.full(len(idx), i, dtype=np.int16),
                               "nearest_map_index": nearest_index, "nearest_query_row": chosen[idx],
                               "relation": codes, "range_gap_m": gap[idx], "angular_separation_deg": angle,
                               "nearest_tie_count": ties[idx], "same_acquisition_scan": source_frame[idx] == i}.items():
                arrays[key].append(value)
            queries.append({"frame": i, "events": len(idx),
                            "counts": {name: int((codes == c).sum()) for c, name in enumerate(RELATIONS)}})
        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        expected_count = evidence_record["groups"]["moving_common_miss"]["range_opportunities"]["nearer_return"]
        if len(arrays["relation"]) != expected_count:
            raise ValueError("nearer-return population changed")
        groups = {}
        for name, mask in {"all_queries": np.ones(len(arrays["relation"]), bool),
                           "same_acquisition_scan": arrays["same_acquisition_scan"],
                           "other_query_scans": ~arrays["same_acquisition_scan"]}.items():
            groups[name] = summarize_events(*(arrays[k][mask] for k in
                ("relation", "target_map_index", "range_gap_m", "angular_separation_deg")))
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "nearer_return_events.npz", **arrays,
                            point_owner=owners, point_owner_ambiguous=ambiguous, moving_owner_indices=moving_indices)
        report = {"scene": manifest["scene"], "input_hashes": hashes,
                  "source_evidence_trace_sha256": sha256(trace_path), "moving_gt_reconstructed_exactly": True,
                  "moving_track_tokens": sorted(moving), "owner_index_tokens": tokens,
                  "points_in_overlapping_boxes": int(ambiguous.sum()),
                  "common_miss_points_in_overlapping_boxes": int((common & ambiguous).sum()),
                  "populations": groups, "query_frames": queries,
                  "events_sha256": sha256(output / "nearer_return_events.npz")}
    write_new(output / "report.json", report)
    return report


def render(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Same moving track", "Other moving track", "Not moving by window rule",
              "Outside annotation boxes", "Target-box overlap", "Nearest-box overlap", "Nearest-distance tie"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, population, title in zip(axes, ("all_queries", "same_acquisition_scan"),
                                     ("All query scans", "Query = point acquisition scan")):
        counts = np.array([[r["populations"][population]["counts"][key] for key in RELATIONS] for r in records])
        totals = counts.sum(axis=1)
        for i, record in enumerate(records):
            values = np.divide(100 * counts[i], totals[i], out=np.zeros(len(RELATIONS)), where=totals[i] > 0)
            ax.barh(np.arange(len(RELATIONS)) + (i - 1) * .25, values, height=.24,
                    label=f"{record['scene']} (n={totals[i]:,})")
        ax.set_yticks(np.arange(len(RELATIONS)), labels); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_xlabel("Share of nearer-return point/scan events (%)")
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle("Nearest angular-pixel returns in front of common misses\n"
                 "Frozen baseline; moving-box annotations; not exact-ray occlusion or semantic ground truth")
    fig.savefig(output / "nearer_return_attribution.png", dpi=160)
    fig.savefig(output / "nearer_return_attribution.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen-results", "evidence-report", "evidence-root", "manifest-root", "data-root", "output", "report-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.report_json.exists():
            raise ValueError("output/report exists; choose new paths")
        frozen, evidence = json.loads(args.frozen_results.read_text()), json.loads(args.evidence_report.read_text())
        plan = frozen["plan"]
        if evidence["frozen_results_sha256"] != sha256(args.frozen_results) or evidence["plan_sha256"] != frozen["plan_sha256"]:
            raise ValueError("evidence does not reference this frozen comparison")
        for report in (frozen, evidence):
            if [r["scene"] for r in report["scene_results"]] != plan["scenes"]:
                raise ValueError("requires complete frozen scene order")
        for name, expected in {**plan["code_sha256"], **evidence["diagnostic_code_sha256"]}.items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"source code changed: {name}")
        for name, expected in plan["metadata_sha256"].items():
            if sha256(args.data_root / plan["metadata_version"] / (name + ".json")) != expected:
                raise ValueError(f"source metadata changed: {name}")
        tables, selected = selected_tables(args.data_root, plan)
        records = []
        for baseline, trace in zip(frozen["scene_results"], evidence["scene_results"]):
            scene = baseline["scene"]
            records.append(analyze_scene(args.manifest_root / scene / "manifest.json", baseline, trace,
                args.evidence_root / scene / "point_scan_evidence.npz", tables, selected[scene], plan, args.output / scene))
            print(scene + ": moving GT and every nearer-return event match saved evidence", flush=True)
        aggregate = {population: {key: sum(r["populations"][population]["counts"][key] for r in records) for key in RELATIONS}
                     for population in ("all_queries", "same_acquisition_scan", "other_query_scans")}
        report = {"plan_sha256": frozen["plan_sha256"], "evidence_report_sha256": sha256(args.evidence_report),
                  "status": "Post-result annotation attribution; no filter or threshold changes.",
                  "code_sha256": {name: sha256(ROOT / name) for name in
                      ("scripts/attribute_nuscenes_nearer_returns.py", "scripts/visualize_nuscenes_errors.py")},
                  "relation_codes": dict(enumerate(RELATIONS)), "quantile_order": [0, .25, .5, .75, 1],
                  "scene_results": records, "aggregate_event_counts": aggregate,
                  "limitations": "Three correlated scenes from one drive; repeated point/scan events, not distinct objects. Moving/nonmoving refers to >2 m box-center displacement over the selected window; single-observation tracks are not classified moving. Outside annotations is not proven static structure. Box overlaps and exact nearest-range ties remain separate. Target identities come from acquisition-time boxes; nearest identities from query-time boxes. Angular-bin nearest returns do not establish exact-ray occlusion. Undeskewed/offline/moving-box GT limitations remain. Distinct-point counts overlap across relation categories; no GT attribution is used for inference."}
        if args.render:
            render(records, args.output)
        write_new(args.report_json, report)
        print(json.dumps(aggregate, indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
