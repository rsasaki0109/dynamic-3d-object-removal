#!/usr/bin/env python3
"""Add later nuScenes query scans while freezing evaluation map, GT and thresholds."""
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
from scripts import run_nuscenes_benchmark as nuscenes
from scripts.visualize_nuscenes_errors import analyze


def evaluate(points, gt, scans, params, target):
    _, kr = core.clean_map_by_visibility(points, scans, **params["range"])
    _, ks = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    dynamic = ~kr & ~ks
    return {"scan_count": len(scans), "metrics": bench.compute_accuracy_metrics(dynamic, gt),
            "target_track": {"gt_points": int(target.sum()),
                             "removed": int((dynamic & target).sum()),
                             "missed": int((~dynamic & target).sum()),
                             "range_dynamic": int((~kr & target).sum()),
                             "scan_ratio_dynamic": int((~ks & target).sum())}}, dynamic


def run(manifest_path, data_root, validation_root, *, stride=3, track_token=None):
    if stride < 1:
        raise ValueError("stride must be positive")
    manifest = json.loads(manifest_path.read_text())
    report, points, gt, keep, owner, scans = analyze(manifest_path, data_root, validation_root)
    track = next((r for r in report["tracks"] if r["instance_token"] == track_token), None) if track_token else report["tracks"][0]
    if track is None:
        raise ValueError("target track has no moving GT")
    target = owner == track["owner_index"]
    tables = nuscenes._load_tables(data_root)
    samples = []
    token = tables["scene"][manifest["scene"]]["first_sample_token"]
    while token:
        samples.append(token)
        token = tables["sample"][token]["next"]
    time_to_index = {float(tables["lidar"][sample]["timestamp"]) * 1e-6: i for i, sample in enumerate(samples)}
    selected_indices = [time_to_index[f["timestamp_sec"]] for f in manifest["frames"]]
    if selected_indices != list(range(selected_indices[0], selected_indices[-1] + 1, stride)):
        raise ValueError("stride differs from baseline acquisition selection")
    future_indices = list(range(selected_indices[-1] + stride, len(samples), stride))
    if not future_indices:
        raise ValueError("no later same-cadence keyframes available")
    baseline, baseline_dynamic = evaluate(points, gt, scans, report["parameters"], target)
    np.testing.assert_array_equal(baseline_dynamic, ~keep)
    records = [{"added_scans": 0, **baseline}]
    extra_scans = []
    additions = []
    for index in future_indices:
        sample = samples[index]
        data = tables["lidar"][sample]
        ego = tables["ego"][data["ego_pose_token"]]
        calibrated = tables["cs"][data["calibrated_sensor_token"]]
        raw_path = data_root / data["filename"]
        local = np.fromfile(raw_path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(float)
        local = local[local[:, 2] > report["preprocessing"]["ground_z_sensor"]]
        local = nuscenes._remove_close_points(local, report["preprocessing"]["min_distance"])
        if not len(local):
            raise ValueError("follow-up scan is empty after baseline preprocessing")
        r_cs, r_ego = nuscenes._quat_to_rot(*calibrated["rotation"]), nuscenes._quat_to_rot(*ego["rotation"])
        origin = r_ego @ np.asarray(calibrated["translation"]) + np.asarray(ego["translation"])
        rotation = r_ego @ r_cs
        global_points = local @ rotation.T + origin
        extra_scans.append((global_points, origin))
        timestamp = float(data["timestamp"]) * 1e-6
        rp, sp = report["parameters"]["range"], report["parameters"]["scan_ratio"]
        st, sf = core._visibility_votes(points[target], global_points, origin,
                    rp["h_res_deg"], rp["v_res_deg"], rp["range_margin"])
        dyn, obs = core._scan_ratio_dynamic(points, global_points, origin,
                    sp["n_rings"], sp["n_sectors"], sp["max_range"],
                    sp["scan_ratio_threshold"], sp["min_map_height"], sp["ground_margin"])
        _, in_range = core._polar_bins(points[target], origin, sp["n_rings"], sp["n_sectors"], sp["max_range"])
        additions.append({"sample_index": index, "sample_token": sample,
                          "time_after_last_baseline_seconds": timestamp - manifest["frames"][-1]["timestamp_sec"],
                          "points": len(global_points), "raw_cloud_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                          "global_points_sha256": hashlib.sha256(global_points.tobytes()).hexdigest(),
                          "target_evidence": {"range_see_through_points": int(st.sum()),
                                              "range_surface_points": int(sf.sum()),
                                              "scan_ratio_observed_points": int(obs[target].sum()),
                                              "scan_ratio_target_points_in_range": int(in_range.sum()),
                                              "scan_ratio_dynamic_points": int(dyn[target].sum())},
                          "rotation": rotation.tolist(), "sensor_origin": origin.tolist()})
        result, dynamic = evaluate(points, gt, scans + extra_scans, report["parameters"], target)
        records.append({"added_scans": len(extra_scans), **result,
                        "additional_moving_removed": int((dynamic & ~baseline_dynamic & gt).sum()),
                        "additional_static_removed": int((dynamic & ~baseline_dynamic & ~gt).sum()),
                        "lost_moving_removed": int((~dynamic & baseline_dynamic & gt).sum()),
                        "recovered_static_points": int((~dynamic & baseline_dynamic & ~gt).sum())})
    # Prove evaluated arrays stayed fixed throughout every API call.
    for key, values in (("map_sha256", points), ("gt_sha256", gt)):
        if hashlib.sha256(values.tobytes()).hexdigest() != report["input_hashes"][key]:
            raise ValueError("evaluation inputs changed")
    return {"scene": report["scene"], "target_track": track,
            "input_hashes": report["input_hashes"], "parameters": report["parameters"],
            "preprocessing": report["preprocessing"], "baseline_sample_indices": selected_indices,
            "query_stride": stride, "followup_scans": additions, "results": records,
            "evaluation_map_and_gt_unchanged": True,
            "limitations": "One previously examined scene and track, no held-out validation. Later queries change revisit-normalized vote thresholds and surface evidence; thresholds/settings are unchanged but removals need not be monotonic. Evaluation map and moving-box GT are fixed and contain no later points or labels. Offline delay, not real-time validation. Rigid poses without deskew; research API experiment, no production changes."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--track", default=None)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report_json.exists():
        parser.error("report exists; choose a new path")
    report = run(args.manifest, args.data_root, args.validation_root,
                 stride=args.stride, track_token=args.track)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
