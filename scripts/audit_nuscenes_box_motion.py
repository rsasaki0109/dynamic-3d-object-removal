#!/usr/bin/env python3
"""Audit the box-center motion rule behind common-miss barrier-semantic targets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import dynamic_object_removal as core
from scripts.run_nuscenes_benchmark import _yaw_from_quat
from scripts.validate_frozen_av2_suite import write_new
from scripts.validate_frozen_nuscenes_suite import iter_table, selected_tables, sha256
from scripts.visualize_nuscenes_errors import moving_instances


def pose_contract(tables):
    z = [float(row["translation"][2]) for row in tables["ego"].values()]
    if not z or not np.isfinite(z).all():
        raise ValueError("requires finite source ego-pose heights")
    return {"selected_ego_pose_records": len(z), "ego_pose_translation_z_values_m": sorted(set(z)),
            "official_schema_url": "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#ego_pose",
            "official_schema_git_blob_sha": "e69415ca97e49f50fd659aac5cafcd39bffa3cf0",
            "official_schema_quotes": ["The localization is 2-dimensional in the x-y plane.", "Note that z is always 0."],
            "interpretation": "The source ego poses do not supply a vertical translation trajectory. Global-Z comparisons and fully registered 3D accumulation require additional validation; z changes alone do not establish physical object motion or annotation-only error."}


def motion_summary(annotations, samples, attributes, threshold=2., sensor_origins=None):
    rows = sorted(annotations, key=lambda a: samples[a["sample_token"]]["timestamp"])
    if not rows:
        raise ValueError("track has no observations")
    times = np.array([samples[a["sample_token"]]["timestamp"] * 1e-6 for a in rows])
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("track observations must have distinct increasing times")
    centers = np.array([a["translation"] for a in rows], float)
    sizes = np.array([a["size"] for a in rows], float)  # width, length, height
    if not np.isfinite(centers).all() or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("invalid box centers or sizes")
    yaw = np.array([_yaw_from_quat(*a["rotation"]) for a in rows])
    first_u = np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    first_v = np.array([-np.sin(yaw[0]), np.cos(yaw[0])])
    u = np.column_stack((np.cos(yaw), np.sin(yaw)))
    v = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    relative_xy = centers[:, :2] - centers[0, :2]
    cu, cv = relative_xy @ first_u, relative_xy @ first_v
    hu = .5 * (sizes[:, 1] * np.abs(u @ first_u) + sizes[:, 0] * np.abs(v @ first_u))
    hv = .5 * (sizes[:, 1] * np.abs(u @ first_v) + sizes[:, 0] * np.abs(v @ first_v))
    bounds = np.column_stack((cu - hu, cu + hu, cv - hv, cv + hv,
                              centers[:, 2] - sizes[:, 2] / 2, centers[:, 2] + sizes[:, 2] / 2))
    displacement = np.linalg.norm(centers - centers[0], axis=1)
    horizontal = np.linalg.norm(centers[:, :2] - centers[0, :2], axis=1)
    vertical = np.abs(centers[:, 2] - centers[0, 2])
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    attr = Counter()
    observations = []
    for i, a in enumerate(rows):
        names = [attributes[t]["name"] for t in a.get("attribute_tokens", [])]
        attr.update(names)
        observations.append({"annotation_token": a["token"], "sample_token": a["sample_token"],
            "relative_seconds": float(times[i] - times[0]), "center_xyz": centers[i].tolist(),
            "size_wlh": sizes[i].tolist(), "yaw_rad": float(yaw[i]), "attributes": names,
            "num_lidar_pts": a.get("num_lidar_pts"),
            "displacement_from_first_m": float(displacement[i]),
            "horizontal_displacement_from_first_m": float(horizontal[i]),
            "vertical_displacement_from_first_m": float(vertical[i]),
            "center_u_from_first_m": float(cu[i]), "projected_bounds_uvz_m": bounds[i].tolist()})
        if sensor_origins is not None:
            origin_z = float(sensor_origins[a["sample_token"]][2])
            observations[-1].update(sensor_origin_z_m=origin_z,
                                    center_minus_sensor_z_m=float(centers[i, 2] - origin_z))
    maximum = float(displacement.max())
    return {"observations": observations, "observation_count": len(rows),
            "moving_by_frozen_center_rule": len(rows) > 1 and maximum > threshold,
            "max_center_displacement_m": maximum, "last_center_displacement_m": float(displacement[-1]),
            "max_horizontal_displacement_m": float(horizontal.max()),
            "max_vertical_displacement_m": float(vertical.max()),
            "horizontal_diagnostic_would_mark_moving": len(rows) > 1 and float(horizontal.max()) > threshold,
            "center_path_length_m": float(steps.sum()),
            "max_annotation_center_step_m": float(steps.max()) if len(steps) else 0.,
            "max_annotation_center_step_rate_m_per_s": float((steps / np.diff(times)).max()) if len(steps) else None,
            "size_min_wlh_m": sizes.min(axis=0).tolist(), "size_max_wlh_m": sizes.max(axis=0).tolist(),
            "projected_bound_spans_uvz_m": np.ptp(bounds, axis=0).tolist(),
            "attribute_observation_counts": dict(attr),
            **({"sensor_origin_z_span_m": float(np.ptp([o["sensor_origin_z_m"] for o in observations])),
                "center_minus_sensor_z_span_m": float(np.ptp([o["center_minus_sensor_z_m"] for o in observations]))}
               if sensor_origins is not None else {})}


def target_memberships(points, source_frame, target, annotations, selected, moving, margin):
    tokens = sorted(moving)
    index = {token: i for i, token in enumerate(tokens)}
    point_rows, track_rows = [], []
    for frame, sample in enumerate(selected):
        rows = np.flatnonzero(target & (source_frame == frame))
        seen = set()
        for a in annotations[sample]:
            token = a["instance_token"]
            if token not in moving:
                continue
            if token in seen:
                raise ValueError("duplicate instance annotation in one sample")
            seen.add(token)
            width, length, height = a["size"]
            box = core.DetectionBox(center=np.asarray(a["translation"]), size=np.array([length, width, height]),
                                    yaw=_yaw_from_quat(*a["rotation"]))
            mask = bench.dynamic_gt_mask(points[rows], [box], margin=(margin,) * 3)
            point_rows.append(rows[mask]); track_rows.append(np.full(int(mask.sum()), index[token], np.int32))
    pairs = np.concatenate(point_rows) if point_rows else np.array([], np.int64)
    tracks = np.concatenate(track_rows) if track_rows else np.array([], np.int32)
    multiplicity = np.bincount(pairs, minlength=len(points))
    np.testing.assert_array_equal(multiplicity > 0, target)
    return tokens, pairs, tracks, multiplicity


def barrier_height_context(points, semantic, annotation, barrier_index, margin):
    """Point-height context inside the box's XY footprint, without a box-Z gate."""
    width, length, _ = annotation["size"]
    yaw = _yaw_from_quat(*annotation["rotation"])
    relative = points[:, :2] - np.asarray(annotation["translation"][:2])
    x = relative[:, 0] * np.cos(yaw) + relative[:, 1] * np.sin(yaw)
    y = -relative[:, 0] * np.sin(yaw) + relative[:, 1] * np.cos(yaw)
    selected = (semantic == barrier_index) & (np.abs(x) <= length / 2 + margin) & (np.abs(y) <= width / 2 + margin)
    z = points[selected, 2]
    return {"barrier_points_in_xy_footprint": int(selected.sum()),
            "barrier_z_quantiles_m": np.quantile(z, [0, .25, .5, .75, 1]).tolist() if len(z) else None}


def analyze_scene(path, baseline, attribution, semantic_record, trace_path, semantic_path, tables, selected,
                  plan, instances, categories, attributes, data_root, output):
    manifest = json.loads(path.read_text())
    if [f["sample_token"] for f in manifest["frames"]] != selected:
        raise ValueError("selected frame order changed")
    if sha256(trace_path) != attribution["source_evidence_trace_sha256"] or sha256(semantic_path) != semantic_record["alignment_sha256"]:
        raise ValueError("point/semantic evidence fingerprint changed")
    scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    gt = np.concatenate([np.load(path.parent / f["point_labels"]) for f in manifest["frames"]]).astype(bool)
    hashes = {"manifest_sha256": sha256(path), "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if hashes != baseline["input_hashes"] or hashes != attribution["input_hashes"] or hashes != semantic_record["input_hashes"]:
        raise ValueError("frozen inputs differ")
    annotations = defaultdict(list)
    for a in tables["ann"]:
        annotations[a["sample_token"]].append(a)
    moving = set(moving_instances(annotations, selected, plan["moving_threshold_m"]))
    if sorted(moving) != attribution["moving_track_tokens"]:
        raise ValueError("moving track set differs from saved attribution")
    with np.load(trace_path) as trace, np.load(semantic_path) as semantics:
        np.testing.assert_array_equal(trace["xyz"], points)
        np.testing.assert_array_equal(trace["gt_dynamic"], gt)
        source = trace["source_frame"]
        semantic_labels = semantics["semantic_per_map_point"]
        target = (trace["group_id"] == 5) & (semantic_labels == plan["audit_semantic_index"])
        n = int(target.sum())
        if n != semantic_record["common_miss_target_semantic_counts"]["movable_object.barrier"]:
            raise ValueError("barrier-semantic target population changed")
        tokens, pairs, memberships, multiplicity = target_memberships(
            points, source, target, annotations, selected, moving, plan["box_margin_m"])
        relevant = {tokens[i] for i in np.unique(memberships)}
        scene_token = tables["scene"][manifest["scene"]]["token"]
        full_scene = defaultdict(list)
        if relevant:
            for a in iter_table(data_root / plan["metadata_version"] / "sample_annotation.json"):
                if a["instance_token"] in relevant and tables["sample"][a["sample_token"]]["scene_token"] == scene_token:
                    full_scene[a["instance_token"]].append(a)
        tracks = []
        origins = {sample: origin for sample, (_, origin) in zip(selected, scans)}
        offsets = np.r_[0, np.cumsum([len(scan) for scan, _ in scans])]
        for token in sorted(relevant):
            ti = tokens.index(token)
            matched = pairs[memberships == ti]
            row = {"instance_token": token, "annotation_category": categories[instances[token]["category_token"]]["name"],
                   "target_point_memberships": len(matched),
                   "uniquely_assigned_target_points": int((multiplicity[matched] == 1).sum()),
                   "shared_target_points": int((multiplicity[matched] > 1).sum()),
                   "selected_window": motion_summary([a for sample in selected for a in annotations[sample] if a["instance_token"] == token],
                                                     tables["sample"], attributes, plan["moving_threshold_m"], origins),
                   "full_scene_context": motion_summary(full_scene[token], tables["sample"], attributes, plan["moving_threshold_m"])}
            if not row["selected_window"]["moving_by_frozen_center_rule"]:
                raise ValueError("target track fails frozen motion rule")
            context = []
            for frame, sample in enumerate(selected):
                annotation = next((a for a in annotations[sample] if a["instance_token"] == token), None)
                if annotation is None:
                    continue
                section = slice(offsets[frame], offsets[frame + 1])
                values = barrier_height_context(points[section], semantic_labels[section],
                    annotation, plan["audit_semantic_index"], plan["box_margin_m"])
                context.append({"sample_token": sample, "acquisition_frame": frame,
                                "annotation_center_z_m": annotation["translation"][2], **values})
            observed_context = [r for r in context if r["barrier_z_quantiles_m"] is not None]
            row["barrier_height_context"] = {"observations": context,
                "frames_with_barrier_points": len(observed_context),
                "median_point_z_span_m": float(np.ptp([r["barrier_z_quantiles_m"][2] for r in observed_context])) if observed_context else None,
                "annotation_minus_median_point_z_span_m": float(np.ptp([r["annotation_center_z_m"] - r["barrier_z_quantiles_m"][2]
                                                                        for r in observed_context])) if observed_context else None}
            tracks.append(row)
        tracks.sort(key=lambda r: (-r["target_point_memberships"], r["instance_token"]))
        horizontal_track_ids = [tokens.index(r["instance_token"]) for r in tracks
                                if r["selected_window"]["horizontal_diagnostic_would_mark_moving"]]
        retained = np.unique(pairs[np.isin(memberships, horizontal_track_ids)])
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "target_memberships.npz", target_map_index=np.flatnonzero(target),
                            point_track_map_index=pairs, point_track_index=memberships,
                            moving_membership_count=multiplicity[target], xyz=points[target], source_frame=source[target])
        result = {"scene": manifest["scene"], "input_hashes": hashes, "barrier_common_miss_points": n,
                  "target_points_with_one_moving_box": int((target & (multiplicity == 1)).sum()),
                  "target_points_with_multiple_moving_boxes": int((target & (multiplicity > 1)).sum()),
                  "diagnostic_only_points_covered_by_any_horizontal_motion_track": int(len(retained)),
                  "diagnostic_only_points_not_covered_by_horizontal_motion_tracks": n - int(len(retained)),
                  "moving_track_index_tokens": tokens, "tracks": tracks,
                  "memberships_sha256": sha256(output / "target_memberships.npz")}
    write_new(output / "report.json", result)
    return result


def render(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = sorted(((scene, track) for scene in records for track in scene["tracks"]),
                   key=lambda x: (-x[1]["target_point_memberships"], x[1]["instance_token"]))[:3]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 3, figsize=(16, 5 * len(cases)), squeeze=False, constrained_layout=True)
    for row, (scene, track) in enumerate(cases):
        observations = track["selected_window"]["observations"]
        centers = np.array([o["center_xyz"] for o in observations]); first = centers[0, :2]
        token_index = scene["moving_track_index_tokens"].index(track["instance_token"])
        with np.load(output / scene["scene"] / "target_memberships.npz") as evidence:
            wanted = evidence["point_track_map_index"][evidence["point_track_index"] == token_index]
            mask = np.isin(evidence["target_map_index"], wanted)
            xyz = evidence["xyz"][mask]; source = evidence["source_frame"][mask]
        ax = axes[row, 0]
        shown = ax.scatter(xyz[:, 0] - first[0], xyz[:, 1] - first[1], c=source, cmap="viridis", s=2, alpha=.45, vmin=0, vmax=11)
        ax.plot(centers[:, 0] - first[0], centers[:, 1] - first[1], "kx-", label="Selected box centers")
        for o, color, label in ((observations[0], "#d24626", "First box"), (observations[-1], "#265cb0", "Last box")):
            width, length, _ = o["size_wlh"]; yaw = o["yaw_rad"]
            u, v = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
            center = np.array(o["center_xyz"][:2]) - first
            xy = np.array([center + a * length / 2 * u + b * width / 2 * v for a, b in ((-1,-1),(-1,1),(1,1),(1,-1),(-1,-1))])
            ax.plot(xy[:, 0], xy[:, 1], color=color, label=label)
        ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=7)
        ax.set_xlabel("Map X relative to first center (m)"); ax.set_ylabel("Map Y relative to first center (m)")
        ax.set_title(f"{scene['scene']} {track['instance_token'][:8]}\n{track['annotation_category']}; {track['target_point_memberships']:,} target memberships")
        fig.colorbar(shown, ax=ax, label="Point acquisition frame")
        t = [o["relative_seconds"] for o in observations]
        ax = axes[row, 1]
        ax.plot(t, [o["displacement_from_first_m"] for o in observations], "kx-", label="3D center displacement")
        ax.plot(t, [o["horizontal_displacement_from_first_m"] for o in observations], label="Horizontal displacement")
        ax.plot(t, [o["vertical_displacement_from_first_m"] for o in observations], label="Absolute vertical displacement")
        if "sensor_origin_z_m" in observations[0]:
            ax.plot(t, [abs(o["sensor_origin_z_m"] - observations[0]["sensor_origin_z_m"]) for o in observations],
                    "--", label="Absolute sensor Z displacement")
        context = {r["sample_token"]: r for r in track["barrier_height_context"]["observations"]
                   if r["barrier_z_quantiles_m"] is not None}
        valid = [o for o in observations if o["sample_token"] in context]
        if valid:
            base_z = context[valid[0]["sample_token"]]["barrier_z_quantiles_m"][2]
            ax.plot([o["relative_seconds"] for o in valid],
                    [abs(context[o["sample_token"]]["barrier_z_quantiles_m"][2] - base_z) for o in valid],
                    ":", label="Absolute barrier-point median Z shift")
        ax.axhline(2, color="#a33333", linestyle=":", label="Frozen 3D cutoff")
        ax.set_xlabel("Selected observation time (s)"); ax.set_ylabel("Displacement from first observation (m)")
        ax.set_title("Annotation and sensor vertical variation"); ax.legend(fontsize=7)
        ax = axes[row, 2]; sizes = np.array([o["size_wlh"] for o in observations])
        for i, label in enumerate(("Width", "Length", "Height")):
            ax.plot(t, sizes[:, i], "o-", label=label)
        ax.set_xlabel("Selected observation time (s)"); ax.set_ylabel("Box size (m)"); ax.legend(fontsize=8)
        ax.set_title(f"Annotation dimensions\nmax center displacement {track['selected_window']['max_center_displacement_m']:.2f} m")
    fig.suptitle("Annotation-motion audit: three largest barrier-semantic membership tracks\n"
                 "Diagnostic selection; box motion is not physical-motion ground truth; frozen GT unchanged")
    fig.savefig(output / "box_motion_audit.png", dpi=160); fig.savefig(output / "box_motion_audit.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen-results", "attribution-report", "semantics-report", "point-evidence-root", "semantic-root", "manifest-root", "data-root", "lidarseg-root", "annotation-metadata-root", "output", "report-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.report_json.exists():
            raise ValueError("output/report exists; choose new paths")
        frozen, attribution, semantics = [json.loads(p.read_text()) for p in
            (args.frozen_results, args.attribution_report, args.semantics_report)]
        plan = dict(frozen["plan"])
        if semantics["attribution_report_sha256"] != sha256(args.attribution_report) or attribution["plan_sha256"] != frozen["plan_sha256"]:
            raise ValueError("source report lineage differs")
        for report in (frozen, attribution, semantics):
            if [r["scene"] for r in report["scene_results"]] != plan["scenes"]:
                raise ValueError("requires all frozen scenes in order")
        for name, expected in {**plan["code_sha256"], **attribution["code_sha256"]}.items():
            if sha256(ROOT / name) != expected:
                raise ValueError("source code changed")
        if sha256(ROOT / "scripts/classify_nuscenes_nearer_semantics.py") != semantics["code_sha256"]:
            raise ValueError("source semantic code changed")
        for name, expected in plan["metadata_sha256"].items():
            if sha256(args.data_root / plan["metadata_version"] / (name + ".json")) != expected:
                raise ValueError("source metadata changed")
        category_path = args.lidarseg_root / plan["metadata_version"] / "category.json"
        if sha256(category_path) != semantics["category_metadata_sha256"]:
            raise ValueError("semantic category mapping changed")
        semantic_categories = json.loads(category_path.read_text())
        plan["audit_semantic_index"] = next(r["index"] for r in semantic_categories if r["name"] == "movable_object.barrier")
        extra = args.annotation_metadata_root / plan["metadata_version"]
        load = lambda name: {r["token"]: r for r in json.loads((extra / (name + ".json")).read_text())}
        instances, categories, attributes = load("instance"), load("category"), load("attribute")
        tables, selected = selected_tables(args.data_root, plan)
        records = []
        for baseline, owners, labels in zip(frozen["scene_results"], attribution["scene_results"], semantics["scene_results"]):
            scene = baseline["scene"]
            records.append(analyze_scene(args.manifest_root / scene / "manifest.json", baseline, owners, labels,
                args.point_evidence_root / scene / "point_scan_evidence.npz", args.semantic_root / scene / "semantic_alignment.npz",
                tables, selected[scene], plan, instances, categories, attributes, args.data_root, args.output / scene))
            print(scene + ": barrier-target memberships and frozen moving-track rule verified", flush=True)
        result = {"plan_sha256": frozen["plan_sha256"], "semantic_report_sha256": sha256(args.semantics_report),
                  "status": "Post-result annotation geometry audit; GT, parameters and filters unchanged.",
                  "code_sha256": sha256(Path(__file__)),
                  "extra_metadata_sha256": {n: sha256(extra / (n + ".json")) for n in ("instance", "category", "attribute")},
                  "projected_bounds_order": ["lower_u", "upper_u", "lower_v", "upper_v", "lower_z", "upper_z"],
                  "pose_contract": pose_contract(tables),
                  "scene_results": records,
                  "barrier_common_miss_points": sum(r["barrier_common_miss_points"] for r in records),
                  "limitations": "Selected audit of barrier-semantic common misses, not all GT or a new accuracy benchmark. Memberships in multiple moving boxes are retained; track membership counts must not be summed as distinct points. Official semantic class is not motion. Center displacement, speed and projected faces describe annotations, not independently measured physical motion. Horizontal-only classification is a diagnostic counterfactual, not a recommended universal rule or changed GT. Sensor/annotation Z covariation does not establish causation or prove stationary objects. Barrier-height context uses all barrier-semantic points inside each annotation XY footprint with the original XY margin and no Z/GT-membership restriction; it is not an independent object identity or point correspondence. Full-scene context is post-result and does not redefine the selected-window GT. One drive, undeskewed input, offline maps and box-GT limitations remain."}
        if args.render:
            render(records, args.output)
        write_new(args.report_json, result)
        print(json.dumps({"barrier_targets": result["barrier_common_miss_points"],
                          "tracks": [{"scene": r["scene"], "tracks": len(r["tracks"])} for r in records]}, indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
