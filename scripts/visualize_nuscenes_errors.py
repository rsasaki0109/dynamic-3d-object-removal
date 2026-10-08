#!/usr/bin/env python3
"""Plot real nuScenes errors and attribute moving GT to annotation tracks."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
import dynamic_object_removal as core
from scripts import run_nuscenes_benchmark as nuscenes


def moving_instances(annotations, selected, threshold=2.0):
    """Use the first selected timestamp, not metadata file order, as motion anchor."""
    centers = collections.defaultdict(list)
    for sample in selected:
        for annotation in annotations[sample]:
            centers[annotation["instance_token"]].append(np.asarray(annotation["translation"]))
    return sorted(token for token, positions in centers.items()
                  if len(positions) > 1 and np.max(np.linalg.norm(np.array(positions) - positions[0], axis=1)) > threshold)


def analyze(manifest_path, data_root, validation_root):
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads((validation_root / "validation.json").read_text())
    scans, _ = core._load_scan_manifest(manifest_path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    gt = np.load(validation_root / "gt.npy")
    keep = np.load(validation_root / "defaults/keep.npy")
    if gt.dtype != bool or keep.dtype != bool or gt.shape != (len(points),) or keep.shape != gt.shape:
        raise ValueError("invalid GT or prediction masks")
    hashes = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if any(reference.get(k) != v for k, v in hashes.items()):
        raise ValueError("input fingerprints changed")
    metrics = bench.compute_accuracy_metrics(~keep, gt)
    if metrics != reference["results"]["defaults"]["metrics"]:
        raise ValueError("prediction metrics differ from recorded baseline")
    tables = nuscenes._load_tables(data_root)
    metadata = data_root / "v1.0-mini"
    instances = {r["token"]: r for r in json.loads((metadata / "instance.json").read_text())}
    categories = {r["token"]: r["name"] for r in json.loads((metadata / "category.json").read_text())}
    samples_by_time = {float(r["timestamp"]) * 1e-6: sample
                       for sample, r in tables["lidar"].items()}
    selected = [samples_by_time[f["timestamp_sec"]] for f in manifest["frames"]]
    scene_samples = set()
    token = tables["scene"][manifest["scene"]]["first_sample_token"]
    while token:
        scene_samples.add(token)
        token = tables["sample"][token]["next"]
    if not set(selected) <= scene_samples:
        raise ValueError("manifest timestamps do not belong to selected scene")
    annotations = collections.defaultdict(list)
    for annotation in tables["ann"]:
        if annotation["sample_token"] in selected:
            annotations[annotation["sample_token"]].append(annotation)
    moving = moving_instances(annotations, selected)
    indices = {token: index for index, token in enumerate(moving)}
    owner = np.full(len(points), -1, dtype=int)
    ambiguity = np.zeros(len(points), dtype=bool)
    cursor = 0
    for sample, (scan, _) in zip(selected, scans):
        end = cursor + len(scan)
        union = np.zeros(len(scan), dtype=bool)
        for annotation in sorted(annotations[sample], key=lambda a: a["instance_token"]):
            token = annotation["instance_token"]
            if token not in indices:
                continue
            width, length, height = annotation["size"]
            box = core.DetectionBox(center=np.asarray(annotation["translation"]),
                size=np.array([length, width, height]), yaw=nuscenes._yaw_from_quat(*annotation["rotation"]))
            mask = bench.dynamic_gt_mask(scan, [box], margin=(.25, .25, .25))
            ambiguity[cursor:end] |= mask & union
            first = mask & ~union
            owner[cursor:end][first] = indices[token]
            union |= mask
        np.testing.assert_array_equal(union, gt[cursor:end])
        cursor = end
    tracks = []
    acquisition_distance = np.concatenate([np.linalg.norm(p - origin, axis=1) for p, origin in scans])
    category_counts = collections.defaultdict(lambda: {"gt_points": 0, "removed": 0, "missed": 0})
    for token, index in indices.items():
        selected_points = owner == index
        count = int(selected_points.sum())
        if not count:
            continue
        category = categories[instances[token]["category_token"]]
        removed = int((selected_points & ~keep).sum())
        missed = count - removed
        row = {"instance_token": token, "category": category, "gt_points": count,
               "removed": removed, "missed": missed, "recall": removed / count,
               "median_acquisition_range_m": float(np.median(acquisition_distance[selected_points])),
               "center_xy": np.mean(points[selected_points, :2], axis=0).tolist()}
        tracks.append(row)
        for key in ("gt_points", "removed", "missed"):
            category_counts[category][key] += row[key]
    tracks.sort(key=lambda r: r["missed"], reverse=True)
    report = {"scene": manifest["scene"], "frames": len(scans), "input_hashes": hashes,
              "parameters": reference["results"]["defaults"]["parameters"],
              "preprocessing": manifest["preprocessing"], "metrics": metrics,
              "tracks": tracks, "categories": dict(category_counts),
              "overlapping_gt_points": int(ambiguity.sum()),
              "limitations": "Moving-instance annotation boxes define GT, not per-point motion or detector outputs. Motion cutoff 2 m and box margin 0.25 m match the baseline. Overlapping boxes are assigned to the first lexicographic instance token. Background only is deterministically sampled for display; counts use all points. Rigid poses without deskew. nuScenes mini CC BY-NC-SA 4.0, https://www.nuscenes.org"}
    if sum(r["gt_points"] for r in tracks) != int(gt.sum()):
        raise ValueError("track counts do not cover all moving GT")
    return report, points, gt, keep, owner, scans


def render(report, points, gt, keep, owner, scans, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    origin = scans[0][1]
    xy = points[:, :2] - origin[:2]
    masks = {"Static kept": ~gt & keep, "Moving removed": gt & ~keep,
             "Moving missed": gt & keep, "Static removed": ~gt & ~keep}
    colors = {"Static kept": "#aab4c0", "Moving removed": "#148a58",
              "Moving missed": "#e57a13", "Static removed": "#9a42c8"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for ax in axes[0]:
        for name, mask in masks.items():
            indices = np.flatnonzero(mask)
            if name == "Static kept" and len(indices) > 15000:
                indices = indices[np.linspace(0, len(indices) - 1, 15000, dtype=int)]
            ax.scatter(xy[indices, 0], xy[indices, 1], s=1 if name == "Static kept" else 4,
                       color=colors[name], alpha=.45 if name == "Static kept" else .8, rasterized=True)
        trajectory = np.array([o[:2] - origin[:2] for _, o in scans])
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="#263e65", lw=1.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Map X relative to first sensor (m)")
        ax.set_ylabel("Map Y relative to first sensor (m)")
        ax.grid(alpha=.15)
    axes[0, 0].set_title("Full accumulated map: baseline error locations")
    worst = report["tracks"][0]
    center = np.array(worst["center_xy"]) - origin[:2]
    axes[0, 1].set_xlim(center[0] - 12, center[0] + 12)
    axes[0, 1].set_ylim(center[1] - 12, center[1] + 12)
    axes[0, 1].set_title(f"Area around highest-miss track: {worst['category']}\n{worst['missed']:,}/{worst['gt_points']:,} GT points missed; 24 m crop")
    handles = [Line2D([], [], color=colors[name], marker="o", ls="", label=f"{name}: {int(mask.sum()):,}")
               for name, mask in masks.items()]
    handles.append(Line2D([], [], color="#263e65", label="Sensor trajectory"))
    axes[0, 0].legend(handles=handles, fontsize=8, loc="upper right")
    rows = sorted(report["categories"].items(), key=lambda item: item[1]["missed"], reverse=True)
    for ax, names, removed, missed, title in [
        (axes[1, 0], [k for k, _ in rows], [v["removed"] for _, v in rows],
         [v["missed"] for _, v in rows], "Moving GT by real annotation category"),
        (axes[1, 1], [f"{r['category'].split('.')[-1]} {r['instance_token'][:6]}" for r in report["tracks"][:8]],
         [r["removed"] for r in report["tracks"][:8]], [r["missed"] for r in report["tracks"][:8]],
         "Most missed annotation tracks")]:
        ax.barh(names, removed, color=colors["Moving removed"], label="Removed")
        ax.barh(names, missed, left=removed, color=colors["Moving missed"], label="Missed")
        ax.invert_yaxis()
        ax.set_xlabel("GT points (full counts, no sampling)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    m = report["metrics"]
    fig.suptitle(f"nuScenes {report['scene']} | {report['frames']} keyframes | recall {m['recall']:.1%} | static kept {m['static_preservation']:.2%}\nMoving-box GT; no deskew; devkit close-point exclusion; gray background sampled", fontsize=13)
    fig.savefig(output / "errors.png", dpi=150)
    fig.savefig(output / "errors.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=nuscenes.ROOT_DIR)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    report, *arrays = analyze(args.manifest, args.data_root, args.validation_root)
    if not report["tracks"]:
        raise ValueError("no moving GT tracks to visualize")
    args.output.mkdir(parents=True, exist_ok=True)
    render(report, *arrays, args.output)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["categories"], indent=2))
    print(f"Saved {args.output / 'errors.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
