#!/usr/bin/env python3
"""Reproducible accuracy benchmark on real Argoverse 2 data.

Downloads a short sequence of consecutive LiDAR sweeps + annotations + ego poses from
the public Argoverse 2 Sensor Dataset (no signup, ``--no-sign-request``), builds a
pose-aligned accumulated map, derives a ground-truth *dynamic* point mask from the
annotation boxes of tracks that **actually moved**, and measures how well this repo's
**detector-free** map cleaner recovers it:

  * ``range``    -- multi-scan range-image visibility cleaner (clean_map_by_visibility)
  * ``temporal`` -- voxel temporal-consistency filter (TemporalConsistencyFilter)

Reports precision / recall / F1 + static preservation as a Markdown table you can paste
into the README. These are **our methods, measured on AV2** -- not a re-run of
ERASOR/Removert.

Why these defaults: visibility removal needs (a) real moving content and (b) a modest
ego baseline so static structure stays consistent. The default scene is a dense-traffic
val log; frames are consecutive 10 Hz sweeps (~1 m apart). Ground points are protected
(sampled at shifting angles between scans). Ground truth counts only tracks whose center
moved > ``--moving-thresh`` m across the window -- a motion-based remover should not be
expected to remove parked cars.

Usage (needs numpy + pyarrow + awscli, e.g. a venv):
    python3 scripts/run_av2_benchmark.py --frames 12
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dynamic_object_removal as core  # noqa: E402
import bench  # noqa: E402

# Dense-traffic Argoverse 2 val log (most moving-object points among sampled logs).
DEFAULT_SCENE = "0b5142c1-420b-3fea-9e98-b87327ae22c6"
S3_ROOT = "s3://argoverse/datasets/av2/sensor/val"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "av2_benchmark"
GROUND_Z = -1.4  # sensor-frame ground height (AV2 LiDAR ~1.73 m above ground)


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        raise SystemExit(1)


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _scene_dir(scene: str) -> Path:
    return OUTPUT_DIR / scene


def _download(scene: str, frames: int, stride: int) -> list[int]:
    import pyarrow.feather as feather

    base = f"{S3_ROOT}/{scene}"
    d = _scene_dir(scene)
    (d / "lidar").mkdir(parents=True, exist_ok=True)
    ann_file = d / "annotations.feather"
    pose_file = d / "city_SE3_egovehicle.feather"
    if not ann_file.exists():
        _run(["aws", "s3", "cp", "--no-sign-request", f"{base}/annotations.feather", str(ann_file)])
    if not pose_file.exists():
        _run(["aws", "s3", "cp", "--no-sign-request", f"{base}/city_SE3_egovehicle.feather", str(pose_file)])

    pose_ts = set(int(t) for t in feather.read_table(pose_file)["timestamp_ns"].to_pylist())
    ann_ts = sorted(int(t) for t in set(feather.read_table(ann_file)["timestamp_ns"].to_pylist()))
    usable = [t for t in ann_ts if t in pose_ts]
    selected = usable[0 : frames * stride : stride]
    for ts in selected:
        f = d / "lidar" / f"{ts}.feather"
        if not f.exists():
            _run(["aws", "s3", "cp", "--no-sign-request", f"{base}/sensors/lidar/{ts}.feather", str(f)])
    return selected


def _load_poses(scene: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import pyarrow.feather as feather

    t = feather.read_table(_scene_dir(scene) / "city_SE3_egovehicle.feather")
    cols = {c: t[c].to_numpy() for c in ["timestamp_ns", "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m"]}
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(t.num_rows):
        R = _quat_to_rot(cols["qw"][i], cols["qx"][i], cols["qy"][i], cols["qz"][i])
        out[int(cols["timestamp_ns"][i])] = (R, np.array([cols["tx_m"][i], cols["ty_m"][i], cols["tz_m"][i]]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accuracy benchmark on real AV2 data.")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="AV2 val log id.")
    parser.add_argument("--frames", type=int, default=12, help="Number of sweeps.")
    parser.add_argument("--stride", type=int, default=3, help="Frame stride (3 = ~3 Hz, gives objects time to move).")
    parser.add_argument("--range-margin", type=float, default=core.DEFAULT_RANGE_MARGIN)
    parser.add_argument("--min-see-through", type=int, default=3, help="Scans that must see a point as free space.")
    parser.add_argument("--max-surface-hits", type=int, default=3, help="Max scans confirming a point as surface (revert guard).")
    parser.add_argument("--h-res", type=float, default=1.0)
    parser.add_argument("--v-res", type=float, default=1.0)
    parser.add_argument("--resolutions", type=float, nargs="+", default=None,
                        help="Multi-resolution consensus (e.g. --resolutions 1.0 2.0): "
                             "higher precision, slightly lower recall. Overrides --h/--v-res.")
    parser.add_argument("--moving-thresh", type=float, default=2.0, help="Track center displacement (m) to count as moving GT.")
    parser.add_argument("--voxel-size", type=float, default=core.DEFAULT_TEMPORAL_VOXEL_SIZE)
    parser.add_argument("--temporal-min-hits", type=int, default=2)
    parser.add_argument("--sr-rings", type=int, default=core.DEFAULT_SR_RINGS)
    parser.add_argument("--sr-sectors", type=int, default=core.DEFAULT_SR_SECTORS)
    parser.add_argument("--sr-max-range", type=float, default=core.DEFAULT_SR_MAX_RANGE)
    parser.add_argument("--sr-ratio", type=float, default=core.DEFAULT_SR_RATIO,
                        help="Column dynamic if query/map height ratio is below this.")
    parser.add_argument("--sr-min-map-height", type=float, default=core.DEFAULT_SR_MIN_MAP_HEIGHT)
    parser.add_argument("--sr-ground-margin", type=float, default=core.DEFAULT_SR_GROUND_MARGIN)
    parser.add_argument("--sr-min-votes", type=int, default=2,
                        help="Scans that must flag a point before the scan-ratio test removes it.")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args(argv)

    import pyarrow.feather as feather

    print(f"Argoverse 2 scene {args.scene}: {args.frames} sweeps (stride {args.stride})...")
    selected = _download(args.scene, args.frames, args.stride)
    poses = _load_poses(args.scene)
    ann_file = _scene_dir(args.scene) / "annotations.feather"
    ann = feather.read_table(ann_file)
    arr = {c: (ann[c].to_pylist() if c in ("track_uuid", "category") else ann[c].to_numpy()) for c in ann.column_names}

    # Pose-aligned accumulated map (ground removed) + per-frame scans/origins/slices.
    chunks: list[np.ndarray] = []
    scans: list[tuple[np.ndarray, np.ndarray]] = []
    slices: list[tuple[int, int]] = []
    cursor = 0
    for ts in selected:
        R, tvec = poses[ts]
        pts_ego = core.load_points(_scene_dir(args.scene) / "lidar" / f"{ts}.feather", fmt="feather")
        pts_ego = pts_ego[pts_ego[:, 2] > GROUND_Z]
        pts_city = pts_ego @ R.T + tvec
        chunks.append(pts_city)
        scans.append((pts_city, tvec))
        slices.append((cursor, cursor + len(pts_city)))
        cursor += len(pts_city)
    acc_map = np.concatenate(chunks, axis=0)

    # Moving tracks: center displacement in the city frame across the window.
    track_centers: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for i in range(ann.num_rows):
        ts = int(arr["timestamp_ns"][i])
        if ts not in poses or ts not in set(selected):
            continue
        R, tvec = poses[ts]
        c = R @ np.array([arr["tx_m"][i], arr["ty_m"][i], arr["tz_m"][i]]) + tvec
        track_centers[arr["track_uuid"][i]].append(c)
    moving = {
        u for u, cs in track_centers.items()
        if len(cs) > 1 and np.max(np.linalg.norm(np.array(cs) - np.array(cs)[0], axis=1)) > args.moving_thresh
    }

    # GT dynamic mask over the accumulated map: points inside a moving track's box.
    gt_chunks = []
    for ts, (s, e) in zip(selected, slices):
        R, tvec = poses[ts]
        ego_yaw = math.atan2(R[1, 0], R[0, 0])
        sub = ann.filter(arr["timestamp_ns"] == ts) if False else ann
        gb: list[core.DetectionBox] = []
        rows = np.where(np.asarray(arr["timestamp_ns"]) == ts)[0]
        for i in rows:
            if arr["track_uuid"][i] not in moving:
                continue
            c = R @ np.array([arr["tx_m"][i], arr["ty_m"][i], arr["tz_m"][i]]) + tvec
            size = np.array([arr["length_m"][i], arr["width_m"][i], arr["height_m"][i]])
            yaw = math.atan2(2 * (arr["qw"][i] * arr["qz"][i] + arr["qx"][i] * arr["qy"][i]),
                             1 - 2 * (arr["qy"][i] ** 2 + arr["qz"][i] ** 2))
            gb.append(core.DetectionBox(center=c, size=size, yaw=yaw + ego_yaw))
        pts = acc_map[s:e]
        gt_chunks.append(bench.dynamic_gt_mask(pts, gb, dynamic_labels=None, margin=(0.2, 0.2, 0.2))
                         if gb else np.zeros(e - s, dtype=bool))
    gt_mask = np.concatenate(gt_chunks)
    print(f"  map: {len(acc_map):,} pts | moving tracks: {len(moving)} | GT dynamic: {int(gt_mask.sum()):,} ({gt_mask.mean()*100:.2f}%)")

    # --- range: multi-scan visibility cleaner ---
    _, keep_range = core.clean_map_by_visibility(
        acc_map, scans,
        h_res_deg=args.h_res, v_res_deg=args.v_res, range_margin=args.range_margin,
        min_see_through=args.min_see_through, max_surface_hits=args.max_surface_hits,
        ground_z=GROUND_Z, resolutions=args.resolutions,
    )
    range_metrics = bench.compute_accuracy_metrics(~keep_range, gt_mask)

    # --- temporal: per-frame voxel consistency assembled over the map ---
    keep_temporal = np.ones(len(acc_map), dtype=bool)
    tfilter = core.TemporalConsistencyFilter(voxel_size=args.voxel_size, window_size=len(selected), min_hits=args.temporal_min_hits)
    for (s, e) in slices:
        tfilter.filter(acc_map[s:e])
    for (s, e) in slices:
        _, keep_f = tfilter.filter(acc_map[s:e])
        keep_temporal[s:e] = keep_f
    temporal_metrics = bench.compute_accuracy_metrics(~keep_temporal, gt_mask)

    # --- scan-ratio: ERASOR-style per-column pseudo-occupancy (a different signal) ---
    _, keep_sr = core.clean_map_by_scan_ratio(
        acc_map, scans,
        n_rings=args.sr_rings, n_sectors=args.sr_sectors, max_range=args.sr_max_range,
        scan_ratio_threshold=args.sr_ratio, min_map_height=args.sr_min_map_height,
        ground_margin=args.sr_ground_margin, min_votes=args.sr_min_votes,
    )
    scanratio_metrics = bench.compute_accuracy_metrics(~keep_sr, gt_mask)

    def row(name: str, m: dict) -> str:
        return (f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                f"{m['static_preservation']:.3f} |")

    table = (
        "\n### Measured on Argoverse 2 (this repo's detector-free methods)\n\n"
        f"Scene `{args.scene}`, {len(selected)} pose-aligned sweeps, {len(acc_map):,} points "
        f"({int(gt_mask.sum()):,} ground-truth points on moving objects).\n\n"
        "| method | precision | recall | F1 | static kept |\n"
        "|---|---|---|---|---|\n"
        f"{row('range-image visibility', range_metrics)}\n"
        f"{row('scan-ratio (pseudo-occupancy)', scanratio_metrics)}\n"
        f"{row('temporal consistency', temporal_metrics)}\n"
    )
    print(table)

    payload = {
        "scene": args.scene, "frames": len(selected), "stride": args.stride,
        "map_points": int(len(acc_map)), "gt_dynamic_points": int(gt_mask.sum()),
        "moving_tracks": len(moving),
        "config": {"h_res": args.h_res, "v_res": args.v_res, "range_margin": args.range_margin,
                   "min_see_through": args.min_see_through, "max_surface_hits": args.max_surface_hits,
                   "ground_z": GROUND_Z, "moving_thresh": args.moving_thresh},
        "range": range_metrics, "temporal": temporal_metrics, "scan_ratio": scanratio_metrics,
    }
    out_json = args.summary_json or str(_scene_dir(args.scene) / "benchmark_result.json")
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")
    print("\nData source: Argoverse 2 Sensor Dataset (CC BY-NC-SA 4.0) https://www.argoverse.org/av2.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
