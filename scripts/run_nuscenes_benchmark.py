#!/usr/bin/env python3
"""Reproducible accuracy benchmark on real nuScenes data.

Downloads the public **nuScenes mini** split (no signup -- it is served anonymously
over HTTPS), accumulates a pose-aligned LiDAR map from a short run of keyframes, derives
a ground-truth *dynamic* point mask from the annotation boxes of tracks that **actually
moved**, and measures how well this repo's **detector-free** map cleaner recovers it:

  * ``range``    -- multi-scan range-image visibility cleaner (clean_map_by_visibility)
  * ``temporal`` -- voxel temporal-consistency filter (TemporalConsistencyFilter)

Reports precision / recall / F1 + static preservation as a Markdown table. These are
**our methods, measured on nuScenes** -- not a re-run of ERASOR/Removert.

Why the defaults differ from the AV2 benchmark (the interesting part):
  nuScenes LiDAR is a 32-beam sensor -- roughly **5x sparser per range-image pixel** than
  Argoverse 2's dense sweep (~5 vs ~27 points per occupied 1.5-degree pixel). With AV2's
  fine range-image resolution, each pixel on nuScenes holds too few points, the per-pixel
  nearest range gets noisy, and static structure gets spuriously "seen through". The fix
  is to **match the range-image resolution to the beam density**: a coarser image
  (``--h-res/--v-res 2.5``) aggregates enough points per pixel. On the dense-traffic
  default scene this lifts F1 from ~0.30 to ~0.63 -- comparable to AV2. Like AV2, the
  method also needs a scene with real moving content; ``scene-0757`` (busy intersection)
  is the densest in the mini split.

Usage (needs numpy + the data; no extra heavy deps -- download is pure stdlib):
    python3 scripts/run_nuscenes_benchmark.py
    python3 scripts/run_nuscenes_benchmark.py --scene scene-0757 --frames 12 --stride 3
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dynamic_object_removal as core  # noqa: E402
import bench  # noqa: E402

DEFAULT_SCENE = "scene-0757"  # busiest intersection in the mini split (most moving points)
# nuScenes mini is served anonymously (no account / signup needed).
MINI_URL = "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-mini.tgz"
ROOT_DIR = Path(__file__).resolve().parents[1] / "data" / "nuscenes_mini"
GROUND_Z_SENSOR = -1.6  # LIDAR_TOP sits ~1.84 m above ground; drop returns below this.


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _ensure_data(root: Path) -> None:
    """Stream the mini tarball and extract only LIDAR_TOP keyframes + metadata.

    ~3.9 GB is streamed once; we keep only ``samples/LIDAR_TOP`` and ``v1.0-mini`` (a few
    hundred MB), skipping cameras, radar, sweeps and maps.
    """
    if (root / "v1.0-mini" / "sample.json").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading nuScenes mini (~3.9 GB stream, keeping LIDAR_TOP + metadata) ...")
    print("  (no signup -- anonymous HTTPS)")
    keep = ("samples/LIDAR_TOP/", "v1.0-mini/")
    req = urllib.request.Request(MINI_URL, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if member.isfile() and member.name.startswith(keep):
                    tar.extract(member, path=root, filter="data")
    print(f"  extracted to {root}")


def _load_tables(root: Path) -> dict:
    meta = root / "v1.0-mini"
    load = lambda n: json.loads((meta / f"{n}.json").read_text())
    sd = load("sample_data")
    return {
        "sample": {s["token"]: s for s in load("sample")},
        "ego": {e["token"]: e for e in load("ego_pose")},
        "cs": {c["token"]: c for c in load("calibrated_sensor")},
        "scene": {s["name"]: s for s in load("scene")},
        "lidar": {d["sample_token"]: d for d in sd
                  if "LIDAR_TOP" in d["filename"] and d["is_key_frame"]},
        "ann": load("sample_annotation"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accuracy benchmark on real nuScenes mini data.")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="nuScenes scene name.")
    parser.add_argument("--frames", type=int, default=12, help="Number of keyframes.")
    parser.add_argument("--stride", type=int, default=3, help="Keyframe stride (2 Hz keyframes; 3 = ~1.5 s apart).")
    # Coarser than AV2 on purpose -- match the range image to the sparse 32-beam sensor.
    parser.add_argument("--h-res", type=float, default=2.5)
    parser.add_argument("--v-res", type=float, default=2.5)
    parser.add_argument("--range-margin", type=float, default=core.DEFAULT_RANGE_MARGIN)
    parser.add_argument("--min-see-through", type=int, default=3)
    parser.add_argument("--max-surface-hits", type=int, default=5)
    parser.add_argument("--resolutions", type=float, nargs="+", default=None,
                        help="Multi-resolution consensus (e.g. --resolutions 2.5 4.0): "
                             "higher precision, slightly lower recall. Overrides --h/--v-res.")
    parser.add_argument("--moving-thresh", type=float, default=2.0, help="Track displacement (m) to count as moving GT.")
    parser.add_argument("--voxel-size", type=float, default=core.DEFAULT_TEMPORAL_VOXEL_SIZE)
    parser.add_argument("--temporal-min-hits", type=int, default=2)
    parser.add_argument("--sr-rings", type=int, default=core.DEFAULT_SR_RINGS)
    parser.add_argument("--sr-sectors", type=int, default=core.DEFAULT_SR_SECTORS)
    parser.add_argument("--sr-max-range", type=float, default=core.DEFAULT_SR_MAX_RANGE)
    parser.add_argument("--sr-ratio", type=float, default=core.DEFAULT_SR_RATIO)
    parser.add_argument("--sr-min-map-height", type=float, default=core.DEFAULT_SR_MIN_MAP_HEIGHT)
    parser.add_argument("--sr-ground-margin", type=float, default=core.DEFAULT_SR_GROUND_MARGIN)
    parser.add_argument("--sr-min-votes", type=int, default=2)
    parser.add_argument("--root", default=str(ROOT_DIR), help="Where the mini data lives / is downloaded.")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args(argv)

    root = Path(args.root)
    _ensure_data(root)
    t = _load_tables(root)
    if args.scene not in t["scene"]:
        print(f"Unknown scene {args.scene}. Available: {', '.join(sorted(t['scene']))}", file=sys.stderr)
        return 2

    # Ordered keyframe tokens for the scene, then strided selection.
    toks: list[str] = []
    cur_tok = t["scene"][args.scene]["first_sample_token"]
    while cur_tok:
        toks.append(cur_tok)
        cur_tok = t["sample"][cur_tok]["next"]
    selected = toks[0 : args.frames * args.stride : args.stride]
    print(f"nuScenes {args.scene}: {len(selected)} keyframes (stride {args.stride})...")

    ann_by_sample: dict[str, list] = collections.defaultdict(list)
    for a in t["ann"]:
        ann_by_sample[a["sample_token"]].append(a)

    # Pose-aligned accumulated map (ground removed) + per-frame scans/origins/slices.
    chunks: list[np.ndarray] = []
    scans: list[tuple[np.ndarray, np.ndarray]] = []
    slices: list[tuple[int, int]] = []
    cursor = 0
    for st in selected:
        d = t["lidar"][st]
        ep = t["ego"][d["ego_pose_token"]]
        csd = t["cs"][d["calibrated_sensor_token"]]
        pts = np.fromfile(root / d["filename"], dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
        pts = pts[pts[:, 2] > GROUND_Z_SENSOR]
        r_cs = _quat_to_rot(*csd["rotation"]); t_cs = np.asarray(csd["translation"])
        r_ego = _quat_to_rot(*ep["rotation"]); t_ego = np.asarray(ep["translation"])
        pts_global = (pts @ r_cs.T + t_cs) @ r_ego.T + t_ego
        origin = r_ego @ t_cs + t_ego
        chunks.append(pts_global)
        scans.append((pts_global, origin))
        slices.append((cursor, cursor + len(pts_global)))
        cursor += len(pts_global)
    acc_map = np.concatenate(chunks, axis=0)
    ground_z = float(np.percentile(acc_map[:, 2], 2))

    # Moving tracks: instance center displacement (global frame) across the window.
    centers: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for st in selected:
        for a in ann_by_sample[st]:
            centers[a["instance_token"]].append(np.asarray(a["translation"]))
    moving = {
        i for i, cs in centers.items()
        if len(cs) > 1 and np.max(np.linalg.norm(np.array(cs) - np.array(cs)[0], axis=1)) > args.moving_thresh
    }

    # GT dynamic mask: points inside a moving instance's box, matched per frame.
    gt_chunks = []
    for st, (s, e) in zip(selected, slices):
        gb: list[core.DetectionBox] = []
        for a in ann_by_sample[st]:
            if a["instance_token"] not in moving:
                continue
            w, l, h = a["size"]  # nuScenes order is width, length, height
            gb.append(core.DetectionBox(
                center=np.asarray(a["translation"]),
                size=np.array([l, w, h]),
                yaw=_yaw_from_quat(*a["rotation"]),
            ))
        gt_chunks.append(bench.dynamic_gt_mask(acc_map[s:e], gb, margin=(0.25, 0.25, 0.25))
                         if gb else np.zeros(e - s, dtype=bool))
    gt_mask = np.concatenate(gt_chunks)
    print(f"  map: {len(acc_map):,} pts | moving tracks: {len(moving)} | "
          f"GT dynamic: {int(gt_mask.sum()):,} ({gt_mask.mean()*100:.2f}%)")

    # --- range: multi-scan visibility cleaner (coarse resolution for the sparse sensor) ---
    _, keep_range = core.clean_map_by_visibility(
        acc_map, scans,
        h_res_deg=args.h_res, v_res_deg=args.v_res, range_margin=args.range_margin,
        min_see_through=args.min_see_through, max_surface_hits=args.max_surface_hits,
        ground_z=ground_z, resolutions=args.resolutions,
    )
    range_metrics = bench.compute_accuracy_metrics(~keep_range, gt_mask)

    # --- temporal: per-frame voxel consistency assembled over the map ---
    keep_temporal = np.ones(len(acc_map), dtype=bool)
    tfilter = core.TemporalConsistencyFilter(
        voxel_size=args.voxel_size, window_size=len(selected), min_hits=args.temporal_min_hits)
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
        "\n### Measured on nuScenes (this repo's detector-free methods)\n\n"
        f"Scene `{args.scene}`, {len(selected)} pose-aligned keyframes, {len(acc_map):,} points "
        f"({int(gt_mask.sum()):,} ground-truth points on moving objects). Range image at "
        f"{args.h_res}deg, coarsened to match the 32-beam sensor.\n\n"
        "| method | precision | recall | F1 | static kept |\n"
        "|---|---|---|---|---|\n"
        f"{row('range-image visibility', range_metrics)}\n"
        f"{row('scan-ratio (pseudo-occupancy)', scanratio_metrics)}\n"
        f"{row('temporal consistency', temporal_metrics)}\n"
    )
    print(table)

    payload = {
        "dataset": "nuscenes-mini", "scene": args.scene, "frames": len(selected), "stride": args.stride,
        "map_points": int(len(acc_map)), "gt_dynamic_points": int(gt_mask.sum()),
        "moving_tracks": len(moving),
        "config": {"h_res": args.h_res, "v_res": args.v_res, "range_margin": args.range_margin,
                   "min_see_through": args.min_see_through, "max_surface_hits": args.max_surface_hits,
                   "ground_z_sensor": GROUND_Z_SENSOR, "moving_thresh": args.moving_thresh},
        "range": range_metrics, "temporal": temporal_metrics, "scan_ratio": scanratio_metrics,
    }
    out_json = args.summary_json or str(root / f"benchmark_{args.scene}.json")
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")
    print("\nData source: nuScenes mini (CC BY-NC-SA 4.0) https://www.nuscenes.org")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
