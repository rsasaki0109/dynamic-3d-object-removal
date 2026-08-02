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
import time
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
MIN_GT_DYNAMIC_POINTS_FOR_MEAN = 5_000

_METRIC_KEYS = ("precision", "recall", "f1", "static_preservation")
_METHOD_LABELS = {
    "range": "range-image visibility",
    "scan_ratio": "scan-ratio (pseudo-occupancy)",
    "range_and_scan_ratio": "range ∧ scan-ratio (intersection)",
    "temporal": "temporal consistency",
    "fusion": "free-space fusion",
}


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


def _resolve_scenes(scene: str | None, scenes: list[str] | None, available: dict[str, dict]) -> list[str]:
    """Resolve the backward-compatible single-scene and multi-scene forms."""
    requested = [scene or DEFAULT_SCENE] if scenes is None else scenes
    if len(requested) == 1 and requested[0].lower() == "all":
        return sorted(available)
    if any(item.lower() == "all" for item in requested):
        raise ValueError("--scenes all cannot be combined with individual scene names")
    unique = list(dict.fromkeys(requested))
    unknown = [item for item in unique if item not in available]
    if unknown:
        raise ValueError(f"Unknown scene(s) {', '.join(unknown)}. Available: {', '.join(sorted(available))}")
    return unique


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _aggregate_scene_results(
    payloads: list[dict],
    method_keys: list[str],
    *,
    min_gt_dynamic_points: int = MIN_GT_DYNAMIC_POINTS_FOR_MEAN,
) -> dict:
    """Compute an unweighted mean of eligible per-scene metric dictionaries."""
    included = [p for p in payloads if p["gt_dynamic_points"] >= min_gt_dynamic_points]
    excluded = [p for p in payloads if p["gt_dynamic_points"] < min_gt_dynamic_points]
    means: dict[str, dict[str, float | None]] = {}
    for key in method_keys:
        means[key] = {
            metric: (sum(float(p["metrics"][key][metric]) for p in included) / len(included)
                     if included else None)
            for metric in _METRIC_KEYS
        }
    return {
        "min_gt_dynamic_points": min_gt_dynamic_points,
        "included_scenes": [p["scene"] for p in included],
        "excluded_scenes": [p["scene"] for p in excluded],
        "methods": means,
    }


def _print_results(payloads: list[dict], aggregate: dict) -> None:
    print("\n### nuScenes per-scene summary\n")
    print("| scene | map points | GT dynamic points | moving tracks | runtime (s) |")
    print("|---|---:|---:|---:|---:|")
    for payload in payloads:
        print(f"| {payload['scene']} | {payload['map_points']:,} | {payload['gt_dynamic_points']:,} | "
              f"{payload['moving_tracks']} | {payload['runtime_seconds']:.1f} |")

    print("\n### nuScenes per-scene metrics\n")
    print("| scene | method | precision | recall | F1 | static kept |")
    print("|---|---|---:|---:|---:|---:|")
    for payload in payloads:
        for key in payload["method_keys"]:
            m = payload["metrics"][key]
            print(f"| {payload['scene']} | {_METHOD_LABELS[key]} | {m['precision']:.3f} | "
                  f"{m['recall']:.3f} | {m['f1']:.3f} | {m['static_preservation']:.3f} |")

    threshold = aggregate["min_gt_dynamic_points"]
    print("\n### nuScenes aggregate mean across scenes\n")
    print(f"Mean is unweighted across scenes with at least {threshold:,} GT dynamic points; "
          "lower-content scenes remain listed above but are excluded from this mean.")
    if aggregate["excluded_scenes"]:
        print(f"Excluded: {', '.join(aggregate['excluded_scenes'])}.")
    print("\n| method | scenes averaged | precision | recall | F1 | static kept |")
    print("|---|---:|---:|---:|---:|---:|")
    for key in payloads[0]["method_keys"]:
        m = aggregate["methods"][key]
        print(f"| {_METHOD_LABELS[key]} | {len(aggregate['included_scenes'])} | "
              f"{_format_metric(m['precision'])} | {_format_metric(m['recall'])} | "
              f"{_format_metric(m['f1'])} | {_format_metric(m['static_preservation'])} |")


def _run_scene(args: argparse.Namespace, root: Path, tables: dict, scene: str) -> dict:
    started = time.perf_counter()
    toks: list[str] = []
    cur_tok = tables["scene"][scene]["first_sample_token"]
    while cur_tok:
        toks.append(cur_tok)
        cur_tok = tables["sample"][cur_tok]["next"]
    selected = toks[0 : args.frames * args.stride : args.stride]
    print(f"nuScenes {scene}: {len(selected)} keyframes (stride {args.stride})...")

    ann_by_sample: dict[str, list] = collections.defaultdict(list)
    for a in tables["ann"]:
        ann_by_sample[a["sample_token"]].append(a)

    # Pose-aligned accumulated map (ground removed) + per-frame scans/origins/slices.
    chunks: list[np.ndarray] = []
    scans: list[tuple[np.ndarray, np.ndarray]] = []
    slices: list[tuple[int, int]] = []
    cursor = 0
    for st in selected:
        d = tables["lidar"][st]
        ep = tables["ego"][d["ego_pose_token"]]
        csd = tables["cs"][d["calibrated_sensor_token"]]
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

    # --- scan-ratio: ERASOR-style per-column pseudo-occupancy (a different signal) ---
    _, keep_sr = core.clean_map_by_scan_ratio(
        acc_map, scans,
        n_rings=args.sr_rings, n_sectors=args.sr_sectors, max_range=args.sr_max_range,
        scan_ratio_threshold=args.sr_ratio, min_map_height=args.sr_min_map_height,
        ground_margin=args.sr_ground_margin, min_votes=args.sr_min_votes,
    )
    scanratio_metrics = bench.compute_accuracy_metrics(~keep_sr, gt_mask)

    # --- range ∧ scan-ratio: intersect the two high-recall channels. Their false
    # positives come from different physics (range-image self-occlusion vs polar-column
    # vacancy), so the intersection trades a little recall for better precision AND
    # better static preservation — the best single number on this sparse sensor.
    combo_metrics = bench.compute_accuracy_metrics(~keep_range & ~keep_sr, gt_mask)

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

    metrics = {
        "range": range_metrics,
        "scan_ratio": scanratio_metrics,
        "range_and_scan_ratio": combo_metrics,
        "temporal": temporal_metrics,
    }
    method_keys = list(metrics)

    # Fusion is structurally unsuitable for sparse 32-beam nuScenes and is opt-in.
    # When requested, use the documented short-window thresholds for comparability.
    if args.include_fusion:
        _, keep_fusion = core.clean_map_by_fusion(
            acc_map, scans,
            free_votes_fraction=args.fusion_free_fraction,
            free_votes_floor=args.fusion_free_floor,
            void_min_scans=args.fusion_void_min_scans,
            workers=args.fusion_workers,
        )
        metrics["fusion"] = bench.compute_accuracy_metrics(~keep_fusion, gt_mask)
        method_keys.append("fusion")

    runtime_seconds = time.perf_counter() - started
    payload = {
        "dataset": "nuscenes-mini", "scene": scene, "frames": len(selected), "stride": args.stride,
        "map_points": int(len(acc_map)), "gt_dynamic_points": int(gt_mask.sum()),
        "moving_tracks": len(moving), "runtime_seconds": runtime_seconds,
        "method_keys": method_keys, "metrics": metrics,
        "config": {
            "h_res": args.h_res, "v_res": args.v_res, "range_margin": args.range_margin,
            "min_see_through": args.min_see_through, "max_surface_hits": args.max_surface_hits,
            "ground_z_sensor": GROUND_Z_SENSOR, "moving_thresh": args.moving_thresh,
            "voxel_size": args.voxel_size, "temporal_min_hits": args.temporal_min_hits,
            "sr_rings": args.sr_rings, "sr_sectors": args.sr_sectors, "sr_max_range": args.sr_max_range,
            "sr_ratio": args.sr_ratio, "sr_min_map_height": args.sr_min_map_height,
            "sr_ground_margin": args.sr_ground_margin, "sr_min_votes": args.sr_min_votes,
            "include_fusion": args.include_fusion,
            "fusion_free_fraction": args.fusion_free_fraction, "fusion_free_floor": args.fusion_free_floor,
            "fusion_void_min_scans": args.fusion_void_min_scans,
        },
    }
    # Retain the single-scene JSON keys used by the original script.
    payload.update(metrics)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accuracy benchmark on real nuScenes mini data.")
    scene_group = parser.add_mutually_exclusive_group()
    scene_group.add_argument("--scene", default=None, help=f"nuScenes scene name (default: {DEFAULT_SCENE}).")
    scene_group.add_argument("--scenes", nargs="+", metavar="SCENE",
                             help="Space-separated scene names, or `all` for all 10 mini scenes.")
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
    parser.add_argument("--sr-min-votes", type=int, default=None,
                        help="Fixed absolute vote threshold (default: normalized, 35%% of each point's column revisits).")
    parser.add_argument("--include-fusion", "--fusion", action="store_true", dest="include_fusion",
                        help="Also run fusion (opt-in: structurally unsuitable/slow on sparse 32-beam data).")
    parser.add_argument("--fusion-workers", type=int, default=6,
                        help="Process pool size for the optional fusion carving channels.")
    # Same short-window adaptation as the AV2 script (12 scans). Note fusion's voxel
    # carving is still a poor fit for this sparse 32-beam sensor: beyond ~13 m the
    # vertical beam spacing exceeds the carving voxel, so a scan's own surface hits no
    # longer protect static structure and it gets carved between beams. Coarser voxels
    # do not recover it (measured F1 stays < 0.3); prefer `range` here.
    parser.add_argument("--fusion-free-fraction", type=float, default=0.7)
    parser.add_argument("--fusion-free-floor", type=int, default=3)
    parser.add_argument("--fusion-void-min-scans", type=int, default=4)
    parser.add_argument("--root", default=str(ROOT_DIR), help="Where the mini data lives / is downloaded.")
    parser.add_argument("--summary-json", default=None,
                        help="Output JSON path (per-scene in single-scene mode; aggregate in multi-scene mode).")
    args = parser.parse_args(argv)

    root = Path(args.root)
    _ensure_data(root)
    tables = _load_tables(root)
    try:
        scene_names = _resolve_scenes(args.scene, args.scenes, tables["scene"])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payloads: list[dict] = []
    for scene in scene_names:
        payload = _run_scene(args, root, tables, scene)
        payloads.append(payload)
        per_scene_path = (Path(args.summary_json) if args.summary_json and len(scene_names) == 1
                          else root / f"benchmark_{scene}.json")
        per_scene_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {per_scene_path}")

    aggregate = _aggregate_scene_results(payloads, payloads[0]["method_keys"])
    _print_results(payloads, aggregate)
    aggregate_payload = {
        "dataset": "nuscenes-mini", "scenes": scene_names,
        "frames": args.frames, "stride": args.stride,
        "config": payloads[0]["config"], "scene_results": payloads,
        "aggregate": aggregate,
    }
    if len(scene_names) == 1 and args.summary_json:
        # The path was already used for the backward-compatible per-scene payload.
        aggregate_path = root / "benchmark_multiscene.json"
    else:
        aggregate_path = Path(args.summary_json) if args.summary_json else root / "benchmark_multiscene.json"
    if len(scene_names) > 1 or args.scenes is not None:
        aggregate_path.write_text(json.dumps(aggregate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved aggregate: {aggregate_path}")
    print("\nData source: nuScenes mini (CC BY-NC-SA 4.0) https://www.nuscenes.org")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
