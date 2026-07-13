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
import os
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


def _export_online_manifest(
    output_path: Path,
    *,
    scene: str,
    stride: int,
    timestamps: list[int],
    local_scans: list[np.ndarray],
    gt_masks: list[np.ndarray],
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Export the exact AV2 selection/GT as an online replay manifest."""
    output_path = output_path.resolve()
    assets = output_path.parent / f"{output_path.stem}_assets"
    assets.mkdir(parents=True, exist_ok=True)
    frames = []
    for ts, points, gt_mask in zip(timestamps, local_scans, gt_masks):
        cloud_path = assets / f"{ts}_cloud.npy"
        labels_path = assets / f"{ts}_labels.npy"
        np.save(cloud_path, np.asarray(points, dtype=np.float64))
        np.save(labels_path, np.asarray(gt_mask, dtype=np.uint8))
        rotation, translation = poses[ts]
        frames.append(
            {
                "cloud": os.path.relpath(cloud_path, output_path.parent),
                "point_labels": os.path.relpath(labels_path, output_path.parent),
                "timestamp_ns": int(ts),
                "timestamp_sec": ts * 1e-9,
                "pose": {
                    "rotation": np.asarray(rotation, dtype=np.float64).tolist(),
                    "translation": np.asarray(translation, dtype=np.float64).tolist(),
                },
            }
        )
    payload = {
        "dataset": "Argoverse 2 Sensor Dataset",
        "scene": scene,
        "sensor_profile": {
            "name": "dual VLP-32C",
            "beams": 64,
            "rate_hz": 10.0 / max(1, stride),
            "deskewed": True,
            "source": "https://argoverse.github.io/user-guide/datasets/sensor.html",
        },
        "frames": frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_indices(mask: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Return deterministic indices for one proof layer without changing its population."""
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if max_points <= 0 or len(indices) <= max_points:
        return indices
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def _render_gt_proof(
    output_path: Path,
    *,
    acc_map: np.ndarray,
    gt_dynamic: np.ndarray,
    keep_mask: np.ndarray,
    metrics: dict[str, float],
    scene: str,
    frames: int,
    moving_tracks: int,
    moving_thresh: float,
    max_points_per_layer: int,
    seed: int,
) -> None:
    """Render a same-pose raw/cleaned proof with explicit moving-track GT and errors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("--proof-png requires matplotlib") from exc

    xyz = np.asarray(acc_map, dtype=np.float64)
    gt = np.asarray(gt_dynamic, dtype=bool)
    keep = np.asarray(keep_mask, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) != len(gt) or len(gt) != len(keep):
        raise ValueError("proof arrays must have matching point counts")

    predicted_dynamic = ~keep
    true_positive = predicted_dynamic & gt
    false_positive = predicted_dynamic & ~gt
    false_negative = keep & gt
    kept_static = keep & ~gt
    static = ~gt
    rng = np.random.default_rng(seed)

    layers = {
        "raw_static": _sample_indices(static, max_points_per_layer, rng),
        "raw_gt": _sample_indices(gt, max_points_per_layer, rng),
        "kept_static": _sample_indices(kept_static, max_points_per_layer, rng),
        "false_negative": _sample_indices(false_negative, max_points_per_layer, rng),
        "true_positive": _sample_indices(true_positive, max_points_per_layer, rng),
        "false_positive": _sample_indices(false_positive, max_points_per_layer, rng),
    }

    # One robust crop and identical axes prevent camera choice from exaggerating a branch.
    xlo, xhi = np.percentile(xyz[:, 0], [0.5, 99.5])
    ylo, yhi = np.percentile(xyz[:, 1], [0.5, 99.5])
    # Put the trajectory's long map dimension on screen horizontally. This is a
    # presentation rotation only; every panel still uses the identical city-frame crop.
    if (yhi - ylo) > (xhi - xlo):
        horizontal_dim, vertical_dim = 1, 0
        hlo, hhi, vlo, vhi = ylo, yhi, xlo, xhi
        horizontal_label, vertical_label = "city y [m]", "city x [m]"
    else:
        horizontal_dim, vertical_dim = 0, 1
        hlo, hhi, vlo, vhi = xlo, xhi, ylo, yhi
        horizontal_label, vertical_label = "city x [m]", "city y [m]"
    xpad = max(1.0, 0.03 * (xhi - xlo))
    ypad = max(1.0, 0.03 * (yhi - ylo))
    hpad = ypad if horizontal_dim == 1 else xpad
    vpad = xpad if vertical_dim == 0 else ypad

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.set_xlim(hlo - hpad, hhi + hpad)
        ax.set_ylim(vlo - vpad, vhi + vpad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.35)
        ax.set_xlabel(horizontal_label)
        ax.set_ylabel(vertical_label)

    def scatter(ax: object, key: str, color: str, size: float, label: str, alpha: float = 0.8) -> None:
        idx = layers[key]
        if len(idx):
            ax.scatter(xyz[idx, horizontal_dim], xyz[idx, vertical_dim], s=size, c=color, alpha=alpha,
                       linewidths=0, rasterized=True, label=label)

    scatter(axes[0], "raw_static", "#64748b", 0.7, "static GT", 0.42)
    scatter(axes[0], "raw_gt", "#dc2626", 2.2, "moving-track GT", 0.9)
    axes[0].set_title(
        f"Raw pose-aligned accumulation\n{int(gt.sum()):,} moving-GT points remain",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(loc="upper right", markerscale=4, framealpha=0.92)

    scatter(axes[1], "kept_static", "#2563eb", 0.7, "kept static", 0.48)
    scatter(axes[1], "false_negative", "#f59e0b", 2.2, "remaining moving GT", 0.92)
    axes[1].set_title(
        "Detector-free fusion cleaned\n"
        f"{100 * metrics['recall']:.1f}% moving GT removed · "
        f"{100 * metrics['static_preservation']:.1f}% static kept",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].legend(loc="upper right", markerscale=4, framealpha=0.92)

    scatter(axes[2], "true_positive", "#dc2626", 2.0, "correctly removed (TP)", 0.88)
    scatter(axes[2], "false_positive", "#7c3aed", 1.8, "static removed (FP)", 0.82)
    axes[2].set_title(
        "Removal audit against GT\n"
        f"precision {100 * metrics['precision']:.1f}% · F1 {100 * metrics['f1']:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    axes[2].legend(loc="upper right", markerscale=4, framealpha=0.92)

    fig.suptitle(
        "Argoverse 2 strict map-cleaning proof — identical poses and frame set",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        f"scene {scene[:8]}… · {frames} sweeps · {moving_tracks} moving tracks · identical pose-aligned input\n"
        f"GT = points inside tracks displaced > {moving_thresh:g} m · "
        "method = numpy free-space fusion · no boxes/detector at inference",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.78, wspace=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


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
    parser.add_argument("--sr-min-votes", type=int, default=None,
                        help="Fixed absolute vote threshold (default: normalized, 35%% of each point's column revisits).")
    parser.add_argument("--fusion-workers", type=int, default=6,
                        help="Process pool size for the fusion carving channels.")
    # Short-window fusion defaults: the library defaults (0.9 / 2 / 11) assume a long
    # KITTI-style sequence. With only 12 sweeps, a single same-scan hit must not veto
    # the fractional vote (0.9 would need 9 frees after 1 hit in 10 observations) and
    # 11 absolute voids can never accumulate.
    parser.add_argument("--fusion-free-fraction", type=float, default=0.7)
    parser.add_argument("--fusion-free-floor", type=int, default=3)
    parser.add_argument("--fusion-void-min-scans", type=int, default=4)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument(
        "--proof-png",
        type=Path,
        default=None,
        help="Write a same-pose raw/cleaned/GT audit image for the fusion result.",
    )
    parser.add_argument("--proof-max-points", type=int, default=180000,
                        help="Maximum plotted points per proof layer (metrics always use all points).")
    parser.add_argument("--proof-seed", type=int, default=7)
    parser.add_argument("--sensor-aware-ablation", action="store_true",
                        help="Report private O1 distance/beam-normalized visibility evidence.")
    parser.add_argument("--sensor-h-spacing", type=float, default=0.2)
    parser.add_argument("--sensor-v-spacing", type=float, default=0.625)
    parser.add_argument("--sensor-support-size", type=float, default=0.5)
    parser.add_argument(
        "--online-manifest",
        type=Path,
        default=None,
        help="Export this exact pose/GT selection for scripts/run_online_benchmark.py.",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Stop after exporting --online-manifest instead of running offline map cleaners.",
    )
    args = parser.parse_args(argv)
    if args.online_only and args.online_manifest is None:
        parser.error("--online-only requires --online-manifest")

    import pyarrow.feather as feather

    print(f"Argoverse 2 scene {args.scene}: {args.frames} sweeps (stride {args.stride})...")
    selected = _download(args.scene, args.frames, args.stride)
    poses = _load_poses(args.scene)
    ann_file = _scene_dir(args.scene) / "annotations.feather"
    ann = feather.read_table(ann_file)
    arr = {c: (ann[c].to_pylist() if c in ("track_uuid", "category") else ann[c].to_numpy()) for c in ann.column_names}

    # Pose-aligned accumulated map (ground removed) + per-frame scans/origins/slices.
    chunks: list[np.ndarray] = []
    local_chunks: list[np.ndarray] = []
    scans: list[tuple[np.ndarray, np.ndarray]] = []
    slices: list[tuple[int, int]] = []
    cursor = 0
    for ts in selected:
        R, tvec = poses[ts]
        pts_ego = core.load_points(_scene_dir(args.scene) / "lidar" / f"{ts}.feather", fmt="feather")
        pts_ego = pts_ego[pts_ego[:, 2] > GROUND_Z]
        pts_city = pts_ego @ R.T + tvec
        local_chunks.append(pts_ego)
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

    if args.online_manifest is not None:
        _export_online_manifest(
            args.online_manifest,
            scene=args.scene,
            stride=args.stride,
            timestamps=selected,
            local_scans=local_chunks,
            gt_masks=gt_chunks,
            poses=poses,
        )
        print(f"  online manifest: {args.online_manifest}")
        if args.online_only:
            return 0

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

    # --- fusion: free-space carving + eroded voids + scan-ratio votes (OR) ---
    _, keep_fusion = core.clean_map_by_fusion(
        acc_map, scans,
        free_votes_fraction=args.fusion_free_fraction,
        free_votes_floor=args.fusion_free_floor,
        void_min_scans=args.fusion_void_min_scans,
        workers=args.fusion_workers,
    )
    fusion_metrics = bench.compute_accuracy_metrics(~keep_fusion, gt_mask)

    if args.proof_png is not None:
        _render_gt_proof(
            args.proof_png,
            acc_map=acc_map,
            gt_dynamic=gt_mask,
            keep_mask=keep_fusion,
            metrics=fusion_metrics,
            scene=args.scene,
            frames=len(selected),
            moving_tracks=len(moving),
            moving_thresh=args.moving_thresh,
            max_points_per_layer=args.proof_max_points,
            seed=args.proof_seed,
        )
        print(f"  GT proof image: {args.proof_png}")

    sensor_aware = None
    if args.sensor_aware_ablation:
        evidence = core._sensor_aware_visibility_evidence(
            acc_map,
            scans,
            h_res_deg=args.h_res,
            v_res_deg=args.v_res,
            range_margin=args.range_margin,
            sensor_h_spacing_deg=args.sensor_h_spacing,
            sensor_v_spacing_deg=args.sensor_v_spacing,
            support_size_m=args.sensor_support_size,
        )
        normalized_range = core._sensor_aware_visibility_dynamic_mask(
            evidence,
            min_raw_see_through=args.min_see_through,
            max_raw_surface_hits=args.max_surface_hits,
            ground_mask=acc_map[:, 2] <= GROUND_Z,
        )
        normalized_metrics = bench.compute_accuracy_metrics(normalized_range, gt_mask)
        sensor_aware = {
            "status": "experimental_not_promoted",
            "profile": {
                "beams": 64,
                "h_spacing_deg": args.sensor_h_spacing,
                "v_spacing_deg": args.sensor_v_spacing,
                "support_size_m": args.sensor_support_size,
            },
            "baseline_strategy": core._sensor_strategy(64, args.sensor_v_spacing),
            "selected_metrics": fusion_metrics,
            "normalized_visibility_candidate": normalized_metrics,
            "evidence": core._sensor_aware_evidence_summary(evidence),
        }

    def row(name: str, m: dict) -> str:
        return (f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                f"{m['static_preservation']:.3f} |")

    table = (
        "\n### Measured on Argoverse 2 (this repo's detector-free methods)\n\n"
        f"Scene `{args.scene}`, {len(selected)} pose-aligned sweeps, {len(acc_map):,} points "
        f"({int(gt_mask.sum()):,} ground-truth points on moving objects).\n\n"
        "| method | precision | recall | F1 | static kept |\n"
        "|---|---|---|---|---|\n"
        f"{row('free-space fusion', fusion_metrics)}\n"
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
        "fusion": fusion_metrics,
        "sensor_aware": sensor_aware,
        "proof": ({
            "task": "offline_map_cleaning",
            "method": "fusion",
            "same_pose_and_frame_set": True,
            "ground_truth": f"points inside tracks displaced > {args.moving_thresh:g} m",
            "inference_uses_boxes": False,
            "image": str(args.proof_png),
            "plot_sampling_only": True,
            "max_points_per_layer": args.proof_max_points,
            "seed": args.proof_seed,
        } if args.proof_png is not None else None),
    }
    out_json = args.summary_json or str(_scene_dir(args.scene) / "benchmark_result.json")
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")
    print("\nData source: Argoverse 2 Sensor Dataset (CC BY-NC-SA 4.0) https://www.argoverse.org/av2.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
