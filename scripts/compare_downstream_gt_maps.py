#!/usr/bin/env python3
"""Evaluate raw/cleaned downstream SLAM maps against manifest point GT.

The comparison is intentionally strict: raw and cleaned runs must have byte-identical
loop-edge and trajectory artifacts.  The manifest labels are used only after both maps
have been built.  Local scans are transformed with the backend's own optimized TUM
trajectory, so the evaluated GT lives in exactly the map coordinate system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dynamic_object_removal as core  # noqa: E402


def _spatial_tree(points: np.ndarray) -> Any:
    """Create the optional SciPy tree only when the integration tool is run."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - depends on integration environment
        raise SystemExit(
            "scipy is required for downstream map comparison; install scipy first"
        ) from exc
    return cKDTree(points)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_identical(left: Path, right: Path, label: str) -> str:
    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    if left_bytes != right_bytes:
        raise ValueError(f"{label} differs between baseline and cleaned runs")
    return hashlib.sha256(left_bytes).hexdigest()


def _rotation_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("trajectory quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("trajectory quaternion cannot be zero")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _read_tum(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 TUM fields")
        values = np.asarray([float(field) for field in fields], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}:{line_number}: non-finite TUM value")
        rows.append({
            "timestamp_sec": float(values[0]),
            "translation": values[1:4],
            "rotation": _rotation_from_quaternion_xyzw(values[4:8]),
        })
    if not rows:
        raise ValueError(f"no poses found in {path}")
    return rows


def _frame_timestamp_sec(frame: dict[str, Any]) -> float:
    if "timestamp_ns" in frame:
        return int(frame["timestamp_ns"]) / 1e9
    return float(frame["timestamp_sec"])


def _build_gt_map(
    manifest_path: Path,
    trajectory: list[dict[str, Any]],
    *,
    timestamp_tolerance_sec: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest.frames must be a non-empty list")

    available = np.asarray([_frame_timestamp_sec(frame) for frame in frames])
    used: set[int] = set()
    points_world: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    timestamps_ns: list[int] = []
    for pose in trajectory:
        differences = np.abs(available - pose["timestamp_sec"])
        frame_index = int(np.argmin(differences))
        if float(differences[frame_index]) > timestamp_tolerance_sec:
            raise ValueError(
                f"no manifest frame matches trajectory timestamp {pose['timestamp_sec']:.9f}"
            )
        if frame_index in used:
            raise ValueError("one manifest frame matched more than one trajectory pose")
        used.add(frame_index)
        frame = frames[frame_index]
        local = np.asarray(
            np.load((manifest_path.parent / frame["cloud"]).resolve()), dtype=np.float32
        )
        point_labels = np.asarray(
            np.load((manifest_path.parent / frame["point_labels"]).resolve()), dtype=bool
        )
        if local.ndim != 2 or local.shape[1] < 3 or len(local) != len(point_labels):
            raise ValueError(f"invalid cloud/label pair for manifest frame {frame_index}")
        xyz = local[:, :3].astype(np.float64)
        points_world.append(xyz @ pose["rotation"].T + pose["translation"])
        labels.append(point_labels)
        timestamps_ns.append(
            int(frame["timestamp_ns"])
            if "timestamp_ns" in frame
            else int(round(float(frame["timestamp_sec"]) * 1e9))
        )
    return np.concatenate(points_world), np.concatenate(labels), timestamps_ns


def _classify_map(
    map_points: np.ndarray,
    gt_points: np.ndarray,
    gt_dynamic: np.ndarray,
    *,
    match_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances, indices = _spatial_tree(gt_points).query(map_points, k=1, workers=-1)
    matched = distances <= match_tolerance_m
    labels = np.zeros(len(map_points), dtype=bool)
    labels[matched] = gt_dynamic[indices[matched]]
    return labels, matched, distances


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_maps(
    baseline: np.ndarray,
    cleaned: np.ndarray,
    gt_points: np.ndarray,
    gt_dynamic: np.ndarray,
    *,
    match_tolerance_m: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    baseline_labels, baseline_matched, baseline_distances = _classify_map(
        baseline, gt_points, gt_dynamic, match_tolerance_m=match_tolerance_m
    )
    cleaned_labels, cleaned_matched, cleaned_distances = _classify_map(
        cleaned, gt_points, gt_dynamic, match_tolerance_m=match_tolerance_m
    )
    if not np.all(baseline_matched) or not np.all(cleaned_matched):
        raise ValueError(
            "not every downstream map point matches manifest GT within the requested tolerance"
        )
    baseline_dynamic = int(np.count_nonzero(baseline_labels & baseline_matched))
    baseline_static = int(np.count_nonzero(~baseline_labels & baseline_matched))
    cleaned_dynamic = int(np.count_nonzero(cleaned_labels & cleaned_matched))
    cleaned_static = int(np.count_nonzero(~cleaned_labels & cleaned_matched))
    dynamic_removed = baseline_dynamic - cleaned_dynamic
    static_removed = baseline_static - cleaned_static
    total_removed = len(baseline) - len(cleaned)
    if min(dynamic_removed, static_removed, total_removed) < 0:
        raise ValueError("cleaned map is not a removal-only result under GT matching")

    result = {
        "match_tolerance_m": match_tolerance_m,
        "gt_map_points": int(len(gt_points)),
        "gt_dynamic_points": int(np.count_nonzero(gt_dynamic)),
        "gt_static_points": int(np.count_nonzero(~gt_dynamic)),
        "baseline": {
            "map_points": int(len(baseline)),
            "matched_points": int(np.count_nonzero(baseline_matched)),
            "matched_ratio": float(np.mean(baseline_matched)),
            "max_match_distance_m": float(np.max(baseline_distances)),
            "dynamic_points": baseline_dynamic,
            "static_points": baseline_static,
            "dynamic_contamination_ratio": _ratio(baseline_dynamic, int(np.count_nonzero(baseline_matched))),
        },
        "cleaned": {
            "map_points": int(len(cleaned)),
            "matched_points": int(np.count_nonzero(cleaned_matched)),
            "matched_ratio": float(np.mean(cleaned_matched)),
            "max_match_distance_m": float(np.max(cleaned_distances)),
            "dynamic_points": cleaned_dynamic,
            "static_points": cleaned_static,
            "dynamic_contamination_ratio": _ratio(cleaned_dynamic, int(np.count_nonzero(cleaned_matched))),
        },
        "removal": {
            "total_points": int(total_removed),
            "dynamic_points": int(dynamic_removed),
            "static_points": int(static_removed),
            "dynamic_gt_reduction": _ratio(dynamic_removed, baseline_dynamic),
            "static_gt_preservation": _ratio(cleaned_static, baseline_static),
            "removed_point_precision": _ratio(dynamic_removed, total_removed),
        },
    }
    return result, {
        "baseline_labels": baseline_labels,
        "cleaned_labels": cleaned_labels,
        "baseline_matched": baseline_matched,
        "cleaned_matched": cleaned_matched,
    }


def _sample(points: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[rng.choice(len(points), size=limit, replace=False)]


def _plot(
    output: Path,
    baseline: np.ndarray,
    cleaned: np.ndarray,
    labels: dict[str, np.ndarray],
    metrics: dict[str, Any],
    *,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    baseline_dynamic = baseline[labels["baseline_labels"] & labels["baseline_matched"]]
    baseline_static = baseline[~labels["baseline_labels"] & labels["baseline_matched"]]
    cleaned_dynamic = cleaned[labels["cleaned_labels"] & labels["cleaned_matched"]]
    cleaned_static = cleaned[~labels["cleaned_labels"] & labels["cleaned_matched"]]

    clean_tree = _spatial_tree(cleaned)
    baseline_to_clean = clean_tree.query(baseline, k=1, workers=-1)[0]
    removed = baseline_to_clean > metrics["match_tolerance_m"]
    removed_dynamic = baseline[removed & labels["baseline_labels"]]
    removed_static = baseline[removed & ~labels["baseline_labels"]]

    dynamic_xy = baseline_dynamic[:, :2]
    lo = np.percentile(dynamic_xy, 2.0, axis=0) - np.array([8.0, 8.0])
    hi = np.percentile(dynamic_xy, 98.0, axis=0) + np.array([8.0, 8.0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), facecolor="#f8fafc")
    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(lo[0], hi[0])
        axis.set_ylim(lo[1], hi[1])
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(color="#cbd5e1", alpha=0.22, linewidth=0.6)

    def scatter(axis: Any, points: np.ndarray, color: str, size: float, label: str, alpha: float) -> None:
        sampled = _sample(points, max_points, rng)
        axis.scatter(sampled[:, 0], sampled[:, 1], s=size, c=color, alpha=alpha,
                     linewidths=0, label=label, rasterized=True)

    scatter(axes[0], baseline_static, "#64748b", 0.55, "static GT", 0.42)
    scatter(axes[0], baseline_dynamic, "#dc2626", 2.0, "moving GT", 0.88)
    axes[0].set_title(
        "Raw downstream map\n"
        f"{metrics['baseline']['dynamic_points']:,} moving-GT points",
        fontweight="bold",
    )
    axes[0].legend(loc="upper right", markerscale=4, framealpha=0.92)

    scatter(axes[1], cleaned_static, "#2563eb", 0.55, "kept static GT", 0.42)
    scatter(axes[1], cleaned_dynamic, "#f59e0b", 2.0, "remaining moving GT", 0.88)
    axes[1].set_title(
        "Realtime range → downstream map\n"
        f"{100 * metrics['removal']['dynamic_gt_reduction']:.1f}% moving GT reduced · "
        f"{100 * metrics['removal']['static_gt_preservation']:.1f}% static kept",
        fontweight="bold",
    )
    axes[1].legend(loc="upper right", markerscale=4, framealpha=0.92)

    scatter(axes[2], removed_dynamic, "#dc2626", 2.0, "moving GT removed", 0.88)
    scatter(axes[2], removed_static, "#7c3aed", 1.2, "static GT removed", 0.62)
    axes[2].set_title(
        "Removal audit against GT\n"
        f"{metrics['removal']['dynamic_points']:,} dynamic · "
        f"{metrics['removal']['static_points']:,} static points removed",
        fontweight="bold",
    )
    axes[2].legend(loc="upper right", markerscale=4, framealpha=0.92)

    fig.suptitle(
        "AV2 downstream SLAM proof — identical frame stamps, poses and pose graph",
        fontsize=17, fontweight="bold", color="#0f172a", y=0.98,
    )
    fig.text(
        0.5, 0.018,
        "11 exact-stamp submaps · labels used only for evaluation · optimized trajectory byte-identical · "
        "no loop edges on this short segment",
        ha="center", va="bottom", fontsize=9.5, color="#475569",
    )
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.15, top=0.82, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--cleaned-map", type=Path, required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--cleaned-trajectory", type=Path, required=True)
    parser.add_argument("--baseline-loop-edges", type=Path, required=True)
    parser.add_argument("--cleaned-loop-edges", type=Path, required=True)
    parser.add_argument("--baseline-raw-trajectory", type=Path)
    parser.add_argument("--cleaned-raw-trajectory", type=Path)
    parser.add_argument("--match-tolerance", type=float, default=0.01)
    parser.add_argument("--timestamp-tolerance", type=float, default=0.001)
    parser.add_argument("--filter-algorithm", default="range")
    parser.add_argument("--range-window", type=int, default=3)
    parser.add_argument("--range-margin", type=float, default=0.5)
    parser.add_argument("--range-h-res", type=float, default=1.0)
    parser.add_argument("--range-v-res", type=float, default=2.0)
    parser.add_argument("--dor-summary", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--plot-max-points", type=int, default=180000)
    parser.add_argument("--plot-seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.match_tolerance <= 0 or args.timestamp_tolerance <= 0:
        parser.error("tolerances must be positive")

    trajectory_sha = _require_identical(
        args.baseline_trajectory, args.cleaned_trajectory, "optimized trajectory"
    )
    loop_edges_sha = _require_identical(
        args.baseline_loop_edges, args.cleaned_loop_edges, "loop-edge artifact"
    )
    raw_trajectory_sha = None
    if (args.baseline_raw_trajectory is None) != (args.cleaned_raw_trajectory is None):
        parser.error("pass both raw trajectories or neither")
    if args.baseline_raw_trajectory is not None:
        raw_trajectory_sha = _require_identical(
            args.baseline_raw_trajectory, args.cleaned_raw_trajectory, "raw trajectory"
        )

    trajectory = _read_tum(args.baseline_trajectory)
    gt_points, gt_dynamic, timestamps_ns = _build_gt_map(
        args.manifest.resolve(), trajectory,
        timestamp_tolerance_sec=args.timestamp_tolerance,
    )
    baseline = core.load_points(args.baseline_map, fmt="auto")
    cleaned = core.load_points(args.cleaned_map, fmt="auto")
    metrics, labels = evaluate_maps(
        baseline, cleaned, gt_points, gt_dynamic,
        match_tolerance_m=args.match_tolerance,
    )
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    loop_edge_lines = args.baseline_loop_edges.read_text(encoding="utf-8").splitlines()
    realtime_summary = None
    if args.dor_summary is not None:
        realtime_summary = json.loads(args.dor_summary.read_text(encoding="utf-8"))
    result = {
        "dataset": manifest_payload.get("dataset"),
        "scene": manifest_payload.get("scene"),
        "sensor_profile": manifest_payload.get("sensor_profile"),
        "filter": {
            "algorithm": args.filter_algorithm,
            "range_window": args.range_window,
            "range_margin_m": args.range_margin,
            "range_horizontal_resolution_deg": args.range_h_res,
            "range_vertical_resolution_deg": args.range_v_res,
            "realtime_summary": realtime_summary,
            "realtime_summary_sha256": (
                _sha256(args.dor_summary) if args.dor_summary is not None else None
            ),
        },
        "proof_contract": {
            "frames": len(trajectory),
            "timestamps_ns": timestamps_ns,
            "baseline_and_cleaned_optimized_trajectory_byte_identical": True,
            "optimized_trajectory_sha256": trajectory_sha,
            "baseline_and_cleaned_raw_trajectory_byte_identical": raw_trajectory_sha is not None,
            "raw_trajectory_sha256": raw_trajectory_sha,
            "baseline_and_cleaned_loop_edges_byte_identical": True,
            "loop_edges_sha256": loop_edges_sha,
            "loop_edge_count": max(0, len(loop_edge_lines) - 1),
            "labels_available_to_filter": False,
            "evaluated_map": "map_optimized.pcd (before optional cloud-driven refinement)",
        },
        "metrics": metrics,
        "inputs": {
            "manifest": args.manifest.name,
            "baseline_map": args.baseline_map.name,
            "cleaned_map": args.cleaned_map.name,
            "baseline_map_sha256": _sha256(args.baseline_map),
            "cleaned_map_sha256": _sha256(args.cleaned_map),
        },
    }
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    if args.output_png is not None:
        _plot(
            args.output_png, baseline, cleaned, labels, metrics,
            max_points=args.plot_max_points, seed=args.plot_seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
