#!/usr/bin/env python3
"""Compare same-pose baseline and filtered Step-A maps without GT claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _read_pcd(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - optional integration dependency
        raise SystemExit("open3d is required to read binary_compressed PCD maps") from exc
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"no XYZ points loaded from {path}")
    return points


def _trajectory_points(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0].startswith("VERTEX_SE3") and len(fields) >= 5:
            points.append([float(fields[2]), float(fields[3]), float(fields[4])])
    return np.asarray(points, dtype=np.float64) if points else None


def compare_points(
    baseline: np.ndarray,
    filtered: np.ndarray,
    *,
    match_radius: float,
    density_radius: float,
    dense_min_neighbors: int,
    trajectory: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    baseline_tree = cKDTree(baseline)
    filtered_tree = cKDTree(filtered)
    baseline_distance = filtered_tree.query(baseline, k=1)[0]
    filtered_distance = baseline_tree.query(filtered, k=1)[0]
    baseline_only = baseline_distance > match_radius
    neighbor_count = baseline_tree.query_ball_point(
        baseline, density_radius, return_length=True
    )
    dense = neighbor_count >= dense_min_neighbors

    result: dict[str, object] = {
        "baseline_points": int(len(baseline)),
        "filtered_points": int(len(filtered)),
        "point_reduction": int(len(baseline) - len(filtered)),
        "point_reduction_ratio": float((len(baseline) - len(filtered)) / len(baseline)),
        "match_radius_m": match_radius,
        "baseline_supported_ratio": float(np.mean(~baseline_only)),
        "filtered_supported_ratio": float(np.mean(filtered_distance <= match_radius)),
        "baseline_only_candidates": int(np.count_nonzero(baseline_only)),
        "baseline_only_candidate_ratio": float(np.mean(baseline_only)),
        "dense_proxy_points": int(np.count_nonzero(dense)),
        "dense_proxy_preserved_ratio": float(
            np.mean(baseline_distance[dense] <= match_radius)
        ) if np.any(dense) else None,
        "sparse_proxy_candidate_ratio": float(
            np.mean(baseline_only[~dense])
        ) if np.any(~dense) else None,
        "dense_proxy_candidate_ratio": float(
            np.mean(baseline_only[dense])
        ) if np.any(dense) else None,
        "baseline_neighbor_count_median": float(np.median(neighbor_count)),
        "candidate_neighbor_count_median": float(
            np.median(neighbor_count[baseline_only])
        ) if np.any(baseline_only) else None,
        "supported_neighbor_count_median": float(
            np.median(neighbor_count[~baseline_only])
        ) if np.any(~baseline_only) else None,
    }
    if trajectory is not None and len(trajectory):
        trajectory_tree = cKDTree(trajectory)
        distance_to_path = trajectory_tree.query(baseline, k=1)[0]
        result["candidate_distance_to_path_median_m"] = float(
            np.median(distance_to_path[baseline_only])
        ) if np.any(baseline_only) else None
        result["supported_distance_to_path_median_m"] = float(
            np.median(distance_to_path[~baseline_only])
        ) if np.any(~baseline_only) else None
    return result, baseline_only


def _plot(
    output: Path,
    baseline: np.ndarray,
    filtered: np.ndarray,
    baseline_only: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    combined = np.vstack([baseline[:, :2], filtered[:, :2]])
    lo = np.percentile(combined, 0.5, axis=0)
    hi = np.percentile(combined, 99.5, axis=0)
    padding = np.maximum((hi - lo) * 0.03, 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = (
        (baseline, "Baseline map", "#666666"),
        (filtered, "Filtered map", "#2676c8"),
        (baseline[baseline_only], "Baseline-only candidates", "#e4572e"),
    )
    for axis, (points, title, color) in zip(axes, panels):
        axis.scatter(points[:, 0], points[:, 1], s=1.2, c=color, alpha=0.75, linewidths=0)
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(lo[0] - padding[0], hi[0] + padding[0])
        axis.set_ylim(lo[1] - padding[1], hi[1] + padding[1])
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(alpha=0.15)
    fig.suptitle("Same-pose TIERS Indoor02 map comparison (robust 0.5–99.5% XY view)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_pcd", type=Path)
    parser.add_argument("filtered_pcd", type=Path)
    parser.add_argument("--pose-graph", type=Path)
    parser.add_argument("--match-radius", type=float, default=0.2)
    parser.add_argument("--density-radius", type=float, default=0.5)
    parser.add_argument("--dense-min-neighbors", type=int, default=8)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-png", type=Path)
    args = parser.parse_args()

    baseline = _read_pcd(args.baseline_pcd)
    filtered = _read_pcd(args.filtered_pcd)
    result, baseline_only = compare_points(
        baseline,
        filtered,
        match_radius=args.match_radius,
        density_radius=args.density_radius,
        dense_min_neighbors=args.dense_min_neighbors,
        trajectory=_trajectory_points(args.pose_graph),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output_json:
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    if args.output_png:
        _plot(args.output_png, baseline, filtered, baseline_only)


if __name__ == "__main__":
    main()
