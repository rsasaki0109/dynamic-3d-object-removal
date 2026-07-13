#!/usr/bin/env python3
"""Experimental BeautyMap-style binary-height candidate generator ablation.

The generator is not a final classifier: low-persistence height voxels propose
candidate points, coarse/fine z bins must agree, ground-adjacent points are reverted,
and the result is intersected with the established visibility + scan-ratio masks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
import dynamic_object_removal as core  # noqa: E402
from scripts import run_online_benchmark as online  # noqa: E402


def _row_keys(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.int64)
    return values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).reshape(-1)


def _membership(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(reference) == 0:
        return np.zeros(len(values), dtype=bool)
    unique = np.unique(reference)
    indices = np.searchsorted(unique, values)
    valid = indices < len(unique)
    result = np.zeros(len(values), dtype=bool)
    result[valid] = unique[indices[valid]] == values[valid]
    return result


def height_persistence_candidate(
    map_points: np.ndarray,
    scans: Sequence[np.ndarray],
    *,
    xy_cell: float,
    z_bin: float,
    min_cell_height: float,
    ground_margin: float,
    min_visits: int,
    max_persistence: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Candidate low-persistence height voxels in repeatedly visited XY cells."""
    map_points = np.asarray(map_points, dtype=np.float64).reshape(-1, 3)
    if xy_cell <= 0.0 or z_bin <= 0.0:
        raise ValueError("xy_cell and z_bin must be positive")
    xy = np.floor(map_points[:, :2] / xy_cell).astype(np.int64)
    xyz = np.column_stack([xy, np.floor(map_points[:, 2] / z_bin).astype(np.int64)])
    xy_keys = _row_keys(xy)
    xyz_keys = _row_keys(xyz)
    unique_xy, xy_inverse = np.unique(xy_keys, return_inverse=True)
    unique_xyz, xyz_inverse = np.unique(xyz_keys, return_inverse=True)
    visits_by_xy = np.zeros(len(unique_xy), dtype=np.int16)
    hits_by_xyz = np.zeros(len(unique_xyz), dtype=np.int16)
    for scan in scans:
        scan = np.asarray(scan, dtype=np.float64).reshape(-1, 3)
        if len(scan) == 0:
            continue
        scan_xy = np.floor(scan[:, :2] / xy_cell).astype(np.int64)
        scan_xyz = np.column_stack([
            scan_xy,
            np.floor(scan[:, 2] / z_bin).astype(np.int64),
        ])
        visits_by_xy += _membership(unique_xy, _row_keys(scan_xy)).astype(np.int16)
        hits_by_xyz += _membership(unique_xyz, _row_keys(scan_xyz)).astype(np.int16)

    visits = visits_by_xy[xy_inverse]
    hits = hits_by_xyz[xyz_inverse]

    low = np.full(len(unique_xy), np.inf)
    high = np.full(len(unique_xy), -np.inf)
    np.minimum.at(low, xy_inverse, map_points[:, 2])
    np.maximum.at(high, xy_inverse, map_points[:, 2])
    height = high[xy_inverse] - low[xy_inverse]
    above_ground = map_points[:, 2] > low[xy_inverse] + ground_margin
    persistence = np.divide(
        hits,
        np.maximum(visits, 1),
        out=np.zeros(len(map_points), dtype=np.float64),
        where=visits > 0,
    )
    candidate = (
        (visits >= max(1, int(min_visits)))
        & (height >= min_cell_height)
        & above_ground
        & (persistence <= max_persistence)
    )
    return candidate, {"visits": visits, "hits": hits, "persistence": persistence,
                       "cell_height": height}


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest.frames must be non-empty")
    fixed_scans = []
    gt_chunks = []
    scans_with_origins = []
    for index, frame in enumerate(frames):
        points = core.load_points(online._resolve_path(path.parent, frame["cloud"]), fmt="auto")
        pose = online._pose_from_payload(frame.get("pose"))
        fixed = points @ pose.rotation.T + pose.translation
        fixed_scans.append(fixed)
        gt_chunks.append(online._load_gt_mask(frame, points, path.parent))
        scans_with_origins.append((fixed, pose.translation))
    return manifest, fixed_scans, np.concatenate(fixed_scans), np.concatenate(gt_chunks), scans_with_origins


def run_ablation(manifest_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest, scans, map_points, gt, scans_with_origins = _load_manifest(manifest_path)
    started = time.perf_counter()
    profile = manifest.get("sensor_profile") if isinstance(manifest.get("sensor_profile"), dict) else {}
    strategy = core._sensor_strategy(profile.get("beams"), profile.get("v_spacing_deg"))
    if strategy == "fusion":
        _, keep_fusion = core.clean_map_by_fusion(
            map_points,
            scans_with_origins,
            free_votes_fraction=args.fusion_free_fraction,
            free_votes_floor=args.fusion_free_floor,
            void_min_scans=args.fusion_void_min_scans,
            workers=args.fusion_workers,
        )
        baseline_dynamic = ~keep_fusion
    else:
        ground_z = float(np.percentile(map_points[:, 2], 2))
        _, keep_range = core.clean_map_by_visibility(
            map_points,
            scans_with_origins,
            h_res_deg=args.h_res,
            v_res_deg=args.v_res,
            range_margin=args.range_margin,
            min_see_through=args.min_see_through,
            max_surface_hits=args.max_surface_hits,
            ground_z=ground_z,
        )
        _, keep_scan_ratio = core.clean_map_by_scan_ratio(map_points, scans_with_origins)
        baseline_dynamic = ~keep_range & ~keep_scan_ratio
        strategy = "range_and_scan_ratio"
    baseline_metrics = bench.compute_accuracy_metrics(baseline_dynamic, gt)
    baseline_completed = time.perf_counter()

    candidates = []
    for xy_cell in args.xy_cells:
        coarse_base, coarse_evidence = height_persistence_candidate(
            map_points, scans,
            xy_cell=xy_cell, z_bin=args.coarse_z_bin,
            min_cell_height=args.min_cell_height,
            ground_margin=args.ground_margin,
            min_visits=args.min_visits,
            max_persistence=1.0,
        )
        fine_base, fine_evidence = height_persistence_candidate(
            map_points, scans,
            xy_cell=xy_cell, z_bin=args.fine_z_bin,
            min_cell_height=args.min_cell_height,
            ground_margin=args.ground_margin,
            min_visits=args.min_visits,
            max_persistence=1.0,
        )
        for max_persistence in args.max_persistence:
            height_candidate = (
                coarse_base
                & fine_base
                & (coarse_evidence["persistence"] <= max_persistence)
                & (fine_evidence["persistence"] <= max_persistence)
            )
            dynamic = baseline_dynamic & height_candidate
            metrics = bench.compute_accuracy_metrics(dynamic, gt)
            candidates.append({
                "config": {"xy_cell": xy_cell, "coarse_z_bin": args.coarse_z_bin,
                           "fine_z_bin": args.fine_z_bin,
                           "max_persistence": max_persistence},
                "candidate_points": int(np.count_nonzero(height_candidate)),
                "metrics": metrics,
                "evidence": {
                    "visits_p50": float(np.percentile(coarse_evidence["visits"], 50)),
                    "persistence_p50": float(np.percentile(coarse_evidence["persistence"], 50)),
                },
            })
    candidates.sort(key=lambda item: item["metrics"]["f1"], reverse=True)
    best = candidates[0]
    finished = time.perf_counter()
    if strategy == "fusion":
        absolute_gate = (
            round(best["metrics"]["f1"], 3) >= 0.657
            and round(best["metrics"]["static_preservation"], 3) >= 0.974
        )
    else:
        absolute_gate = (
            round(best["metrics"]["f1"], 3) >= 0.642
            and round(best["metrics"]["static_preservation"], 3) >= 0.842
        )
    return {
        "task": "offline_map_cleaning",
        "algorithm": "height_candidate_experimental",
        "status": "experimental_not_promoted",
        "sensor_profile": profile,
        "deskew_input_contract_satisfied": bool(profile.get("deskewed", False)),
        "baseline": {"strategy": strategy, "metrics": baseline_metrics},
        "best_candidate": best,
        "candidates": candidates,
        "runtime": {
            "baseline_sec": baseline_completed - started,
            "candidate_grid_sec": finished - baseline_completed,
            "total_sec": finished - started,
        },
        "current_dataset_absolute_gate_pass": absolute_gate,
        "promotion_ready": False,
        "promotion_reason": "Requires cross-dataset non-regression and held-out validation.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--h-res", type=float, default=2.5)
    parser.add_argument("--v-res", type=float, default=2.5)
    parser.add_argument("--range-margin", type=float, default=0.5)
    parser.add_argument("--min-see-through", type=int, default=3)
    parser.add_argument("--max-surface-hits", type=int, default=5)
    parser.add_argument("--xy-cells", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--coarse-z-bin", type=float, default=0.5)
    parser.add_argument("--fine-z-bin", type=float, default=0.25)
    parser.add_argument("--min-cell-height", type=float, default=0.5)
    parser.add_argument("--ground-margin", type=float, default=0.2)
    parser.add_argument("--min-visits", type=int, default=3)
    parser.add_argument("--max-persistence", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--fusion-workers", type=int, default=6)
    parser.add_argument("--fusion-free-fraction", type=float, default=0.7)
    parser.add_argument("--fusion-free-floor", type=int, default=3)
    parser.add_argument("--fusion-void-min-scans", type=int, default=4)
    parser.add_argument("--summary-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_ablation(args.manifest.resolve(), args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    base = result["baseline"]["metrics"]
    best = result["best_candidate"]["metrics"]
    print(f"baseline f1={base['f1']:.3f} static={base['static_preservation']:.3f}")
    print(f"height candidate f1={best['f1']:.3f} static={best['static_preservation']:.3f}")
    print(f"Saved: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
