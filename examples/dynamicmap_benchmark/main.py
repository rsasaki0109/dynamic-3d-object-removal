#!/usr/bin/env python3
"""DynamicMap_Benchmark adapter for dynamic-object-removal (numpy-only).

Usage (from a cloned DynamicMap_Benchmark repo, after
``pip install "dynamic-object-removal>=0.5"``):
    python main.py --data_dir /path/to/00 --algorithm fusion
    python main.py --data_dir /path/to/00 --algorithm range
    python main.py --data_dir /path/to/00 --algorithm scan_ratio
    python main.py --data_dir /path/to/00 --algorithm temporal

Writes ``dor_<algorithm>_output.pcd`` into ``data_dir`` (cleaned accumulated map).
Run ``export_eval_pcd`` + ``evaluate_all.py`` in the benchmark repo for SA/DA/AA/HA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import dynamic_object_removal as core
except ImportError as exc:
    raise SystemExit(
        'Install first: pip install "dynamic-object-removal>=0.5"'
    ) from exc


def _load_sequence(pcd_dir: Path) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int]]]:
    scan_files = sorted(pcd_dir.glob("*.pcd"))
    if not scan_files:
        raise SystemExit(f"No scans in {pcd_dir}")

    chunks: list[np.ndarray] = []
    scans: list[tuple[np.ndarray, np.ndarray]] = []
    slices: list[tuple[int, int]] = []
    cursor = 0
    for path in scan_files:
        scan = core.load_pcd_scan(path)
        # PCD defaults VIEWPOINT to identity, so a header at the default value and
        # a scan genuinely captured at the world origin are indistinguishable —
        # accept both as a zero origin.
        origin = scan.viewpoint[:3] if scan.viewpoint is not None else np.zeros(3)
        n = len(scan.points)
        chunks.append(scan.points)
        scans.append((scan.points, origin))
        slices.append((cursor, cursor + n))
        cursor += n
    return np.concatenate(chunks, axis=0), scans, slices


def _clean(
    algorithm: str,
    acc_map: np.ndarray,
    scans: list[tuple[np.ndarray, np.ndarray]],
    slices: list[tuple[int, int]],
    *,
    h_res: float,
    v_res: float,
    min_see_through: int,
    max_surface_hits: int,
    voxel_size: float,
    temporal_min_hits: int,
    sr_min_votes: int | None,
    fusion_workers: int,
) -> np.ndarray:
    if algorithm == "fusion":
        # OR-fusion of free-space carving, DUFOMap-style eroded voids, and
        # scan-ratio votes — the library's highest-accuracy map cleaner.
        _, keep = core.clean_map_by_fusion(acc_map, scans, workers=fusion_workers)
    elif algorithm == "range":
        ground_z = float(np.percentile(acc_map[:, 2], 2))
        _, keep = core.clean_map_by_visibility(
            acc_map,
            scans,
            h_res_deg=h_res,
            v_res_deg=v_res,
            min_see_through=min_see_through,
            max_surface_hits=max_surface_hits,
            ground_z=ground_z,
        )
    elif algorithm == "scan_ratio":
        # min_votes=None: the library normalizes votes per point by the number of
        # scans that revisit its polar column (majority rule).
        _, keep = core.clean_map_by_scan_ratio(acc_map, scans, min_votes=sr_min_votes)
    elif algorithm == "temporal":
        keep = np.ones(len(acc_map), dtype=bool)
        tfilter = core.TemporalConsistencyFilter(
            voxel_size=voxel_size,
            window_size=len(scans),
            min_hits=temporal_min_hits,
        )
        # Two passes over the same scans: the first fills the hit counter
        # (window_size == number of scans, so nothing is evicted), the second
        # judges each scan against the full-sequence counts — re-inserting a
        # scan evicts its own first-pass copy, leaving the counter unchanged.
        for s, e in slices:
            tfilter.filter(acc_map[s:e])
        for s, e in slices:
            _, keep_f = tfilter.filter(acc_map[s:e])
            keep[s:e] = keep_f
    else:
        raise SystemExit(f"Unknown algorithm: {algorithm}")
    return acc_map[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description="DynamicMap_Benchmark adapter (numpy-only).")
    parser.add_argument("--data_dir", required=True, help="Sequence folder with pcd/ and gt_cloud.pcd.")
    parser.add_argument("--algorithm", choices=["fusion", "range", "scan_ratio", "temporal"], default="fusion")
    parser.add_argument("--h-res", type=float, default=1.0)
    parser.add_argument("--v-res", type=float, default=1.0)
    # 3/3 (vs the library's 2/2 default) is the setting behind the README range
    # numbers, matching the upstream library's run_dynamicmap_benchmark.py.
    parser.add_argument("--min-see-through", type=int, default=3)
    parser.add_argument("--max-surface-hits", type=int, default=3)
    parser.add_argument("--voxel-size", type=float, default=core.DEFAULT_TEMPORAL_VOXEL_SIZE)
    parser.add_argument("--temporal-min-hits", type=int, default=2)
    parser.add_argument("--sr-min-votes", type=int, default=None,
                        help="Fixed absolute vote threshold (default: majority of each point's column revisits).")
    parser.add_argument("--fusion-workers", type=int, default=6,
                        help="Process pool size for the fusion carving channels.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pcd_dir = data_dir / "pcd"
    acc_map, scans, slices = _load_sequence(pcd_dir)
    cleaned = _clean(
        args.algorithm,
        acc_map,
        scans,
        slices,
        h_res=args.h_res,
        v_res=args.v_res,
        min_see_through=args.min_see_through,
        max_surface_hits=args.max_surface_hits,
        voxel_size=args.voxel_size,
        temporal_min_hits=args.temporal_min_hits,
        sr_min_votes=args.sr_min_votes,
        fusion_workers=args.fusion_workers,
    )
    out = data_dir / f"dor_{args.algorithm}_output.pcd"
    core.save_points(out, cleaned, fmt="pcd")
    print(f"Wrote {len(cleaned):,} points -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
