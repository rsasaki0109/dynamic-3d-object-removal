#!/usr/bin/env python3
"""Reproducible benchmark on KTH-RPL DynamicMap_Benchmark (Semantic-KITTI).

Downloads the public Zenodo teaser sequences (no signup), loads pose-attached
per-scan PCDs (sensor pose in the PCD VIEWPOINT field), accumulates a raw map,
runs this repo's detector-free cleaners (``range``, ``scan_ratio``, ``temporal``),
and reports DynamicMap metrics (SA / DA / AA / HA) against the bundled ``gt_cloud.pcd``.

These are **our methods on the standard benchmark format** — not a re-run of
ERASOR/Removert/DUFOMap.

Usage (numpy only; stdlib download):
    python3 scripts/run_dynamicmap_benchmark.py
    python3 scripts/run_dynamicmap_benchmark.py --sequences 00 05
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench  # noqa: E402
import dynamic_object_removal as core  # noqa: E402
from scripts.run_height_candidate_ablation import height_persistence_candidate  # noqa: E402

ZENODO_BASE = "https://zenodo.org/records/10886629/files"
DEFAULT_SEQUENCES = ("00", "05")
ROOT_DIR = Path(__file__).resolve().parents[1] / "data" / "dynamicmap"
MIN_FREE_AFTER_EXTRACT = 256 * 1024 * 1024


def _archive_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as zf:
        return sum(info.file_size for info in zf.infolist())


def _require_extract_capacity(zip_path: Path, root: Path) -> None:
    required = _archive_uncompressed_bytes(zip_path)
    free = shutil.disk_usage(root).free
    if required + MIN_FREE_AFTER_EXTRACT > free:
        raise SystemExit(
            f"Refusing to extract {zip_path.name}: {required / 2**30:.2f} GiB expanded "
            f"with only {free / 2**30:.2f} GiB free; keep at least "
            f"{MIN_FREE_AFTER_EXTRACT / 2**30:.2f} GiB free."
        )


def _download_sequence(seq: str, root: Path) -> Path:
    seq_dir = root / seq
    if (seq_dir / "gt_cloud.pcd").exists() and (seq_dir / "pcd").is_dir():
        return seq_dir

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{seq}.zip"
    url = f"{ZENODO_BASE}/{seq}.zip"
    if not zip_path.exists():
        print(f"Downloading Semantic-KITTI seq {seq} from Zenodo ({url}) ...")
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req) as resp, zip_path.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        print(f"  saved {zip_path}")

    print(f"Extracting {zip_path.name} ...")
    _require_extract_capacity(zip_path, root)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not (seq_dir / "gt_cloud.pcd").exists():
        raise SystemExit(f"Expected {seq_dir}/gt_cloud.pcd after extract")
    return seq_dir


def _load_sequence(seq_dir: Path) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int]]]:
    pcd_dir = seq_dir / "pcd"
    scan_files = sorted(pcd_dir.glob("*.pcd"))
    if not scan_files:
        raise SystemExit(f"No PCD scans in {pcd_dir}")

    chunks: list[np.ndarray] = []
    scans: list[tuple[np.ndarray, np.ndarray]] = []
    slices: list[tuple[int, int]] = []
    cursor = 0

    for path in scan_files:
        scan = core.load_pcd_scan(path)
        if scan.viewpoint is None:
            raise SystemExit(f"Missing VIEWPOINT pose in {path}")
        origin = scan.viewpoint[:3]
        n = len(scan.points)
        chunks.append(scan.points)
        scans.append((scan.points, origin))
        slices.append((cursor, cursor + n))
        cursor += n

    acc_map = np.concatenate(chunks, axis=0)
    return acc_map, scans, slices


def _run_methods(
    acc_map: np.ndarray,
    scans: list[tuple[np.ndarray, np.ndarray]],
    slices: list[tuple[int, int]],
    *,
    h_res: float,
    v_res: float,
    range_margin: float,
    min_see_through: int,
    max_surface_hits: int,
    resolutions: list[float] | None,
    voxel_size: float,
    temporal_min_hits: int,
    sr_min_votes: int,
    fusion_workers: int,
    height_candidate_ablation: bool,
    height_xy_cell: float,
    height_coarse_z_bin: float,
    height_fine_z_bin: float,
    height_min_cell_height: float,
    height_ground_margin: float,
    height_min_visits: int,
    height_max_persistence: float,
    methods: list[str] | None,
) -> dict[str, np.ndarray]:
    ground_z = float(np.percentile(acc_map[:, 2], 2))
    selected = set(methods or ("range", "scan_ratio", "fusion", "temporal"))
    cleaned: dict[str, np.ndarray] = {}

    if "range" in selected:
        print("  running range-image visibility ...", flush=True)
        _, keep_range = core.clean_map_by_visibility(
            acc_map,
            scans,
            h_res_deg=h_res,
            v_res_deg=v_res,
            range_margin=range_margin,
            min_see_through=min_see_through,
            max_surface_hits=max_surface_hits,
            ground_z=ground_z,
            resolutions=resolutions,
        )
        print(f"    kept {int(keep_range.sum()):,} / {len(acc_map):,}", flush=True)
        cleaned["range"] = acc_map[keep_range]

    if "scan_ratio" in selected:
        print("  running scan-ratio ...", flush=True)
        _, keep_sr = core.clean_map_by_scan_ratio(
            acc_map,
            scans,
            min_votes=sr_min_votes,
        )
        print(f"    kept {int(keep_sr.sum()):,} / {len(acc_map):,}", flush=True)
        cleaned["scan_ratio"] = acc_map[keep_sr]

    keep_fusion = None
    if "fusion" in selected or height_candidate_ablation:
        print("  running free-space fusion ...", flush=True)
        _, keep_fusion = core.clean_map_by_fusion(
            acc_map,
            scans,
            workers=fusion_workers,
        )
        print(f"    kept {int(keep_fusion.sum()):,} / {len(acc_map):,}", flush=True)
        if "fusion" in selected:
            cleaned["fusion"] = acc_map[keep_fusion]

    if "temporal" in selected:
        print("  running temporal consistency ...", flush=True)
        keep_temporal = np.ones(len(acc_map), dtype=bool)
        tfilter = core.TemporalConsistencyFilter(
            voxel_size=voxel_size,
            window_size=len(scans),
            min_hits=temporal_min_hits,
        )
        for s, e in slices:
            tfilter.filter(acc_map[s:e])
        for s, e in slices:
            _, keep_f = tfilter.filter(acc_map[s:e])
            keep_temporal[s:e] = keep_f
        print(f"    kept {int(keep_temporal.sum()):,} / {len(acc_map):,}", flush=True)
        cleaned["temporal"] = acc_map[keep_temporal]
    if height_candidate_ablation:
        assert keep_fusion is not None
        print("  running private height-candidate ablation ...", flush=True)
        scan_points = [points for points, _ in scans]
        coarse, coarse_evidence = height_persistence_candidate(
            acc_map,
            scan_points,
            xy_cell=height_xy_cell,
            z_bin=height_coarse_z_bin,
            min_cell_height=height_min_cell_height,
            ground_margin=height_ground_margin,
            min_visits=height_min_visits,
            max_persistence=height_max_persistence,
        )
        fine, fine_evidence = height_persistence_candidate(
            acc_map,
            scan_points,
            xy_cell=height_xy_cell,
            z_bin=height_fine_z_bin,
            min_cell_height=height_min_cell_height,
            ground_margin=height_ground_margin,
            min_visits=height_min_visits,
            max_persistence=height_max_persistence,
        )
        candidate = (
            coarse
            & fine
            & (coarse_evidence["persistence"] <= height_max_persistence)
            & (fine_evidence["persistence"] <= height_max_persistence)
        )
        dynamic = ~keep_fusion & candidate
        cleaned["fusion_height_candidate"] = acc_map[~dynamic]
        print(
            f"    proposed {int(candidate.sum()):,}; removed {int(dynamic.sum()):,} / {len(acc_map):,}",
            flush=True,
        )
    return cleaned


def _evaluate_sequence(
    seq: str,
    seq_dir: Path,
    *,
    eval_max_dist: float,
    **method_kw,
) -> dict[str, dict[str, float]]:
    print(f"\nSemantic-KITTI seq {seq}: loading scans ...", flush=True)
    acc_map, scans, slices = _load_sequence(seq_dir)
    print(f"  raw map: {len(acc_map):,} pts | {len(scans)} scans", flush=True)

    gt = core.load_pcd_scan(seq_dir / "gt_cloud.pcd")
    if gt.intensity is None:
        raise SystemExit(f"{seq_dir}/gt_cloud.pcd has no intensity labels")
    gt_labels = (gt.intensity > 0.5).astype(np.int64)
    print(
        f"  gt_cloud: {len(gt.points):,} pts | "
        f"static {int(np.count_nonzero(gt_labels == 0)):,} | "
        f"dynamic {int(np.count_nonzero(gt_labels == 1)):,}",
        flush=True,
    )

    cleaned = _run_methods(acc_map, scans, slices, **method_kw)
    results: dict[str, dict[str, float]] = {}
    for name, kept_xyz in cleaned.items():
        print(f"  evaluating {name} ({len(kept_xyz):,} kept pts) ...", flush=True)
        est = bench.export_dynamicmap_eval_labels(gt.points, kept_xyz, max_dist=eval_max_dist)
        metrics = bench.compute_dynamicmap_metrics(est, gt_labels)
        results[name] = metrics
    return results


def _print_table(seq: str, results: dict[str, dict[str, float]]) -> str:
    rows = []
    for method, m in results.items():
        rows.append(
            f"| {method} | {m['SA']:.2f} | {m['DA']:.2f} | {m['AA']:.2f} | {m['HA']:.2f} |"
        )
    table = (
        f"\n### Measured on Semantic-KITTI seq {seq} (DynamicMap_Benchmark format)\n\n"
        "| method | SA [%] ↑ | DA [%] ↑ | AA [%] ↑ | HA [%] ↑ |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )
    print(table)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark on DynamicMap_Benchmark Semantic-KITTI data.")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES), help="Sequence ids (00, 05, ...).")
    parser.add_argument("--data-root", default=str(ROOT_DIR), help="Download/extract root directory.")
    parser.add_argument("--eval-max-dist", type=float, default=0.05, help="Nearest-neighbor match threshold (m).")
    parser.add_argument("--h-res", type=float, default=1.0, help="Range-image azimuth resolution (VLP-64).")
    parser.add_argument("--v-res", type=float, default=1.0, help="Range-image elevation resolution (VLP-64).")
    parser.add_argument("--range-margin", type=float, default=core.DEFAULT_RANGE_MARGIN)
    parser.add_argument("--min-see-through", type=int, default=3)
    parser.add_argument("--max-surface-hits", type=int, default=3)
    parser.add_argument("--resolutions", type=float, nargs="+", default=None)
    parser.add_argument("--voxel-size", type=float, default=core.DEFAULT_TEMPORAL_VOXEL_SIZE)
    parser.add_argument("--temporal-min-hits", type=int, default=2)
    parser.add_argument("--sr-min-votes", type=int, default=None,
                        help="Fixed absolute vote threshold (default: normalized, majority of each point's column revisits).")
    parser.add_argument("--fusion-workers", type=int, default=6,
                        help="Process pool size for the free-space fusion channels.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("range", "scan_ratio", "fusion", "temporal"),
        default=None,
        help="Run only selected methods (default: all). Useful for a targeted gate rerun.",
    )
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--sensor-aware-ablation", action="store_true",
                        help="Record the private O1 dense-sensor selector decision.")
    parser.add_argument("--height-candidate-ablation", action="store_true",
                        help="Evaluate the private O2 candidate intersected with fusion.")
    parser.add_argument("--height-xy-cell", type=float, default=2.0)
    parser.add_argument("--height-coarse-z-bin", type=float, default=0.5)
    parser.add_argument("--height-fine-z-bin", type=float, default=0.25)
    parser.add_argument("--height-min-cell-height", type=float, default=0.5)
    parser.add_argument("--height-ground-margin", type=float, default=0.2)
    parser.add_argument("--height-min-visits", type=int, default=3)
    parser.add_argument("--height-max-persistence", type=float, default=1.0)
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    method_kw = {
        "h_res": args.h_res,
        "v_res": args.v_res,
        "range_margin": args.range_margin,
        "min_see_through": args.min_see_through,
        "max_surface_hits": args.max_surface_hits,
        "resolutions": args.resolutions,
        "voxel_size": args.voxel_size,
        "temporal_min_hits": args.temporal_min_hits,
        "sr_min_votes": args.sr_min_votes,
        "fusion_workers": args.fusion_workers,
        "height_candidate_ablation": args.height_candidate_ablation,
        "height_xy_cell": args.height_xy_cell,
        "height_coarse_z_bin": args.height_coarse_z_bin,
        "height_fine_z_bin": args.height_fine_z_bin,
        "height_min_cell_height": args.height_min_cell_height,
        "height_ground_margin": args.height_ground_margin,
        "height_min_visits": args.height_min_visits,
        "height_max_persistence": args.height_max_persistence,
        "methods": args.methods,
    }

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for seq in args.sequences:
        seq_dir = _download_sequence(seq, root)
        all_results[seq] = _evaluate_sequence(
            seq,
            seq_dir,
            eval_max_dist=args.eval_max_dist,
            **method_kw,
        )
        _print_table(seq, all_results[seq])

    payload = {
        "dataset": "DynamicMap_Benchmark/Semantic-KITTI",
        "sequences": args.sequences,
        "config": {
            "eval_max_dist": args.eval_max_dist,
            **method_kw,
        },
        "results": all_results,
        "sensor_aware": ({
            "status": "experimental_not_promoted",
            "profile": {"beams": 64, "v_spacing_deg": 0.4},
            "baseline_strategy": core._sensor_strategy(64, 0.4),
            "selected_method": "fusion",
        } if args.sensor_aware_ablation else None),
    }
    out_json = args.summary_json or str(root / "benchmark_result.json")
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")
    print(
        "\nData source: KTH-RPL DynamicMap_Benchmark (Zenodo record 10886629) "
        "https://github.com/KTH-RPL/DynamicMap_Benchmark"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
