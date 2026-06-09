#!/usr/bin/env python3
"""Build ``demo/sample_nuscenes_range.npz`` for the browser playground.

Uses nuScenes mini scene-0757 (busy intersection), 6 pose-aligned keyframes,
centroid-centered like ``sample_av2_range.npz``. Target size: ~1.5 MB (same order
as the AV2 multi-scan preset).

Usage:
    python3 scripts/build_playground_nuscenes_sample.py
    python3 scripts/build_playground_nuscenes_sample.py --output demo/sample_nuscenes_range.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_nuscenes_script():
    path = ROOT / "scripts" / "run_nuscenes_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_nuscenes_benchmark", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nb = _load_nuscenes_script()

DEFAULT_SCENE = "scene-0757"
DEFAULT_FRAMES = 6
DEFAULT_STRIDE = 2
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "demo" / "sample_nuscenes_range.npz"


def build(
    *,
    scene: str,
    frames: int,
    stride: int,
    root: Path,
    output: Path,
) -> None:
    nb._ensure_data(root)
    tables = nb._load_tables(root)
    if scene not in tables["scene"]:
        raise SystemExit(f"Unknown scene {scene}")

    toks: list[str] = []
    cur = tables["scene"][scene]["first_sample_token"]
    while cur:
        toks.append(cur)
        cur = tables["sample"][cur]["next"]
    selected = toks[0 : frames * stride : stride]

    chunks: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    splits = [0]
    for st in selected:
        d = tables["lidar"][st]
        ep = tables["ego"][d["ego_pose_token"]]
        csd = tables["cs"][d["calibrated_sensor_token"]]
        pts = np.fromfile(root / d["filename"], dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
        pts = pts[pts[:, 2] > nb.GROUND_Z_SENSOR]
        r_cs = nb._quat_to_rot(*csd["rotation"])
        t_cs = np.asarray(csd["translation"])
        r_ego = nb._quat_to_rot(*ep["rotation"])
        t_ego = np.asarray(ep["translation"])
        pts_global = (pts @ r_cs.T + t_cs) @ r_ego.T + t_ego
        origin = r_ego @ t_cs + t_ego
        chunks.append(pts_global)
        origins.append(origin)
        splits.append(splits[-1] + len(pts_global))

    scans_cat = np.concatenate(chunks, axis=0)
    centroid = scans_cat.mean(axis=0)
    scans_cat = scans_cat - centroid
    origins_arr = np.stack(origins, axis=0) - centroid
    ground_z = float(np.percentile(scans_cat[:, 2], 2))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        scans=scans_cat.astype(np.float32),
        splits=np.asarray(splits, dtype=np.int64),
        origins=origins_arr.astype(np.float32),
        ground_z=np.float32(ground_z),
        h_res=np.float32(2.5),
        v_res=np.float32(2.5),
    )
    mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({mb:.2f} MB, {len(scans_cat):,} pts, {len(selected)} scans)")
    print("Data source: nuScenes mini (CC BY-NC-SA 4.0) https://www.nuscenes.org")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build nuScenes playground multi-scan preset.")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--root", default=str(nb.ROOT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    build(
        scene=args.scene,
        frames=args.frames,
        stride=args.stride,
        root=Path(args.root),
        output=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
