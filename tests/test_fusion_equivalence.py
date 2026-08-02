"""Result-preservation tests for fusion's optimized and spawn paths.

The reference helpers below are the pre-Task-E sorted-set implementation.  Keep
them in the test module so a future optimization has a local, executable oracle
instead of only metric-level regression coverage.
"""

from __future__ import annotations

import multiprocessing
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

import dynamic_object_removal as core


def _reference_in_sorted(values: np.ndarray, sorted_arr: np.ndarray) -> np.ndarray:
    if sorted_arr.size == 0:
        return np.zeros(values.shape, dtype=bool)
    idx = np.clip(np.searchsorted(sorted_arr, values), 0, sorted_arr.size - 1)
    return sorted_arr[idx] == values


def _reference_carve_free_scan(
    map_keys: np.ndarray,
    pts: np.ndarray,
    origin: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, float],
    ground: tuple[np.ndarray, np.ndarray, float, np.ndarray],
    *,
    step: float,
    carve_margin: float,
    ground_margin: float,
    min_range: float,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    mins, dims, voxel = grid
    vec = pts - origin
    r = np.linalg.norm(vec, axis=1)
    keep = r > min_range
    u = vec[keep] / r[keep, None]
    r = np.minimum(r[keep], max_range)
    endpts = pts[keep]

    ts = np.arange(min_range, max_range, step)
    free_chunks = []
    chunk = 20000
    for s in range(0, len(r), chunk):
        uc = u[s:s + chunk]
        rc = r[s:s + chunk]
        tmask = ts[None, :] < (rc[:, None] - carve_margin)
        if not tmask.any():
            continue
        n_t = int(tmask.sum(axis=1).max())
        samp = origin[None, None, :] + uc[:, None, :] * ts[None, :n_t, None]
        samp = samp[tmask[:, :n_t]]
        gz = core._ground_z_at(samp[:, :2], ground)
        samp = samp[samp[:, 2] > gz + ground_margin]
        if not len(samp):
            continue
        k, ok = core._voxel_keys(samp, mins, dims, voxel)
        free_chunks.append(np.unique(k[ok]))
    free = np.unique(np.concatenate(free_chunks)) if free_chunks else np.empty(0, np.int64)
    hit_keys, hok = core._voxel_keys(endpts, mins, dims, voxel)
    hit = np.unique(hit_keys[hok])
    free = np.setdiff1d(free, hit, assume_unique=True)
    freed = _reference_in_sorted(map_keys, free)
    observed = freed | _reference_in_sorted(map_keys, hit)
    return freed, observed


def _reference_carve_void_scan(
    map_keys: np.ndarray,
    pts: np.ndarray,
    origin: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, float],
    *,
    step: float,
    hit_inflation: float,
    min_range: float,
    max_range: float,
) -> np.ndarray:
    mins, dims, voxel = grid
    vec = pts - origin
    r = np.linalg.norm(vec, axis=1)
    keep = (r > min_range) & (r < max_range)
    u = vec[keep] / r[keep, None]
    r = r[keep]
    endpts = pts[keep]

    hit_keys, hok = core._voxel_keys(endpts, mins, dims, voxel)
    hit_list = [hit_keys[hok]]
    n_inf = max(1, int(np.ceil(hit_inflation / step)))
    for j in range(1, n_inf + 1):
        t = r - j * step
        good = t > min_range
        smp = origin[None, :] + u[good] * t[good, None]
        k, ok2 = core._voxel_keys(smp, mins, dims, voxel)
        hit_list.append(k[ok2])
    hit = np.unique(np.concatenate(hit_list))

    ts = np.arange(min_range + step, max_range, step)
    miss_list, ext_list = [], []
    chunk = 8000
    n_ext = 2
    for s in range(0, len(r), chunk):
        uc = u[s:s + chunk]
        rc = r[s:s + chunk]
        lim = rc - hit_inflation
        if not len(rc):
            continue
        n_t = int(min(len(ts), np.ceil((lim.max() - ts[0]) / step) + n_ext + 1))
        if n_t <= 0:
            continue
        tg = ts[:n_t]
        samp = origin[None, None, :] + uc[:, None, :] * tg[None, :, None]
        flat = samp.reshape(-1, 3)
        k, ok2 = core._voxel_keys(flat, mins, dims, voxel)
        k = k.reshape(len(rc), n_t)
        ok2 = ok2.reshape(len(rc), n_t)
        within = tg[None, :] < lim[:, None]
        blocked = _reference_in_sorted(k.ravel(), hit).reshape(len(rc), n_t) & within
        first_hit = np.where(blocked.any(axis=1), blocked.argmax(axis=1), n_t)
        col = np.arange(n_t)[None, :]
        miss_mask = within & ok2 & (col < first_hit[:, None])
        stop = np.minimum(
            np.where(
                blocked.any(axis=1),
                first_hit,
                np.ceil((lim - ts[0]) / step).astype(np.int64),
            ),
            n_t,
        )
        ext_mask = ok2 & (col >= stop[:, None]) & (col < (stop + n_ext)[:, None])
        miss_list.append(np.unique(k[miss_mask]))
        ext_list.append(np.unique(k[ext_mask]))
    miss = np.unique(np.concatenate(miss_list)) if miss_list else np.empty(0, np.int64)
    miss = np.setdiff1d(miss, hit, assume_unique=True)
    ext = np.unique(np.concatenate(ext_list)) if ext_list else np.empty(0, np.int64)
    observed = np.union1d(np.union1d(miss, hit), ext)

    noff = core._NB26[:, 0] * int(dims[1] * dims[2]) + core._NB26[:, 1] * int(dims[2]) + core._NB26[:, 2]
    confirmed = np.ones(miss.shape, dtype=bool)
    for off in noff:
        confirmed &= _reference_in_sorted(miss + off, observed)
        if not confirmed.any():
            break
    return _reference_in_sorted(map_keys, miss[confirmed])


def _reference_fusion(
    map_points: np.ndarray,
    scans: Sequence[tuple[np.ndarray, Sequence[float]]],
    *,
    free_votes_fraction: float = 0.7,
    free_votes_floor: int = 3,
    void_min_scans: int = 4,
) -> np.ndarray:
    """Run the pre-Task-E sorted membership implementation sequentially."""
    map_points = np.asarray(map_points, dtype=np.float64)
    free_mins, free_dims = core._voxel_grid(map_points, 0.5)
    void_mins, void_dims = core._voxel_grid(map_points, 0.4)
    free_keys, _ = core._voxel_keys(map_points, free_mins, free_dims, 0.5)
    void_keys, _ = core._voxel_keys(map_points, void_mins, void_dims, 0.4)
    params: dict[str, Any] = {
        "min_range": 0.5,
        "max_range": 12.0,
        "free_grid": (free_mins, free_dims, 0.5),
        "void_grid": (void_mins, void_dims, 0.4),
        "ground": core._ground_min_grid(map_points),
        "free_keys": free_keys,
        "void_keys": void_keys,
        "free_step": 0.5,
        "free_carve_margin": 0.7,
        "free_ground_margin": 0.25,
        "void_step": 0.2,
        "void_hit_inflation": 0.4,
        "n_rings": 10,
        "n_sectors": 48,
        "sr_max_range": 12.0,
        "scan_ratio_threshold": 0.2,
        "min_map_height": 0.5,
        "sr_ground_margin": 0.2,
    }
    n = len(map_points)
    sr_votes = np.zeros(n, dtype=np.int32)
    sr_obs = np.zeros(n, dtype=np.int32)
    free_votes = np.zeros(n, dtype=np.int32)
    free_obs = np.zeros(n, dtype=np.int32)
    void_votes = np.zeros(n, dtype=np.int32)
    for points, origin_value in scans:
        points = np.asarray(points, dtype=np.float64)
        origin = np.asarray(origin_value, dtype=np.float64)
        dyn, obs = core._scan_ratio_dynamic(
            map_points, points, origin,
            params["n_rings"], params["n_sectors"], params["sr_max_range"],
            params["scan_ratio_threshold"], params["min_map_height"], params["sr_ground_margin"],
        )
        sr_votes += dyn.astype(np.int32)
        sr_obs += obs.astype(np.int32)
        freed, fobs = _reference_carve_free_scan(
            free_keys, points, origin, params["free_grid"], params["ground"],
            step=params["free_step"], carve_margin=params["free_carve_margin"],
            ground_margin=params["free_ground_margin"], min_range=params["min_range"],
            max_range=params["max_range"],
        )
        free_votes += freed.astype(np.int32)
        free_obs += fobs.astype(np.int32)
        voided = _reference_carve_void_scan(
            void_keys, points, origin, params["void_grid"],
            step=params["void_step"], hit_inflation=params["void_hit_inflation"],
            min_range=params["min_range"], max_range=params["max_range"],
        )
        void_votes += voided.astype(np.int32)

    sr_dyn = sr_votes >= np.maximum(3, np.ceil(0.7 * sr_obs).astype(np.int64))
    free_dyn = free_votes >= np.maximum(
        max(1, min(int(free_votes_floor), len(scans))),
        np.ceil(float(free_votes_fraction) * free_obs).astype(np.int64),
    )
    void_dyn = void_votes >= max(1, min(int(void_min_scans), len(scans)))
    return ~(sr_dyn | free_dyn | void_dyn)


def _random_scene(seed: int) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    map_points = rng.uniform(
        [-8.0, -8.0, -1.0], [8.0, 8.0, 3.0], size=(180 + seed * 7, 3)
    ).astype(np.float64)
    scans: list[tuple[np.ndarray, np.ndarray]] = []
    for scan_index in range(4):
        origin = rng.uniform(-1.0, 1.0, size=3).astype(np.float64)
        sampled = map_points[rng.choice(len(map_points), size=90, replace=False)]
        noise = rng.normal(0.0, 0.025, size=sampled.shape)
        extra = rng.uniform([-10.0, -10.0, -2.0], [10.0, 10.0, 4.0], size=(35, 3))
        points = np.vstack([sampled + noise, extra]).astype(np.float64)
        if scan_index == 0:
            points = np.vstack([points, map_points[:12]])
        scans.append((points, origin))
    return map_points, scans


def _run_new(map_points: np.ndarray, scans: Sequence[tuple[np.ndarray, np.ndarray]], workers: int) -> np.ndarray:
    _, keep = core.clean_map_by_fusion(
        map_points,
        scans,
        min_range=0.5,
        max_range=12.0,
        free_voxel=0.5,
        free_step=0.5,
        free_carve_margin=0.7,
        void_voxel=0.4,
        void_step=0.2,
        void_hit_inflation=0.4,
        free_votes_fraction=0.7,
        free_votes_floor=3,
        void_min_scans=4,
        n_rings=10,
        n_sectors=48,
        sr_max_range=12.0,
        workers=workers,
    )
    return keep


def test_randomized_fusion_matches_pre_task_e_reference() -> None:
    for seed in range(1, 6):
        map_points, scans = _random_scene(seed)
        expected = _reference_fusion(map_points, scans)
        actual = _run_new(map_points, scans, workers=1)
        assert np.array_equal(actual, expected), f"sequential mismatch for seed {seed}"


def test_randomized_fusion_spawn_matches_reference(monkeypatch) -> None:
    map_points, scans = _random_scene(17)
    expected = _reference_fusion(map_points, scans)
    spawn_context = multiprocessing.get_context("spawn")
    monkeypatch.setattr(core, "_fusion_pool_context", lambda: spawn_context)
    actual = _run_new(map_points, scans, workers=2)
    assert np.array_equal(actual, expected)


def test_fusion_context_falls_back_to_spawn(monkeypatch) -> None:
    real_get_context = multiprocessing.get_context
    calls: list[str] = []

    def fake_get_context(method: str):
        calls.append(method)
        if method == "fork":
            raise ValueError("fork unavailable in test")
        return real_get_context(method)

    monkeypatch.setattr(multiprocessing, "get_context", fake_get_context)
    context = core._fusion_pool_context()
    assert context.get_start_method() == "spawn"
    assert calls == ["fork", "spawn"]


def test_fusion_context_preserves_fork_selection(monkeypatch) -> None:
    fake_context = SimpleNamespace(get_start_method=lambda: "fork")
    calls: list[str] = []

    def fake_get_context(method: str):
        calls.append(method)
        return fake_context

    monkeypatch.setattr(multiprocessing, "get_context", fake_get_context)
    assert core._fusion_pool_context() is fake_context
    assert calls == ["fork"]
