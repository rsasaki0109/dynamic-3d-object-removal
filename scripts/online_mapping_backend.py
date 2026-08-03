"""Private bounded back-end used by the Task F online mapping experiment.

This module is deliberately outside the public library API.  It contains the
smallest stateful piece needed for the experiment: per-voxel free-space
evidence, a bounded recent map-point index, and a sliced retroactive
re-judgment queue.

The evidence rule follows the measured fusion free-space channel.  A scan's
ray samples are collected first, endpoint hits take precedence over free
samples from that same scan, and a voxel is eligible only after the configured
fraction/floor of observed scans voted free.  Ground returns are protected
using the same local-minimum grid used by the fusion implementation.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np

import dynamic_object_removal as core


_KEY_DTYPE = np.dtype((np.void, 3 * np.dtype(np.int64).itemsize))


def _pack_voxel_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, 3)
    return rows.view(_KEY_DTYPE).reshape(-1)


def _unpack_voxel_keys(keys: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(keys).view(np.int64).reshape(-1, 3)


def _key(row: Sequence[int]) -> tuple[int, int, int]:
    return int(row[0]), int(row[1]), int(row[2])


def _unique_rows(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    packed = np.unique(_pack_voxel_rows(rows))
    return _unpack_voxel_keys(packed)


def _carve_free_voxels(
    points: np.ndarray,
    origin: np.ndarray,
    *,
    voxel_size: float,
    step: float,
    carve_margin: float,
    ground_margin: float,
    min_range: float,
    max_range: float,
    target_keys: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(free, hit)`` voxel rows for one pose-aligned scan.

    This is the bounded-back-end equivalent of the existing fusion helper.  It
    intentionally does not retain the ray samples: only unique voxel rows are
    returned to the bounded evidence store.
    """
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if points.size == 0 or len(points) == 0:
        empty = np.empty((0, 3), dtype=np.int64)
        return empty, empty
    if target_keys is not None and len(target_keys) == 0:
        empty = np.empty((0, 3), dtype=np.int64)
        return empty, empty
    if origin.shape != (3,):
        raise ValueError("origin must have shape (3,)")

    vectors = points - origin
    ranges = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(ranges) & (ranges > min_range) & (ranges < max_range)
    if not valid.any():
        empty = np.empty((0, 3), dtype=np.int64)
        return empty, empty

    valid_points = points[valid]
    unit = vectors[valid] / ranges[valid, None]
    valid_ranges = ranges[valid]

    hit_rows = np.floor(valid_points / voxel_size).astype(np.int64)
    if target_keys is not None:
        target_packed = np.asarray(target_keys, dtype=_KEY_DTYPE).reshape(-1)
        hit_packed = _pack_voxel_rows(hit_rows)
        hit = _unpack_voxel_keys(np.intersect1d(hit_packed, target_packed, assume_unique=False))
    else:
        target_packed = None
        hit = _unique_rows(hit_rows)

    # The current fusion path uses a local XY minimum grid to protect ground
    # returns.  It is temporary per scan, so it does not consume persistent
    # back-end state.
    ground = core._ground_min_grid(points)
    sample_distances = np.arange(min_range, max_range, step, dtype=np.float64)
    free_parts: list[np.ndarray] = []
    chunk_size = 20_000
    for start in range(0, len(valid_ranges), chunk_size):
        unit_chunk = unit[start : start + chunk_size]
        range_chunk = valid_ranges[start : start + chunk_size]
        traversed = sample_distances[None, :] < (range_chunk[:, None] - carve_margin)
        if not traversed.any():
            continue
        max_samples = int(traversed.sum(axis=1).max())
        samples = origin[None, None, :] + unit_chunk[:, None, :] * sample_distances[
            None, :max_samples, None
        ]
        samples = samples[traversed[:, :max_samples]]
        if samples.size == 0:
            continue
        ground_z = core._ground_z_at(samples[:, :2], ground)
        samples = samples[samples[:, 2] > ground_z + ground_margin]
        if samples.size:
            sample_rows = np.floor(samples / voxel_size).astype(np.int64)
            sample_packed = _pack_voxel_rows(sample_rows)
            if target_packed is not None:
                sample_packed = sample_packed[np.isin(sample_packed, target_packed)]
                if sample_packed.size:
                    free_parts.append(np.unique(sample_packed))
            else:
                free_parts.append(_unique_rows(sample_rows))

    if free_parts:
        free_packed = np.unique(np.concatenate(free_parts, axis=0))
        # Hit precedence: a voxel hit by this scan cannot also be a free vote
        # from the same scan.
        free_packed = np.setdiff1d(free_packed, _pack_voxel_rows(hit), assume_unique=True)
        free = _unpack_voxel_keys(free_packed)
    else:
        free = np.empty((0, 3), dtype=np.int64)
    return free, hit


class _BoundedVoxelEvidenceStore:
    """An LRU-bounded, saturating voxel evidence table.

    The Python mapping is intentionally capped by ``max_voxels``.  Counters
    saturate instead of growing with replay length, so both the number of
    records and the per-record state have explicit bounds.
    """

    def __init__(self, max_voxels: int, *, counter_max: int = 255) -> None:
        if max_voxels <= 0:
            raise ValueError("max_voxels must be positive")
        if counter_max <= 0:
            raise ValueError("counter_max must be positive")
        self.max_voxels = int(max_voxels)
        self.counter_max = int(counter_max)
        self._items: OrderedDict[tuple[int, int, int], list[int]] = OrderedDict()
        self.evictions = 0
        self.counter_saturations = 0
        self.peak_size = 0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return int(value[0]), int(value[1]), int(value[2]), int(value[3])

    def _ensure(self, key: tuple[int, int, int], frame_index: int) -> list[int]:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
            return value
        if len(self._items) >= self.max_voxels:
            self._items.popitem(last=False)
            self.evictions += 1
        value = [0, 0, 0, int(frame_index)]  # free, observed, hit, last frame
        self._items[key] = value
        self.peak_size = max(self.peak_size, len(self._items))
        return value

    def _increment(self, value: list[int], index: int) -> None:
        if value[index] >= self.counter_max:
            self.counter_saturations += 1
            value[index] = self.counter_max
        else:
            value[index] += 1

    def update(
        self,
        free_rows: np.ndarray,
        hit_rows: np.ndarray,
        *,
        frame_index: int,
    ) -> list[tuple[int, int, int]]:
        """Apply one scan and return touched keys in deterministic order."""
        touched: list[tuple[int, int, int]] = []
        # ``hit_rows`` is processed first.  ``free_rows`` has already had hit
        # precedence applied by the carving helper, but the order makes the
        # semantics explicit and keeps the method safe for direct unit tests.
        hit_keys = [_key(row) for row in np.asarray(hit_rows, dtype=np.int64).reshape(-1, 3)]
        hit_set = set(hit_keys)
        for voxel_key in hit_keys:
            value = self._ensure(voxel_key, frame_index)
            self._increment(value, 1)  # observed
            self._increment(value, 2)  # surface confirmation
            value[3] = int(frame_index)
            touched.append(voxel_key)
        for row in np.asarray(free_rows, dtype=np.int64).reshape(-1, 3):
            voxel_key = _key(row)
            if voxel_key in hit_set:
                continue
            value = self._ensure(voxel_key, frame_index)
            self._increment(value, 0)  # see-through/free vote
            self._increment(value, 1)  # observation
            value[3] = int(frame_index)
            touched.append(voxel_key)
        return touched

    def is_free_enough(
        self,
        key: tuple[int, int, int],
        *,
        free_fraction: float,
        free_floor: int,
    ) -> bool:
        value = self._items.get(key)
        if value is None or value[0] <= 0:
            return False
        required = max(int(free_floor), int(math.ceil(free_fraction * value[1])))
        # ``value[1]`` includes hits.  Therefore surface confirmations reduce
        # the free ratio, while same-scan hit precedence is already enforced by
        # the absence of a simultaneous free vote.
        return value[0] >= required

    def memory_bound_bytes(self) -> int:
        # This is a conservative logical bound, not a process RSS claim.  It
        # includes the fixed per-record evidence payload plus a generous bound
        # for the capped Python key/mapping entry.
        return int(self.max_voxels * 160 + 4096)


@dataclass(frozen=True)
class _BackendConfig:
    voxel_size: float = 0.30
    free_step: float = 0.30
    free_carve_margin: float = 0.60
    free_ground_margin: float = 0.25
    min_range: float = 1.0
    max_range: float = 60.0
    free_fraction: float = 0.70
    free_floor: int = 3
    rejudge_every: int = 3
    slice_budget_points: int = 20_000
    max_slices_per_frame: int = 1
    max_voxels: int = 250_000
    max_recent_points: int = 500_000
    max_recent_frames: int = 12
    max_queue_points: int = 500_000
    max_pending_voxels: int = 250_000

    def validate(self) -> None:
        if self.voxel_size <= 0.0 or self.free_step <= 0.0:
            raise ValueError("backend voxel_size and free_step must be positive")
        if self.free_carve_margin < 0.0 or self.free_ground_margin < 0.0:
            raise ValueError("backend margins must be non-negative")
        if self.min_range < 0.0 or self.max_range <= self.min_range:
            raise ValueError("backend range limits are invalid")
        if not 0.0 < self.free_fraction <= 1.0:
            raise ValueError("backend free_fraction must be in (0, 1]")
        for name in (
            "free_floor",
            "rejudge_every",
            "slice_budget_points",
            "max_slices_per_frame",
            "max_voxels",
            "max_recent_points",
            "max_recent_frames",
            "max_queue_points",
            "max_pending_voxels",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"backend {name} must be positive")


class _BoundedFreeSpaceBackend:
    """Experimental bounded asynchronous map back-end.

    ``add_map_points`` receives only points accepted by the unchanged
    front-end. ``observe_scan`` updates persistent free/hit evidence from the
    raw current scan. ``service`` schedules at most the configured number of
    bounded slices and mutates the supplied active map mask. The map itself is
    owned by the replay harness; only a bounded recent reference index is kept
    here.
    """

    def __init__(self, config: _BackendConfig | None = None) -> None:
        self.config = config or _BackendConfig()
        self.config.validate()
        self.evidence = _BoundedVoxelEvidenceStore(self.config.max_voxels)
        self._recent_frames: deque[tuple[int, list[tuple[tuple[int, int, int], np.ndarray]], int]] = deque()
        self._recent_by_voxel: dict[
            tuple[int, int, int], deque[tuple[int, np.ndarray]]
        ] = {}
        self._recent_points = 0
        self.peak_recent_points = 0
        self.peak_recent_voxels = 0

        self._pending: deque[tuple[int, int, int]] = deque()
        self._pending_set: set[tuple[int, int, int]] = set()
        self.pending_drops = 0
        self._candidate_chunks: deque[tuple[np.ndarray, tuple[int, int, int]]] = deque()
        self._candidate_points = 0
        self.peak_queue_points = 0

        self._ground_z: float | None = None
        self._update_durations_ms: list[float] = []
        self._slice_durations_ms: list[float] = []
        self._slice_points: list[int] = []
        self._backend_removed = 0
        self._frames_observed = 0

    @property
    def pending_voxels(self) -> int:
        return len(self._pending)

    @property
    def queued_points(self) -> int:
        return self._candidate_points

    def _mark_pending(self, keys: Iterable[tuple[int, int, int]]) -> None:
        for voxel_key in keys:
            if voxel_key in self._pending_set:
                continue
            if len(self._pending) >= self.config.max_pending_voxels:
                dropped = self._pending.popleft()
                self._pending_set.discard(dropped)
                self.pending_drops += 1
            self._pending.append(voxel_key)
            self._pending_set.add(voxel_key)

    def _remove_recent_frame(
        self,
        frame: tuple[int, list[tuple[tuple[int, int, int], np.ndarray]], int],
    ) -> None:
        frame_index, groups, count = frame
        self._recent_points -= count
        for voxel_key, _ in groups:
            bucket = self._recent_by_voxel.get(voxel_key)
            if not bucket:
                continue
            if bucket and bucket[0][0] == frame_index:
                bucket.popleft()
            else:
                for position, (stored_frame, _) in enumerate(bucket):
                    if stored_frame == frame_index:
                        del bucket[position]
                        break
            if not bucket:
                self._recent_by_voxel.pop(voxel_key, None)

    def add_map_points(
        self,
        points: np.ndarray,
        map_indices: np.ndarray,
        *,
        frame_index: int,
    ) -> None:
        """Index a bounded suffix of the accepted map points for re-judgment."""
        points = np.asarray(points, dtype=np.float64)
        map_indices = np.asarray(map_indices, dtype=np.int64).reshape(-1)
        if len(points) != len(map_indices):
            raise ValueError("points and map_indices must have equal length")
        if points.size == 0 or len(points) == 0:
            return
        if len(points) > self.config.max_recent_points:
            points = points[-self.config.max_recent_points :]
            map_indices = map_indices[-self.config.max_recent_points :]

        voxels = np.floor(points / self.config.voxel_size).astype(np.int64)
        packed = _pack_voxel_rows(voxels)
        unique, inverse = np.unique(packed, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        groups: list[tuple[tuple[int, int, int], np.ndarray]] = []
        start = 0
        for group_index in range(len(unique)):
            while start < len(order) and inverse[order[start]] == group_index:
                start += 1
            group_indices = map_indices[order[np.searchsorted(inverse[order], group_index, side="left") : start]]
            # ``order`` is sorted by inverse, so the search above is stable and
            # keeps map indices in source order.  The explicit slice avoids a
            # Python loop over points.
            voxel_key = _key(_unpack_voxel_keys(unique[group_index : group_index + 1])[0])
            groups.append((voxel_key, group_indices.copy()))

        # Evict before insertion so the persistent reference index never
        # exceeds either capacity, even transiently at a frame boundary.
        while self._recent_frames and (
            len(self._recent_frames) >= self.config.max_recent_frames
            or self._recent_points + len(map_indices) > self.config.max_recent_points
        ):
            self._remove_recent_frame(self._recent_frames.popleft())

        self._recent_frames.append((int(frame_index), groups, len(map_indices)))
        self._recent_points += len(map_indices)
        for voxel_key, group_indices in groups:
            self._recent_by_voxel.setdefault(voxel_key, deque()).append(
                (int(frame_index), group_indices)
            )
        self.peak_recent_points = max(self.peak_recent_points, self._recent_points)
        self.peak_recent_voxels = max(self.peak_recent_voxels, len(self._recent_by_voxel))

    def observe_scan(self, points: np.ndarray, origin: Sequence[float], *, frame_index: int) -> dict[str, int]:
        """Carve one scan and update bounded evidence; return scan counts."""
        started = time.perf_counter()
        points = np.asarray(points, dtype=np.float64)
        origin = np.asarray(origin, dtype=np.float64)
        finite_z = points[:, 2][np.isfinite(points[:, 2])] if points.size else np.empty(0)
        if finite_z.size:
            self._ground_z = float(np.percentile(finite_z, 2.0))
        recent_keys = (
            np.asarray(tuple(self._recent_by_voxel), dtype=np.int64).reshape(-1, 3)
            if self._recent_by_voxel
            else np.empty((0, 3), dtype=np.int64)
        )
        recent_packed = np.unique(_pack_voxel_rows(recent_keys)) if len(recent_keys) else np.empty(0, dtype=_KEY_DTYPE)
        free, hit = _carve_free_voxels(
            points,
            origin,
            voxel_size=self.config.voxel_size,
            step=self.config.free_step,
            carve_margin=self.config.free_carve_margin,
            ground_margin=self.config.free_ground_margin,
            min_range=self.config.min_range,
            max_range=self.config.max_range,
            target_keys=recent_packed,
        )
        touched = self.evidence.update(free, hit, frame_index=frame_index)
        self._mark_pending(touched)
        self._frames_observed += 1
        self._update_durations_ms.append((time.perf_counter() - started) * 1000.0)
        return {
            "free_voxels": int(len(free)),
            "hit_voxels": int(len(hit)),
            "touched_voxels": int(len(touched)),
        }

    def _enqueue_pending_candidates(self) -> None:
        while self._pending:
            voxel_key = self._pending.popleft()
            self._pending_set.discard(voxel_key)
            if self.evidence.get(voxel_key) is None:
                continue
            bucket = self._recent_by_voxel.get(voxel_key)
            if not bucket:
                continue
            for _, map_indices in bucket:
                available = self.config.max_queue_points - self._candidate_points
                if available <= 0:
                    self.pending_drops += 1
                    return
                chunk = map_indices[:available]
                if len(chunk):
                    self._candidate_chunks.append((chunk, voxel_key))
                    self._candidate_points += len(chunk)
                    self.peak_queue_points = max(self.peak_queue_points, self._candidate_points)

    def _run_slices(
        self,
        map_points: np.ndarray,
        active_mask: np.ndarray,
        *,
        max_slices: int,
    ) -> dict[str, Any]:
        durations: list[float] = []
        processed_points = 0
        removed_points = 0
        for _ in range(max(0, int(max_slices))):
            if not self._candidate_chunks:
                break
            started = time.perf_counter()
            remaining = self.config.slice_budget_points
            slice_points = 0
            slice_removed = 0
            while remaining > 0 and self._candidate_chunks:
                indices, voxel_key = self._candidate_chunks[0]
                take = min(remaining, len(indices))
                current = indices[:take]
                if take == len(indices):
                    self._candidate_chunks.popleft()
                else:
                    self._candidate_chunks[0] = (indices[take:], voxel_key)
                self._candidate_points -= take
                remaining -= take
                slice_points += take
                if not self.evidence.is_free_enough(
                    voxel_key,
                    free_fraction=self.config.free_fraction,
                    free_floor=self.config.free_floor,
                ):
                    continue
                keepable = np.ones(take, dtype=bool)
                if self._ground_z is not None:
                    keepable &= map_points[current, 2] > self._ground_z + self.config.free_ground_margin
                eligible = current[keepable]
                if eligible.size:
                    was_active = active_mask[eligible].copy()
                    active_mask[eligible] = False
                    slice_removed += int(np.count_nonzero(was_active))
            elapsed = (time.perf_counter() - started) * 1000.0
            durations.append(elapsed)
            self._slice_durations_ms.append(elapsed)
            self._slice_points.append(slice_points)
            processed_points += slice_points
            removed_points += slice_removed
            self._backend_removed += slice_removed
        return {
            "slices": len(durations),
            "processed_points": processed_points,
            "removed_points": removed_points,
            "durations_ms": durations,
        }

    def service(
        self,
        map_points: np.ndarray,
        active_mask: np.ndarray,
        *,
        frame_index: int,
        force: bool = False,
    ) -> dict[str, Any]:
        """Schedule/replay at most bounded slices for one frame."""
        if force or (int(frame_index) + 1) % self.config.rejudge_every == 0:
            self._enqueue_pending_candidates()
        return self._run_slices(
            np.asarray(map_points, dtype=np.float64),
            np.asarray(active_mask, dtype=bool),
            max_slices=self.config.max_slices_per_frame,
        )

    def drain(self, map_points: np.ndarray, active_mask: np.ndarray) -> dict[str, Any]:
        """Finish queued evidence after the last frame for final-map scoring."""
        self._enqueue_pending_candidates()
        slices = int(math.ceil(self._candidate_points / self.config.slice_budget_points))
        return self._run_slices(map_points, active_mask, max_slices=slices)

    def memory_report(self) -> dict[str, int]:
        evidence_bound = self.evidence.memory_bound_bytes()
        recent_bound = self.config.max_recent_points * 40 + self.config.max_recent_points * 8
        queue_bound = self.config.max_queue_points * 16
        pending_bound = self.config.max_pending_voxels * 40
        return {
            "logical_state_bound_bytes": int(evidence_bound + recent_bound + queue_bound + pending_bound),
            "evidence_capacity_voxels": int(self.config.max_voxels),
            "evidence_peak_voxels": int(self.evidence.peak_size),
            "evidence_evictions": int(self.evidence.evictions),
            "recent_point_capacity": int(self.config.max_recent_points),
            "recent_point_peak": int(self.peak_recent_points),
            "recent_voxel_peak": int(self.peak_recent_voxels),
            "candidate_queue_capacity_points": int(self.config.max_queue_points),
            "candidate_queue_peak_points": int(self.peak_queue_points),
            "pending_voxel_capacity": int(self.config.max_pending_voxels),
            "pending_drops": int(self.pending_drops),
        }

    @staticmethod
    def _summary(values: Sequence[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        return {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "status": "experimental_private",
            "config": {
                "voxel_size": self.config.voxel_size,
                "free_step": self.config.free_step,
                "free_carve_margin": self.config.free_carve_margin,
                "free_ground_margin": self.config.free_ground_margin,
                "min_range": self.config.min_range,
                "max_range": self.config.max_range,
                "free_fraction": self.config.free_fraction,
                "free_floor": self.config.free_floor,
                "rejudge_every_frames": self.config.rejudge_every,
                "slice_budget_points": self.config.slice_budget_points,
                "max_slices_per_frame": self.config.max_slices_per_frame,
                "max_recent_frames": self.config.max_recent_frames,
            },
            "frames_observed": int(self._frames_observed),
            "backend_removed_points": int(self._backend_removed),
            "amortized_cost": {
                "backend_update_per_frame": self._summary(self._update_durations_ms),
                "slice": self._summary(self._slice_durations_ms),
                "slice_count": len(self._slice_durations_ms),
                "slice_budget_points": int(self.config.slice_budget_points),
                "max_slice_points": int(max(self._slice_points) if self._slice_points else 0),
                "total_backend_ms": float(sum(self._update_durations_ms) + sum(self._slice_durations_ms)),
                "amortized_backend_ms_per_input_frame": float(
                    (sum(self._update_durations_ms) + sum(self._slice_durations_ms))
                    / max(1, self._frames_observed)
                ),
            },
            "ground": {
                "source": "latest-scan 2nd percentile for re-judgment; local XY minimum grid for carving",
                "latest_ground_z": self._ground_z,
            },
            "memory": self.memory_report(),
        }


__all__ = ["_BackendConfig", "_BoundedFreeSpaceBackend", "_BoundedVoxelEvidenceStore"]
