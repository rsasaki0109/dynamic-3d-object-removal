#!/usr/bin/env python3
"""Experimental learning-free three-scan range residual + Beta evidence ablation.

This is deliberately not a public library mode. Frame t is classified only after
frame t+1 arrives (one-frame delay). Previous/next scans cast see-through or surface
votes on frame-t points, a Beta posterior converts those votes to dynamic probability,
and connected range-image components suppress isolated candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
import dynamic_object_removal as core  # noqa: E402
from scripts import run_online_benchmark as online  # noqa: E402


@dataclass(frozen=True)
class _Frame:
    index: int
    timestamp: float
    local_points: np.ndarray
    fixed_points: np.ndarray
    origin: np.ndarray
    gt_dynamic: np.ndarray


def _cluster_range_candidates(
    points: np.ndarray,
    origin: np.ndarray,
    candidate: np.ndarray,
    *,
    h_res_deg: float,
    v_res_deg: float,
    min_cluster_points: int,
) -> np.ndarray:
    """Keep candidate points in 8-connected occupied range-image components."""
    candidate = np.asarray(candidate, dtype=bool)
    if not candidate.any() or min_cluster_points <= 1:
        return candidate.copy()
    _, col, row, valid = core._spherical_pixels(points, origin, h_res_deg, v_res_deg)
    n_cols = int(np.ceil(360.0 / h_res_deg))
    n_rows = int(np.ceil(180.0 / v_res_deg))
    chosen = np.flatnonzero(candidate & valid)
    pixel = core._pixel_indices(col[chosen], row[chosen], n_cols, n_rows)
    members: dict[int, list[int]] = {}
    for point_index, pixel_index in zip(chosen.tolist(), pixel.tolist()):
        members.setdefault(pixel_index, []).append(point_index)

    keep = np.zeros(len(points), dtype=bool)
    unseen = set(members)
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            current_row, current_col = divmod(current, n_cols)
            for dr in (-1, 0, 1):
                next_row = current_row + dr
                if not 0 <= next_row < n_rows:
                    continue
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    next_col = (current_col + dc) % n_cols
                    neighbor = next_row * n_cols + next_col
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
                        component.append(neighbor)
        count = sum(len(members[p]) for p in component)
        if count >= min_cluster_points:
            for p in component:
                keep[members[p]] = True
    return keep


def range_prob_mask(
    previous: _Frame,
    target: _Frame,
    following: _Frame,
    *,
    h_res_deg: float,
    v_res_deg: float,
    range_margin: float,
    alpha_prior: float,
    beta_prior: float,
    probability_threshold: float,
    min_dynamic_votes: int,
    min_cluster_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return experimental dynamic mask and per-point Beta posterior mean."""
    dynamic_votes = np.zeros(len(target.fixed_points), dtype=np.int16)
    surface_votes = np.zeros(len(target.fixed_points), dtype=np.int16)
    for neighbor in (previous, following):
        seen, surface = core._visibility_votes(
            target.fixed_points,
            neighbor.fixed_points,
            neighbor.origin,
            h_res_deg,
            v_res_deg,
            range_margin,
        )
        dynamic_votes += seen.astype(np.int16)
        surface_votes += surface.astype(np.int16)
    alpha = float(alpha_prior) + dynamic_votes
    beta = float(beta_prior) + surface_votes
    probability = alpha / (alpha + beta)
    candidate = (
        (dynamic_votes >= max(1, int(min_dynamic_votes)))
        & (probability >= float(probability_threshold))
    )
    clustered = _cluster_range_candidates(
        target.fixed_points,
        target.origin,
        candidate,
        h_res_deg=h_res_deg,
        v_res_deg=v_res_deg,
        min_cluster_points=min_cluster_points,
    )
    return clustered, probability


def _load_frames(manifest: dict[str, Any], root: Path, args: argparse.Namespace) -> list[_Frame]:
    payloads = manifest.get("frames")
    if not isinstance(payloads, list) or len(payloads) < 3:
        raise ValueError("range_prob requires at least three manifest frames")
    rng = np.random.default_rng(args.seed)
    frames = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict) or "cloud" not in payload:
            raise ValueError(f"invalid frame {index}")
        points = core.load_points(online._resolve_path(root, payload["cloud"]), fmt="auto")
        gt = online._load_gt_mask(payload, points, root)
        pose_payload = payload.get("pose")
        if pose_payload is None:
            raise ValueError(f"frame {index} has no pose")
        pose = online._perturb_pose(
            online._pose_from_payload(pose_payload),
            translation_sigma=args.pose_noise_translation,
            yaw_sigma_deg=args.pose_noise_yaw,
            rng=rng,
        )
        frames.append(_Frame(
            index=index,
            timestamp=float(payload.get("timestamp_sec", index)),
            local_points=points,
            fixed_points=points @ pose.rotation.T + pose.translation,
            origin=pose.translation,
            gt_dynamic=gt,
        ))
    return frames


def run_ablation(manifest: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    frames = _load_frames(manifest, root, args)
    predicted = [np.zeros(len(frame.local_points), dtype=bool) for frame in frames]
    probability = [np.full(len(frame.local_points), 0.5) for frame in frames]
    latencies = [0.0] * len(frames)
    # Sequential one-pass semantics: frame t is classified when t+1 is available.
    for index in range(1, len(frames) - 1):
        start = time.perf_counter()
        predicted[index], probability[index] = range_prob_mask(
            frames[index - 1], frames[index], frames[index + 1],
            h_res_deg=args.h_res,
            v_res_deg=args.v_res,
            range_margin=args.range_margin,
            alpha_prior=args.alpha_prior,
            beta_prior=args.beta_prior,
            probability_threshold=args.probability_threshold,
            min_dynamic_votes=args.min_dynamic_votes,
            min_cluster_points=args.min_cluster_points,
        )
        latencies[index] = (time.perf_counter() - start) * 1000.0

    evaluated_pred = np.concatenate(predicted[1:-1])
    evaluated_gt = np.concatenate([f.gt_dynamic for f in frames[1:-1]])
    all_pred = np.concatenate(predicted)
    all_gt = np.concatenate([f.gt_dynamic for f in frames])
    interior_latency = latencies[1:-1]
    profile = manifest.get("sensor_profile") if isinstance(manifest.get("sensor_profile"), dict) else {}
    input_contract_satisfied = bool(profile.get("deskewed", False))
    rate_hz = float(args.rate_hz or profile.get("rate_hz") or 0.0)
    period_ms = 1000.0 / rate_hz if rate_hz > 0.0 else None
    per_frame = []
    for frame, pred, prob, latency in zip(frames, predicted, probability, latencies):
        metrics = bench.compute_accuracy_metrics(pred, frame.gt_dynamic)
        per_frame.append({
            "index": frame.index,
            "timestamp_sec": frame.timestamp,
            "boundary_unscored": frame.index in {0, len(frames) - 1},
            "points": len(pred),
            "removed_points": int(np.count_nonzero(pred)),
            "mean_dynamic_probability": float(prob.mean()) if len(prob) else 0.0,
            "filter_latency_ms": latency,
            "metrics": metrics,
        })
    return {
        "task": "online_moving_object_segmentation",
        "algorithm": "range_prob_experimental",
        "status": "experimental_not_promoted",
        "delay_frames": 1,
        "boundary_unscored_frames": 2,
        "frames": len(frames),
        "evaluated_frames": len(frames) - 2,
        "evaluated_metrics": bench.compute_accuracy_metrics(evaluated_pred, evaluated_gt),
        "all_frame_metrics_with_boundary_fail_open": bench.compute_accuracy_metrics(all_pred, all_gt),
        "filter_latency": online._latency_summary(interior_latency),
        "period_ms": period_ms,
        "deadline_misses": (
            int(np.count_nonzero(np.asarray(interior_latency) > period_ms))
            if period_ms is not None else None
        ),
        "sensor_profile": profile,
        "deskew_input_contract_satisfied": input_contract_satisfied,
        "promotion_eligible": input_contract_satisfied,
        "pose_noise": {
            "translation_sigma_m": args.pose_noise_translation,
            "yaw_sigma_deg": args.pose_noise_yaw,
            "seed": args.seed,
        },
        "config": {
            "h_res_deg": args.h_res,
            "v_res_deg": args.v_res,
            "range_margin": args.range_margin,
            "alpha_prior": args.alpha_prior,
            "beta_prior": args.beta_prior,
            "probability_threshold": args.probability_threshold,
            "min_dynamic_votes": args.min_dynamic_votes,
            "min_cluster_points": args.min_cluster_points,
        },
        "per_frame": per_frame,
    }


def compare_with_range_baseline(
    result: dict[str, Any], baseline_summary: dict[str, Any]
) -> dict[str, Any]:
    """Compare against the same interior frames of an online range summary."""
    scenarios = baseline_summary.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("baseline summary has no scenarios")
    frames = scenarios[0].get("per_frame")
    if not isinstance(frames, list) or len(frames) < 3:
        raise ValueError("baseline summary needs per-frame confusion counts")
    interior = frames[1:-1]
    counts = {
        key: sum(int(frame[key]) for frame in interior)
        for key in ("true_positive", "false_positive", "false_negative", "true_negative")
    }
    tp, fp, fn, tn = (counts[k] for k in (
        "true_positive", "false_positive", "false_negative", "true_negative"
    ))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    static = tn / (tn + fp) if tn + fp else 1.0
    baseline = {**counts, "precision": precision, "recall": recall, "f1": f1,
                "static_preservation": static}
    candidate = result["evaluated_metrics"]
    period = result.get("period_ms")
    latency_pass = period is None or result["filter_latency"]["p95_ms"] < period
    f1_delta = candidate["f1"] - baseline["f1"]
    static_delta = candidate["static_preservation"] - baseline["static_preservation"]
    # Improvement in one metric cannot buy an arbitrary collapse in the other.
    # A one-point absolute tolerance is the experimental non-regression guard.
    accuracy_pass = (
        (f1_delta > 0.0 or static_delta > 0.0)
        and f1_delta >= -0.01
        and static_delta >= -0.01
    )
    return {
        "baseline_algorithm": baseline_summary.get("algorithm"),
        "baseline_interior_metrics": baseline,
        "delta_f1": f1_delta,
        "delta_static_preservation": static_delta,
        "metric_non_regression_tolerance": 0.01,
        "accuracy_gate_pass": accuracy_pass,
        "latency_gate_pass": latency_pass,
        "input_contract_gate_pass": result["deskew_input_contract_satisfied"],
        "single_dataset_gate_pass": bool(
            accuracy_pass and latency_pass and result["deskew_input_contract_satisfied"]
        ),
        "promotion_ready": False,
        "reason": "Cross-sensor and pose-noise gates must be evaluated separately.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--h-res", type=float, default=1.0)
    parser.add_argument("--v-res", type=float, default=1.0)
    parser.add_argument("--range-margin", type=float, default=0.5)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--beta-prior", type=float, default=1.0)
    parser.add_argument("--probability-threshold", type=float, default=0.7)
    parser.add_argument("--min-dynamic-votes", type=int, default=2)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--pose-noise-translation", type=float, default=0.0)
    parser.add_argument("--pose-noise-yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate-hz", type=float, default=None)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.alpha_prior <= 0.0 or args.beta_prior <= 0.0:
        raise SystemExit("Beta priors must be positive")
    if not 0.0 <= args.probability_threshold <= 1.0:
        raise SystemExit("--probability-threshold must be between 0 and 1")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = run_ablation(manifest, manifest_path.parent, args)
    if args.baseline_summary is not None:
        baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        result["baseline_comparison"] = compare_with_range_baseline(result, baseline)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = result["evaluated_metrics"]
    latency = result["filter_latency"]
    print(
        f"range_prob: frames={result['evaluated_frames']} precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
        f"static={metrics['static_preservation']:.3f} p95={latency['p95_ms']:.3f}ms"
    )
    print(f"Saved: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
