#!/usr/bin/env python3
"""Trace one annotation track's baseline votes across selected nuScenes frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dynamic_object_removal as core
from scripts.visualize_nuscenes_errors import analyze
from scripts.diagnose_common_misses import height_gates


def trace(manifest, data_root, validation_root, token=None):
    report, points, gt, keep, owner, scans = analyze(manifest, data_root, validation_root)
    track = next((r for r in report["tracks"] if r["instance_token"] == token), None) if token else report["tracks"][0]
    if track is None:
        raise ValueError("track has no attributed moving GT")
    selected = owner == track["owner_index"]
    source_frames = np.repeat(np.arange(len(scans)), [len(p) for p, _ in scans])[selected]
    manifest_frames = json.loads(manifest.read_text())["frames"]
    first_time = manifest_frames[0]["timestamp_sec"]
    times = [f["timestamp_sec"] - first_time for f in manifest_frames]
    rp, sp = report["parameters"]["range"], report["parameters"]["scan_ratio"]
    if rp["resolutions"] is not None or sp["min_votes"] is not None:
        raise ValueError("trace supports single-resolution normalized voting")
    see, surface, sr_votes, revisits = [], [], [], []
    query_frames = []
    for index, (scan, origin) in enumerate(scans):
        st, sf = core._visibility_votes(points[selected], scan, origin,
                    rp["h_res_deg"], rp["v_res_deg"], rp["range_margin"])
        dyn, obs = core._scan_ratio_dynamic(points, scan, origin,
                    sp["n_rings"], sp["n_sectors"], sp["max_range"],
                    sp["scan_ratio_threshold"], sp["min_map_height"], sp["ground_margin"])
        low, ratio, ground = height_gates(points, scan, origin, sp, obs, dyn)
        see.append(st); surface.append(sf)
        sr_votes.append(dyn[selected]); revisits.append(obs[selected])
        query_frames.append({"frame": index, "time_seconds": times[index], "range_see_through_points": int(st.sum()),
            "range_surface_points": int(sf.sum()), "scan_ratio_observed_points": int(obs[selected].sum()),
            "scan_ratio_dynamic_points": int(dyn[selected].sum()),
            "map_height_too_small": int(low[selected].sum()),
            "height_ratio_too_large": int(ratio[selected].sum()),
            "ground_reverted": int(ground[selected].sum())})
    see, surface, sr_votes, revisits = map(np.asarray, (see, surface, sr_votes, revisits))
    range_dynamic = (see.sum(axis=0) >= rp["min_see_through"]) & (surface.sum(axis=0) <= rp["max_surface_hits"])
    if rp["ground_z"] is not None:
        range_dynamic &= points[selected, 2] > rp["ground_z"]
    floor = max(1, min(sp["votes_floor"], len(scans)))
    threshold = np.maximum(floor, np.ceil(sp["votes_fraction"] * revisits.sum(axis=0)))
    sr_dynamic = sr_votes.sum(axis=0) >= threshold
    _, kr = core.clean_map_by_visibility(points, scans, **rp)
    _, ks = core.clean_map_by_scan_ratio(points, scans, **sp)
    np.testing.assert_array_equal(range_dynamic, ~kr[selected])
    np.testing.assert_array_equal(sr_dynamic, ~ks[selected])
    np.testing.assert_array_equal(range_dynamic & sr_dynamic, ~keep[selected])
    acquired = []
    for frame in range(len(scans)):
        group = source_frames == frame
        acquired.append({"frame": frame, "time_seconds": times[frame], "gt_points": int(group.sum()),
                         "removed": int((group & ~keep[selected]).sum()),
                         "missed": int((group & keep[selected]).sum())})
    count = lambda mask: int(mask.sum())
    result = {"scene": report["scene"], "track": track, "input_hashes": report["input_hashes"],
              "parameters": report["parameters"], "query_frames": query_frames, "acquisition_frames": acquired,
              "range_blockers": {"insufficient_see_through": count(see.sum(axis=0) < rp["min_see_through"]),
                                 "surface_guard": count(surface.sum(axis=0) > rp["max_surface_hits"])},
              "scan_ratio_blockers": {"no_dynamic_votes": count(sr_votes.sum(axis=0) == 0),
                                      "insufficient_revisits": count(revisits.sum(axis=0) < floor),
                                      "below_vote_threshold": count(~sr_dynamic)},
              "channel_decisions": {"both_dynamic": count(range_dynamic & sr_dynamic),
                                    "range_only": count(range_dynamic & ~sr_dynamic),
                                    "scan_ratio_only": count(~range_dynamic & sr_dynamic),
                                    "neither": count(~range_dynamic & ~sr_dynamic)},
              "limitations": "Votes are against the full accumulated map, including future scans: offline analysis, not an online replay. Rows are query scans; columns are acquisition frames. Fractions normalize by target GT points in each acquisition frame. Range blocker masks overlap. Moving-box GT does not assert per-point motion. No deskew or filter changes."}
    matrices = {}
    for name, votes in (("see_through", see), ("scan_ratio_dynamic", sr_votes)):
        values = np.full((len(scans), len(scans)), np.nan)
        for source in range(len(scans)):
            group = source_frames == source
            if group.any():
                values[:, source] = votes[:, group].mean(axis=1)
        matrices[name] = values
    return result, matrices


def render(report, matrices, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, name, title in ((axes[0, 0], "see_through", "Range: fraction seen through"),
                            (axes[0, 1], "scan_ratio_dynamic", "Scan-ratio: fraction receiving a dynamic vote")):
        shown = ax.imshow(matrices[name], vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Point acquisition frame")
        ax.set_ylabel("Voting query frame")
        ax.set_xticks(range(len(report["query_frames"]))); ax.set_yticks(range(len(report["query_frames"])))
        fig.colorbar(shown, ax=ax, label="Fraction of target GT points (0-1)")
    rows = report["acquisition_frames"]
    x = np.arange(len(rows))
    axes[1, 0].bar(x, [r["removed"] for r in rows], color="#148a58", label="Removed")
    axes[1, 0].bar(x, [r["missed"] for r in rows], bottom=[r["removed"] for r in rows], color="#e57a13", label="Missed")
    axes[1, 0].set_title("Final baseline result by point acquisition frame")
    axes[1, 0].set_xlabel("Point acquisition frame"); axes[1, 0].set_ylabel("Target GT points")
    axes[1, 0].set_xticks(x); axes[1, 0].legend()
    query = report["query_frames"]
    bottom = np.zeros(len(query))
    for name, color, label in (("scan_ratio_dynamic_points", "#148a58", "Dynamic vote"),
                              ("height_ratio_too_large", "#e57a13", "Height ratio fails"),
                              ("ground_reverted", "#9a42c8", "Ground protected"),
                              ("map_height_too_small", "#657486", "Map too flat")):
        values = np.array([r[name] for r in query])
        axes[1, 1].bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[1, 1].set_title("Scan-ratio stages among observed target map points")
    axes[1, 1].set_xlabel("Voting query frame"); axes[1, 1].set_ylabel("Observed target map points")
    axes[1, 1].set_xticks(x); axes[1, 1].legend(fontsize=8)
    t = report["track"]
    fig.suptitle(f"{report['scene']} | car track {t['instance_token'][:8]} | {t['missed']:,}/{t['gt_points']:,} missed\nOffline full-map votes; moving-box GT; no deskew; frames are 0-based\nWhite columns: no target GT acquired in that frame", fontsize=13)
    fig.savefig(output / "track_evidence.png", dpi=150)
    fig.savefig(output / "track_evidence.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--track", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    report, matrices = trace(args.manifest, args.data_root, args.validation_root, args.track)
    args.output.mkdir(parents=True, exist_ok=True)
    render(report, matrices, args.output)
    np.savez(args.output / "evidence_fractions.npz", **matrices)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("range_blockers", "scan_ratio_blockers", "channel_decisions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
