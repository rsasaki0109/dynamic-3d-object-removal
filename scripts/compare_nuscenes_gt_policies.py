#!/usr/bin/env python3
"""Export named GT policies and compare them with identical frozen predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import dynamic_object_removal as core
from scripts.experiment_neighbor_columns import clean
from scripts.nuscenes_gt_policies import POLICIES, definition, select_tracks
from scripts.run_nuscenes_benchmark import _yaw_from_quat
from scripts.validate_frozen_av2_suite import MODES, write_new
from scripts.validate_frozen_nuscenes_suite import selected_tables, sha256
from scripts.visualize_nuscenes_errors import moving_instances


def labels_from_tracks(scans, selected, annotations, moving, margin):
    if len(scans) != len(selected) or not scans:
        raise ValueError("scans and selected frames must be nonempty and row-aligned")
    chunks = []
    for sample, (scan, _) in zip(selected, scans):
        boxes = []
        for a in annotations.get(sample, []):
            if a["instance_token"] not in moving:
                continue
            width, length, height = a["size"]
            boxes.append(core.DetectionBox(center=np.asarray(a["translation"]), size=np.array([length, width, height]),
                                            yaw=_yaw_from_quat(*a["rotation"])))
        chunks.append(bench.dynamic_gt_mask(scan, boxes, margin=(margin,) * 3))
    return np.concatenate(chunks)


def export_manifest(source_path, manifest, gt, frame_sizes, output, gt_definition, parent_plan):
    if len(frame_sizes) != len(manifest["frames"]) or gt.ndim != 1 or sum(frame_sizes) != len(gt):
        raise ValueError("label rows and frame sizes differ")
    output.mkdir(parents=True, exist_ok=False)
    clone = json.loads(json.dumps(manifest))
    clone.pop("frozen_plan_sha256", None)
    clone.update(source_frozen_plan_sha256=parent_plan, parent_manifest_sha256=sha256(source_path),
                 gt_definition=gt_definition)
    cursor = 0
    for i, (frame, size) in enumerate(zip(clone["frames"], frame_sizes)):
        frame["cloud"] = os.path.relpath((source_path.parent / frame["cloud"]).resolve(), output.resolve())
        frame["point_labels"] = f"labels_{i:04d}.npy"
        np.save(output / frame["point_labels"], gt[cursor:cursor + size].astype(np.uint8))
        cursor += size
    if cursor != len(gt):
        raise ValueError("label rows and frame sizes differ")
    np.save(output / "gt.npy", gt)
    write_new(output / "manifest.json", clone)


def compare_scene(path, reference, tables, selected, attribute_names, categories, instances,
                  policies, plan, output):
    manifest = json.loads(path.read_text())
    if manifest["scene"] != reference["scene"] or [f["sample_token"] for f in manifest["frames"]] != selected:
        raise ValueError("frozen sample selection differs")
    scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    labels = [np.load(path.parent / f["point_labels"]) for f in manifest["frames"]]
    if any(g.shape != (len(scan),) or not np.isin(g, [0, 1]).all() for g, (scan, _) in zip(labels, scans)):
        raise ValueError("invalid source GT")
    legacy = np.concatenate(labels).astype(bool)
    hashes = {"manifest_sha256": sha256(path), "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(legacy.tobytes()).hexdigest()}
    if hashes != reference["input_hashes"]:
        raise ValueError("frozen input fingerprints differ")
    annotations = defaultdict(list)
    for a in tables["ann"]:
        annotations[a["sample_token"]].append(a)
    legacy_selection = select_tracks(annotations, selected, policy="xyz", threshold=plan["moving_threshold_m"],
                                     attribute_names=attribute_names)
    if legacy_selection["moving"] != set(moving_instances(annotations, selected, plan["moving_threshold_m"])):
        raise ValueError("legacy track selector differs from original benchmark")
    reconstructed = labels_from_tracks(scans, selected, annotations, legacy_selection["moving"], plan["box_margin_m"])
    np.testing.assert_array_equal(reconstructed, legacy)
    params = reference["parameters"]
    _, kr = core.clean_map_by_visibility(points, scans, **params["range"])
    _, ks = core.clean_map_by_scan_ratio(points, scans, **params["scan_ratio"])
    predictions = {}
    for mode in MODES:
        sr_dynamic, inferred = clean(points, scans, params["scan_ratio"], mode, params["range"])
        if mode == "none":
            np.testing.assert_array_equal(sr_dynamic, ~ks)
        prediction = ~kr & sr_dynamic
        if bench.compute_accuracy_metrics(prediction, legacy) != reference["results"][mode]["metrics"]:
            raise ValueError("prediction metrics differ from frozen comparison")
        if inferred != reference["results"][mode]["inferred_point_scan_observations"]:
            raise ValueError("candidate evidence population changed")
        predictions[mode] = prediction
    output.mkdir(parents=True, exist_ok=False)
    for mode, mask in predictions.items():
        np.save(output / f"prediction_{mode}.npy", mask)
    decisions = {}
    category = lambda token: categories[instances[token]["category_token"]]["name"]
    for policy in policies:
        decision = select_tracks(annotations, selected, policy=policy, threshold=plan["moving_threshold_m"],
                                 attribute_names=attribute_names)
        gt = labels_from_tracks(scans, selected, annotations, decision["moving"], plan["box_margin_m"])
        if policy == "xy" and np.any(gt & ~legacy):
            raise ValueError("XY displacement labels must be a subset of legacy XYZ labels")
        gt_definition = definition(policy, plan["moving_threshold_m"], plan["box_margin_m"])
        export_manifest(path, manifest, gt, [len(p) for p, _ in scans], output / policy, gt_definition, plan["parent_plan_sha256"])
        removed = sorted(legacy_selection["moving"] - decision["moving"])
        added = sorted(decision["moving"] - legacy_selection["moving"])
        decisions[policy] = {"gt_definition": gt_definition, "moving_gt_points": int(gt.sum()),
            "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest(), "moving_tracks": sorted(decision["moving"]),
            "removed_track_categories": {t: category(t) for t in removed},
            "added_track_categories": {t: category(t) for t in added},
            "label_changes_from_legacy": {"moving_to_static": int((legacy & ~gt).sum()),
                                          "static_to_moving": int((~legacy & gt).sum())},
            "track_decisions": {t: dict(d, category=category(t)) for t, d in decision["tracks"].items()},
            "metrics_with_identical_predictions": {mode: bench.compute_accuracy_metrics(mask, gt) for mode, mask in predictions.items()}}
    return {"scene": reference["scene"], "input_hashes": hashes, "map_points": len(points),
            "legacy_gt_reproduced_exactly": True, "frozen_metrics_and_evidence_counts_reproduced": True,
            "prediction_masks_shared_across_gt_policies": True,
            "prediction_sha256": {mode: hashlib.sha256(mask.tobytes()).hexdigest() for mode, mask in predictions.items()},
            "parameters": params, "policies": decisions}


def render(records, policies, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(records)); width = .8 / len(policies)
    for j, policy in enumerate(policies):
        dx = (j - (len(policies) - 1) / 2) * width
        axes[0].bar(x + dx, [r["policies"][policy]["moving_gt_points"] for r in records], width=width, label=policy)
        axes[1].bar(x + dx, [r["policies"][policy]["metrics_with_identical_predictions"]["none"]["f1"] for r in records], width=width, label=policy)
    for ax in axes:
        ax.set_xticks(x, [r["scene"] for r in records]); ax.legend(fontsize=8)
    axes[0].set_ylabel("Points labelled moving"); axes[0].set_title("GT definitions change the evaluated positives")
    axes[1].set_ylabel("F1 of identical native prediction"); axes[1].set_title("Changed GT only; no filter improvement")
    fig.suptitle("nuScenes GT-policy sensitivity on the same three previously examined scenes\n"
                 "No registration correction, threshold tuning or production default change")
    fig.savefig(output / "gt_policy_comparison.png", dpi=160)
    fig.savefig(output / "gt_policy_comparison.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen-results", "manifest-root", "data-root", "annotation-metadata-root", "output", "report-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["xyz"])
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        if len(args.policies) != len(set(args.policies)):
            raise ValueError("policies must be unique")
        if args.output.exists() or args.report_json.exists():
            raise ValueError("output/report exists; choose new paths")
        frozen = json.loads(args.frozen_results.read_text()); plan = dict(frozen["plan"])
        if plan["dataset"] != "nuscenes-trainval" or [r["scene"] for r in frozen["scene_results"]] != plan["scenes"]:
            raise ValueError("requires the complete frozen trainval scene set")
        for name, digest in plan["code_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError("frozen filter/exporter code changed")
        for name, digest in plan["metadata_sha256"].items():
            if sha256(args.data_root / plan["metadata_version"] / (name + ".json")) != digest:
                raise ValueError("frozen metadata changed")
        plan["parent_plan_sha256"] = frozen["plan_sha256"]
        extra = args.annotation_metadata_root / plan["metadata_version"]
        load = lambda n: {r["token"]: r for r in json.loads((extra / (n + ".json")).read_text())}
        attributes, categories, instances = load("attribute"), load("category"), load("instance")
        attribute_names = {k: v["name"] for k, v in attributes.items()}
        tables, selected = selected_tables(args.data_root, plan)
        records = []
        for baseline in frozen["scene_results"]:
            scene = baseline["scene"]
            records.append(compare_scene(args.manifest_root / scene / "manifest.json", baseline, tables, selected[scene],
                attribute_names, categories, instances, args.policies, plan, args.output / scene))
            print(scene + ": legacy GT matches; frozen metrics/evidence counts replayed; shared predictions and GT policies exported", flush=True)
        metrics = ("precision", "recall", "f1", "iou", "static_preservation")
        means = {policy: {mode: {k: float(np.mean([r["policies"][policy]["metrics_with_identical_predictions"][mode][k] for r in records]))
                                 for k in metrics} for mode in MODES} for policy in args.policies}
        result = {"task": "nuscenes_gt_policy_sensitivity", "status": "Post-audit GT definition comparison; identical filter predictions.",
            "parent_plan_sha256": frozen["plan_sha256"], "frozen_results_sha256": sha256(args.frozen_results),
            "code_sha256": {n: sha256(ROOT / n) for n in ("scripts/nuscenes_gt_policies.py", "scripts/compare_nuscenes_gt_policies.py")},
            "extra_metadata_sha256": {n: sha256(extra / (n + ".json")) for n in ("attribute", "category", "instance")},
            "policy_order": args.policies, "scene_results": records,
            "aggregate": {"included_scenes": plan["scenes"], "excluded_scenes": [], "mean_metrics": means},
            "limitations": "Previously examined one-drive sample; not prospective or representative validation. GT definitions change while all point coordinates, filter parameters and prediction masks stay fixed. Metric differences are not filter improvements. Historical candidate masks were not archived; their replay checks frozen source code, input arrays, reported metrics and inferred-observation counts. All new policies use the same saved prediction masks. Zero-positive scenes remain included with the existing zero precision/recall/F1 convention. Attribute policy is track-window-level and admits single-observation explicit movers; non-moving attributes and rider presence alone do not imply motion. No ego-Z/3D registration or deskew correction is applied. Original benchmark defaults, files and historical results are retained."}
        if args.render:
            render(records, args.policies, args.output)
        write_new(args.report_json, result)
        print(json.dumps(means, indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
