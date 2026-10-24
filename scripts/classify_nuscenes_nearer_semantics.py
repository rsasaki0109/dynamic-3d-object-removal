#!/usr/bin/env python3
"""Join official lidarseg labels to saved nearer-return events; no inference changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dynamic_object_removal as core
from scripts.attribute_nuscenes_nearer_returns import RELATIONS
from scripts.validate_frozen_av2_suite import write_new
from scripts.validate_frozen_nuscenes_suite import sha256

ARCHIVE_URL = "https://motional-nuscenes.s3.amazonaws.com/public/nuscenes-lidarseg-v1.0/nuScenes-lidarseg-all-v1.0.tar.bz2"


def align_labels(raw, labels, ground_z, min_distance):
    if raw.ndim != 2 or raw.shape[1] != 5 or labels.shape != (len(raw),):
        raise ValueError("raw points and lidarseg labels must be row-aligned")
    points = raw[:, :3].astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError("nonfinite raw coordinates")
    keep = (points[:, 2] > ground_z) & ~((np.abs(points[:, 0]) < min_distance) &
                                              (np.abs(points[:, 1]) < min_distance))
    return points[keep], labels[keep], np.flatnonzero(keep)


def counts(labels, categories):
    if not set(np.unique(labels).tolist()) <= set(categories):
        raise ValueError("label ID absent from official category mapping")
    return {name: int((labels == index).sum()) for index, name in categories.items()}


def summarize(labels, relation, own_scan, categories):
    if not (labels.shape == relation.shape == own_scan.shape):
        raise ValueError("event arrays differ in length")
    if np.any(relation < 0) or np.any(relation >= len(RELATIONS)):
        raise ValueError("unknown annotation relation")
    return {population: {
        "events": int(mask.sum()), "semantic_counts": counts(labels[mask], categories),
        "outside_boxes_semantic_counts": counts(labels[mask & (relation == 3)], categories),
        "per_relation_semantic_counts": {name: counts(labels[mask & (relation == i)], categories)
                                         for i, name in enumerate(RELATIONS)}}
        for population, mask in {"all_queries": np.ones(len(labels), bool),
                                 "same_acquisition_scan": own_scan,
                                 "other_query_scans": ~own_scan}.items()}


def analyze_scene(manifest_path, record, events_path, point_trace_path, data_root, label_root,
                  label_lookup, categories, output):
    if sha256(events_path) != record["events_sha256"] or sha256(point_trace_path) != record["source_evidence_trace_sha256"]:
        raise ValueError("saved event or point trace changed")
    manifest = json.loads(manifest_path.read_text())
    scans, _ = core._load_scan_manifest(manifest_path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    gt = np.concatenate([np.load(manifest_path.parent / f["point_labels"]) for f in manifest["frames"]]).astype(bool)
    hashes = {"manifest_sha256": sha256(manifest_path), "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    if hashes != record["input_hashes"] or manifest["scene"] != record["scene"]:
        raise ValueError("source input fingerprints differ")
    aligned, raw_rows, sources = [], [], []
    for frame in manifest["frames"]:
        source = data_root / frame["source_cloud"]
        if sha256(source) != frame["source_cloud_sha256"]:
            raise ValueError("raw source point cloud changed")
        raw = np.fromfile(source, dtype=np.float32).reshape(-1, 5)
        label_record = label_lookup[frame["sample_data_token"]]
        label_path = label_root / label_record["filename"]
        labels = np.fromfile(label_path, dtype=np.uint8)
        counts(labels, categories)  # Verify the explicit label mapping, including discarded rows.
        local, filtered_labels, rows = align_labels(raw, labels, manifest["preprocessing"]["ground_z_sensor"],
                                                    manifest["preprocessing"]["min_distance"])
        np.testing.assert_array_equal(local, np.load(manifest_path.parent / frame["cloud"]))
        aligned.append(filtered_labels); raw_rows.append(rows)
        sources.append({"sample_data_token": frame["sample_data_token"], "filename": label_record["filename"],
                        "sha256": sha256(label_path), "raw_rows": len(raw), "retained_rows": len(rows)})
    semantic = np.concatenate(aligned)
    with np.load(events_path) as events, np.load(point_trace_path) as trace:
        np.testing.assert_array_equal(points, trace["xyz"])
        np.testing.assert_array_equal(gt, trace["gt_dynamic"])
        index = events["nearest_map_index"]
        if np.any(index < 0) or np.any(index >= len(semantic)):
            raise ValueError("nearest event row outside input map")
        event_semantic = semantic[index]
        populations = summarize(event_semantic, events["relation"], events["same_acquisition_scan"], categories)
        for population, values in populations.items():
            if values["events"] != record["populations"][population]["events"]:
                raise ValueError("event population changed")
            for relation in RELATIONS:
                if sum(values["per_relation_semantic_counts"][relation].values()) != record["populations"][population]["counts"][relation]:
                    raise ValueError("annotation relation count changed")
        common_counts = counts(semantic[trace["group_id"] == 5], categories)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "semantic_alignment.npz", semantic_per_map_point=semantic,
                            raw_source_row=np.concatenate(raw_rows), nearer_event_semantic=event_semantic)
    result = {"scene": record["scene"], "input_hashes": hashes, "populations": populations,
              "common_miss_target_semantic_counts": common_counts, "label_sources": sources,
              "filtered_clouds_equal_raw_selection": True,
              "alignment_sha256": sha256(output / "semantic_alignment.npz")}
    write_new(output / "report.json", result)
    return result


def render(records, categories, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(categories.values())
    totals = {name: sum(r["populations"]["all_queries"]["outside_boxes_semantic_counts"][name] for r in records) for name in names}
    names = sorted((name for name in names if totals[name]), key=totals.get, reverse=True)
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for i, record in enumerate(records):
        values = record["populations"]["all_queries"]["outside_boxes_semantic_counts"]
        n = sum(values.values())
        ax.barh(np.arange(len(names)) + (i - 1) * .25,
                [100 * values[name] / n if n else 0 for name in names], height=.24,
                label=f"{record['scene']} (n={n:,})")
    ax.set_yticks(np.arange(len(names)), names); ax.invert_yaxis()
    ax.set_xlabel("Share of outside-box nearer-return point/scan events (%)")
    ax.set_title("Official lidarseg classes of nearer returns outside all annotation boxes\n"
                 "Frozen baseline; point/scan events repeat points; semantic classes do not establish motion")
    ax.legend(); ax.set_xlim(0, 100)
    fig.savefig(output / "outside_box_semantics.png", dpi=160)
    fig.savefig(output / "outside_box_semantics.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attribution-report", "attribution-root", "point-evidence-root", "manifest-root", "data-root", "lidarseg-root", "label-archive", "output", "report-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.report_json.exists():
            raise ValueError("output/report exists; choose new paths")
        attribution = json.loads(args.attribution_report.read_text())
        for name, expected in attribution["code_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError("source attribution code changed")
        category_path = args.lidarseg_root / "v1.0-trainval/category.json"
        lookup_path = args.lidarseg_root / "v1.0-trainval/lidarseg.json"
        category_rows = json.loads(category_path.read_text())
        categories = {row["index"]: row["name"] for row in sorted(category_rows, key=lambda x: x["index"])}
        if len(categories) != len(category_rows) or len(set(categories.values())) != len(categories):
            raise ValueError("duplicate category index/name")
        lookup_rows = json.loads(lookup_path.read_text())
        lookup = {row["token"]: row for row in lookup_rows}
        if len(lookup) != len(lookup_rows):
            raise ValueError("duplicate lidarseg sample-data token")
        records = []
        for record in attribution["scene_results"]:
            scene = record["scene"]
            records.append(analyze_scene(args.manifest_root / scene / "manifest.json", record,
                args.attribution_root / scene / "nearer_return_events.npz",
                args.point_evidence_root / scene / "point_scan_evidence.npz", args.data_root,
                args.lidarseg_root, lookup, categories, args.output / scene))
            print(scene + ": raw-row alignment and saved event counts match", flush=True)
        aggregate = {population: {field: {name: sum(r["populations"][population][field][name] for r in records)
                                          for name in categories.values()}
                                  for field in ("semantic_counts", "outside_boxes_semantic_counts")}
                     for population in ("all_queries", "same_acquisition_scan", "other_query_scans")}
        result = {"attribution_report_sha256": sha256(args.attribution_report),
                  "status": "Post-result official semantic-label attribution; no filter or GT changes.",
                  "source_archive_url": ARCHIVE_URL, "source_archive_sha256": sha256(args.label_archive),
                  "category_metadata_sha256": sha256(category_path), "lidarseg_metadata_sha256": sha256(lookup_path),
                  "code_sha256": sha256(Path(__file__)), "scene_results": records, "aggregate_counts": aggregate,
                  "common_miss_target_semantic_counts": {name: sum(r["common_miss_target_semantic_counts"][name] for r in records) for name in categories.values()},
                  "limitations": "Official lidarseg labels identify semantic classes, not motion. Moving-box GT is retained unchanged and may contain points of other semantic classes. Raw cloud and label sample-data tokens and row counts must align; z/near-sensor exclusions preserve original raw row indices. Counts are repeated nearer-return events unless marked target points. Three correlated scenes, undeskewed/offline geometry; nearest angular bins are not exact rays. Labels are diagnostic only, never inference inputs."}
        if args.render:
            render(records, categories, args.output)
        write_new(args.report_json, result)
        print(json.dumps(aggregate["all_queries"]["outside_boxes_semantic_counts"], indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
