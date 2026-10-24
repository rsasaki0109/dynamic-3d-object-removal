#!/usr/bin/env python3
"""Prospective neighbor-candidate evaluation on new nuScenes trainval scenes.

Data acquisition is separate. Metadata must be official v1.0-trainval; mini
scenes are excluded. Scene availability is defined before inference by the log
names in the first 8 MiB of the public trainval01 LiDAR archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import dynamic_object_removal as core
from scripts import run_nuscenes_benchmark as nuscenes
from scripts.experiment_neighbor_columns import compare_scene
from scripts.validate_frozen_av2_suite import MODES, aggregate, write_new

META = "v1.0-trainval"
ARCHIVE_URL = "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs_lidar.tgz"
TABLES = ("scene", "log", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sample_annotation")
CODE_FILES = ("dynamic_object_removal.py", "bench.py", "scripts/run_nuscenes_benchmark.py",
              "scripts/experiment_neighbor_columns.py", "scripts/validate_frozen_av2_suite.py",
              "scripts/validate_frozen_nuscenes_suite.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_table(path, chunk_size=64 * 1024):
    """Read a JSON array incrementally, without loading multi-GB tables."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer = ""
        eof = False

        def replenish():
            nonlocal buffer, eof
            chunk = stream.read(chunk_size)
            buffer += chunk
            eof = not chunk

        def token():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                replenish()
                buffer = buffer.lstrip()
            if not buffer:
                raise ValueError(f"truncated metadata array: {path}")
            return buffer[0]

        replenish()
        if token() != "[":
            raise ValueError("metadata table must be a JSON array")
        buffer = buffer[1:]
        if token() == "]":
            buffer = buffer[1:]
        else:
            while True:
                token()
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer)
                        # A scalar may end at a chunk boundary; tables require objects.
                        if not isinstance(value, dict):
                            raise ValueError("metadata rows must be objects")
                        break
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError(f"invalid or truncated metadata: {path}") from exc
                        replenish()
                buffer = buffer[end:]
                yield value
                separator = token()
                buffer = buffer[1:]
                if separator == "]":
                    break
                if separator != "," or token() == "]":
                    raise ValueError("invalid metadata array separator")
        if buffer.strip() or stream.read().strip():
            raise ValueError("trailing data after metadata array")


def archive_logs(prefix_path):
    if prefix_path.stat().st_size != 8 * 1024 * 1024:
        raise ValueError("archive inventory must be the first 8 MiB")
    logs = set()
    try:
        with tarfile.open(prefix_path, mode="r|gz") as archive:
            for member in archive:
                if member.name.startswith("samples/LIDAR_TOP/") and member.isfile():
                    logs.add(Path(member.name).name.split("__LIDAR_TOP__")[0])
    except (EOFError, tarfile.ReadError):
        # The fixed compressed prefix ends inside a tar member by design.
        pass
    if not logs:
        raise ValueError("archive prefix contains no LiDAR keyframe inventory")
    return sorted(logs)


def select_scenes(scenes, logs, available_logs, seen, count):
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    names = [s["name"] for s in scenes]
    if len(names) != len(set(names)):
        raise ValueError("duplicate scene names")
    eligible = sorted(s["name"] for s in scenes
                      if s["name"] not in seen and logs[s["log_token"]]["logfile"] in available_logs)
    if len(eligible) < count:
        raise ValueError("not enough unexamined scenes in the frozen archive inventory")
    return eligible[:count]


def freeze(root, prefix, reference_path, count):
    source = json.loads(reference_path.read_text())
    prior = source["nuscenes_scene_results"]
    params = prior[0]["parameters"]
    if any(r["parameters"] != params for r in prior):
        raise ValueError("prior scenes do not share fixed parameters")
    if params["range"]["ground_z"] is not None or params["scan_ratio"]["min_votes"] is not None:
        raise ValueError("requires no absolute map ground guard and normalized votes")
    scenes = list(iter_table(root / META / "scene.json"))
    logs = {x["token"]: x for x in iter_table(root / META / "log.json")}
    available = archive_logs(prefix)
    seen = sorted(r["scene"] for r in prior)
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "nuscenes-trainval", "metadata_version": META,
            "archive_url": ARCHIVE_URL, "archive_prefix_sha256": sha256(prefix),
            "archive_inventory_logs": available, "previously_examined_scenes": seen,
            "scenes": select_scenes(scenes, logs, available, seen, count),
            "selection": "First lexicographic non-mini scenes whose logs occur in the fixed first 8 MiB of trainval01 LiDAR archive; no GT or result screening, no replacement.",
            "frames": 12, "stride": 3, "moving_threshold_m": 2.0, "box_margin_m": 0.25,
            "min_distance": 1.0, "sensor_ground_z": -1.6, "deskewed": False,
            "parameters": params, "parameters_source_sha256": sha256(reference_path),
            "modes": list(MODES),
            "metadata_sha256": {name: sha256(root / META / (name + ".json")) for name in TABLES},
            "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
            "aggregation": "Unweighted mean of all selected scenes, including zero-GT scenes; per-scene deltas and worst static preservation; no eligibility threshold."}


def verify(plan, plan_path, expected_digest, root):
    if sha256(plan_path) != expected_digest:
        raise ValueError("plan digest changed after freeze")
    if (plan.get("schema_version"), plan.get("dataset"), plan.get("metadata_version"), plan.get("modes")) != (1, "nuscenes-trainval", META, list(MODES)):
        raise ValueError("requires the frozen trainval plan")
    for name, expected in plan["code_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"code changed after freeze: {name}")
    if set(plan["code_sha256"]) != set(CODE_FILES) or set(plan["metadata_sha256"]) != set(TABLES):
        raise ValueError("incomplete fingerprints")
    for name, expected in plan["metadata_sha256"].items():
        if sha256(root / META / (name + ".json")) != expected:
            raise ValueError(f"source metadata changed: {name}")
    scenes = list(iter_table(root / META / "scene.json"))
    logs = {x["token"]: x for x in iter_table(root / META / "log.json")}
    if select_scenes(scenes, logs, plan["archive_inventory_logs"], plan["previously_examined_scenes"], len(plan["scenes"])) != plan["scenes"]:
        raise ValueError("frozen selection changed")


def selected_tables(root, plan):
    load = lambda name: iter_table(root / META / (name + ".json"))
    scenes = {s["name"]: s for s in load("scene") if s["name"] in plan["scenes"]}
    samples = {s["token"]: s for s in load("sample")}
    selected = {}
    for name in plan["scenes"]:
        tokens = []
        token = scenes[name]["first_sample_token"]
        for _ in range(plan["frames"] * plan["stride"]):
            if not token:
                break
            tokens.append(token)
            if samples[token]["scene_token"] != scenes[name]["token"]:
                raise ValueError("sample chain crosses scene boundary")
            token = samples[token]["next"]
        selected[name] = tokens[::plan["stride"]]
        if len(selected[name]) != plan["frames"]:
            raise ValueError("selected scene lacks 12 stride-3 frames; do not replace it")
    wanted = set(t for seq in selected.values() for t in seq)
    lidar = {d["sample_token"]: d for d in load("sample_data")
             if d["sample_token"] in wanted and d["is_key_frame"] and "LIDAR_TOP" in d["filename"]}
    if set(lidar) != wanted:
        raise ValueError("missing selected LiDAR sample metadata")
    pose_tokens = {d["ego_pose_token"] for d in lidar.values()}
    cs_tokens = {d["calibrated_sensor_token"] for d in lidar.values()}
    tables = {"scene": scenes, "sample": samples, "lidar": lidar,
              "ego": {e["token"]: e for e in load("ego_pose") if e["token"] in pose_tokens},
              "cs": {c["token"]: c for c in load("calibrated_sensor") if c["token"] in cs_tokens},
              "ann": [a for a in load("sample_annotation") if a["sample_token"] in wanted]}
    return tables, selected


def export_manifests(root, manifest_root, plan, digest, tables, selected):
    if manifest_root.exists():
        raise ValueError("manifest root exists; choose a new directory")
    for scene in plan["scenes"]:
        path = manifest_root / scene / "manifest.json"
        args = SimpleNamespace(frames=12, stride=3, moving_thresh=2.0, min_distance=1.0,
                               online_manifest=path, online_only=True)
        nuscenes._run_scene(args, root, tables, scene)
        manifest = json.loads(path.read_text())
        manifest.update(dataset="nuscenes-trainval", metadata_version=META, frozen_plan_sha256=digest)
        for frame, token in zip(manifest["frames"], selected[scene]):
            raw = tables["lidar"][token]
            frame.update(sample_token=token, sample_data_token=raw["token"],
                         source_cloud=raw["filename"], source_cloud_sha256=sha256(root / raw["filename"]))
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def evaluate_manifest(path, root, plan, digest, tables, expected):
    manifest = json.loads(path.read_text())
    if (manifest.get("dataset"), manifest.get("metadata_version"), manifest.get("frozen_plan_sha256")) != ("nuscenes-trainval", META, digest):
        raise ValueError("manifest provenance differs from frozen trainval plan")
    if [f["sample_token"] for f in manifest["frames"]] != expected:
        raise ValueError("manifest frame selection differs")
    if manifest["preprocessing"]["min_distance"] != 1 or manifest["sensor_profile"]["deskewed"] is not False:
        raise ValueError("wrong close-point selection or deskew status")
    for frame, token in zip(manifest["frames"], expected):
        source = tables["lidar"][token]
        if frame["timestamp_sec"] != float(source["timestamp"]) * 1e-6 or frame["source_cloud"] != source["filename"] or frame["source_cloud_sha256"] != sha256(root / source["filename"]):
            raise ValueError("raw cloud or timestamp mismatch")
    scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    labels = [np.load(path.parent / f["point_labels"]) for f in manifest["frames"]]
    if any(g.shape != (len(scan),) or not np.isin(g, [0, 1]).all() for g, (scan, _) in zip(labels, scans)):
        raise ValueError("invalid row-aligned GT")
    gt = np.concatenate(labels).astype(bool)
    _, kr = core.clean_map_by_visibility(points, scans, **plan["parameters"]["range"])
    _, ks = core.clean_map_by_scan_ratio(points, scans, **plan["parameters"]["scan_ratio"])
    hashes = {"manifest_sha256": sha256(path), "map_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
              "gt_sha256": hashlib.sha256(gt.tobytes()).hexdigest()}
    reference = {"config": dict(hashes, parameters={"range_scan_ratio": plan["parameters"]}),
                 "metrics": {"range_scan_ratio": bench.compute_accuracy_metrics(~kr & ~ks, gt)}}
    result = compare_scene(path, reference, MODES)
    result.update(map_points=len(points), native_api_mask_verified=True,
                  raw_cloud_sha256={f["source_cloud"]: f["source_cloud_sha256"] for f in manifest["frames"]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    for name in ("root", "archive-prefix", "reference-json", "plan-json"):
        f.add_argument("--" + name, type=Path, required=True)
    f.add_argument("--count", type=int, default=3)
    for action in ("inventory", "export", "evaluate"):
        p = sub.add_parser(action)
        for name in ("root", "plan-json"):
            p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--plan-sha256", required=True)
        if action != "inventory":
            p.add_argument("--manifest-root", type=Path, required=True)
        p.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "freeze":
            plan = freeze(args.root, args.archive_prefix, args.reference_json, args.count)
            write_new(args.plan_json, plan)
            print(json.dumps({"scenes": plan["scenes"], "plan_sha256": sha256(args.plan_json)}), flush=True)
            return 0
        if args.report_json.exists():
            raise ValueError("report exists; choose a new path")
        plan = json.loads(args.plan_json.read_text())
        verify(plan, args.plan_json, args.plan_sha256, args.root)
        tables, selected = selected_tables(args.root, plan)
        if args.action == "inventory":
            write_new(args.report_json, {"plan_sha256": args.plan_sha256, "scenes": selected,
                      "clouds": [tables["lidar"][t]["filename"] for seq in selected.values() for t in seq]})
        elif args.action == "export":
            export_manifests(args.root, args.manifest_root, plan, args.plan_sha256, tables, selected)
            write_new(args.report_json, {"plan_sha256": args.plan_sha256, "exported_scenes": plan["scenes"]})
        else:
            records = []
            for scene in plan["scenes"]:
                print("Evaluating " + scene, flush=True)
                records.append(evaluate_manifest(args.manifest_root / scene / "manifest.json", args.root,
                                                 plan, args.plan_sha256, tables, selected[scene]))
            report = {"plan_sha256": args.plan_sha256, "plan": plan, "scene_results": records,
                      "aggregate": aggregate(records),
                      "limitations": "Three archive-scoped trainval scenes, not representative or an official held-out test split. Candidates previously tuned on mini; these scenes are separate. Undeskewed rigid-pose keyframes, moving-box GT with >2 m displacement and 0.25 m margin, offline maps including future points. Visibility support reuses angular-bin range evidence. All selected scenes retained; zero-GT precision/recall/F1 use the existing zero convention. No production default changes or automatic promotion."}
            write_new(args.report_json, report)
            print(json.dumps(report["aggregate"], indent=2), flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
