#!/usr/bin/env python3
"""Evaluate frozen neighbor candidates on an AV2 log absent from prior records."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
import dynamic_object_removal as core
from scripts.experiment_neighbor_columns import compare_scene


def require_unseen(scene, previously_examined):
    if scene in previously_examined:
        raise ValueError('scene was previously examined; choose a new log before evaluating')


def validate(manifest_path, prior_path, previously_examined, data_root):
    source = json.loads(prior_path.read_text())
    template = source['av2_scene_result']
    manifest = json.loads(manifest_path.read_text())
    require_unseen(manifest['scene'], set(previously_examined) | {template['scene']})
    if len(manifest['frames']) != 12:
        raise ValueError('requires the frozen 12-frame selection')
    import pyarrow.feather as feather
    annotation = data_root / manifest['scene'] / 'annotations.feather'
    poses = data_root / manifest['scene'] / 'city_SE3_egovehicle.feather'
    at = set(map(int, feather.read_table(annotation, columns=['timestamp_ns'])['timestamp_ns'].to_pylist()))
    pt = set(map(int, feather.read_table(poses, columns=['timestamp_ns'])['timestamp_ns'].to_pylist()))
    expected_ts = sorted(at & pt)[:36:3]
    if [f['timestamp_ns'] for f in manifest['frames']] != expected_ts:
        raise ValueError('manifest does not match frozen first 12 frames at stride 3')
    scans, _ = core._load_scan_manifest(manifest_path)
    points = np.concatenate([p for p, _ in scans])
    labels = []
    for frame, (scan, _) in zip(manifest['frames'], scans):
        values = np.load(manifest_path.parent / frame['point_labels'])
        if values.shape != (len(scan),) or not np.isin(values, [0, 1]).all():
            raise ValueError('invalid labels')
        labels.append(values.astype(bool))
    gt = np.concatenate(labels)
    if not gt.any() or gt.all():
        raise ValueError('requires both moving-box GT and static points')
    params = template['parameters']
    _, kr = core.clean_map_by_visibility(points, scans, **params['range'])
    _, ks = core.clean_map_by_scan_ratio(points, scans, **params['scan_ratio'])
    hashes = dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  map_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                  gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest())
    reference = {'config': dict(hashes, parameters={'range_scan_ratio': params}),
                 'metrics': {'range_scan_ratio': bench.compute_accuracy_metrics(~kr & ~ks, gt)}}
    result = compare_scene(manifest_path, reference, ('none', 'either', 'visibility_supported'))
    assert result['parameters'] == params
    assert result['input_hashes'] == hashes
    diagnostic_params = json.loads(json.dumps(params))
    diagnostic_params['range']['ground_z'] = None
    _, diagnostic_range = core.clean_map_by_visibility(points, scans, **diagnostic_params['range'])
    diagnostic_reference = {'config': dict(hashes, parameters={'range_scan_ratio': diagnostic_params}),
                            'metrics': {'range_scan_ratio': bench.compute_accuracy_metrics(~diagnostic_range & ~ks, gt)}}
    diagnostic = compare_scene(manifest_path, diagnostic_reference, ('none', 'either', 'visibility_supported'))
    return {'coordinate_diagnostic': {
                'map_z_percentiles': np.percentile(points[:, 2], [0, 25, 50, 75, 100]).tolist(),
                'points_at_or_below_fixed_ground_z': int((points[:, 2] <= params['range']['ground_z']).sum()),
                'moving_gt_at_or_below_fixed_ground_z': int((gt & (points[:, 2] <= params['range']['ground_z'])).sum()),
                'native_range_dynamic': int((~kr).sum()), 'native_scan_ratio_dynamic': int((~ks).sum()),
                'description': 'Post-result diagnostic only: disable absolute map-Z ground guard without other parameter changes. Not a frozen held-out result or a proposed production setting.',
                'without_absolute_ground_guard': diagnostic},
            'dataset': 'argoverse-2-sensor-val', 'frames': len(scans), 'stride': 3,
            'map_points': len(points), 'gt_dynamic_points': int(gt.sum()),
            'parameters_source_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest(),
            'parameters_source_scene': template['scene'],
            'annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
            'poses_sha256': hashlib.sha256(poses.read_bytes()).hexdigest(),
            'timestamps_ns': expected_ts,
            'previously_examined_scenes': sorted(set(previously_examined) | {template['scene']}),
            'selection': 'First lexicographic AV2 val log absent from repository scene references and local prior evaluation; chosen before reading candidate metrics. First 12 usable annotation/pose timestamps at stride 3. No reselection based on GT or results.',
            'native_api_mask_verified': True, 'result': result,
            'limitations': 'One newly examined log, not a representative held-out dataset or official AV2 test split. Parameters frozen from prior AV2 comparison; no tuning after this result. Moving annotation-box GT uses >2 m displacement and 0.2 m box margin; not per-point motion labels. Existing AV2 preprocessing excludes ego Z <= -1.4 m. Visibility support reuses baseline angular-bin range evidence, not exact rays or an independent measurement. Offline accumulated map includes future points.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--prior-experiment', required=True, type=Path)
    parser.add_argument('--previously-examined', required=True, nargs='+')
    parser.add_argument('--report-json', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report_json.exists():
        parser.error('report exists; choose a new path')
    report = validate(args.manifest, args.prior_experiment, args.previously_examined, args.data_root)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['result']['results'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
