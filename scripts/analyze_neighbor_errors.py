#!/usr/bin/env python3
"""Trace changed neighbor-column decisions on fingerprint-verified fixed maps."""
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
from scripts.experiment_neighbor_columns import scan_votes


def evidence(points, scans, params, support):
    votes = np.zeros(len(points), dtype=int)
    observed = votes.copy()
    inferred = votes.copy()
    inferred_votes = votes.copy()
    for scan, origin in scans:
        dynamic, obs, borrowed = scan_votes(points, scan, origin, params, support)
        votes += dynamic
        observed += obs
        inferred += borrowed
        inferred_votes += dynamic & borrowed
    threshold = np.maximum(max(1, min(params['votes_floor'], len(scans))),
                           np.ceil(params['votes_fraction'] * observed).astype(int))
    return dict(votes=votes, observed=observed, inferred=inferred,
                inferred_votes=inferred_votes, threshold=threshold)


def changed_groups(native, candidate, gt):
    return {'additional_moving_removed': candidate & ~native & gt,
            'additional_static_removed': candidate & ~native & ~gt,
            'lost_moving_removed': native & ~candidate & gt,
            'recovered_static_points': native & ~candidate & ~gt}


def quantiles(values):
    return dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                    map(float, np.percentile(values, [0, 25, 50, 75, 100])))) if len(values) else None


def analyze(path, record, output):
    manifest = json.loads(path.read_text())
    scans, _ = core._load_scan_manifest(path, allow_undeskewed=True)
    points = np.concatenate([p for p, _ in scans])
    labels = [np.load(path.parent / frame['point_labels']) for frame in manifest['frames']]
    for label, (scan, _) in zip(labels, scans):
        if label.shape != (len(scan),) or not np.isin(label, [0, 1]).all():
            raise ValueError('invalid GT labels')
    gt = np.concatenate(labels).astype(bool)
    hashes = dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  map_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                  gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest())
    if hashes != record['input_hashes']:
        raise ValueError('input fingerprints differ')
    params = record['parameters']
    if params['scan_ratio']['min_votes'] is not None:
        raise ValueError('requires normalized voting')
    _, range_keep = core.clean_map_by_visibility(points, scans, **params['range'])
    _, sr_keep = core.clean_map_by_scan_ratio(points, scans, **params['scan_ratio'])
    native = evidence(points, scans, params['scan_ratio'], 'none')
    candidate = evidence(points, scans, params['scan_ratio'], 'either')
    np.testing.assert_array_equal(native['votes'] >= native['threshold'], ~sr_keep)
    masks = [~range_keep & (e['votes'] >= e['threshold']) for e in (native, candidate)]
    for mode, mask in zip(('none', 'either'), masks):
        if bench.compute_accuracy_metrics(mask, gt) != record['results'][mode]['metrics']:
            raise ValueError('recorded metrics differ')
    groups = changed_groups(*masks, gt)
    distance = np.concatenate([np.linalg.norm(p - origin, axis=1) for p, origin in scans])
    frame_index = np.repeat(np.arange(len(scans)), [len(p) for p, _ in scans])
    fixed = ~range_keep & (candidate['votes'] >= native['threshold'])
    fixed_groups = changed_groups(masks[0], fixed, gt)
    report = dict(scene=record['scene'], input_hashes=hashes, parameters=params, groups={},
                  native_threshold_counterfactual={
                      'metrics': bench.compute_accuracy_metrics(fixed, gt),
                      'changes': {k: int(v.sum()) for k, v in fixed_groups.items()},
                      'description': 'Either-neighbor dynamic votes with native-only observation threshold; research counterfactual, not production behavior.'})
    for name, mask in groups.items():
        count = int(mask.sum())
        if count != record['results']['either'][name]:
            raise ValueError('changed point count differs')
        report['groups'][name] = dict(count=count,
            acquisition_range_m=quantiles(distance[mask]),
            acquisition_frame_counts=np.bincount(frame_index[mask], minlength=len(scans)).tolist(),
            native={k: quantiles(v[mask]) for k, v in native.items()},
            either={k: quantiles(v[mask]) for k, v in candidate.items()},
            threshold_increased=int((mask & (candidate['threshold'] > native['threshold'])).sum()),
            dynamic_votes_decreased=int((mask & (candidate['votes'] < native['votes'])).sum()),
            no_inferred_dynamic_votes=int((mask & (candidate['inferred_votes'] == 0)).sum()))
    report['limitations'] = ('Annotation-box GT, not point motion labels or static object categories. '
        'Acquisition range is 3D distance to the point source sensor. Quantiles describe changed '
        'points, not predictive gates. Same examined data; no held-out validation. '
        'Neighbor support adds inferred observations and affects normalized thresholds. '
        'Full offline accumulated map includes future points.')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    render(points, scans, groups, output, record['scene'])
    return report


def render(points, scans, groups, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    # Deterministic background subsample only; every changed point is shown.
    background = points[::max(1, int(np.ceil(len(points) / 15000)))]
    trajectory = np.array([origin for _, origin in scans])
    for ax, (name, mask), color in zip(axes.flat, groups.items(), ('green', 'purple', 'orange', 'blue')):
        ax.scatter(background[:, 0], background[:, 1], s=.3, c='gray', alpha=.15, rasterized=True)
        ax.plot(trajectory[:, 0], trajectory[:, 1], c='black', lw=.7)
        ax.scatter(points[mask, 0], points[mask, 1], s=2, c=color, rasterized=True)
        ax.set_title(f"{name.replace('_', ' ')}: {mask.sum():,}")
        ax.set_aspect('equal'); ax.set_xlabel('Map X (m)'); ax.set_ylabel('Map Y (m)')
    fig.suptitle(f'{title}: either-neighbor changes from native\nAll changed points shown; gray background capped at 15,000')
    fig.savefig(output / 'changes.png', dpi=160)
    fig.savefig(output / 'changes.svg')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-json', type=Path, required=True)
    parser.add_argument('--validation-root', type=Path, required=True)
    parser.add_argument('--av2-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    source = json.loads(args.experiment_json.read_text())
    records = source['nuscenes_scene_results']
    # Select useful diagnostic examples without presenting them as an unbiased aggregate.
    names = {'scene-0796', 'scene-0757', max(records, key=lambda r: r['results']['either']['additional_static_removed'])['scene']}
    selected = [(args.validation_root / r['scene'] / 'manifest.json', r) for r in records if r['scene'] in names]
    selected.append((args.av2_manifest, source['av2_scene_result']))
    reports = []
    for path, record in selected:
        report = analyze(path, record, args.output / record['scene'])
        reports.append(report)
        print(json.dumps({'scene': record['scene'], 'groups': report['groups']}), flush=True)
    (args.output / 'report.json').write_text(json.dumps({'selection': 'Previously examined target, largest moving gain, largest additional static removal, and AV2 comparison; selected examples, not aggregate.', 'scenes': reports}, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
