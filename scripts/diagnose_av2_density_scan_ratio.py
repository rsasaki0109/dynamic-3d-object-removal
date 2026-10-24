#!/usr/bin/env python3
"""Explain frozen laser-density scan-ratio decisions without changing thresholds."""
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
from scripts.experiment_av2_laser_density import read_inputs, subset
from scripts.validate_frozen_av2_suite import write_new
from scripts.validate_frozen_nuscenes_suite import sha256

STATES = ('outside_polar_domain', 'empty_query_column', 'map_height_too_small',
          'height_ratio_too_large', 'ground_reverted', 'dynamic_vote')


def profile(points, origin, params):
    flat, valid = core._polar_bins(points, origin, params['n_rings'], params['n_sectors'], params['max_range'])
    size = params['n_rings'] * params['n_sectors']
    low, high = np.full(size, np.inf), np.full(size, -np.inf)
    count = np.bincount(flat[valid], minlength=size)
    np.minimum.at(low, flat[valid], points[valid, 2])
    np.maximum.at(high, flat[valid], points[valid, 2])
    height = np.zeros(size)
    height[count > 0] = high[count > 0] - low[count > 0]
    return flat, valid, height, count


def trace(points, scans, params):
    if params['min_votes'] is not None:
        raise ValueError('normalized voting required')
    states, mh, qh, mc, qc = [], [], [], [], []
    for query, origin in scans:
        bins, valid, map_h, map_n = profile(points, origin, params)
        _, _, query_h, query_n = profile(query, origin, params)
        observed = np.zeros(len(points), bool)
        observed[valid] = query_n[bins[valid]] > 0
        low = np.zeros(len(points), bool)
        low[observed] = map_h[bins[observed]] <= params['min_map_height']
        ratio_fail = np.zeros(len(points), bool)
        eligible = observed & ~low
        ratio_fail[eligible] = (query_h[bins[eligible]] / np.maximum(map_h[bins[eligible]], 1e-9)
                                >= params['scan_ratio_threshold'])
        dynamic, actual_observed = core._scan_ratio_dynamic(points, query, origin,
            params['n_rings'], params['n_sectors'], params['max_range'],
            params['scan_ratio_threshold'], params['min_map_height'], params['ground_margin'])
        np.testing.assert_array_equal(observed, actual_observed)
        if np.any(dynamic & (~eligible | ratio_fail)):
            raise ValueError('API dynamic vote outside height gates')
        code = np.zeros(len(points), np.uint8)
        code[valid] = 1; code[low] = 2; code[ratio_fail] = 3
        code[eligible & ~ratio_fail] = 4; code[dynamic] = 5
        states.append(code); mh.append(map_h); qh.append(query_h); mc.append(map_n); qc.append(query_n)
    states = np.asarray(states)
    votes, observed = (states == 5).sum(axis=0), (states >= 2).sum(axis=0)
    threshold = np.maximum(max(1, min(params['votes_floor'], len(scans))),
                           np.ceil(params['votes_fraction'] * observed).astype(int))
    removed = votes >= threshold
    _, keep = core.clean_map_by_scan_ratio(points, scans, **params)
    np.testing.assert_array_equal(removed, ~keep)
    return dict(states=states, votes=votes, observed=observed, threshold=threshold, removed=removed,
                map_height=np.asarray(mh), query_height=np.asarray(qh),
                map_count=np.asarray(mc), query_count=np.asarray(qc))


def summarize(dense, sparse, indices, gt):
    ds, ss = dense['states'][:, indices], sparse['states']
    if ds.shape != ss.shape or ss.shape[1] != len(gt):
        raise ValueError('unmatched populations')
    # Thinning cannot create a previously unobserved polar column.
    if np.any((ss >= 2) & (ds < 2)):
        raise ValueError('subset gained observed columns')
    if np.any(sparse['threshold'] > dense['threshold'][indices]):
        raise ValueError('subset increased normalized threshold')
    a, b = dense['removed'][indices], sparse['removed']
    masks = {'all_retained': np.ones(len(gt), bool), 'moving_lost_removal': gt & a & ~b,
             'moving_gained_removal': gt & ~a & b, 'static_added_removal': ~gt & ~a & b,
             'static_recovered': ~gt & a & ~b}
    out = {}
    for name, mask in masks.items():
        left, right = ds[:, mask], ss[:, mask]
        transition = np.bincount((left.astype(int) * 6 + right).ravel(), minlength=36).reshape(6, 6)
        lost = (left == 5) & (right != 5)
        gained = (left != 5) & (right == 5)
        out[name] = {'points': int(mask.sum()), 'point_scan_opportunities': int(left.size),
            'stage_transition_dense_to_sparse': transition.tolist(),
            'lost_vote_destination': {s: int((lost & (right == i)).sum()) for i,s in enumerate(STATES)},
            'gained_vote_source': {s: int((gained & (left == i)).sum()) for i,s in enumerate(STATES)},
            'lost_votes': int(lost.sum()), 'gained_votes': int(gained.sum()),
            'dense_votes': int((left == 5).sum()), 'sparse_votes': int((right == 5).sum()),
            'threshold_lowered_points': int((sparse['threshold'][mask] < dense['threshold'][indices][mask]).sum()),
            'dense_threshold_sum': int(dense['threshold'][indices][mask].sum()),
            'sparse_threshold_sum': int(sparse['threshold'][mask].sum())}
        if out[name]['sparse_votes'] - out[name]['dense_votes'] != int(gained.sum()) - int(lost.sum()):
            raise ValueError('vote accounting failed')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['freeze', 'evaluate'])
    p.add_argument('--density-results', type=Path, required=True)
    p.add_argument('--frozen-results', type=Path, required=True)
    p.add_argument('--density-root', type=Path, required=True)
    p.add_argument('--manifest-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan-json', type=Path, required=True)
    p.add_argument('--plan-sha256')
    p.add_argument('--output', type=Path)
    p.add_argument('--report-json', type=Path)
    args = p.parse_args(argv)
    density = json.loads(args.density_results.read_text())
    frozen = json.loads(args.frozen_results.read_text())
    code_files = [str(Path(__file__).relative_to(ROOT)), 'dynamic_object_removal.py',
                  'scripts/experiment_av2_laser_density.py', 'scripts/validate_frozen_av2_suite.py',
                  'scripts/validate_frozen_nuscenes_suite.py']
    fingerprints = {'density_results': sha256(args.density_results), 'frozen_results': sha256(args.frozen_results),
                    'code': {s: sha256(ROOT / s) for s in code_files}}
    if args.action == 'freeze':
        write_new(args.plan_json, {'schema_version': 1, 'fingerprints': fingerprints,
            'scope': 'All three previously examined AV2 scenes and all six thinned phases; no tuning or scene/phase exclusion.',
            'method': 'GT-free scan-ratio stage replay; compare identical source rows. GT is used only for post-replay grouping.',
            'states': STATES, 'limitations': 'Descriptive stage transitions, not an intervention separating map and query thinning. No new generalization or physical-sensor claim.'})
        print(sha256(args.plan_json)); return
    if not args.plan_sha256 or sha256(args.plan_json) != args.plan_sha256:
        p.error('plan digest mismatch')
    plan = json.loads(args.plan_json.read_text())
    if plan['fingerprints'] != fingerprints:
        p.error('source/code fingerprints changed')
    if not args.output or not args.report_json or args.output.exists() or args.report_json.exists():
        p.error('fresh output and report paths required')
    if density['plan']['scene_order'] != frozen['plan']['scenes']:
        p.error('scene sets differ')
    for filename,digest in density['plan']['code_sha256'].items():
        if sha256(ROOT/filename) != digest: p.error('density experiment source changed')
    args.output.mkdir(parents=True)
    records = []
    for reference, prior in zip(frozen['scene_results'], density['scene_results']):
        if prior['scene'] != reference['scene']: p.error('scene order mismatch')
        scene = reference['scene']; target = args.output / scene; target.mkdir()
        scans, gt, ids, raw = read_inputs(args.manifest_root / scene / 'manifest.json', reference, args.data_root)
        if raw != prior['raw_files']: p.error('raw source mismatch')
        points = np.concatenate([x for x,_ in scans]); params = reference['parameters']['scan_ratio']
        dense = trace(points, scans, params)
        np.savez_compressed(target/'dense_evidence.npz', **dense)
        for variant in density['plan']['variants']:
            chosen, labels, indices = subset(scans, gt, ids, variant)
            saved = args.density_root / scene / variant['name']
            np.testing.assert_array_equal(indices, np.load(saved/'source_map_indices.npy'))
            np.testing.assert_array_equal(labels, np.load(saved/'gt.npy'))
            sparse = dense if variant['modulus'] == 1 else trace(np.concatenate([x for x,_ in chosen]), chosen, params)
            for res in density['plan']['resolutions_deg']:
                np.testing.assert_array_equal(sparse['removed'], np.load(saved/f'removed_{res}_scan_ratio.npy'))
            if variant['modulus'] == 1: continue
            np.savez_compressed(target/(variant['name']+'_evidence.npz'), source_map_indices=indices, gt=labels, **sparse)
            report = summarize(dense, sparse, indices, labels)
            prior_case = next(v for v in prior['variants'] if v['variant'] == variant)
            expected = prior_case['conditions']['1.0']['scan_ratio']['decision_changes']
            for group,key in [('moving_lost_removal','lost_moving_removed'),('moving_gained_removal','added_moving_removed'),
                              ('static_added_removal','added_static_removed'),('static_recovered','recovered_static_points')]:
                if report[group]['points'] != expected[key]: p.error('prior decision counts differ')
            records.append({'scene': scene, 'variant': variant, 'groups': report})
            print(scene[:8], variant['name'], 'verified', flush=True)
    totals = {}
    for n in [32,16]:
        cases = [r for r in records if r['variant']['nominal_ids'] == n]
        groups = {}
        for group in cases[0]['groups']:
            parts = [r['groups'][group] for r in cases]
            groups[group] = {k: (np.sum([x[k] for x in parts], axis=0).tolist() if k == 'stage_transition_dense_to_sparse'
                else {s: sum(x[k][s] for x in parts) for s in STATES} if isinstance(parts[0][k],dict)
                else sum(x[k] for x in parts)) for k in parts[0]}
        totals[str(n)] = {'scene_phase_cases': len(cases), 'groups': groups}
    write_new(args.report_json, {'plan_sha256': args.plan_sha256, 'plan':plan, 'records':records, 'totals':totals,
        'verified': 'Every replay matches API masks, archived density masks at both resolutions and prior decision counts. Original source/GT and laser metadata verified. Threshold never increases under subset thinning.'})
    print(json.dumps({n: x['groups']['moving_lost_removal'] for n,x in totals.items()},indent=2))


if __name__ == '__main__':
    main()
