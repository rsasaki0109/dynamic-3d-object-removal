# GT-free relative vertical-alignment experiment

**Outcome: not adopted.** This fixed-configuration experiment did not establish a useful vertical correction on the previously examined three-scene sample. Two scenes fell back to unchanged poses; the supported scene had essentially unchanged holdout median error, worse holdout p90 and lower F1. No production filter, default, GT, historical result or public demo changes.

[Pre-run plan and code/input fingerprints](nuscenes_vertical_alignment_plan.json) · [All edges, offsets and measured results](nuscenes_vertical_alignment.json) · [Source GT-policy comparison](GT_POLICY_COMPARISON.md)

## Fixed method

- Input: the same 524,495 points in scene-0004/0005/0006, using the explicit `xy_or_moving_attribute` GT definition for reporting only.
- Fit input: XYZ coordinates only, with no GT, boxes, semantic classes, attributes, predictions or scores. XY positions and rotations are fixed; estimate one relative Z translation per frame.
- Build 1 m XY cells and a 10th-percentile Z profile for cells with at least three points. The original preprocessing remains in place, including sensor-Z and near-square exclusions; no discarded raw points are reintroduced.
- Withhold one fifth of 5-by-5-cell spatial blocks using `(block_x + 3*block_y) mod 5 == 0`. The same blocks remain withheld across frame pairs. Their height values never enter offset fitting.
- Compare adjacent and lag-2 frames. Each pair requires at least 30 training cells, five holdout cells, 60% robust support and 5 m inlier spans in both XY axes. Reject scaled MAD >0.5 m; inliers use a 3-scaled-MAD band with a 0.1 m floor.
- Solve supported pair constraints by NumPy weighted least squares, anchored at frame 0, with weights proportional to the square root of inlier support.
- Require every frame to connect to frame 0. If not, apply **zero corrections to the entire scene**, retaining the scene in all summaries. No re-anchoring or selection of a better-looking subgraph after observing the result.

The plan SHA-256 was recorded before estimation/evaluation: `c6fcb049f55a74ebedff177622979d4ed3624481eaca7336926475dfe5c4364b`. Parameters and estimator/filter byte hashes are frozen in the plan. The design followed earlier audits of this sample; this is post-audit exploration, not an unseen-scene experiment. No parameters were retuned after inspecting these results.

## Geometry results

| Scene | Supported edges / attempted | Frames connected to anchor | Applied for experiment | Holdout median abs residual before → after (m) | Holdout p90 before → after (m) |
|---|---:|---:|---|---|---|
| scene-0004 | 5 / 21 | 4 / 12 | No; whole-scene fallback | 0.354212 → 0.354212 | 2.736188 → 2.736188 |
| scene-0005 | 15 / 21 | 12 / 12 | Yes | 0.292216 → 0.291695 | 2.384701 → 2.401867 |
| scene-0006 | 15 / 21 | 1 / 12 | No; whole-scene fallback | 0.591588 → 0.591588 | 2.368074 → 2.368074 |

Rejected real-data edges all exceeded the fixed scaled-MAD limit, indicating inconsistent shared-cell height differences for this estimator. Scene-0004 connects only frames 0–3. Scene-0006 has 15 supported edges elsewhere, but neither edge from frame 0 is supported, so its anchored component contains only frame 0. Both receive zero offsets.

Scene-0005 receives a maximum correction of about **+0.234 m** relative to frame 0. Its holdout median changes by only about **0.00052 m**, while p90 worsens by about **0.0172 m**. These are overlapping-XY-profile consistency measurements, not physical pose error. Holdout counts pool pair/cell observations; a spatial cell can appear in multiple frame pairs, so they are not counts of distinct physical surfaces.

## Filtering with the same GT labels

After fitting without labels, the native `range ∩ scan_ratio` filter is rerun with the same parameters. All original point identities and labels are retained. Points and sensor origins receive the same per-frame Z translation through newly exported manifests; label boxes are not reevaluated against their old global coordinates. Source native predictions match the saved mask hashes from the GT-policy comparison exactly.

| Scene | Source F1 | Experimental F1 | Source static-labelled points kept | Experimental static-labelled points kept |
|---|---:|---:|---:|---:|
| scene-0004 | 0.035520 | 0.035520 | 99.7080% | 99.7080% |
| scene-0005 | 0.026523 | 0.025182 | 98.8764% | 98.8896% |
| scene-0006 | 0.008549 | 0.008549 | 97.0705% | 97.0705% |

All-scene unweighted mean F1: **0.023531 → 0.023084**; recall: 0.017113 → 0.016817. No scene is excluded because estimation fails. These remain measurements against the explicit box/attribute GT proxy, not independently measured physical motion.

Only scene-0005 changes predictions. It adds eight moving-GT removals but loses 16 (net eight fewer), while adding 100 static-GT false removals and recovering 125 (net 25 fewer false removals). A small net static-keep increase does not compensate for the missing geometric/accuracy evidence.

The result does not support promoting this single-Z-shift method. It also does not prove that vertical alignment is unnecessary or impossible: lower-quantile cells can observe different surfaces, vegetation or moving traffic; some graph components lack a reliable connection; no rotation or full 6-DoF registration is estimated. The next useful direction needs independently reliable 3D pose evidence or a separately justified registration model, rather than selecting permissive thresholds on this examined GT.

## Artifacts and reproduction

`scripts/experiment_nuscenes_vertical_alignment.py` has separate `freeze` and `evaluate` actions. Each scene export includes a new posed `manifest.json`, `alignment.json`, and source/experimental prediction masks. The manifest retains the exact GT definition and source label files, adds `experimental_alignment` with offsets and plan digest, and records its source-manifest hash. XYZ point rows, XY coordinates, rotations and labels remain unchanged except for the declared vertical pose translation. Large exports and PNG/SVG plots remain ignored under `output/`.

To replay the checked-in plan, retain the exact source manifests from the GT-policy comparison. Freeze a new plan before evaluation if reexporting manifests changes their byte-level paths/provenance. Use fresh output/report paths:

```bash
mkdir -p work/mpl-cache work/font-cache
MPLCONFIGDIR=work/mpl-cache XDG_CACHE_HOME=work/font-cache \
python scripts/experiment_nuscenes_vertical_alignment.py evaluate \
  --policy-results examples/cli_validation/nuscenes_gt_policy_comparison.json \
  --source-root output/nuscenes_gt_policy_comparison_final \
  --plan-json examples/cli_validation/nuscenes_vertical_alignment_plan.json \
  --plan-sha256 c6fcb049f55a74ebedff177622979d4ed3624481eaca7336926475dfe5c4364b \
  --output output/vertical-alignment-replay \
  --report-json output/vertical-alignment-replay.json --render
```

For a new pre-run record with the same explicit configuration:

```bash
python scripts/experiment_nuscenes_vertical_alignment.py freeze \
  --policy-results examples/cli_validation/nuscenes_gt_policy_comparison.json \
  --source-root output/nuscenes_gt_policy_comparison_final \
  --plan-json work/new-vertical-plan.json
```

Record the printed digest before evaluation. NumPy is sufficient without `--render`; Matplotlib is optional for PNG/SVG. Tests cover known offsets, spatial-holdout isolation, robust outlier rejection, disconnected and narrow-support fallback, coordinate/origin/label preservation, and invalid settings. All **343 repository tests passed**. The synthetic holdout corruption test confirms that changing held-out Z values cannot change the fitted offsets and still appears in validation errors.

Limits: one previously examined drive, rigid undeskewed keyframe poses, 2D source ego localization, lower-quantile cell correspondence assumptions and an offline graph using future scans. Anchoring removes only a gauge freedom; it does not provide absolute altitude truth. No physical registration accuracy or generalization claim is made.

Follow-up: the [controlled AV2 laser-ID ablation](LASER_DENSITY_ABLATION.md) measures thinning with inherited 3D source poses and identical-row references. It isolates a density effect within AV2 without claiming to resolve nuScenes pose errors.
