# Explicit nuScenes GT-policy comparison

Three named moving-track GT policies are implemented in `scripts/nuscenes_gt_policies.py`. The dedicated `scripts/compare_nuscenes_gt_policies.py` CLI exports policy-labelled manifests and row-aligned labels, and measures them against identical saved predictions. The original benchmark/exporter, production library, defaults and historical records are retained.

[Comparison JSON, track decisions, label flips and fingerprints](nuscenes_gt_policy_comparison.json) · [Motion/vertical-coordinate audit](BOX_MOTION_AUDIT.md)

## Policies

| CLI policy | Qualifying track evidence | Single observation |
|---|---|---|
| `xyz` (default) | Maximum 3D center displacement from first selected observation >2 m | Excluded |
| `xy` | Maximum XY center displacement from first selected observation >2 m | Excluded |
| `xy_or_moving_attribute` | XY displacement >2 m **or** an official `vehicle.moving` / `pedestrian.moving` attribute in any selected observation | May qualify by explicit moving attribute |

Thresholds use strict `>` comparisons and the first observation in the selected sample order, matching the legacy anchor. The hybrid policy admits slow movers and explicitly annotated single-observation movers. `cycle.with_rider`, parked/stopped attributes and object categories alone do not trigger its attribute branch. Missing attribute mappings are rejected rather than guessed.

These remain track-window policies: qualifying a track labels points inside its annotation boxes at matching frames throughout the selected window, including observations without a moving attribute. They are not per-point physical-motion truth or per-frame attribute-only segmentation. The same 0.25 m box margins are used for every policy.

## Fixed-input comparison

This is a post-audit sensitivity comparison on the previously examined scene-0004/0005/0006 (one drive), not a fresh held-out experiment. All 524,495 point coordinates, poses, preprocessing and filter parameters are fixed. Each method produces one saved boolean prediction mask per scene, shared across all GT policies. The native, either-neighbor and visibility-supported methods are replayed with the original frozen code and inputs.

The recomputed `xyz` labels match the original GT **point for point**, and its moving-track set matches the original selector. Replayed precision/recall/F1/static-preservation values and inferred-observation counts match the frozen candidate comparison. Historical candidate masks were not archived; no bytewise comparison to those unavailable masks is claimed. All newly saved masks have explicit hashes and are shared unchanged across policy evaluations.

All three selected scenes remain in every mean, including scene-0006 after its revised GT count drops below 5,000. No GT-content threshold is used to change which scenes are averaged. A zero-positive scene would also remain included, with the existing zero precision/recall/F1 convention.

| GT policy | Total moving-labelled points | Native mean F1 | Native mean recall | Native mean static kept |
|---|---:|---:|---:|---:|
| `xyz` | 43,555 | 0.0221 | 0.0128 | 98.386% |
| `xy` | 15,788 | 0.0232 | 0.0192 | 98.555% |
| `xy_or_moving_attribute` | 16,775 | 0.0235 | 0.0171 | 98.552% |

**The metric differences are caused by GT definitions, not improved removal.** The same prediction can change precision, recall and even static preservation when positive/negative labels change. Do not select a GT policy by maximizing these scores. Low recall remains under all three definitions, so the GT change alone does not resolve the sparse-data failure.

| Scene | GT policy | Moving-labelled points | Moving tracks | Original positive → negative | Original negative → positive | Native F1 |
|---|---|---:|---:|---:|---:|---:|
| scene-0004 | `xyz` | 5,697 | 19 | 0 | 0 | 0.0357 |
| scene-0004 | `xy` | 5,697 | 19 | 0 | 0 | 0.0357 |
| scene-0004 | `xy_or_moving_attribute` | 5,730 | 25 | 0 | 33 | 0.0355 |
| scene-0005 | `xyz` | 9,018 | 10 | 0 | 0 | 0.0265 |
| scene-0005 | `xy` | 9,018 | 10 | 0 | 0 | 0.0265 |
| scene-0005 | `xy_or_moving_attribute` | 9,025 | 12 | 0 | 7 | 0.0265 |
| scene-0006 | `xyz` | 28,840 | 52 | 0 | 0 | 0.0040 |
| scene-0006 | `xy` | 1,073 | 6 | 27,767 | 0 | 0.0073 |
| scene-0006 | `xy_or_moving_attribute` | 2,020 | 14 | 27,051 | 231 | 0.0085 |

The large label change is in scene-0006: `xy` removes 46 legacy moving tracks; the hybrid removes 44 and adds six previously unqualified tracks there. Its complete decision records explain displacement and attribute qualification independently. Across all scenes the hybrid adds 14 tracks relative to legacy, including slow pedestrians/workers and cars. This is a policy consequence, not independent verification that every changed point is correctly classified.

For completeness, the other two methods use the same GT definitions and fixed predictions:

| GT policy | Either-neighbor mean F1 | Visibility-supported mean F1 |
|---|---:|---:|
| `xyz` | 0.0413 | 0.0279 |
| `xy` | 0.0428 | 0.0289 |
| `xy_or_moving_attribute` | 0.0430 | 0.0292 |

## Exported labels and provenance

Each output scene contains shared `prediction_none.npy`, `prediction_either.npy` and `prediction_visibility_supported.npy`. Each requested policy has its own directory containing:

- `gt.npy`: boolean labels in the original accumulated-map row order.
- `labels_0000.npy`, etc.: per-frame uint8 labels with the exact original cloud row order.
- `manifest.json`: references the original preprocessed clouds through relative paths and retains the same poses and undeskewed sensor profile. Its `gt_definition` records axes, threshold, anchor, attributes, window scope, margin and pose limitation.

A policy export carries `parent_manifest_sha256` and `source_frozen_plan_sha256`. It removes the original `frozen_plan_sha256` field so the changed labels do not impersonate the original frozen export. The source clouds, labels, manifests and historical benchmark JSON are never overwritten.

This comparison JSON has task `nuscenes_gt_policy_sensitivity`; it is not a regression-comparator benchmark result. Downstream accuracy exports must retain the manifest’s GT definition, and regression comparisons must hold that definition and GT/input hashes fixed.

## Run the comparison or export one policy

Use the existing frozen manifests and official trainval metadata, including the attribute/category/instance tables extracted for the [motion audit](BOX_MOTION_AUDIT.md#reproduce). Run from the repository root with fresh output/report paths:

```bash
mkdir -p work/mpl-cache work/font-cache
MPLCONFIGDIR=work/mpl-cache XDG_CACHE_HOME=work/font-cache \
python scripts/compare_nuscenes_gt_policies.py \
  --frozen-results examples/cli_validation/nuscenes_frozen_suite_results.json \
  --manifest-root output/nuscenes_frozen_suite/manifests \
  --data-root data/nuscenes_trainval \
  --annotation-metadata-root data/nuscenes_annotation_audit \
  --policies xyz xy xy_or_moving_attribute \
  --output output/gt-policy-replay \
  --report-json output/gt-policy-replay.json --render
```

To generate just the attribute-aware policy, use `--policies xy_or_moving_attribute`. Omitting `--policies` selects legacy `xyz`. Omit `--render` to use NumPy only; Matplotlib is optional for the PNG/SVG comparison figure. Large predictions/label files and plots remain ignored under `output/`, while compact comparison evidence is retained here.

Seven new tests cover vertical-only motion, strict chronological displacement rules, slow/single-observation attribute movers, parked/rider exclusions, malformed mappings, label/prediction isolation, metadata lineage, and invalid export lengths. All **333 repository tests passed**. The legacy 3D policy reproduces the original real-data GT exactly on all three scenes.

## Remaining limitations

The subsequent [GT-free vertical-alignment experiment](VERTICAL_ALIGNMENT_EXPERIMENT.md) fixes the GT policy and tests a supported Z-only correction with spatial holdout cells. It did not establish a useful correction and was not adopted.

The [nuScenes pose audit](BOX_MOTION_AUDIT.md) still applies: raw ego Z is fixed to zero by the dataset’s 2D localization contract, and this sample has changing annotation and point-height populations. None of these policies estimates missing vertical ego motion, corrects 3D registration, deskews keyframes or resolves angular occlusion. The next geometric evaluation must address or explicitly constrain those assumptions; changing GT alone is not a 3D map-cleaning fix.

Source: nuScenes Dataset, https://www.nuscenes.org. Original GT and every historical accuracy figure remain available under their original definitions.
