# Attribution of nearer returns in front of common misses

This post-result analysis identifies which annotated tracks and official lidarseg classes generated the nearer angular-pixel returns. It retains the frozen inputs, GT and filter parameters. It is not a new inference method or accuracy improvement.

[Previous point/scan evidence](COMMON_MISS_EVIDENCE.md) · [Track attribution and fingerprints](nuscenes_nearer_return_attribution.json) · [Official semantic-label attribution and raw-row fingerprints](nuscenes_nearer_return_semantics.json)

## What is counted

The population is the **242,504 point/scan events** where a baseline common-miss GT point has a query return at least the strict range margin closer in its 2.5-degree angular pixel (range gap >0.5 m). The same target or return may appear repeatedly across scans; these are not distinct object counts. All three scenes belong to one drive.

Target track identity uses boxes at the target point’s acquisition frame. Nearest-return identity uses boxes at the query frame. Moving tracks follow the unchanged >2 m center-displacement rule over the selected 12 keyframes. Box margins remain 0.25 m. The union of moving boxes reproduces the saved GT exactly for every input point.

## Track and box attribution

| Relation | Events | Share of all nearer-return events |
|---|---:|---:|
| Same uniquely attributed moving track | 25,641 | 10.57% |
| Other uniquely attributed moving track | 30,018 | 12.38% |
| Annotated track not moving by window rule | 7,000 | 2.89% |
| Outside all annotation boxes | 138,221 | 57.00% |
| Target overlaps multiple boxes | 37,383 | 15.42% |
| Nearest return overlaps multiple boxes | 4,241 | 1.75% |
| Exact nearest-range tie | 0 | 0.00% |

Unclassified-by-motion tracks include tracks seen only once and tracks whose center displacement stays within the threshold; they are not proven physically static. Outside-box points were initially left unclassified. Target box overlaps override all identity assignments; otherwise exact nearest-range ties override nearest-box overlaps. Unique track relations are assigned only after these exclusions. No arbitrary nearest row is treated as a confident identity when a range tie exists.

| Scene | All nearer-return events | Same moving track | Other moving track | Outside all boxes | Same-acquisition query events |
|---|---:|---:|---:|---:|---:|
| scene-0004 | 13,064 | 1,894 | 5,033 | 1,614 | 591 |
| scene-0005 | 72,309 | 6,791 | 148 | 64,134 | 1,838 |
| scene-0006 | 157,131 | 16,956 | 24,837 | 72,473 | 4,117 |

When the query is the target’s own acquisition scan, there are 6,546 nearer-return events: 1,833 (28.0%) share a unique moving track, and 2,854 (43.6%) lie outside all boxes. Even these same-scan collisions occur inside angular bins and do not establish exact-ray physical occlusion. Per-relation range-gap and angular-separation quantiles are recorded without fitting a new angular cutoff.

## Resolving outside-box returns with official point labels

Official lidarseg labels resolve the semantic classes of the **138,221 outside-box events**. These shares use the outside-box population, rather than all nearer-return events.

| Official lidarseg class | Outside-box events | Share within outside-box population |
|---|---:|---:|
| `static.vegetation` | 112,378 | 81.303% |
| `static.manmade` | 17,997 | 13.020% |
| `flat.terrain` | 5,964 | 4.315% |
| `flat.driveable_surface` | 1,258 | 0.910% |
| `movable_object.barrier` | 466 | 0.337% |
| `movable_object.debris` | 96 | 0.069% |
| `flat.other` | 26 | 0.019% |
| `vehicle.ego` | 23 | 0.017% |
| `noise` | 6 | 0.004% |
| `vehicle.car` | 5 | 0.004% |
| `flat.sidewalk` | 2 | 0.001% |

The large outside-box population is mainly vegetation and manmade surfaces. This supports investigating environmental observations and angular-pixel grouping rather than assuming all closer returns come from the moving object itself. It does not prove that every nearer return physically occludes the exact target ray or that existing protection is erroneous. Semantic class labels do not measure motion. No semantic labels or GT identities enter production inference.

## Target-label audit clue

The subsequent [box-motion and vertical-coordinate audit](BOX_MOTION_AUDIT.md) checks all contributing tracks. It finds that the original 3D motion proxy must be interpreted in light of nuScenes' 2D ego localization, with raw ego-pose Z fixed to zero.

The same row alignment also labels the **distinct 33,679 common-miss target points**, rather than repeated nearer-return events. The largest classes are:

| Common-miss target semantic class | Distinct target points |
|---|---:|
| `movable_object.barrier` | 13,998 |
| `vehicle.bus.rigid` | 5,890 |
| `vehicle.car` | 4,805 |
| `vehicle.truck` | 4,258 |
| `flat.terrain` | 1,201 |
| `human.pedestrian.adult` | 1,090 |
| `vehicle.construction` | 868 |
| `human.pedestrian.construction_worker` | 500 |
| `static.vegetation` | 497 |
| `flat.driveable_surface` | 308 |
| `static.manmade` | 155 |
| `static.other` | 57 |
| `vehicle.bicycle` | 40 |
| `noise` | 5 |
| `flat.sidewalk` | 4 |
| `movable_object.trafficcone` | 3 |

`movable_object.barrier` accounts for 13,998 / 33,679 points (41.6%), all in scene-0006. This is an audit clue, not proof that those points are stationary or that the original GT is wrong. Semantic class, box membership and per-point motion are different labels. Before altering filter protections, audit which annotation tracks supply those target points and whether their >2 m center displacement represents motion, partial observations, or box changes. The original moving-box GT remains unchanged.

## Source and alignment checks

- Official source: https://motional-nuscenes.s3.amazonaws.com/public/nuscenes-lidarseg-v1.0/nuScenes-lidarseg-all-v1.0.tar.bz2
- Archive size: 162,788,682 bytes; SHA-256 `0b6483af5cbe404688017f885f42c0aa13914763451392f5861f4a52759967f8`.
- Each label file is selected by the manifest’s sample-data token through official `v1.0-trainval/lidarseg.json`. The explicit category `index` fields supply the label mapping; numeric IDs are not guessed.
- Every raw BIN cloud’s hash matches the earlier export, and its label byte count equals its raw point count. Raw indices are retained through the same strict sensor-Z and close-square exclusions.
- The resulting local XYZ arrays match all 36 saved manifest clouds exactly, proving row alignment after preprocessing. Label metadata and every label file have recorded hashes.
- Event and original point/scan NPZ hashes match prior records; all original event/relation counts are reproduced. Both scripts reject existing outputs and changed source evidence.

Generated `nearer_return_events.npz` files retain target and query map indices, query frame, chosen nearest row, ties, relation labels, angular separations and range gaps. Their owner arrays map to tokens in JSON. Generated `semantic_alignment.npz` files retain each map point’s semantic label and raw-source row, plus a label for every nearer-return event. PNG/SVG comparison plots and NPZ files remain in ignored `output/`; compact JSON evidence is retained here.

## Reproduce

First reproduce or retain the [fixed point/scan traces](COMMON_MISS_EVIDENCE.md#reproduce). Run track attribution with the existing trainval metadata and new output paths:

```bash
mkdir -p work/mpl-cache work/font-cache
MPLCONFIGDIR=work/mpl-cache XDG_CACHE_HOME=work/font-cache \
python scripts/attribute_nuscenes_nearer_returns.py \
  --frozen-results examples/cli_validation/nuscenes_frozen_suite_results.json \
  --evidence-report examples/cli_validation/nuscenes_common_miss_evidence.json \
  --evidence-root output/nuscenes_common_miss_evidence \
  --manifest-root output/nuscenes_frozen_suite/manifests \
  --data-root data/nuscenes_trainval \
  --output output/nearer-attribution-replay \
  --report-json output/nearer-attribution-replay.json --render
```

For semantics, obtain the official archive above. Preserve `v1.0-trainval/category.json`, `v1.0-trainval/lidarseg.json` and the 36 required `lidarseg/v1.0-trainval/<sample_data_token>_lidarseg.bin` files under `data/nuscenes_lidarseg/`. Full extraction also works; label data remains ignored. Use the attribution record and event directory from the same run:

```bash
MPLCONFIGDIR=work/mpl-cache XDG_CACHE_HOME=work/font-cache \
python scripts/classify_nuscenes_nearer_semantics.py \
  --attribution-report output/nearer-attribution-replay.json \
  --attribution-root output/nearer-attribution-replay \
  --point-evidence-root output/nuscenes_common_miss_evidence \
  --manifest-root output/nuscenes_frozen_suite/manifests \
  --data-root data/nuscenes_trainval --lidarseg-root data/nuscenes_lidarseg \
  --label-archive work/nuScenes-lidarseg-all-v1.0.tar.bz2 \
  --output output/nearer-semantics-replay \
  --report-json output/nearer-semantics-replay.json --render
```

Ten new tests cover stable nearest rows/ties, box ambiguity, query-time identities, GT reconstruction failures, empty populations, strict raw-row filtering, unknown label IDs and unchanged subgroup counts. All **318 repository tests passed**. These diagnostics preserve the library, public demos, parameters and original benchmark results.

Source: nuScenes Dataset and nuScenes-lidarseg, https://www.nuscenes.org. Three correlated scenes, undeskewed poses, offline future points, box-based motion labels and angular pixels remain limits on interpretation.
