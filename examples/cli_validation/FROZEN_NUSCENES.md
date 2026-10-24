# Frozen sparse-sensor validation on new nuScenes scenes

Candidates tuned on mini were evaluated on new trainval scenes with settings and scene selection fixed before selected-cloud extraction and inference. Production filters and defaults remain unchanged.

The later [motion-label/pose audit](BOX_MOTION_AUDIT.md) identifies a vertical-coordinate limitation in the original 3D moving-box proxy: nuScenes ego localization is 2D with Z fixed to zero, while annotation and point-height populations vary. The numbers below remain the frozen original-policy results; they are not independently validated physical-motion or fully registered 3D-map accuracy.

## Selection and inputs

- Source: official nuScenes v1.0-trainval metadata and trainval01 LiDAR archive, served publicly by Motional.
- Availability inventory: log names found in the first 8 MiB of the compressed LiDAR archive, fixed before selection. This scopes the experiment to eight archive logs, rather than claiming a random sample of trainval.
- Select the first three lexicographic scene names in those logs after excluding all ten previously examined mini scenes. No moving-content screening, minimum-GT exclusion, or result-based replacements.
- Selected: `scene-0004`, `scene-0005`, `scene-0006`. All three come from **one drive**, `n015-2018-07-18-11-18-34+0800`, absent from the ten mini-scene source logs. Treat them as correlated scenes, not three independent drives.
- Each scene uses its first 12 keyframes at stride 3. Actual raw clouds contain ring IDs 0–31, confirming 32-beam inputs. The 36 raw clouds contain 1,250,080 points.
- Preprocessing matches the earlier devkit-selection experiment: exclude sensor Z <= -1.6 m before rigid pose alignment and the near-sensor square `abs(x)<1 AND abs(y)<1`; range has no absolute map-Z guard.
- Clouds remain **undeskewed**. One rigid pose per keyframe does not correct intra-sweep motion. Results do not establish deskewed or live moving-platform behavior.
- GT: points in per-frame boxes of instances displaced >2 m over the evaluated window, with 0.25 m box margin. These are moving-box labels, not per-point motion annotations.

After preprocessing: **524,495 map points**, **43,555 moving-GT points**. All selected scenes are included.

Frozen plan SHA-256: `c19221c2e5cd89b0ec6678ae72ac3ba4400337557521b5a4a361051973aaa391`.

[Frozen plan, settings, metadata/code fingerprints](nuscenes_frozen_suite_plan.json) · [Full per-scene results and raw-cloud fingerprints](nuscenes_frozen_suite_results.json) · [Post-result channel diagnostic](nuscenes_frozen_suite_diagnostic.json)

## Prospective candidate comparison

| Method | Mean precision | Mean recall | Mean F1 | Mean static kept | Worst-scene static kept |
|---|---:|---:|---:|---:|---:|
| Native baseline | 0.0974 | 0.0128 | 0.0221 | 98.386% | 96.574% |
| Either neighbor | 0.1197 | 0.0250 | 0.0413 | 98.200% | 96.330% |
| Same-scan visibility support | 0.1192 | 0.0161 | 0.0279 | 98.363% | 96.484% |

| Scene | Map points | Moving GT | Method | F1 | Static kept | Extra moving removed | Lost moving removed | Extra static removed | Recovered static |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| scene-0004 | 145,804 | 5,697 | Native baseline | 0.0357 | 99.708% | 0 | 0 | 0 | 0 |
| scene-0004 | 145,804 | 5,697 | Either neighbor | 0.0946 | 99.421% | 254 | 42 | 424 | 22 |
| scene-0004 | 145,804 | 5,697 | Same-scan visibility support | 0.0515 | 99.694% | 93 | 42 | 38 | 18 |
| scene-0005 | 199,133 | 9,018 | Native baseline | 0.0265 | 98.876% | 0 | 0 | 0 | 0 |
| scene-0005 | 199,133 | 9,018 | Either neighbor | 0.0256 | 98.850% | 3 | 8 | 176 | 126 |
| scene-0005 | 199,133 | 9,018 | Same-scan visibility support | 0.0281 | 98.911% | 10 | 2 | 4 | 69 |
| scene-0006 | 179,558 | 28,840 | Native baseline | 0.0040 | 96.574% | 0 | 0 | 0 | 0 |
| scene-0006 | 179,558 | 28,840 | Either neighbor | 0.0039 | 96.330% | 9 | 10 | 603 | 235 |
| scene-0006 | 179,558 | 28,840 | Same-scan visibility support | 0.0040 | 96.484% | 0 | 0 | 326 | 191 |

Means are unweighted over these three scenes. Ungated support improves mean F1 through scene-0004 but loses F1 on 0005/0006 and damages static preservation in all three. Visibility-supported inference loses F1 on 0006 and static preservation on 0004/0006. Both baseline and candidates have low recall here. The earlier AV2 gains do not establish robust sparse-sensor transfer; neither candidate is promoted.

Native research scan-ratio masks match the production API exactly in all three scenes. Each candidate uses fingerprint-identical input maps, labels, range settings, and normalized voting. Zero-GT scenes would remain included with the existing zero precision/recall/F1 convention; none of these three has zero GT.

## Post-result diagnostic

The subsequent [point/scan evidence replay](COMMON_MISS_EVIDENCE.md) separates empty angular pixels, nearer returns, height-ratio failures and ground reversion, with API-equivalent masks and saved traces for every evaluated point.

The following channel-only measurements were made after the candidate results, with the map, GT and parameters held fixed. They are diagnostic observations, not extra prospective candidate trials or retuned settings. Baseline intersection metrics were replayed exactly and input hashes rechecked.

| Scene | Range recall | Scan-ratio recall | Range-only moving removals | Scan-ratio-only moving removals | Moving points missed by both |
|---|---:|---:|---:|---:|---:|
| scene-0004 | 39.740% | 3.756% | 2,153 | 103 | 3,330 |
| scene-0005 | 2.428% | 14.294% | 69 | 1,139 | 7,660 |
| scene-0006 | 3.866% | 17.698% | 1,047 | 5,036 | 22,689 |

Both channels miss **33,679 / 43,555 moving-GT points (77.3%)**. Scene-0004 is strongly limited by scan-ratio, while 0005/0006 have low range recall as well. Simply adding inferred columns does not solve the shared evidence gap. This does not prove the existing guards are wrong: occlusion, sparse observations and box-based GT all remain possible contributors. The next diagnostic should reconstruct point/scan visibility and height-ratio evidence on these fixed inputs before changing thresholds.

## Reproduce

Use the repository source matching the plan’s code hashes. Only NumPy is needed after acquiring the dataset; metadata is read incrementally without loading multi-GB JSON arrays into memory. This run used Python 3.12 and NumPy 2.5.3. All **302 tests** passed, including 11 new metadata/selection/provenance tests.

Place official metadata JSON under `data/nuscenes_trainval/v1.0-trainval/`. Source archives:

- https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval_meta.tgz
- https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs_lidar.tgz

The metadata archive SHA-256 is `db48746b10e3544d5ef619eaa3d687e3960626fe1b4422ed856711da5aa7325b`. The plan records hashes for the seven actual metadata tables used. Acquisition in this run read 1,404,221,440 compressed LiDAR bytes before finding all 36 selected keyframes; bounded prefetch fetched additional archive blocks. Full-archive download is not required for the evaluator. Dataset/cache files remain ignored.

Generate the exact required file inventory from metadata before extracting point clouds:

```bash
python scripts/validate_frozen_nuscenes_suite.py inventory \
  --root data/nuscenes_trainval \
  --plan-json examples/cli_validation/nuscenes_frozen_suite_plan.json \
  --plan-sha256 c19221c2e5cd89b0ec6678ae72ac3ba4400337557521b5a4a361051973aaa391 \
  --report-json output/nuscenes-inventory.json
```

Extract the listed `clouds` from the official LiDAR archive into `data/nuscenes_trainval/`, preserving `samples/LIDAR_TOP/...` paths. The inventory depends on sample chains and sensor metadata, not candidate results. Then export manifests and run every frozen candidate:

```bash
python scripts/validate_frozen_nuscenes_suite.py export \
  --root data/nuscenes_trainval \
  --plan-json examples/cli_validation/nuscenes_frozen_suite_plan.json \
  --plan-sha256 c19221c2e5cd89b0ec6678ae72ac3ba4400337557521b5a4a361051973aaa391 \
  --manifest-root output/nuscenes-replay/manifests \
  --report-json output/nuscenes-replay/export.json

python scripts/validate_frozen_nuscenes_suite.py evaluate \
  --root data/nuscenes_trainval \
  --plan-json examples/cli_validation/nuscenes_frozen_suite_plan.json \
  --plan-sha256 c19221c2e5cd89b0ec6678ae72ac3ba4400337557521b5a4a361051973aaa391 \
  --manifest-root output/nuscenes-replay/manifests \
  --report-json output/nuscenes-replay/results.json
```

Use new manifest/report paths. Changed plan digests, code, metadata, raw cloud hashes, frame selections or timestamps fail validation; missing/failing scenes abort rather than disappearing from the mean. Manifests retain the true `nuscenes-trainval` dataset identity, sample/source tokens, raw cloud hashes and undeskewed sensor status. The unchanged nuScenes benchmark exporter supplies the geometry/GT pipeline; its generated dataset label is explicitly replaced with trainval provenance.

For a new prospective experiment, use `freeze --root ... --archive-prefix ... --reference-json examples/error_visualization/neighbor_visibility_supported.json --count 3 --plan-json NEW_PATH`. The prefix must be the first 8 MiB of the official trainval01 LiDAR archive; record the emitted digest before selected-cloud extraction or evaluating candidates.

Source: nuScenes Dataset, https://www.nuscenes.org, CC BY-NC-SA 4.0. This archive-scoped, one-drive check is not an official held-out test split or evidence of general performance across environments.
