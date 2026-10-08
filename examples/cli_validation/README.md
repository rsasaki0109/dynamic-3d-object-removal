# Multi-scan CLI validation

[nuscenes_10_scenes.json](nuscenes_10_scenes.json) records all 10 real nuScenes
mini scenes: 12 keyframes each, stride 3, 2,720,477 accumulated points total.
The sparse 2.5° CLI defaults and a separate ground-protected configuration both
produced exactly the API's point clouds and masks in every scene. The protected
`scene-0757` metrics also matched the existing benchmark exactly.

| Configuration | Eligible scenes | Precision | Recall | F1 | Static kept |
|---|---:|---:|---:|---:|---:|
| CLI defaults | 6 | 0.2971 | 0.2632 | 0.2401 | 93.084% |
| Map-Z ground protection | 6 | 0.2970 | 0.2631 | 0.2401 | 93.087% |

Four scenes with fewer than 5,000 moving-GT points are excluded from the
unweighted mean and retained in the record. Default CLI filter time totaled
6.60 seconds across all 10 scenes; this excludes I/O and process startup and
is one measurement. These are rigid keyframe poses with **no intra-sweep
deskew**. Low accuracy remains visible; this validates implementation
equivalence rather than suitability for live moving-platform use.

```bash
python scripts/validate_nuscenes_cli.py --scenes all --frames 12 --stride 3 \
  --output output/cli_nuscenes_validation
```

This command retains full clouds, GT, masks, commands and summaries in the
ignored output directory. The checked-in JSON keeps metrics, parameters and
input hashes only. Use a new output directory when repeating it.

## Sparse-sensor error diagnosis

### Recall diagnosis with devkit selection fixed

[nuscenes_recall_analysis.json](nuscenes_recall_analysis.json) keeps the same
devkit-selected input hashes and replays baseline metrics in all ten scenes.
On the six eligible scenes, range alone recalls 50.35% of moving GT, while
scan-ratio alone recalls 25.95%; their intersection recalls 20.44%.
Across all scenes, 34,109 moving-GT points are marked dynamic by range but
kept by scan-ratio, versus 4,250 with the opposite disagreement. Both methods
miss another 28,530 moving-GT points.

| Scan-ratio change (range fixed) | Precision | Recall | F1 | Static kept |
|---|---:|---:|---:|---:|
| Baseline | 0.5866 | 0.2044 | 0.2925 | 99.476% |
| Height ratio threshold 0.3 | 0.5736 | 0.2260 | 0.3148 | 99.361% |
| Height ratio threshold 0.4 | 0.5517 | 0.2478 | 0.3296 | 99.092% |
| Revisit vote fraction 0.35 | 0.5607 | 0.2752 | 0.3420 | 99.283% |
| Vote floor 2 | 0.6616 | 0.2492 | 0.3539 | 99.411% |
| 216 sectors | 0.5337 | 0.1791 | 0.2619 | 99.436% |

Vote floor 2 has the largest mean F1 among these one-factor experiments.
[nuscenes_votes_floor_2.json](nuscenes_votes_floor_2.json) verifies its API/CLI
point clouds and masks exactly in all ten scenes. It retains the same input
hashes and moving GT. Static preservation is an average, not a per-scene
guarantee: `scene-0916` falls from 96.82% to 96.71%, and two eligible scenes
fall slightly below 99%. This is exploratory mini-scene tuning without a
held-out dataset; production defaults remain at vote floor 3.

```bash
python scripts/analyze_nuscenes_recall.py \
  --validation-root output/cli_nuscenes_devkit_selection \
  --report-json output/cli_nuscenes_recall.json
python scripts/validate_nuscenes_cli.py --scenes all --frames 12 --stride 3 \
  --min-distance 1 --scan-ratio-votes-floor 2 \
  --output output/cli_nuscenes_votes_floor_2
```

For direct CLI use, `--scan-ratio-votes-floor 2` is already available; the
validator now accepts the same explicit override and labels the resulting
unprotected run `votes_floor_2` rather than `defaults`.

### Cross-dataset check of the frozen vote-floor candidate

[av2_votes_floor_2.json](av2_votes_floor_2.json) tests the nuScenes-selected
vote floor 2 on the previously recorded real AV2 scene without retuning.
The 12-sweep map, GT and manifest hashes exactly match the earlier AV2 record.
Range settings stay at the AV2 reference's 1° resolution, 3 see-through scans,
3 maximum surface confirmations and ground Z -1.4 m. Only the scan-ratio vote
floor changes. Both runs have exact API/CLI masks, points and parameters.

| AV2 configuration | Precision | Recall | F1 | Static kept | Moving GT removed |
|---|---:|---:|---:|---:|---:|
| Vote floor 3 | 0.992429 | 0.161393 | 0.277635 | 99.990965% | 13,633 |
| Vote floor 2 | 0.992432 | 0.161452 | 0.277723 | 99.990965% | 13,638 |

The candidate adds only five true positives and no false positives in this
scene. The larger nuScenes recall gain therefore does not recur here. One
previously benchmarked AV2 scene is not broad held-out validation; keep the
candidate explicit and avoid a universal default change. API timings cover
scan-ratio only; CLI timings cover both filters.

```bash
python scripts/validate_vote_floor.py \
  --manifest output/cli_av2/manifest.json \
  --baseline-summary examples/cli_validation/av2_12_sweeps.json \
  --candidate-floor 2 --output output/cli_av2_votes_floor_2
```

The validator rejects changed input fingerprints, replays baseline metrics,
and retains CLI commands, masks and summaries in a new output directory.

### Moving GT missed by both baseline filters

[common_miss_diagnosis.json](common_miss_diagnosis.json) reconstructs range
and scan-ratio votes, requiring exact agreement with public API masks and
recorded baseline metrics on all ten devkit-selected nuScenes scenes and the
AV2 scene. Inputs are fingerprint-verified and defaults are unchanged.

| Common-miss diagnosis | nuScenes, all 10 scenes | AV2, one scene |
|---|---:|---:|
| Moving GT missed by both | 28,530 | 37,274 |
| Fewer than 3 range see-through votes | 27,933 (97.9%) | 25,021 (67.1%) |
| Range surface guard active | 1,935 | 14,477 |
| Fewer revisits than scan-ratio vote floor | 10,839 | 1,092 |
| No scan-ratio dynamic votes | 18,783 | 15,871 |

Range reason masks overlap; do not add those rows. These are counts across
all ten scenes, including the four excluded from accuracy means.

For these common misses, each observed point-scan opportunity is assigned
to exactly one scan-ratio stage. This denominator counts repeated observations
of points, not unique points:

| Scan-ratio stage | nuScenes opportunities | AV2 opportunities |
|---|---:|---:|
| Map height too small | 3,812 | 691 |
| Query/map height ratio too large | 67,977 (53.4%) | 319,532 (79.0%) |
| Ground reversion | 37,819 (29.7%) | 33,494 (8.3%) |
| Dynamic vote produced | 17,603 | 50,554 |
| Total observed opportunities | 127,211 | 404,271 |

nuScenes is often blocked by too few range see-through votes; AV2 also has
substantial surface protection. In both datasets the scan-ratio height-ratio
test often prevents votes from being produced. Lowering the final vote floor
cannot rescue a point with zero votes. These measurements identify gates,
not proof that a gate is wrong: occlusion and ground protection can be
appropriate, and GT uses moving-instance boxes rather than per-point motion.
Further experiments should target evidence formation while tracking static
false removals, rather than assuming one threshold fits both datasets.

```bash
python scripts/diagnose_common_misses.py \
  --validation-root output/cli_nuscenes_devkit_selection \
  --av2-manifest output/cli_av2/manifest.json \
  --av2-baseline examples/cli_validation/av2_12_sweeps.json \
  --report-json output/common_misses.json
```

### One-factor evidence experiments

[evidence_ablations.json](evidence_ablations.json) compares six observation
conditions on the same fingerprint-verified inputs, keeping voting and the
other channel fixed. nuScenes values are the usual six-scene unweighted mean;
AV2 is the single reference scene. Baseline replay and vote reconstruction
still agree exactly with the public APIs.

| Evidence change | nuScenes recall | nuScenes static kept | AV2 recall | AV2 static kept |
|---|---:|---:|---:|---:|
| Baseline | 20.440% | 99.476% | 16.139% | 99.991% |
| See-through scans 3 → 2 | 22.265% | 99.115% | 16.506% | 99.989% |
| Range margin 0.5 → 0.25 m | 20.691% | 99.427% | 16.381% | 99.990% |
| Both angular resolutions halved | 16.861% | 99.521% | 16.573% | 99.987% |
| Both angular resolutions doubled | 17.528% | 99.658% | 15.636% | 99.996% |
| Minimum map height 0.5 → 0.25 m | 20.440% | 99.475% | 16.139% | 99.991% |
| Ground margin 0.2 → 0.1 m | 21.237% | 99.380% | 16.594% | 99.990% |

Reducing the see-through requirement improves mean recall in both datasets,
but also removes more static points. Finer resolution moves recall in opposite
directions between datasets. Lowering the minimum map height has essentially
no benefit. These changes do not provide a large shared recall gain, so no
production defaults or presets change. All variants are exploratory comparisons
on already examined data, without held-out validation. Mean static retention
is not a per-scene guarantee. Only baseline CLI equivalence is claimed here;
the experimental variants are API measurements.

```bash
python scripts/diagnose_common_misses.py \
  --validation-root output/cli_nuscenes_devkit_selection \
  --av2-manifest output/cli_av2/manifest.json \
  --av2-baseline examples/cli_validation/av2_12_sweeps.json \
  --ablate-evidence --report-json output/evidence_ablations.json
```

### Sensor-axis range-image experiment

[range_frame_comparison.json](range_frame_comparison.json) compares the current
map-axis spherical pixels with per-scan sensor-axis pixels. Each map point is
transformed by `(map_point - translation) @ sensor_to_map_rotation`; query
points use their native sensor coordinates. Both use the same angular
resolution, range margin and voting thresholds. Scan-ratio remains unchanged,
and the ground guard still uses map-frame Z. No deskew is added.

All ten nuScenes inputs and the AV2 scene retain identical fingerprints, and
map-axis baseline metrics replay exactly. The synthetic tests check posed
visibility, map-row ordering, map-frame ground protection and quaternion/matrix
rotation agreement.

| Frame / method | nuScenes recall | nuScenes F1 | nuScenes static kept | AV2 recall | AV2 F1 | AV2 static kept |
|---|---:|---:|---:|---:|---:|---:|
| Map axes, range only | 50.349% | 0.4726 | 96.120% | 53.702% | 0.5998 | 98.139% |
| Sensor axes, range only | 48.290% | 0.4475 | 95.637% | 53.111% | 0.5854 | 97.920% |
| Map axes, intersection | 20.440% | 0.2925 | 99.476% | 16.139% | 0.2776 | 99.991% |
| Sensor axes, intersection | 21.053% | 0.2995 | 99.467% | 16.233% | 0.2790 | 99.992% |

The intersection gains a little recall, but range alone loses F1 and static
preservation in both datasets. This does not support replacing the production
projection. These are API research measurements on already examined data;
the sensor-axis helper is not a new public API or CLI option. nuScenes means
use the same six eligible scenes; AV2 is one scene. Neither is held-out
validation of the selected experiment.

```bash
python scripts/compare_range_frames.py \
  --validation-root output/cli_nuscenes_devkit_selection \
  --av2-manifest output/cli_av2/manifest.json \
  --av2-baseline examples/cli_validation/av2_12_sweeps.json \
  --report-json output/range_frame_comparison.json
```

### Close-point preprocessing record

The follow-up [nuscenes_devkit_selection.json](nuscenes_devkit_selection.json)
uses the official nuScenes devkit multisweep close-point rule before pose
alignment: discard points where **both** `abs(sensor_x) < 1 m` and
`abs(sensor_y) < 1 m`. This is an XY square, independent of Z, not a 2 m
spherical guard. The source is
[PointCloud.from_file_multisweep/remove_close](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py).
The benchmark had omitted this preprocessing. Close points are consistent
with self returns or invalid near-sensor returns; their semantic identity is
not proven by the geometry alone.

This removes 1,054,321 points, leaving 1,666,156 across all ten scenes. Moving
GT point counts are unchanged in every scene, and both CLI configurations
match their APIs exactly. The six eligible scenes remain the same.

| Selection | Precision | Recall | F1 | Static kept |
|---|---:|---:|---:|---:|
| Legacy, close points included | 0.2971 | 0.2632 | 0.2401 | 93.084% |
| Devkit close-point exclusion | 0.5866 | 0.2044 | 0.2925 | 99.476% |

These are different input populations, not a same-input regression comparison.
Close-point exclusion changes the range and scan-ratio evidence as well as
the static evaluation denominator; it is not equivalent to preserving those
points after filtering. All accuracy and timing limitations above still apply.
`--min-distance 0` preserves legacy benchmark selection. Core filter defaults
are unchanged; `--min-distance 1` is explicit in the new record.

```bash
python scripts/validate_nuscenes_cli.py --scenes all --frames 12 --stride 3 \
  --min-distance 1 --output output/cli_nuscenes_devkit_selection
```

[nuscenes_error_analysis.json](nuscenes_error_analysis.json) replays the baseline
exactly in all 10 scenes and measures distance/height bins and four exploratory
alternatives. Of 244,581 baseline false positives across all scenes, 236,778
(96.8%) were acquired less than 2 m from their source sensor. The worst scene
is `scene-0916`, with 78,661 false positives, including 74,875 inside 2 m.
This is consistent with ego-vehicle returns contaminating the static category,
but source identity has not been verified. Here "static" means outside the
moving-instance annotation boxes, not a semantic static label.

| Configuration | Precision | Recall | F1 | Static kept |
|---|---:|---:|---:|---:|
| Baseline | 0.2971 | 0.2632 | 0.2401 | 93.084% |
| Maximum surface confirmations: 2 | 0.3368 | 0.1415 | 0.1642 | 93.374% |
| Minimum see-through scans: 5 | 0.3041 | 0.1839 | 0.2132 | 97.035% |
| Resolution consensus: 1° and 2.5° | 0.3457 | 0.1634 | 0.1947 | 96.483% |
| Protect acquisition range below 2 m (diagnostic) | 0.6920 | 0.2632 | 0.3393 | 99.836% |

Means use the same six eligible scenes. The last experiment needs each map
point's source scan, which an arbitrary accumulated map does not provide.
It preserves near-sensor points rather than proving they are static; it is
not a new CLI option or a recommendation for production. These exploratory
measurements use the same mini scenes without held-out validation. Existing
defaults remain unchanged. Verify ego returns and define an explicit
evaluation exclusion before drawing conclusions about static preservation.

```bash
python scripts/analyze_nuscenes_errors.py \
  --validation-root output/cli_nuscenes_validation \
  --report-json output/cli_nuscenes_error_analysis.json
```

[av2_12_sweeps.json](av2_12_sweeps.json) records a real AV2 CLI validation:
12 sweeps, stride 3, 1,235,563 map points, 84,471 moving-GT points.
Both CLI modes produced exactly the same point clouds and boolean keep masks as
their API equivalents. Fusion, range, and scan-ratio reference metrics matched
the existing AV2 benchmark.

| Method | Precision | Recall | F1 | Static kept | Filter time |
|---|---:|---:|---:|---:|---:|
| fusion, short-window | 0.651 | 0.663 | 0.657 | 97.388% | 126.82 s |
| range ∩ scan-ratio, AV2 settings | 0.992 | 0.161 | 0.278 | 99.991% | 5.00 s |

This is one dense 64-beam scene. The intersection used the AV2 reference's
1° range image and ground settings, not the sparse-sensor CLI defaults.
Its low recall is retained in the report. Timings are one measurement with
two fusion workers, excluding process startup and I/O.

Reproduce from the repository root:

```bash
python scripts/run_av2_benchmark.py --frames 12 --stride 3 --fusion-workers 2 \
  --online-manifest output/cli_av2/manifest.json \
  --summary-json output/cli_av2/reference.json
python scripts/validate_multiscan_cli.py \
  --manifest output/cli_av2/manifest.json --stride 3 --workers 2 \
  --reference-summary output/cli_av2/reference.json \
  --output-dir output/cli_av2/validation
python scripts/compare_benchmark_results.py \
  --baseline examples/cli_validation/av2_12_sweeps.json \
  --candidate output/cli_av2/validation/candidate_cli.json \
  --report-json output/cli_av2/validation/regression_report.json
```

Use a new or empty validation output directory. Install the benchmark extras
first (`pip install -e ".[benchmarks]"`). Public unsigned S3 access is needed
for acquisition; no dataset is committed here. The JSON retains all effective
parameters and hashes of the manifest, map, and GT. Regenerated inputs must
match those fingerprints for a regression comparison.

## Coordinate-safe AV2 ground preprocessing

New AV2 benchmark runs exclude points with ego-frame Z <= -1.4 m **before**
pose transformation. The range channel then uses `ground_z=None`: the ego cutoff
must not be reused as an absolute city-map elevation. Benchmark configuration
now distinguishes `sensor_ground_z=-1.4` and
`ground_preprocessing_frame=ego_before_pose_transform` from `ground_z=null`.
The optional sensor-aware ablation likewise does not classify retained map
points by absolute city Z.

`validate_multiscan_cli.py` uses no absolute range ground guard for new runs
without a reference. When replaying a reference, it honors the recorded
`ground_z`, including legacy numeric values (or the historical -1.4 fallback
for old references lacking the key). Old records remain unchanged and must not
be mixed with new metrics without acknowledging the configuration change.
This changes AV2 benchmark/validation defaults, not core API/CLI defaults.
Source sensor-height exclusion is still a simple heuristic, not a local terrain
estimator; the library scan-ratio channel retains its per-column ground logic.

A regression scene with a removable moving point produces identical range plus
scan-ratio masks after translating all poses 20 m downward. Both translated and
untranslated runs also verify exact API/CLI mask agreement. The new real log
is replayed to validate the corrected path separately from the earlier frozen
neighbor evaluation. Generated outputs remain ignored.

```bash
python scripts/validate_multiscan_cli.py \
  --manifest output/av2_unseen_02a/manifest.json \
  --output-dir output/av2_unseen_02a/coordinate_fix_validation --workers 2
python scripts/run_av2_benchmark.py \
  --scene 02a00399-3857-444e-8db3-a8f58489c394 --frames 12 --stride 3 \
  --fusion-workers 2 \
  --summary-json output/av2_unseen_02a/coordinate_fix_benchmark.json
```

Use new output locations. The export manifest can be reproduced with the
command above in the new-log evaluation section; the fix itself does not
require that evaluation PR's helper.

[av2_coordinate_ground_fix.json](av2_coordinate_ground_fix.json) records the
corrected benchmark and API/CLI replay on `02a00399…`: range plus scan-ratio
recall 67.440%, F1 0.7832, static preservation 99.955%, 7,502 moving removals
and 532 static false removals. All API/CLI point masks match. This reproduces
the earlier diagnostic result; it is a corrected replay, not fresh held-out
validation. The full benchmark method metrics and coordinate-specific settings
are included so historical configurations remain distinguishable.
