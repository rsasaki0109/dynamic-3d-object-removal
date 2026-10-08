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
