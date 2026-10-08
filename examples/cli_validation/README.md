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
