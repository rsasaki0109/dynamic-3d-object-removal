# Frozen three-log AV2 validation

Candidates and parameters were frozen before downloading the selected logs or inspecting their metrics. All selected scenes remain in the report, including zero-moving-GT scenes. No production filters or defaults change.

## Selection and input contract

- Source: the public AV2 Sensor Dataset val catalogue (150 logs).
- Excluded: scenes mentioned in tracked repository text, local `data/av2_benchmark` directory names, and local `output` paths/manifests (five catalogue logs). This records project-local prior exposure, not proof about all external use.
- Selection: first three remaining log IDs in lexicographic order, without moving-content screening or result-based replacement.
- Each log: first 12 usable annotation/pose timestamps, stride 3. Ground preprocessing excludes ego-frame Z <= -1.4 m before alignment; the map-frame range ground guard is `null`.
- GT: points in annotation boxes of tracks displaced >2 m over the window, with 0.2 m box margin. These are box-based motion labels, not per-point motion annotations.
- Parameters come from the corrected AV2 reference. All three candidates share inputs, range settings, and normalized voting. Visibility support reuses the range evidence; it is not an independent measurement.

Frozen plan SHA-256: `e20ef5ba45dfc23397ca6810e30beaaea0f9853d52952d4ed26f604280c2d271`.

[Plan, selected/excluded logs, settings, and code hashes](av2_frozen_suite_plan.json) · [Full results and input hashes](av2_frozen_suite_results.json) · [Corrected parameter reference](av2_coordinate_ground_fix.json)

## Results

Means are unweighted over all three selected scenes, with no minimum-GT exclusion. Zero-GT scenes use the existing metric convention of zero precision/recall/F1 and remain explicitly listed. Static preservation is also reported per scene. These conditions differ from earlier screened AV2 and eligible nuScenes means.

| Method | Mean precision | Mean recall | Mean F1 | Mean static kept | Worst-scene static kept |
|---|---:|---:|---:|---:|---:|
| Native baseline | 0.9715 | 0.6157 | 0.7529 | 99.939% | 99.873% |
| Either neighbor | 0.9730 | 0.6327 | 0.7661 | 99.941% | 99.878% |
| Same-scan visibility support | 0.9720 | 0.6184 | 0.7551 | 99.940% | 99.875% |

| Scene | Map points | Moving GT | Method | F1 | Static kept | Extra moving removed | Lost moving removed | Extra static removed | Recovered static |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `070bbf42…` | 1,187,917 | 22,957 | Native baseline | 0.7934 | 99.958% | 0 | 0 | 0 | 0 |
| `070bbf42…` | 1,187,917 | 22,957 | Either neighbor | 0.8037 | 99.958% | 383 | 47 | 18 | 16 |
| `070bbf42…` | 1,187,917 | 22,957 | Same-scan visibility support | 0.7966 | 99.958% | 158 | 58 | 2 | 8 |
| `0aa4e8f5…` | 1,235,633 | 19,444 | Native baseline | 0.7493 | 99.986% | 0 | 0 | 0 | 0 |
| `0aa4e8f5…` | 1,235,633 | 19,444 | Either neighbor | 0.7493 | 99.986% | 0 | 0 | 0 | 1 |
| `0aa4e8f5…` | 1,235,633 | 19,444 | Same-scan visibility support | 0.7493 | 99.986% | 0 | 0 | 0 | 0 |
| `0b86f508…` | 1,083,848 | 54,750 | Native baseline | 0.7162 | 99.873% | 0 | 0 | 0 | 0 |
| `0b86f508…` | 1,083,848 | 54,750 | Either neighbor | 0.7453 | 99.878% | 2,234 | 238 | 22 | 79 |
| `0b86f508…` | 1,083,848 | 54,750 | Same-scan visibility support | 0.7194 | 99.875% | 325 | 119 | 7 | 34 |

Any negative F1 or static-preservation delta is listed in the full report, even when a mean improves. No pass/fail threshold or production promotion is inferred after inspecting these results. Native research masks must match the production scan-ratio API exactly; all candidates are evaluated on fingerprint-identical arrays.

Both candidates improve mean F1 on these three logs. Ungated neighbor support has the larger gain but increases net static false removals by two points in `070bbf42…` (18 added, 16 recovered). Visibility-supported inference has no F1 or static-preservation regression in this sample, with a smaller recall gain. Earlier AV2 records contain recall losses for that candidate, so these results do not establish a universal improvement. Neither experiment validates sparse-sensor transfer or justifies changing defaults.

## Reproduce

Install benchmark extras (`pip install -e ".[benchmarks]"`) and use the source revision whose byte-level code hashes match the plan. Acquisition requires public unsigned S3 access. Dataset and generated cloud files stay in ignored `data/` and `output/`. Commands below run from the repository root.

This run used Python 3.12, NumPy 2.5.3 and PyArrow 25.0.1. All 291 repository tests passed, including five new tests for deterministic selection, frozen-plan/code validation, zero-GT inclusion, per-scene regression reporting, and overwrite protection.

```bash
for scene in 070bbf42-31d3-3aa9-aca4-c262afc9077d \
             0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2 \
             0b86f508-5df9-4a46-bc59-5b9536dbde9f; do
  python scripts/run_av2_benchmark.py --scene "$scene" \
    --frames 12 --stride 3 --moving-thresh 2 \
    --online-manifest "output/av2_frozen_suite/$scene/manifest.json" --online-only
done

python scripts/validate_frozen_av2_suite.py evaluate \
  --plan-json examples/cli_validation/av2_frozen_suite_plan.json \
  --plan-sha256 e20ef5ba45dfc23397ca6810e30beaaea0f9853d52952d4ed26f604280c2d271 \
  --manifest-root output/av2_frozen_suite --data-root data/av2_benchmark \
  --report-json output/av2_frozen_suite/replayed.json
```

Choose a new report path. Evaluation rejects a changed plan digest, changed filter/exporter/evaluator code, different scene ordering, different timestamps, malformed GT, or existing report. Missing/failing scenes abort the evaluation instead of silently shrinking the sample. NumPy asset hashes and source annotation/pose hashes are recorded; absolute manifest paths are not needed for metric reproduction.

To register a future experiment before acquisition, supply a public catalogue JSON (`source_url`, `scenes`), a JSON list of previously examined scene IDs, and the corrected reference:

```bash
python scripts/validate_frozen_av2_suite.py freeze \
  --catalogue-json work/catalogue.json --seen-json work/seen.json \
  --reference-json examples/cli_validation/av2_coordinate_ground_fix.json \
  --count 3 --plan-json work/new-plan.json
```

Record the printed digest before acquiring those inputs. The freeze helper trusts the supplied catalogue/exposure list; build that list from repository references and local prior records before looking at candidate results. The checked-in plan retains the exact catalogue and exclusions used in this run.

## Limits

Three new dense-sensor AV2 val logs are a transfer check, not an official test split, a representative dataset, or sparse-LiDAR validation. Offline maps include future points. Borrowed columns can cross object boundaries, and extra inferred revisits change normalized vote thresholds. The visibility gate is an angular-bin approximation, not exact ray traversal. A default change needs broader validation with settings fixed in advance.
