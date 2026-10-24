# Scan-ratio evidence under AV2 laser-ID thinning

**Outcome:** moving-GT removal losses are dominated by empty query columns, while additional static-GT removals gain votes mainly through the height-ratio gate. Total dynamic votes actually increase under thinning. A blanket reduction of vote thresholds would not address both effects safely; no filter or default is changed.

[Fixed diagnostic plan](av2_density_scan_ratio_plan.json) · [Every scene/phase and stage transition](av2_density_scan_ratio.json) · [Preceding density comparison](LASER_DENSITY_ABLATION.md)

## Fixed replay

Replay all three previously examined AV2 scenes and every complementary 32-ID/16-ID phase from the density experiment. Keep original poses, preprocessing, polar grid, height and voting thresholds. Trace inference without GT; use GT only afterward to group decision changes. Full-density predictions are restricted to exactly the same retained source rows as each sparse prediction.

Each point/query-frame opportunity receives one disjoint state: outside polar domain, empty query column, insufficient map height, rejected height ratio, ground reverted, or dynamic vote. Map/query per-bin height and point counts are retained alongside the states. The dynamic state uses the production ground fit. Aggregate decisions are reconstructed from these votes and normalized thresholds, then verified against the production API and every archived scan-ratio mask at both angular resolutions. Scan-ratio itself has no range-image angular-resolution parameter.

The plan digest recorded before this replay is `c4304dfb1b2c7db7fc2031d6e61d8f37e40029567b3ec54b6749f05a3eebc95a`. This is descriptive post-ablation diagnosis on examined scenes, not a new performance experiment.

## Moving-GT removals lost

| Nominal retained IDs | Scene/phase cases | Points no longer removed | Lost vote events | Empty query column | Ground revert | Height-ratio rejection | Insufficient map height |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 6 | 12,492 | 65,422 | 61,548 (94.1%) | 3,493 | 360 | 21 |
| 16 | 12 | 24,446 | 163,524 | 153,959 (94.1%) | 8,211 | 544 | 810 |

Events count repeated point/query-frame opportunities, not distinct objects or columns. The point groups here contain **only previously removed moving-GT points whose final removal is lost**. They also gain 1,373 and 6,725 votes elsewhere, respectively; the JSON records both directions and the full transition matrices.

Thinning cannot create a previously unobserved query column. Observed revisit counts and normalized thresholds never increase; this is checked for every retained point. Among the lost moving-GT points, thresholds decrease for 10,747/12,492 points at 32 IDs and 22,850/24,446 at 16 IDs. Lost removals therefore cannot be explained by a threshold increase. The unchanged absolute vote floor can still constrain poorly observed points; this record does not test a different floor.

## Additional static-GT removals

| Nominal retained IDs | Newly removed static-GT points | Recovered static-GT points | Net extra static-GT removals | Gained vote events in newly removed group | Previously height-ratio rejected |
|---|---:|---:|---:|---:|---:|
| 32 | 16,687 | 4,413 | 12,274 | 57,681 | 57,268 (99.3%) |
| 16 | 53,107 | 6,311 | 46,796 | 256,160 | 255,058 (99.6%) |

These are scan-ratio channel changes, not intersection changes. Moving-GT gains are 3,897 and 5,843 points; they do not offset the removal losses above.

Across **all retained points**, total dynamic votes rise from 1,106,831 to 1,227,761 at 32 IDs and to 1,537,787 at 16 IDs. Complementary phases partition the source rows, so the dense total is identical at both levels. Sparse evidence does not simply mean fewer dynamic votes: empty columns destroy useful votes for some points, while height-ratio transitions create other votes and false removals. Thinning changes both map and query height profiles; these state transitions alone do not isolate their separate causal contributions or prove any physical object motion.

## Artifacts, validation and closing scope

`scripts/diagnose_av2_density_scan_ratio.py` provides `freeze` and `evaluate`. It verifies source manifests/map/GT/raw laser files against the previous frozen results and records code/input fingerprints. Ignored `output/av2_density_scan_ratio/<scene>/` exports contain `dense_evidence.npz` and each phase's evidence: per-point states, votes, revisits, thresholds and removed masks; per-bin map/query heights and counts; and sparse source indices/GT. Heights are indexed by query-frame and polar bin, states by query-frame and map row.

Replay with the cached source data and fresh output paths:

```bash
python scripts/diagnose_av2_density_scan_ratio.py evaluate \
  --density-results examples/cli_validation/av2_laser_density.json \
  --frozen-results examples/cli_validation/av2_frozen_suite_results.json \
  --density-root output/av2_laser_density --manifest-root output/av2_frozen_suite \
  --data-root data/av2_benchmark \
  --plan-json examples/cli_validation/av2_density_scan_ratio_plan.json \
  --plan-sha256 c4304dfb1b2c7db7fc2031d6e61d8f37e40029567b3ec54b6749f05a3eebc95a \
  --output output/density-diagnosis-replay \
  --report-json output/density-diagnosis-replay.json
```

For a new plan, use `freeze` with the same input arguments and a new `--plan-json`; omit evaluation-only digest/output/report options. NumPy and PyArrow suffice. All **358 tests pass**, including known ground/body separation, missing/low/height-ratio stages, bin profiles, transition accounting and observation/threshold monotonicity. Every real replay also matches previous decision-change counts.

This closes the current README and validation work: the README remains 78.4% shorter with all seven original media references retained; detailed usage and evidence live in linked documents. Existing algorithms, defaults and public demo media remain unchanged. Negative results are retained rather than promoted.

The next development proposal is a separately frozen comparison that distinguishes map-only from query-only thinning before proposing a confidence gate for low-support columns. Any candidate must report both moving-GT recall and static preservation, then pass fresh-scene validation. No such candidate is introduced in this change.
