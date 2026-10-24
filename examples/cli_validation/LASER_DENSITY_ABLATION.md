# Controlled AV2 laser-ID density ablation

**Outcome:** thinning alone reduces `range ∩ scan_ratio` F1 on these AV2 logs. Increasing the range-image bin size to 2.5° does not recover intersection F1. No production defaults change.

[Pre-run plan](av2_laser_density_plan.json) · [All scene/phase/channel results](av2_laser_density.json) · [Source frozen AV2 suite](FROZEN_AV2.md)

## Comparison contract

Use all three previously examined AV2 logs, their same 12 scans each, full source poses, sensor-Z preprocessing and point-attached GT. There are 3,507,398 source map points. Read row-aligned `laser_number` from the original Feather files and verify the retained XYZ rows exactly against the frozen manifests. Raw file, manifest, map, GT and code fingerprints are recorded.

Before evaluation, freeze full IDs 0–63, both modulo-2 phases (32 IDs each), and all four modulo-4 phases (16 IDs each). Include every scene/phase, with no selection by GT content or score. These are nominal ID counts, not equal point counts: source sampling and preprocessing vary by laser. All cases have moving GT. Scene/phase means give equal weight to each case, not each point.

Run fixed 1° and 2.5° range-image bins. Every other range and scan-ratio parameter comes from the frozen native AV2 reference; there is no neighbor fusion or tuning after results. Scan-ratio has no angular-bin parameter and its predictions are shared between resolutions. At 1°, all three full-density intersection baselines exactly reproduce the frozen metrics.

Thinning changes the evaluated point population. For each retained population, compare its rerun against **full-density predictions restricted to precisely the same retained rows**, at the same angular resolution. The matched reference prevents a favorable subset of GT from being mistaken for an algorithm improvement. Complementary phases partition the original rows exactly once at each density.

Plan SHA-256: `19fabd3cc644f120fb43a84cff7612316d4970f02a1c18604a993e6f75b2fbf3`.

## Measured intersection results

| Bins | Retained IDs | Scene/phase cases | Rerun F1 | Dense F1 on identical rows | F1 difference | Static-labelled points kept |
|---|---:|---:|---:|---:|---:|---:|
| 1° | 64 | 3 | 0.752938 | 0.752938 | 0 | 99.9388% |
| 1° | 32 | 6 | 0.685088 | 0.749482 | −0.064394 | 99.8901% |
| 1° | 16 | 12 | 0.535782 | 0.748498 | −0.212716 | 99.7793% |
| 2.5° | 64 | 3 | 0.669877 | 0.669877 | 0 | 99.9620% |
| 2.5° | 32 | 6 | 0.625132 | 0.666532 | −0.041399 | 99.9394% |
| 2.5° | 16 | 12 | 0.528475 | 0.665110 | −0.136635 | 99.8763% |

Both channels lose F1 under thinning. At 1°, the 16-ID range channel loses 0.1843 against its matched reference; scan-ratio loses 0.2861. Thus sparse range-image coverage is not the only affected component. Larger bins improve the range channel's absolute F1, but the intersection's F1 remains lower than with 1° at every density. The smaller thinning penalty at 2.5° is measured relative to a weaker dense intersection baseline and is not recovery of the 1° result.

This supports controlled density sensitivity within AV2. It does **not** reproduce a physical 32/16-beam device: field of view, beam elevations, timing, range distribution and calibration remain AV2's. These already examined logs are exploratory controls, not fresh held-out evidence. Source box GT is a proxy for motion. The result cannot apportion the nuScenes gap between density, pose limitations and other dataset differences. The recorded source-origin Z spans are 0.4314, 0.0066 and 0.0860 m; this does not establish perfect pose accuracy.

A useful next development item is a fixed, label-free diagnostic of lost scan-ratio evidence under thinning, using the same matched rows. Changing only range bins does not address that channel's degradation. Any resulting filter proposal needs a separately frozen comparison and fresh scenes before adoption.

## Reproduction and checks

The new script requires NumPy and PyArrow; plotting additionally requires Matplotlib. Retain the original AV2 source files/manifests and frozen code. Use fresh output paths:

```bash
MPLCONFIGDIR=/workspace/work/mpl-cache XDG_CACHE_HOME=/workspace/work/font-cache \
python scripts/experiment_av2_laser_density.py evaluate \
  --frozen-results examples/cli_validation/av2_frozen_suite_results.json \
  --manifest-root output/av2_frozen_suite --data-root data/av2_benchmark \
  --plan-json examples/cli_validation/av2_laser_density_plan.json \
  --plan-sha256 19fabd3cc644f120fb43a84cff7612316d4970f02a1c18604a993e6f75b2fbf3 \
  --output output/av2-laser-density-replay \
  --report-json output/av2-laser-density-replay.json --render
```

For a new pre-run record, use `freeze` with `--frozen-results`, `--manifest-root`, `--data-root`, and a new `--plan-json`; record the printed digest before `evaluate`.

Ignored exports under `output/av2_laser_density/` contain source row indices, subset GT, and Boolean **removed** masks for all methods/resolutions, plus PNG/SVG comparison plots. Masks can be compared directly to `full64` masks indexed by the exported source rows. The checked-in JSON records counts, hashes, per-channel metrics and moving/static decision changes.

All **353 repository tests passed**. Ten new tests cover complementary partitions, invalid IDs, row/GT/origin preservation, empty-frame rejection and matched-population scoring. Independent checks regenerated metrics from every saved mask, verified all subset GT against source rows, checked complementary partitions and aggregate means, and confirmed every original README image/GIF reference remains present.

Follow-up: the [scan-ratio stage diagnosis](DENSITY_SCAN_RATIO_DIAGNOSIS.md) records lost/gained votes, query-column availability and unchanged or lowered thresholds on the same retained rows.
