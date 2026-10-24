# Point/scan evidence for frozen nuScenes common misses

This is a post-result diagnosis of the fixed scene-0004/0005/0006 baseline, not a new candidate comparison. Maps, GT, sensor preprocessing, thresholds and production filters are unchanged. All three scenes share one drive.

[Frozen comparison](FROZEN_NUSCENES.md) · [Full evidence counts and code/input fingerprints](nuscenes_common_miss_evidence.json)

## Findings

- Common misses: **33,679** moving-box GT points, with **404,148 point/scan opportunities** across 12 query frames.
- Every common miss has fewer than three range see-through votes. **22,327 (66.3%)** have none. Only six also exceed the surface-confirmation guard; no point is blocked by that guard alone.
- Of the **305,062 observed scan-ratio opportunities**, **210,969 (69.2%)** fail the height-ratio test; **19,050 (6.2%)** are ground-reverted. Only 448 fail minimum map height.
- **6,630** common-miss points have zero scan-ratio dynamic votes. The other **27,049** have positive votes below their normalized threshold. Only **1,254 (3.7%)** have fewer than three column revisits; this overlaps the vote categories.

The immediate bottleneck is evidence formation: sufficient range see-through observations and height-column evidence are missing. These counts do not establish that the guards are incorrect, nor justify relaxing them. Empty pixels and nearer returns are different situations, and inferred neighboring heights are not direct observations of free space.

## Range observations

These categories partition all common-miss point/scan opportunities. They use the same map-axis angular bins, nearest-return rule and strict margin comparisons as production. A nearer return in a shared pixel does **not** prove physical occlusion along the exact ray. Different directions can fall in one 2.5-degree pixel.

| Range state | Opportunities | Share |
|---|---:|---:|
| `invalid_map_range` | 0 | 0.00% |
| `empty_angular_pixel` | 84,175 | 20.83% |
| `nearer_return` | 242,504 | 60.00% |
| `margin_boundary` | 0 | 0.00% |
| `surface` | 62,414 | 15.44% |
| `see_through` | 15,055 | 3.73% |

`see_through` uses query-nearest range minus map-point range > 0.5 m. `surface` uses absolute difference < 0.5 m. Equality at either margin is `margin_boundary`; `nearer_return` uses a difference < -0.5 m. Invalid map ranges and empty angular pixels are kept separate. Per-query see-through and surface masks match the production helper exactly.

## Scene-level blockers

| Scene | Common-miss points | Zero see-through points | Empty-pixel opportunities | Nearer-return opportunities | Height-ratio failures / observed SR opportunities | Ground-reverted opportunities |
|---|---:|---:|---:|---:|---:|---:|
| scene-0004 | 3,330 | 942 | 18,705 | 13,064 | 7,796 / 11,588 (67.3%) | 1,945 |
| scene-0005 | 7,660 | 6,379 | 3,039 | 72,309 | 58,500 / 79,882 (73.2%) | 9,607 |
| scene-0006 | 22,689 | 15,006 | 62,431 | 157,131 | 144,673 / 213,592 (67.7%) | 7,498 |

Only 726 common-miss points (2.2%) were acquired in the final two frames. The population is not concentrated only at the sequence endpoint; this is an acquisition-count observation, not an experiment proving whether later scans help.

## Auditable traces

The diagnostic preserves **all 524,495 evaluated points** and partitions them into six groups:

| Group | Points |
|---|---:|
| `static_kept` | 473,231 |
| `static_removed` | 7,709 |
| `moving_both_dynamic` | 329 |
| `moving_range_only` | 3,269 |
| `moving_scan_ratio_only` | 6,278 |
| `moving_common_miss` | 33,679 |

The static-removed control group means false positives against the existing moving-box GT; it is not a semantic inventory of static objects. Group differences are selected by final outcomes and must not be treated as independent evidence for a newly fitted threshold.

Each generated `scene-XXXX/point_scan_evidence.npz` contains:

- `map_index`, map-frame `xyz`, `gt_dynamic`, and `group_id` for every point.
- `source_frame` and `source_row`: acquisition frame and row in its **preprocessed manifest cloud**, not its original unfiltered raw BIN file.
- `range_state` and `scan_ratio_state`: matrices with query scans as rows and input-map points as columns. State-code legends are in the JSON report.
- Range see-through/surface vote counts; scan-ratio votes, revisits and per-point normalized thresholds; reconstructed final channel masks.
- Query timestamps relative to the first frame and query-by-acquisition matrices for common misses.

Scan-ratio ground fits always use the full accumulated map, retaining static context. State partitions cover every point/scan pair. Reconstructed final range and scan-ratio masks match both public APIs exactly; intersection metrics match the frozen baseline. Manifest/map/GT hashes must match the earlier result before any trace is written.

The PNG/SVG heatmap has query scans as rows and point-acquisition frames as columns. Each cell is a fraction of the common-miss points acquired in that column’s frame; zero-population columns are undefined/white, not zero. It is an offline full-map analysis, including future points. PNG/SVG and per-point NPZ assets stay in ignored `output/`; the compact JSON record is retained here.

## Reproduce

The subsequent [nearest-return attribution](NEARER_RETURN_ATTRIBUTION.md) matches these events to query-time annotation tracks and official lidarseg classes, while retaining ambiguous cases separately.

Use the manifests from the [frozen sparse-sensor reproduction](FROZEN_NUSCENES.md#reproduce). No dataset download or metadata re-export is needed if those manifests remain locally available. NumPy is the only required dependency for diagnosis; `--render` additionally needs Matplotlib.

```bash
mkdir -p work/mpl-cache work/font-cache
MPLCONFIGDIR=work/mpl-cache XDG_CACHE_HOME=work/font-cache \
python scripts/diagnose_frozen_nuscenes_misses.py \
  --frozen-results examples/cli_validation/nuscenes_frozen_suite_results.json \
  --manifest-root output/nuscenes_frozen_suite/manifests \
  --output output/nuscenes_common_miss_evidence_replay \
  --report-json output/nuscenes_common_miss_evidence_replay.json --render
```

Choose new output/report paths. The tool rejects changed frozen code, incomplete/reordered scene sets, changed input fingerprints, invalid GT, or mismatched baseline metrics. The summary records diagnostic-code hashes and SHA-256 for every generated point/scan NPZ. Six new tests cover strict range margins, nearest-return selection, stage partitions, empty heatmap populations, source-row alignment, fingerprint/metric failure and incomplete-scene rejection. All **308 repository tests passed**.

Limits: three correlated scenes, moving-instance-box GT, undeskewed keyframes, map-axis angular pixels and offline future-point evidence. This diagnoses where votes are lost; it does not prove why the real sensor lacked evidence or that either protection should be weakened. The next geometry check should distinguish empty pixels from closer returns and test whether those returns belong to the same tracked object or nearby structures, while preserving the fixed baseline for comparison.
