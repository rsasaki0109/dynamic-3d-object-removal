# Annotation-based error visualization

[scene_0796.json](scene_0796.json) records an exact baseline replay on real
nuScenes mini `scene-0796`: 12 keyframes, devkit close-point exclusion,
166,983 map points, 9,308 moving-box GT points. The intersection removes 331
moving-GT points, misses 8,977 and removes 796 points outside moving boxes.

| Actual annotation category | GT points | Removed | Missed |
|---|---:|---:|---:|
| Car | 7,695 | 318 | 7,377 |
| Rigid bus | 710 | 13 | 697 |
| Adult pedestrian | 549 | 0 | 549 |
| Truck | 240 | 0 | 240 |
| Motorcycle | 114 | 0 | 114 |

One car track (`69385845cb9747b7afe095177cc405b5`) accounts for 5,064 misses
out of 5,356 GT points. Its median acquisition range is only 3.23 m. Its map
location is far from the first sensor because the accumulated sequence covers
a moving sensor trajectory; map coordinates are not acquisition distance.
This track is a concrete next target for per-frame evidence analysis.

The plot contains full-map error locations, a 24 m crop around the most missed
track, category totals and the top eight missed tracks. Gray static-kept
background is deterministically capped at 15,000 displayed points; all error
points are displayed and all counts use the complete map. Blue is the sensor
trajectory, green removed moving GT, orange missed moving GT and purple
removed points outside moving boxes. The crop may include neighboring tracks.

GT is reconstructed from the real selected-frame annotations and must match
the saved boolean mask exactly. Motion is measured from the first selected
timestamp using the benchmark's strict >2 m cutoff, not annotation file order.
Boxes use the benchmark's 0.25 m margin. Six GT points overlap multiple moving
boxes and are attributed to the first lexicographic instance token so category
and track counts remain additive. This is annotation-box GT, not a detector
output or a per-point motion label. Sweeps remain undeskewed.

Reproduce after obtaining the earlier devkit-selected validation inputs:

```bash
python -m pip install matplotlib
python scripts/visualize_nuscenes_errors.py \
  --manifest output/cli_nuscenes_devkit_selection/scene-0796/manifest.json \
  --validation-root output/cli_nuscenes_devkit_selection/scene-0796/validation \
  --data-root data/nuscenes_mini \
  --output output/scene_0796_errors
```

Use a new output directory. The script saves `errors.png`, `errors.svg` and
`report.json`. It verifies manifest/map/GT hashes and prediction metrics before
rendering. Matplotlib is optional and imported only when rendering; it is not
a core library dependency. Set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable
directories if running in a read-only home environment. Generated plots and
dataset files remain in ignored `output/` and `data/`, not in public demos.

Source: nuScenes mini, [nuScenes](https://www.nuscenes.org), CC BY-NC-SA 4.0.

## Frame-by-frame trace of the highest-miss car

[scene_0796_track_trace.json](scene_0796_track_trace.json) records the votes
for car `69385845cb9747b7afe095177cc405b5`. Evidence reconstruction must match
both public API channel masks and the saved intersection mask exactly.
Each scan votes against the full accumulated map; this includes future map
points and is an offline analysis, not an online replay.

Of this track's 5,356 GT points, range marks 3,160 dynamic and scan-ratio only
399. Their intersection removes 292 and misses 5,064. Range alone flags 2,868
points that scan-ratio protects; 2,089 are kept by both. The range surface guard
protects none of this track; 2,196 have fewer than three see-through votes.
Scan-ratio produces no dynamic votes for 1,985 points and stays below its
final vote threshold for 4,957. Only 142 have fewer than three revisits.

The last two acquisition frames (indices 10 and 11) contain 4,813 GT points,
of which 4,610 are missed: **91.0% of this track's misses**. This makes the
sequence endpoint a concrete next hypothesis. Test additional later query
scans while keeping the evaluated map/GT fixed, rather than mixing a longer
map into the comparison. It remains a hypothesis, not a proven cause.

Across 16,609 observed point-scan opportunities for this track, 5,634 fail
the height-ratio test, 4,391 are ground-reverted, 137 lack sufficient map
height and 6,447 yield a dynamic vote. These are repeated observations of
target map points, not unique points or annotation visibility counts.

```bash
python scripts/trace_nuscenes_track.py \
  --manifest output/cli_nuscenes_devkit_selection/scene-0796/manifest.json \
  --validation-root output/cli_nuscenes_devkit_selection/scene-0796/validation \
  --data-root data/nuscenes_mini \
  --track 69385845cb9747b7afe095177cc405b5 \
  --output output/scene_0796_track_trace
```

The output contains PNG/SVG heatmaps, acquisition-frame removal counts,
query-frame scan-ratio gate counts, fraction matrices in NPZ and the full
report. Heatmap rows are voting query frames, columns are point acquisition
frames, and values are fractions of the target GT in each acquisition frame.
White columns have no target GT and are undefined, not zero or perfect votes.
Frame numbers start at zero; relative timestamp seconds are recorded in JSON.
No filter behavior changes.

## Fixed-map follow-up scan experiment

[scene_0796_followup_scans.json](scene_0796_followup_scans.json) tests the
sequence-end hypothesis without changing the evaluated map, GT, track or
filter settings. Add the next same-cadence keyframes, scene sample indices
36 and 39, about 1.50 and 3.00 seconds after the final baseline scan.
Hashes prove evaluation arrays remain fixed before and after all API calls;
later points and annotations are not added to the evaluated population.

| Added queries | Scene moving GT removed | Scene static falsely removed | Target car removed | Target car missed |
|---|---:|---:|---:|---:|
| 0 (12 queries) | 331 | 796 | 292 | 5,064 |
| 1 (13 queries) | 331 | 803 | 292 | 5,064 |
| 2 (14 queries) | 331 | 803 | 292 | 5,064 |

The target receives 156 range see-through votes from the first extra scan
and none from the second. Neither extra scan revisits any target point's
scan-ratio column. Most target points are still within the 80 m polar range
(5,164 and 5,129 of 5,356 respectively), so the lack of revisits is not simply
that every target point is out of range. Preprocessed query columns are empty
at the target bins. This does not prove physical occlusion or sensor blindness;
the observation definition includes the fixed discretization and preprocessing.

The target's range dynamic count increases only 3,160 → 3,161, while its
scan-ratio dynamic count stays at 399, so the intersection cannot recover
additional target GT. This limited 3-second experiment fails to support a
simple wait-for-two-more-scans solution. Across the scene, the second variant
adds 9 static false removals and recovers 2 baseline false removals, a net +7;
additional evidence is not guaranteed to make removal monotonic.

```bash
python scripts/evaluate_followup_scans.py \
  --manifest output/cli_nuscenes_devkit_selection/scene-0796/manifest.json \
  --validation-root output/cli_nuscenes_devkit_selection/scene-0796/validation \
  --data-root data/nuscenes_mini --stride 3 \
  --track 69385845cb9747b7afe095177cc405b5 \
  --report-json output/scene_0796_followup_scans.json
```

No production changes are made. This is one previously examined track and
scene, not held-out validation or proof about longer delays. Later queries
still affect normalized vote thresholds and surface evidence, while the
parameter values remain fixed. The record includes query file and transformed
point hashes, poses, relative times and individual target evidence counts.

## Empty-column neighbor experiment

[neighbor_columns.json](neighbor_columns.json) tests a research-only support
rule for query columns with no points. Pool height extrema from the immediately
adjacent angular sectors within the same radial ring. Angular adjacency wraps
at 360°, radial rings never mix, and occupied center columns are unchanged.
Compare requiring both adjacent columns to contain points versus allowing
either one. These are inferred column observations, not direct measurements;
they also change the revisit-normalized vote threshold. Range, map/GT inputs,
and all numerical thresholds remain fixed.

| Query support | nuScenes precision | Recall | F1 | Static kept | AV2 recall | AV2 static kept |
|---|---:|---:|---:|---:|---:|---:|
| Native only | 0.5866 | 20.440% | 0.2925 | 99.476% | 16.139% | 99.991% |
| Both adjacent columns | 0.5877 | 20.424% | 0.2924 | 99.475% | 16.046% | 99.991% |
| Either adjacent column | 0.6431 | 30.492% | 0.3898 | 99.384% | 16.093% | 99.991% |

nuScenes means use the same six eligible scenes. On `scene-0796`, either-side
support raises scene recall 3.56% → 18.62%, while static preservation falls
99.50% → 99.23%. This is scene-level evidence, not a claim about the individual
car track. On AV2 it adds 90 true positives but loses 129 baseline true positives,
for a net loss of 39. More inferred revisits can raise a point's threshold,
so additional support does not imply monotonic removal.

The native-only research implementation must match the production per-point
scan-ratio masks exactly on all ten mini scenes and AV2 before results are
accepted. Inputs and baseline metrics are fingerprint-verified. Unit tests
cover angular wrapping, radial isolation, preservation of occupied columns,
and native-mask equivalence with nonzero dynamic points.

The either-side result is promising for the sparse mini data but not a general
default improvement. Scene means are not per-scene guarantees (`scene-0916`
still retains only about 96.78% of static points). These are experiments on
already examined data, with no held-out validation. Borrowed heights can
cross object boundaries and do not prove free space in the empty column.
The helper is not a public API or CLI option; production filters are unchanged.

```bash
python scripts/experiment_neighbor_columns.py \
  --validation-root output/cli_nuscenes_devkit_selection \
  --av2-manifest output/cli_av2/manifest.json \
  --av2-baseline examples/cli_validation/av2_12_sweeps.json \
  --report-json output/neighbor_columns.json
```
