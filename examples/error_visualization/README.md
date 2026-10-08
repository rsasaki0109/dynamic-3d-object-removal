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
