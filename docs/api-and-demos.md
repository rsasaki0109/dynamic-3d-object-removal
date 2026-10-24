# Python API and demo regeneration

Run commands from the repository root. [Back to README](../README.md).

## Python API

```python
from pathlib import Path
from dynamic_object_removal import load_points, load_boxes, remove_points_in_boxes, save_points

points = load_points(Path("/path/to/scan.pcd"), fmt="auto")
boxes = load_boxes(Path("/path/to/objects.json"), fmt="auto", skip_invalid=True)
kept, keep_mask = remove_points_in_boxes(points, boxes, margin=(0.05, 0.05, 0.05))

save_points(Path("/path/to/output.xyz"), kept, fmt="auto")

# The same APIs also support the minimal form used by the CLI examples:
points = load_points(Path("scan.pcd"), fmt="auto")
boxes = load_boxes(Path("objects.json"), fmt="auto")
cleaned, keep = remove_points_in_boxes(points, boxes)
save_points(Path("cleaned.pcd"), cleaned, fmt="auto")
```

Main public APIs:

- `load_points(path, fmt="auto")` / `load_boxes(path, fmt="auto", skip_invalid=False)` / `save_points(path, fmt="auto")`
- `remove_points_in_boxes(points, boxes, margin=(0.05, 0.05, 0.05))`
- `TemporalConsistencyFilter(voxel_size=0.10, window_size=5, min_hits=3, visibility=False)`
- `remove_ghost_by_range_image(map_points, query_points, sensor_origin, range_margin=0.5)` — single map-vs-scan visibility removal
- `clean_map_by_visibility(map_points, scans, min_see_through=2, max_surface_hits=2, ground_z=None, resolutions=None)` — multi-scan map cleaner (remove + revert)
- `remove_dynamic_by_scan_ratio(map_points, query_points, sensor_origin, scan_ratio_threshold=0.2, ground_margin=0.2)` — single map-vs-scan scan-ratio removal
- `clean_map_by_scan_ratio(map_points, scans, scan_ratio_threshold=0.2, min_votes=None, votes_fraction=0.5, votes_floor=3)` — multi-scan scan-ratio cleaner (`min_votes=None` = majority of each point's column revisits)
- `clean_map_by_fusion(map_points, scans, workers=1)` — highest-accuracy map cleaner
- `RangeImageGhostFilter(window_size=5, range_margin=0.5)` — streaming range-image filter for ROS2

### Range-image visibility removal

```python
# scans: list of (points_in_map_frame, sensor_origin) from the sweeps that built the map.
kept, keep_mask = clean_map_by_visibility(
    map_points, scans,
    range_margin=0.5, min_see_through=2, max_surface_hits=2, ground_z=-1.4,
)
```

A point is removed only when enough scans see *through* it **and** few confirm it as a real surface (the Removert-style *revert* guard). For higher precision pass `resolutions=[2.5, 4.0]` (multi-resolution consensus: AV2 precision 0.68 → 0.78). Try it in the playground's **Range mode**.

### Scan-ratio (pseudo-occupancy) removal

```python
kept, keep_mask = clean_map_by_scan_ratio(
    map_points, scans,
    scan_ratio_threshold=0.2, min_map_height=0.5, ground_margin=0.2,
)
```

ERASOR-style and independent of visibility: a polar column that is tall in the map but flat in a live sweep held a moving object; above-ground points are removed, the ground reverted by a per-column plane fit. Strongest on dense (64-beam+) LiDAR; on sparse sensors prefer `range` or raise `votes_fraction`.

### Free-space fusion (highest accuracy)

```python
kept, keep_mask = clean_map_by_fusion(map_points, scans, workers=6)
```

Fusion ORs three independent dynamic-evidence channels: ray-sampled free-space carving with per-scan hit precedence, DUFOMap-style eroded-void confirmation, and scan-ratio votes. Fractional free-space voting catches transient traffic while absolute void counts catch slower movers; the union reaches KITTI AA **98.6 / 98.0**. Carving is the cost: minutes per hundred 64-beam scans with `workers=6`, versus seconds for `range`/`scan_ratio`. For short windows (~12 scans), use `free_votes_fraction=0.7`, `free_votes_floor=3`, and `void_min_scans=4`; on sparse 32-beam sensors, prefer `range`.

Main APIs: `TemporalConsistencyFilter`, `RangeImageGhostFilter`, `clean_map_by_visibility`, `clean_map_by_scan_ratio`, and `clean_map_by_fusion`. See their docstrings and [`dynamic_object_removal.py`](../dynamic_object_removal.py).

Supported point clouds: PCD (ASCII/binary), CSV, TXT, XYZ, NPY, KITTI BIN, and AV2 Feather. Boxes: JSON, CSV, KITTI `label_2`, and AV2 Feather. `PCD DATA binary_compressed` is not supported.

## Demo Regeneration

```bash
python3 demo/run_scan_demo.py \
  --input-cloud demo/actual_scan_20240820_cloud.pcd \
  --input-objects demo/actual_scan_20240820_objects.json \
  --max-render-points 220000 \
  --output-scene demo/demo_scene_single_scan.json \
  --output-html demo/index_3d_standalone.html

python3 demo/run_scan_sequence_demo.py \
  --input-glob "/path/to/graph/*/cloud.pcd" \
  --frame-count 12 --stride 1 --max-render-points 9000 --fps 4 \
  --voxel-size 0.35 --window-size 5 --min-hits 3 \
  --output-html demo/index_3d_sequence_standalone.html
```

The checked-in HTML demos are self-contained and embed sampled point data.

More demos: [single scan](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html) · [temporal sequence](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_sequence_standalone.html).
