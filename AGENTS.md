# Repository Guidance

## Purpose

This repository is a small library plus public demos for dynamic object removal in 3D point clouds.

The most important public artifact is the GitHub Pages sequence demo:

- `demo/index_3d_sequence_standalone.html`

Treat it as a proof demo, not as a generic point cloud viewer.

## Read First

Before making demo changes, read:

1. `README.md`
2. `plan.md`

`plan.md` is the handoff note that explains the current proof structure, what is already solved, and what work is still high value.

## Main Files

Core library:

- `dynamic_object_removal.py`

Single-scan demo:

- `demo/run_scan_demo.py`
- `demo/index_3d_standalone.html`
- `demo/index_3d_scan_standalone.html`

Sequence proof demo:

- `demo/run_scan_sequence_demo.py`
- `demo/index_3d_sequence_standalone.html`

Landing pages:

- `README.md`
- `index.html`
- `demo/index.html`

## Working Rules

### Preserve the thesis

The sequence demo should prove:

- raw accumulation creates ghost contamination
- cleaned accumulation reduces that contamination
- stable structure is preserved

Do not drift into viewer work that does not strengthen that argument.

### Prefer data improvements over UI polishing

The highest-value next step is to replace the checked-in sampled preview path with real per-frame detections or box annotations if such data exists locally.

If real per-frame boxes do not exist, avoid low-value cosmetic iteration.

### Be precise about what the checked-in sequence is

At the current handoff state, the checked-in sequence is a real multi-frame sequence, but its cleaned side may still be driven by temporal consistency plus derived auto transient boxes when real per-frame boxes are absent.

Do not claim the checked-in sequence uses real detections unless the data and regeneration path actually do.

### Keep labels honest

There was earlier confusion between single scans and map point clouds.

Do not relabel map data as scans or scans as maps.

### Tune for proof clarity, not brute force

Do not switch public demos to misleading "full point cloud" defaults just because higher density is available.

The public pages should stay usable on GitHub Pages and should communicate the point quickly.

## Regeneration Commands

Single-scan demo:

```bash
python3 demo/run_scan_demo.py \
  --input-cloud demo/actual_scan_20240820_cloud.pcd \
  --input-objects demo/actual_scan_20240820_objects.json \
  --max-render-points 220000 \
  --output-scene demo/demo_scene_single_scan.json \
  --output-html demo/index_3d_standalone.html
```

Sequence demo:

```bash
python3 demo/run_scan_sequence_demo.py \
  --input-glob "/path/to/graph/*/cloud.pcd" \
  --frame-count 12 \
  --stride 1 \
  --max-render-points 9000 \
  --fps 4 \
  --voxel-size 0.35 \
  --window-size 5 \
  --min-hits 3 \
  --output-html demo/index_3d_sequence_standalone.html
```

If real per-frame boxes are available, pass:

```bash
--input-objects /path/to/objects.json
```

## Visual Verification

When changing public-facing demo visuals, verify with screenshots instead of assuming correctness.

Local check:

```bash
python3 -m http.server 8765
```

Then capture:

```bash
npx playwright screenshot --device="Desktop Chrome" --wait-for-timeout=2200 --full-page http://127.0.0.1:8765/demo/index_3d_sequence_standalone.html /tmp/local_sequence.png
```

Use a longer wait for story-mode timing checks when needed.

## Commit Hygiene

Do not commit generated cache directories:

- `__pycache__/`
- `demo/__pycache__/`

If you update README or landing-page copy, sharpen the message instead of expanding it without clear proof value.
