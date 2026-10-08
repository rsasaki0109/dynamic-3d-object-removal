# Clean your first map

This guide uses the current source checkout and Python 3.9 or newer. It needs
only NumPy, no GPU, detector, ROS2, account or dataset download. Run commands
from the repository root. The multi-scan CLI may be newer than the PyPI release.

## Install from source

```bash
git clone https://github.com/rsasaki0109/dynamic-3d-object-removal.git
cd dynamic-3d-object-removal
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate` instead. `python -m` ensures
you run the CLI from the same Python environment used to install dependencies.

## Run a download-free example

```bash
python examples/quickstart/create_sample.py --output output/first-map
python examples/quickstart/prepare_map.py \
  --manifest output/first-map/scans.json --output output/first-map/map.npy
python -m dynamic_object_removal \
  --algorithm range_scan_ratio \
  --input-map output/first-map/map.npy --input-manifest output/first-map/scans.json \
  --output-cloud output/first-map/cleaned.npy \
  --output-mask output/first-map/keep.npy --summary-json output/first-map/summary.json
python examples/quickstart/inspect_result.py \
  --input-map output/first-map/map.npy --cleaned output/first-map/cleaned.npy \
  --mask output/first-map/keep.npy
```

Expected: **4 scans, 9 input points, 8 kept, 1 removed**. The last command checks
that saved points exactly equal the input selected by the keep mask.

This tiny synthetic column exercises the command and file formats. It is not
real LiDAR, a realistic sensor simulation, or an accuracy benchmark. Use a new
sample directory when rerunning: the input preparation helpers reject existing
outputs. The CLI itself can overwrite outputs, so use a fresh output location
when comparing settings.

## Replace the sample with your scans

You need multiple scans of an overlapping region, plus known sensor poses.
For the simplest input, save each **deskewed sensor-frame** scan as a finite
`N x 3` NumPy array of XYZ in meters. No labels or intensity column are needed.
PCD, XYZ and other supported cloud formats can also be referenced by the
manifest. The prepared accumulated map is saved as XYZ-only NPY.

Create `data/my-scans/scans.json` with one entry per scan:

```json
{
  "sensor_profile": {"name": "my LiDAR", "deskewed": true},
  "frames": [
    {
      "cloud": "000.npy",
      "pose": {"translation": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}
    },
    {
      "cloud": "001.npy",
      "pose": {"translation": [1, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}
    }
  ]
}
```

Those poses are format examples: replace them with your actual poses and add
the remaining scans. Two entries alone are insufficient for the default
three-vote removal floor. Start with a sequence with several revisits; about
12 overlapping scans is a useful first experiment, not a quality guarantee.

Paths are relative to the JSON file. The pose maps sensor coordinates into a
common fixed map frame: `p_map = R @ p_sensor + translation`. Quaternion order
is **x, y, z, w**. Use a proper 3x3 `rotation` instead if that is how your system
exports poses. For scans already transformed into the map frame, use
`"sensor_origin": [x, y, z]` instead of `pose`; the origin must still be the
sensor's actual position in that frame. Do not transform a cloud twice.

`deskewed: true` is a statement about your input, not an instruction to correct
it. These tools do not estimate poses or deskew rotating-LiDAR sweeps. If those
are missing, obtain them from your acquisition/SLAM pipeline before relying
on moving-platform results. Do not set the field to true just to pass validation.

```bash
python examples/quickstart/prepare_map.py \
  --manifest data/my-scans/scans.json --output output/my-first-run/map.npy
python -m dynamic_object_removal \
  --algorithm range_scan_ratio \
  --input-map output/my-first-run/map.npy --input-manifest data/my-scans/scans.json \
  --output-cloud output/my-first-run/cleaned.npy \
  --output-mask output/my-first-run/keep.npy --summary-json output/my-first-run/summary.json
python examples/quickstart/inspect_result.py \
  --input-map output/my-first-run/map.npy --cleaned output/my-first-run/cleaned.npy \
  --mask output/my-first-run/keep.npy
```

This accumulation preserves all points and duplicates, in scan order. It does
not downsample or estimate ground. If you already have an aligned accumulated
map, you can use it directly as `--input-map`; the manifest's scans and origins
must be in that same frame. The mask then follows that map's original point order.

## Decide whether the result is useful

`range_scan_ratio` removes only points both range visibility and scan-ratio
mark dynamic. It is a conservative starting workflow with defaults evaluated
on sparse 32-beam data; match range resolution to your sensor and inspect the
result. For dense 64-beam-class data, compare `fusion --preset short-window`
with about 12 scans, or `fusion --preset long-map` for long sequences. See the
[method and preset guidance](../../README.md#clean-a-map-using-multiple-scans).

The mask is `true` for kept points. The summary records effective settings and
removal counts, **not accuracy**. Inspect removed points as well as cleaned
points: look for reduced moving-object trails and preserved walls, poles and
parked vehicles. Zero removals can mean insufficient revisits or that neither
channel has enough evidence. High removal counts can include static damage.
Without ground truth, do not interpret a removal percentage as recall.

For an external point viewer, export raw, kept and removed XYZ files separately:

```python
import numpy as np
points = np.load("output/my-first-run/map.npy")
keep = np.load("output/my-first-run/keep.npy")
np.savetxt("output/my-first-run/raw.xyz", points, fmt="%.6f")
np.savetxt("output/my-first-run/kept.xyz", points[keep], fmt="%.6f")
np.savetxt("output/my-first-run/removed.xyz", points[~keep], fmt="%.6f")
```

Large text files can be slow to export/open. XYZ contains coordinates only;
these helpers do not preserve intensity, timestamps or other point attributes.
Record settings and use fresh output paths when comparing algorithms. Public
real-data accuracy records are in [CLI validation](../cli_validation/README.md).
