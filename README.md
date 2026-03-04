# Dynamic 3D Object Removal

動的物体検出結果の 3D バウンディングボックスを使って点群を除去するライブラリです。

## インストール

```bash
git clone git@github.com:rsasaki0109/dynamic-3d-object-removal.git
cd dynamic-3d-object-removal
python3 -m pip install -e .
```

## ライブラリ API

```python
from pathlib import Path
from dynamic_object_removal import load_points, load_boxes, remove_points_in_boxes, save_points

points = load_points(Path("demo/demo_input.xyz"), fmt="auto")
boxes = load_boxes(Path("demo/demo_objects.json"), fmt="auto", skip_invalid=True)
kept, keep_mask = remove_points_in_boxes(points, boxes, margin=(0.05, 0.05, 0.05))
removed = points[~keep_mask]

save_points(Path("demo/demo_output.xyz"), kept, fmt="auto")
```

公開 API（主要）
- `load_points(path, fmt="auto")`
- `load_boxes(path, fmt="auto", skip_invalid=False)`
- `remove_points_in_boxes(points, boxes, margin=(0.05,0.05,0.05))`
- `TemporalConsistencyFilter(voxel_size=0.10, window_size=5, min_hits=3)`  
- `save_points(path, points, fmt="auto")`

## CLI で使う

```bash
dynamic-object-removal \
  --input-cloud demo/demo_input.xyz \
  --input-objects demo/demo_objects.json \
  --output-cloud demo/demo_output.xyz
```

```bash
dynamic-object-removal --help
```

## 3Dデモ

- GitHub Pages: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html
- ローカル: `python3 demo/run_demo.py` を実行して `demo/demo_...` を更新、ブラウザで `demo/index_3d_standalone.html` を開く

## 参考アルゴリズム

- [UTS-RI/dynamic_object_detection](https://github.com/UTS-RI/dynamic_object_detection)
