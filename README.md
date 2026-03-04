# Dynamic 3D Object Removal

動的物体検出結果の 3D バウンディングボックスを使って点群を除去するライブラリです。

## 3Dデモ（最初にここ）

- GitHub Pages: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html
- 単一スキャン用: `demo/index_3d_scan_standalone.html`（`run_scan_demo.py` で生成）

### 単一スキャンでの検証候補（tsukubachallenge）

- [map_utsukuba22_university_of_tsukuba.pcd](https://drive.google.com/file/d/1mSi6OP2p4jFwK3vVgvFIbCdxPmN4UcVg/view?usp=sharing)
  - 目安: 224 MB / 14.6M 点 / 2022年（扱いやすい）
- [map_tc19_furo.pcd](https://drive.google.com/file/d/1mH20dXpnBBlQ6hMKJZqdVhphrffsvWK_/view?usp=sharing)
  - 目安: 683 MB / 22.3M 点 / 2019年（街中シーン）
- [map_tc18_furo.pcd](https://drive.google.com/file/d/1c7Vd4vkMudAHyxc0ZOZCbTgx8ZFZ_Slx/view?usp=sharing)
  - 目安: 519 MB / 17.0M 点 / 2018年

どれも `tsukubachallenge/tc-datasets` 側の配布形式に合わせて取得してください。
現状の `run_scan_demo.py` は `dynamic_object_removal.load_points()` を経由するため、`PCD` は
ASCII 形式を前提にしています（`DATA binary` の場合は事前変換が必要です）。

### 外部点群のデモ化（新）

```bash
python3 demo/run_scan_demo.py \
  --input-cloud /path/to/map_utsukuba22_university_of_tsukuba.pcd \
  --input-objects /path/to/objects.json \
  --max-render-points 220000 \
  --output-scene demo/demo_scene_single_scan.json \
  --output-html demo/index_3d_scan_standalone.html
```

- `--input-objects` は省略可能です（省略時は「除去なしの可視化」）
- 出力ファイル `--output-html` を GitHub Pages で置く場合は、リンクを更新してください

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
