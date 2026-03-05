# Dynamic 3D Object Removal

動的物体検出結果の 3D バウンディングボックスを使って点群を除去するライブラリです。

## 3Dデモ（最初にここ）

- GitHub Pages / 単発実スキャン: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html
- GitHub Pages / 連続フレーム: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_sequence_standalone.html
- 単発 checked-in 版は実スキャン `demo/actual_scan_20240820_cloud.pcd` と検出 box `demo/actual_scan_20240820_objects.json` から生成
- 連続 checked-in 版は local multi-frame sequence `graph/*/cloud.pcd` を sampled 埋め込みして再生します

![actual scan removal preview](demo/actual_scan_result_overview.png)

現在の単発 checked-in デモ結果:
- 入力 24,224 点
- 除去後 23,909 点
- 除去 315 点（実スキャン中の vehicle box 1件）

### 外部地図点群での検証候補（tsukubachallenge map）

- [map_utsukuba22_university_of_tsukuba.pcd](https://drive.google.com/file/d/1mSi6OP2p4jFwK3vVgvFIbCdxPmN4UcVg/view?usp=sharing)
  - 目安: 224 MB / 14.6M 点 / 2022年（扱いやすい）
- [map_tc19_furo.pcd](https://drive.google.com/file/d/1mH20dXpnBBlQ6hMKJZqdVhphrffsvWK_/view?usp=sharing)
  - 目安: 683 MB / 22.3M 点 / 2019年（街中シーン）
- [map_tc18_furo.pcd](https://drive.google.com/file/d/1c7Vd4vkMudAHyxc0ZOZCbTgx8ZFZ_Slx/view?usp=sharing)
  - 目安: 519 MB / 17.0M 点 / 2018年

どれも `tsukubachallenge/tc-datasets` 側の配布形式に合わせて取得してください。
現状の `run_scan_demo.py` / `run_scan_sequence_demo.py` は `dynamic_object_removal.load_points()` を経由するため、`PCD` は
ASCII / binary に対応しています（`DATA binary_compressed` は未対応です）。

### 単一スキャンの再生成

```bash
python3 demo/run_scan_demo.py \
  --input-cloud demo/actual_scan_20240820_cloud.pcd \
  --input-objects demo/actual_scan_20240820_objects.json \
  --max-render-points 220000 \
  --output-scene demo/demo_scene_single_scan.json \
  --output-html demo/index_3d_standalone.html
```

### 連続フレームの再生成

```bash
python3 demo/run_scan_sequence_demo.py \
  --input-glob "/path/to/graph/*/cloud.pcd" \
  --frame-count 12 \
  --stride 1 \
  --max-render-points 12000 \
  --fps 4 \
  --output-html demo/index_3d_sequence_standalone.html
```

- `--input-objects` を付けると、global boxes または `frame_name -> boxes` JSON を使って kept / removed / 3D box も連続再生できます
- checked-in 版は Pages でそのまま再生できるように sampled point 群を HTML に内包しています

### 外部点群の単発デモ化

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

points = load_points(Path("/path/to/scan.pcd"), fmt="auto")
boxes = load_boxes(Path("/path/to/objects.json"), fmt="auto", skip_invalid=True)
kept, keep_mask = remove_points_in_boxes(points, boxes, margin=(0.05, 0.05, 0.05))
removed = points[~keep_mask]

save_points(Path("/path/to/output.xyz"), kept, fmt="auto")
```

公開 API（主要）
- `load_points(path, fmt="auto")`
- `load_boxes(path, fmt="auto", skip_invalid=False)`
- `remove_points_in_boxes(points, boxes, margin=(0.05,0.05,0.05))`
- `TemporalConsistencyFilter(voxel_size=0.10, window_size=5, min_hits=3)`
- `save_points(path, fmt="auto")`

## CLI で使う

```bash
dynamic-object-removal \
  --input-cloud /path/to/scan.pcd \
  --input-objects /path/to/objects.json \
  --output-cloud /path/to/output.xyz
```

```bash
dynamic-object-removal --help
```

## 参考アルゴリズム

- [UTS-RI/dynamic_object_detection](https://github.com/UTS-RI/dynamic_object_detection)
