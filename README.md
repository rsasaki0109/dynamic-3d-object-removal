# Dynamic 3D Object Removal

動的物体検出結果の 3D バウンディングボックスを使って点群を除去するライブラリです。

## 3Dデモ（最初にここ）

- GitHub Pages / 単発実スキャン: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html
- GitHub Pages / 連続比較デモ: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_sequence_standalone.html
- 単発 checked-in 版は実スキャン `demo/actual_scan_20240820_cloud.pcd` と検出 box `demo/actual_scan_20240820_objects.json` から生成
- 連続 checked-in 版は real multi-frame sequence `graph/*/cloud.pcd` を使い、`raw accumulation vs cleaned accumulation` を Pages 上でそのまま比較します
- 連続デモには `ghost hotspot` と `static structure preserved` の証拠パネルを追加し、必要性と安全性を同じ画面で確認できます
- 連続 checked-in 版は、repo に per-frame box が入っていないため temporal consistency で transient outlier を拾い、そこから自動提案した `auto transient boxes` で sampled box-removal preview を作っています

![actual scan removal preview](demo/actual_scan_result_overview.png)

![sequence proof overview](demo/sequence_proof_overview.png)

現在の単発 checked-in デモ結果:
- 入力 24,224 点
- 除去後 23,909 点
- 除去 315 点（実スキャン中の vehicle box 1件）

### 連続デモの見どころ

- `Story mode` を押すと、全体比較 -> ghost hotspot -> static preserved の順に自動で寄って主張を通します

- 左: `Raw accumulation`
  - 各フレームで観測した点をそのまま積むので、動的物体や transient clutter が尾を引きます
- 右: `Cleaned accumulation`
  - checked-in 版では auto transient boxes で cropped した sampled point 群を積むので、`box で消すと map contamination がどう変わるか` をそのまま見られます
- 下段左: `Ghost hotspot`
  - final accumulation の中で raw-only occupancy が最も濃い領域を crop し、raw 側にだけ残る汚染を局所比較します
  - checked-in 版では current frame の transient cluster から自動提案した box も重ねて、どこを box-removal したいのかをその場で示します
- 下段右: `Static structure preserved`
  - cleaned 後も footprint が残る静的領域を crop し、「cleaning がただ削っているだけではない」ことを見せます
- つまり主張は `removed points が赤く見える` ことではなく、`時間方向に積むと地図の締まり方が変わる` ことと `静的構造は残る` ことです

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

### 連続比較デモの再生成

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

- `--input-objects` を渡すと、連続デモの cleaned 側を per-frame box 除去ベースで作れます
- `--input-objects` を渡さない場合は temporal consistency で cleaned accumulation を作ります
- checked-in 版は sampled point 群を HTML に内包しているので、GitHub Pages 上でそのまま再生できます

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
