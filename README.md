# Dynamic 3D Object Removal

動的物体の 3D バウンディングボックスを使って、点群から対象領域を除去するライブラリと可視化デモです。

## まずはこれ

メインの公開デモ:

- Sequence proof demo: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_sequence_standalone.html

単発スキャン版:

- Single-scan demo: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html

Sequence demo では次の 3 点を見せます。

- 動的物体による不要な点群が accumulated map に残る
- cleaned accumulation はそれを抑える
- 静的構造は残る

![actual scan removal preview](demo/actual_scan_result_overview.png)

![sequence proof overview](demo/sequence_proof_overview.png)

補足:

- 公開中の sequence demo は repo 内に per-frame box JSON が無いため、cleaned 側を `temporal consistency` ベースで作っています
- per-frame box がある場合は `--input-objects` を渡して box-driven な sequence を再生成できます

## インストール

```bash
git clone git@github.com:rsasaki0109/dynamic-3d-object-removal.git
cd dynamic-3d-object-removal
python3 -m pip install -e .
```

## デモ再生成

### 単発スキャン

```bash
python3 demo/run_scan_demo.py \
  --input-cloud demo/actual_scan_20240820_cloud.pcd \
  --input-objects demo/actual_scan_20240820_objects.json \
  --max-render-points 220000 \
  --output-scene demo/demo_scene_single_scan.json \
  --output-html demo/index_3d_standalone.html
```

### Sequence

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

- `--input-objects` を渡すと、cleaned 側を per-frame box 除去ベースで生成できます
- `--input-objects` は 1 つの box payload でも `frame name -> payload` の map JSON でも受けられます
- checked-in HTML は sampled point 群を内包する self-contained 形式です

## CLI

```bash
dynamic-object-removal \
  --input-cloud /path/to/scan.pcd \
  --input-objects /path/to/objects.json \
  --output-cloud /path/to/output.xyz
```

```bash
dynamic-object-removal --help
```

## ライブラリ API

```python
from pathlib import Path
from dynamic_object_removal import load_points, load_boxes, remove_points_in_boxes, save_points

points = load_points(Path("/path/to/scan.pcd"), fmt="auto")
boxes = load_boxes(Path("/path/to/objects.json"), fmt="auto", skip_invalid=True)
kept, keep_mask = remove_points_in_boxes(points, boxes, margin=(0.05, 0.05, 0.05))

save_points(Path("/path/to/output.xyz"), kept, fmt="auto")
```

主な公開 API:

- `load_points(path, fmt="auto")`
- `load_boxes(path, fmt="auto", skip_invalid=False)`
- `remove_points_in_boxes(points, boxes, margin=(0.05, 0.05, 0.05))`
- `TemporalConsistencyFilter(voxel_size=0.10, window_size=5, min_hits=3)`
- `save_points(path, fmt="auto")`

## 対応形式

- `PCD` の ASCII / binary に対応
- `DATA binary_compressed` は未対応

## 参考

- [UTS-RI/dynamic_object_detection](https://github.com/UTS-RI/dynamic_object_detection)
