# Dynamic 3D Object Removal

動的物体検出の 3D バウンディングボックスを使って、点群から除去対象点を取り除きます。

## 最短セットアップ

```bash
git clone git@github.com:rsasaki0109/dynamic-3d-object-removal.git
cd dynamic-3d-object-removal
python3 -m pip install -e .
# ROS2を使う場合
# python3 -m pip install -e .[ros2]
```

## 使い方

```bash
dynamic-object-removal \
  --input-cloud path/to/input.xyz \
  --input-objects path/to/objects.json \
  --output-cloud path/to/output.xyz
```

## デモ（再現）

```bash
cd demo
python3 run_demo.py
python3 -m http.server 4173
```

ブラウザで `http://127.0.0.1:4173/index_3d.html` を開く。

最新を再生成して開き直すには:

```bash
cd demo
./open_latest_report.sh --safe
```

## 対応データ

- 点群: `*.xyz`, `*.pcd`, `*.csv`, `*.txt`, `*.npy`
- 物体: JSON 配列、または `{ "objects": [...] }`

物体1件例:
- `center`: `[x, y, z]`
- `size` または `dimensions`: `[dx, dy, dz]`
- `yaw` または `orientation`

## 3Dビュー

- `demo/index_3d_standalone.html`: ファイルを直接開いて確認
- `demo/index_3d.html`: `demo_scene.json` を読み込むサーバ版

どちらも、
- Input/Kept/Removed 表示切替
- BBox 表示切替
- 点サイズ/透過率調整
- 回転・ズーム
- PNG 保存

を備えています。

## ライセンス

MIT
