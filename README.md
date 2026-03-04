# Dynamic Object Removal (Python)

`dynamic-object-removal` は、`dynamic_object_removal` 系の手法をベースにした
**簡易版の点群動的物体消去 CLI**です。
検出物体の 3D バウンディングボックス内に含まれる点を取り除き、静的点群を残します。

## インストール

```bash
cd /workspace/ai_coding_ws/academic_research_ws/oss/dynamic_object_removal
python3 -m pip install -e .
```

ROS2版を使う場合:

```bash
python3 -m pip install -e .[ros2]
```

## 使い方

```bash
dynamic-object-removal \
  --input-cloud /path/to/map.xyz \
  --input-objects /path/to/objects.json \
  --output-cloud /path/to/map_no_dynamic.xyz
```

### リアルタイム版

ROS2 の `PointCloud2`（＋検出 JSON）向けノード版を用意しています。

```bash
dynamic-object-removal-realtime \
  --algorithm box \
  --pointcloud-topic /points_raw \
  --objects-topic /dynamic_objects \
  --objects-msg-type std_msgs.msg.String \
  --output-topic /points_static
```

`objects` トピックは `std_msgs/msg/String` の JSON 以外に、  
`detection`系メッセージ（`Detection3DArray` や `BoundingBoxArray`）から
座標/サイズ/向き情報を取り出せるフォーマットにも対応しやすいよう、
`--objects-msg-type` で型を固定指定できます（例: `std_msgs.msg.String`, `vision_msgs.msg.Detection3DArray`）。

検出メッセージ側で `header.stamp` や `stamp` を持っている場合は、
点群時刻と近いものを拾って使うようにしているので、少しズレたトピックでも実用的に動きやすくなっています。

### ベンチマーク

1ファイルでオフライン性能を測るコマンドです。

```bash
dynamic-object-removal-bench \
  --algorithm box \
  --input-cloud /path/to/map.xyz \
  --input-objects /path/to/objects.json \
  --iterations 300
```

### デモ（サンプルデータ）

同梱のスクリプトで、入力点群・除去対象・除去後点群、および比較画像を生成できます。

```bash
cd <repo_root>/oss/dynamic_object_removal
python3 demo/run_demo.py
```

ブラウザで確認するなら:

```bash
cd <repo_root>/oss/dynamic_object_removal/demo
python3 -m http.server 4173
```

その後ブラウザで `http://localhost:4173/index.html` を開く。

旧レポートを閉じて最新を開くには、更新と同時にクリーン実行するユーティリティも使えます。

```bash
cd <repo_root>/oss/dynamic_object_removal/demo
./open_latest_report.sh
```

このスクリプトは、先に以下を実行してからレポートを再生成します。

- `python3 -m http.server 4173` を終了
- 以前の `localhost:4173/index.html` / `index_3d(_standalone).html` を参照しているブラウザを終了
- `demo/run_demo.py` 実行後、`demo/index_3d.html`（HTTPサーバ経由）を開く
- 必要なら `demo/index_3d_standalone.html`（ファイル直接開き）を使う

既定は安全優先(`--safe`)で、対象がデモ関連ページに限定されます。必要なら `--force` を指定して
広めにブラウザ終了対象を含めることもできます。

```bash
./open_latest_report.sh --safe
./open_latest_report.sh --force
./open_latest_report.sh --help
```

2つ表示され続ける場合は、`--force` で旧プロセスを広めに除去してから表示してください。

出力:

- `demo/demo_input.xyz`
- `demo/demo_objects.json`
- `demo/demo_output.xyz`
- `demo/demo_before_after.png`
- `demo/demo_comparison.png`
- `demo/demo_before_after_3d.png`
- `demo/demo_scene.json`
- `demo/index_3d_standalone.html`
- `demo/index_3d.html`

サンプル実行の例では、`11000 -> 8633` 点を残し、`2367` 点が削除されています。

![除去前後の比較（2分割）](demo/demo_before_after.png)

![全体比較図（要約）](demo/demo_comparison.png)

![3D除去結果](demo/demo_before_after_3d.png)

インタラクティブ3D（回転・ズーム可）:

- `demo/index_3d.html`
- `demo/index_3d_standalone.html`

`index_3d_standalone.html` は Three.js 製の Web3D スタジオ形式です。  
`index_3d.html` も同じUI構成（点サイズ、透過率、レイヤーON/OFF、bbox、PNG保存）を使った
サーバ版です。  
`index_3d.html` は `demo_scene.json` を `HTTP` で読み込むため、同一オリジン配信（`python3 -m http.server`）で開いてください。

- input/kept/removed の表示ON/OFF
- bbox表示ON/OFF
- 点サイズ調整

サーバが使えない環境では:

- `demo/index_3d_standalone.html`（ファイルを直接開いて表示）

### 主要オプション

- `--cloud-format {auto,csv,pcd,text,npy}`
- `--objects-format {auto,json,csv}`
- `--box-margin x y z`  
  バウンディングボックスに加える安全マージン（m）
- `--min-size`  
  この値より小さいサイズを持つ物体を除外
- `--skip-invalid`  
  不正な検出行をスキップ
- `--summary-json path`  
  処理結果を保存
- `--algorithm {box,temporal}`  
  `box` は bbox 消去、`temporal` はフレーム履歴による簡易動的除去

### リアルタイム版専用オプション

- `--pointcloud-topic`, `--objects-topic`, `--output-topic`
- `--objects-msg-type`  
  `objects` トピックの ROS メッセージ型を指定（未指定時は `std_msgs.msg.String`）
  `JSON` の場合は `stamp` を含めると点群時刻と高精度に同期できます。
- `--max-object-history`  
  受信した検出結果のキャッシュ件数
- `--algorithm {box,temporal}`
- `--box-stale-time`
- `--voxel-size`, `--temporal-window`, `--temporal-min-hits`
- `--stats-period`, `--summary-json`

## 入力: point cloud

- `*.csv` / `*.txt` / `*.xyz` / `*.pts` / `*.pcd`（ASCII） / `*.npy`
- `xyz` 以外の列が含まれている場合でも先頭3列を `x,y,z` として扱います。

## 入力: objects

### JSON 形式

配列または以下のどれか:
- `{"objects":[...]}`
- `{"detections":[...]}`
- `{"boxes":[...]}`

各要素は下記キーを含む辞書を想定しています。

```json
[
  {
    "center": [1.2, 0.8, 0.0],
    "size": [2.0, 1.0, 1.8],
    "yaw": 0.12,
    "label": "car"
  },
  {
    "position": {"x": 5.0, "y": -1.1, "z": 0.0},
    "dimensions": {"x": 1.6, "y": 1.8, "z": 1.5},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.03, "w": 0.999},
    "label": "pedestrian"
  }
]
```

- `center`/`position` のいずれかでボックス中心を指定
- `size`/`dimensions`/`length,width,height` などでサイズを指定
- `yaw` もしくは `orientation`（quaternion）で姿勢を指定

### 代替アルゴリズム補足（temporal）

- `box`  
  検出ボックス内を除去する既定方式。
- `temporal`  
  直近 `window` フレームで同一ボクセルに継続出現した点のみ残し、短時間だけ現れる動的点を除去する方式。

### CSV 形式（ヘッダあり）

最小ヘッダ例:

```csv
x,y,z,length,width,height,yaw,label
1.2,0.8,0.0,2.0,1.0,1.8,0.12,car
```

## 参考

本実装は、検出された物体バウンディングボックスを用いて点群をクロップし除去する既存手法を参考にしています。

## OSS 公開向けメモ（簡易スタック）

- 目的
  - 点群から動的物体を除去するデモと、3D可視化（比較ビュー）を一つの手順で再現できるようにすること。
- 依存
  - Python（実行環境）
  - 追加外部サービスは不要（`index_3d_standalone.html` は単一HTMLで動作）
  - 3D対話表示は `index_3d.html` を `HTTP` 経由で配信して表示

### クイックリプレイ

```bash
git clone <repo-url>
cd <repo_root>/oss/dynamic_object_removal/demo
python3 run_demo.py
python3 -m http.server 4173
```

ブラウザで `http://127.0.0.1:4173/index_3d.html` を開く。

- `run_demo.py` の出力物
  - `demo_scene.json`（`index_3d.html` の入力データ）
  - `index_3d_standalone.html`（ファイル直接開封用）
  - `demo_before_after_3d.png`（静的比較画像）

### 開発者向け運用（推奨）

- `./open_latest_report.sh --safe`
  - 旧レポート用プロセスを除去し、最新データを再生成して `index_3d.html` を開きます。
- `./open_latest_report.sh --force`
  - 同上。より広い既知ブラウザ/プロセスを除去したい場合。

### OSS公開時の運用方針

- 可能なら `index_3d_standalone.html` / `index_3d.html` のUI仕様を固定
- `demo_scene.json` のフォーマットを軽量維持（必要最小項目）
- 再現性重視で、サンプル実行のデータ生成と可視化を README で同じ順序にする
