# dynamic-3d-object-removal plan

Last updated: 2026-03-26 (Asia/Tokyo)
Repo: `rsasaki0109/dynamic-3d-object-removal`
Branch: `master`
Latest pushed commit: `9e257c8`

---

## What this project is

LiDAR 点群から動的物体（車両・歩行者・自転車など）を除去するライブラリ。
**deep learning を使わない** — 幾何的な 3D bounding box crop と voxel-based temporal consistency filter のみ。
依存は numpy だけ（pyarrow は Argoverse 2 形式を読む場合のみ必要）。

3 つの形態で提供:

1. **Python ライブラリ** (`dynamic_object_removal.py`, 843 行)
2. **CLI** (`dynamic-object-removal`)
3. **ROS2 リアルタイムノード** (`realtime.py`, 835 行)

ベンチマーク (`bench.py`) とテスト (`tests/`, 57 テスト) 付き。

---

## Architecture

```
dynamic_object_removal.py   # コアライブラリ + CLI
├── load_points()           # PCD, CSV, TXT, XYZ, NPY, BIN(KITTI), Feather(AV2)
├── load_boxes()            # JSON, CSV, KITTI label_2, Feather(AV2)
├── remove_points_in_boxes()  # 幾何的 box crop (yaw 対応, margin 付き)
├── TemporalConsistencyFilter # voxel hit-count ベースの temporal filter
└── save_points()           # PCD, CSV, TXT, NPY

realtime.py                 # ROS2 PointCloud2 subscriber/publisher ノード
bench.py                    # ベンチマーク (box / temporal)

demo/
├── run_scan_demo.py        # 単発スキャン → standalone HTML 生成
├── run_scan_sequence_demo.py # マルチフレーム → sequence HTML 生成
├── index_3d_standalone.html  # 単発スキャンデモ (GitHub Pages)
├── index_3d_sequence_standalone.html  # sequence デモ (GitHub Pages)
├── index_3d_sequence_av2.html  # AV2 public sequence デモ (GitHub Pages)
├── index_3d_av2.html       # Argoverse 2 単発デモ
├── av2_before_after.png    # README hero (3-panel: Before/Ghost/After)
├── av2_zoom.png            # README hero (zoomed close-up)
└── story_mode.gif          # プライベートデータの sequence アニメーション

scripts/
├── download_av2_sample.py  # Argoverse 2 サンプル DL (登録不要, ~1.3MB)
└── download_kitti_sample.py # KITTI 合成サンプル生成 (5フレーム)

tests/
├── conftest.py             # 共有 fixture
├── test_dynamic_object_removal.py  # コアライブラリの回帰テスト
└── test_sequence_demo.py   # sequence demo の pose / AV2 回帰テスト
```

---

## Current state (2026-03-26)

### 完了済み

- [x] コアライブラリ: box crop + temporal consistency filter
- [x] CLI: `--input-cloud`, `--input-objects`, `--output-cloud`, `--summary-json` 等
- [x] KITTI 形式対応: `.bin` 点群, `label_2` ラベル, calibration パース, camera→velodyne 座標変換
- [x] Argoverse 2 形式対応: `.feather` 点群/annotations, quaternion→yaw, `--timestamp-ns` フィルタ
- [x] Quick start: AV2 公開データ (登録不要) で 3 コマンドで体験可能
- [x] README hero image: 交通量の多い AV2 シーン (99 objects/frame) の accumulated map Before/After
  - 3-panel (Before / Ghost only / After) + zoomed close-up
  - 2M 点, 233k ghost points (11.9%) 除去
- [x] GitHub Pages デモ: sequence + 単発スキャン + AV2
- [x] AV2 public sequence demo: `annotations.feather` + `city_SE3_egovehicle.feather` から 20 フレームの pose-aligned / box-driven HTML を生成
- [x] テスト 64 件 (DetectionBox, load_points, load_boxes, KITTI, remove, temporal filter, save, CLI, sequence demo pose / AV2)
- [x] Dogfooding: fresh clone → install → quick start の全フローを検証済み
  - 発見・修正: output dir 自動作成, margin デフォルト値, timestamp_ns 未指定時の warning
- [x] メッセージ整合: README / index.html / demo/index.html で temporal consistency ベースを明示
- [x] CI: `.github/workflows/test.yml` で `pytest` を実行

### 未完了

- [ ] hero image のインパクト向上の余地（後述）
- [ ] PyPI publish (0.1.0 は未 publish)

---

## Design decisions & rationale

### なぜ deep learning を使わないか

- LiDAR SLAM の後処理として使う想定。検出器は別にある（or 3D box annotation が既にある）
- 除去自体は幾何的な box crop で十分 — 高価な GPU 推論は不要
- numpy only → pip install して即使える。Docker も GPU もいらない
- ベンチマーク: 24k 点で 1.5ms (box crop), CPU のみ
- **リアルタイム処理が可能**: ROS2 ノードとして PointCloud2 を受けて即座に filter → publish
  - box mode: 外部検出器の結果を subscribe して除去
  - temporal mode: 検出器なしで動的物体を voxel hit-count ベースで除去

### ポジショニング

| | このプロジェクト | DL ベースの手法 |
|---|---|---|
| 入力 | 点群 + 3D box (or 検出器出力) | 点群のみ |
| GPU | 不要 | 必須 |
| 推論速度 | 1.5ms / 24k点 | 数十〜数百 ms |
| 依存 | numpy | PyTorch, CUDA, 学習済みモデル |
| ROS2 | リアルタイムノード同梱 | 個別実装が必要 |
| 精度 | 検出器に依存 | end-to-end で最適化 |

### なぜ Argoverse 2 を選んだか

- **登録不要** で S3 から直接ダウンロードできる唯一の大規模 LiDAR データセット
- 64-beam, ~95k 点/フレーム, 3D cuboid annotation 付き
- CC BY-NC-SA 4.0 ライセンス
- KITTI は登録必須、Waymo は GCS 認証必要、nuScenes は公式には登録必要

### hero image のシーン選定

- scene `04994d08-156c-3018-9717-ba0e29be8153`: 平均 99 objects/frame (車両 8437 + 歩行者 5662 over 156 frames)
- 他のシーンは 34 objects/frame 程度 — ghost trail が少なく差がわかりにくかった
- 20 フレーム accumulated → 233k ghost points (11.9%) で十分な視覚的インパクト

---

## Key file details

### dynamic_object_removal.py (843 行)

- `DetectionBox`: frozen dataclass (center, size, yaw, label)
- `load_points()`: 7 形式対応。auto-detect by extension
- `load_boxes()`: 4 形式対応 + AV2 の `timestamp_ns` フィルタ
  - KITTI: camera→velodyne 座標変換を calibration ファイルから計算
  - AV2: quaternion→yaw, `_KITTI_DYNAMIC_CLASSES` / AV2 category でフィルタ
- `remove_points_in_boxes()`: yaw 回転対応の axis-aligned crop。margin デフォルト `(0.05, 0.05, 0.05)`
- `TemporalConsistencyFilter`: voxel hit-count, sliding window, min_hits threshold
- `main()`: CLI エントリポイント。output dir 自動作成

### realtime.py (835 行)

- ROS2 ノード。`sensor_msgs/PointCloud2` を subscribe → filter → publish
- box / temporal の 2 アルゴリズム切り替え
- `--box-stale-time`, `--max-object-history` で検出の鮮度管理
- rclpy 必須（ROS2 環境外ではテスト不可）

### demo/ 系

- `run_scan_demo.py`: 単発スキャン → Three.js standalone HTML 生成
- `run_scan_sequence_demo.py`: multi-frame → アニメーション HTML 生成
- checked-in HTML は点群データを JSON で内包する self-contained 形式
- `story_mode.gif` はプライベートデータから生成 — 再生成には `/media/sasaki/aiueo/rosbag/GT/` が必要

---

## Confirmed facts

### Checked-in sequence source

`demo/index_3d_sequence_standalone.html` は以下の 12 フレームから再生成すると一致する:

```bash
/media/sasaki/aiueo/rosbag/GT/2025-05-28-12-48-29/verify_1_16_5_final/graph/*/cloud.pcd
```

再生成パラメータ: `--frame-count 12 --stride 1 --max-render-points 9000 --fps 4 --voxel-size 0.35 --window-size 5 --min-hits 3`

### Per-frame box JSON (2026-03-23 確定)

`verify_1_16_5_final` 配下に per-frame detection / box JSON は存在しない。
`graph/*/data.txt` はカメラ姿勢・変換行列。
checked-in sequence の cleaned 側は temporal consistency ベースで確定。

### AV2 hero image source (2026-03-26 確定)

scene: `04994d08-156c-3018-9717-ba0e29be8153` (val split)
20 フレーム, 1,957,497 raw points, 233,123 ghost points removed (11.9%)
データは `/tmp/av2_dense/` にダウンロード済み（永続化されていない）

再生成手順:
```bash
# 1. データ取得
export SCENE=04994d08-156c-3018-9717-ba0e29be8153
aws s3 cp --no-sign-request --recursive s3://argoverse/datasets/av2/sensor/val/${SCENE}/sensors/lidar/ /tmp/av2_dense/lidar/ # 最初の20件
aws s3 cp --no-sign-request s3://argoverse/datasets/av2/sensor/val/${SCENE}/annotations.feather /tmp/av2_dense/
aws s3 cp --no-sign-request s3://argoverse/datasets/av2/sensor/val/${SCENE}/city_SE3_egovehicle.feather /tmp/av2_dense/

# 2. accumulated map 生成 → matplotlib で Before/After 画像生成
# (scipy.spatial.transform.Rotation が必要)
# 具体的なスクリプトは commit 5036f8b のコンテキストを参照
```

---

## Priority (next steps)

### 1. README / GitHub About の非 DL ポジショニング強化

README 冒頭で「GPU 不要・numpy only・幾何ベース」を明示する。
GitHub repo の About (description + topics) を設定する。

ステータス: **今回対応する**

### 2. hero image の改善

現状の 3-panel + zoom は十分だが、さらに改善するなら:
- ghost trail だけを isolated で見せるアニメーション GIF
- フレーム番号付きで ghost が蓄積していく過程を見せる

優先度: 低（現状で十分なインパクト）

### 3. PyPI publish

`pyproject.toml` は整備済み。`twine upload` するだけ。
バージョンは 0.1.0。

優先度: 低

### 4. 検出器との統合例

CenterPoint / PointPillars の出力を `load_boxes()` に食わせるチュートリアル。
これがあると「検出→除去」のエンドツーエンドが見えて、ユーザーの導入障壁が下がる。

優先度: 低（ユーザーからの要望次第）

### 5. 非ディープ手法の比較検証

ERASOR, Removert, dynamic_object_detection などの non-DL / geometry-heavy 系手法との比較は、外向きの説得力を上げる余地がある。

ただし、この repo をすぐに「いろいろな手法を載せる総合 benchmark repo」に広げるのは避ける。
このプロジェクトの主語は `numpy-only` / `small` / `public proof demo` のまま保つ方が価値が高い。

比較検証をやるなら:
- 第一候補は `bench/` 以下に adapter / metric / dataset runner を置く
- より大きくするなら、この repo とは別の benchmark repo に切り出す
- この repo 本体には third-party 実装や重い依存を持ち込まない

優先度: 低（今は core repo の sharpness を維持する方が重要）

---

## Do not do

- generic viewer controls を増やす
- panel を増やして同じ話を繰り返す
- `real detections` が無いのにそう読める表現にする
- `__pycache__` / `data/` / `*.egg-info` を commit する (.gitignore 設定済み)
- deep learning の依存を追加する（このプロジェクトの差別化ポイントは DL 不要であること）
- 比較検証の名目で、この repo を third-party 手法の寄せ集めにしない

---

## Useful commands

### AV2 Quick start

```bash
python3 scripts/download_av2_sample.py
dynamic-object-removal \
  --input-cloud data/av2_sample/lidar/315969904359876000.feather \
  --input-objects data/av2_sample/annotations.feather \
  --timestamp-ns 315969904359876000 \
  --output-cloud output/av2_cleaned.pcd
```

### KITTI Quick start

```bash
python3 scripts/download_kitti_sample.py
dynamic-object-removal \
  --input-cloud data/kitti_sample/velodyne/000000.bin \
  --input-objects data/kitti_sample/label_2/000000.txt \
  --objects-format kitti \
  --calib-path data/kitti_sample/calib/000000.txt \
  --output-cloud output/kitti_cleaned.pcd
```

### Sequence repro (プライベートデータ)

```bash
python3 demo/run_scan_sequence_demo.py \
  --input-glob "/media/sasaki/aiueo/rosbag/GT/2025-05-28-12-48-29/verify_1_16_5_final/graph/*/cloud.pcd" \
  --frame-count 12 --stride 1 --max-render-points 9000 --fps 4 \
  --voxel-size 0.35 --window-size 5 --min-hits 3 \
  --output-html demo/index_3d_sequence_standalone.html
```

### テスト

```bash
python3 -m pytest tests/ -v
```

### ベンチマーク

```bash
dynamic-object-removal-bench \
  --input-cloud demo/actual_scan_20240820_cloud.pcd \
  --input-objects demo/actual_scan_20240820_objects.json \
  --algorithm box --iterations 100 --skip-invalid
```

### Visual verification

```bash
python3 -m http.server 8765
npx playwright screenshot --device="Desktop Chrome" --wait-for-timeout=2200 --full-page \
  http://127.0.0.1:8765/demo/index_3d_av2.html /tmp/av2_screenshot.png
```

### GitHub About 更新

```bash
gh repo edit rsasaki0109/dynamic-3d-object-removal \
  --description "Remove dynamic objects from LiDAR point clouds using 3D bounding boxes. No deep learning — numpy only, geometry-based." \
  --homepage "https://rsasaki0109.github.io/dynamic-3d-object-removal/" \
  --add-topic lidar --add-topic point-cloud --add-topic slam \
  --add-topic dynamic-object-removal --add-topic mapping \
  --add-topic argoverse --add-topic kitti --add-topic ros2
```
