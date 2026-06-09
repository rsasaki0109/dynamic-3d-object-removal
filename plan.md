# dynamic-3d-object-removal plan

Last updated: 2026-06-10 (Asia/Tokyo)
Repo: `rsasaki0109/dynamic-3d-object-removal`
Branch: `master`
Latest pushed commit: `47393b2` (v0.2.0)
Stars: **74 / 100 (目標)** — fork 4, created 2026-03-05

---

## What this project is

LiDAR 点群から動的物体（車両・歩行者・自転車など）を除去するライブラリ。
**deep learning を使わない** — 幾何ベースのみ。依存は numpy だけ
（pyarrow は Argoverse 2 形式を読む場合のみ必要）。

アルゴリズムは 4 つ、すべて numpy:

1. **box** — 検出 3D box による per-scan crop（検出器 or annotation が必要）
2. **temporal** — voxel hit-count の時系列一貫性（検出器不要、最も単純・高 recall）
3. **range** — range-image 可視性（Removert 系 remove + revert、検出器不要）
4. **scan_ratio** — 極座標カラムの擬似 occupancy（ERASOR 系 scan-ratio + ground revert、検出器不要）

3 つの形態で提供:

1. **Python ライブラリ** (`dynamic_object_removal.py`, 1421 行)
2. **CLI** (`dynamic-object-removal`) — PyPI 公開済み (`pip install dynamic-object-removal`)
3. **ROS2 リアルタイムノード** (`realtime.py`, 859 行) — box / temporal / range 対応

ベンチマーク (`bench.py` + `scripts/run_av2_benchmark.py` + `scripts/run_nuscenes_benchmark.py`)
とテスト (`tests/`, 84 passed + 1 skipped) 付き。
ブラウザ Playground（Pyodide で実ライブラリがクライアントサイド実行、Box / Range / Temporal の 3 モード）が GitHub Pages にある。

---

## Architecture

```
dynamic_object_removal.py   # コアライブラリ + CLI (v0.2.0)
├── load_points()           # PCD, CSV, TXT, XYZ, NPY, BIN(KITTI), Feather(AV2)
├── load_boxes()            # JSON, CSV, KITTI label_2, Feather(AV2)
├── remove_points_in_boxes()
├── TemporalConsistencyFilter
├── remove_ghost_by_range_image() / clean_map_by_visibility()   # range (multi-resolution consensus 対応)
├── remove_dynamic_by_scan_ratio() / clean_map_by_scan_ratio()  # scan_ratio
├── RangeImageGhostFilter   # ROS2 用ストリーミング range filter
└── save_points()

realtime.py                 # ROS2 PointCloud2 subscriber/publisher ノード (box/temporal/range)
bench.py                    # 速度ベンチマーク

demo/
├── playground.html         # Pyodide Playground (704 行, GitHub Pages の主力導線)
├── run_scan_demo.py / run_scan_sequence_demo.py
├── index_3d_*.html         # checked-in self-contained デモ群
├── sample_av2_cloud.npy / sample_av2_objects.json / sample_av2_range.npz  # Playground 用データ
└── av2_before_after.png / av2_zoom.png / playground_demo.gif / story_mode.gif

scripts/
├── download_av2_sample.py / download_kitti_sample.py
├── run_av2_benchmark.py      # 再現可能な AV2 ベンチ (12 sweeps, P/R/F1)
└── run_nuscenes_benchmark.py # nuScenes mini ベンチ (32-beam 汎化の証明)
```

---

## Current state (2026-06-10)

### 完了済み（前回 plan 以降の追加分を含む）

- [x] コアライブラリ: box + temporal + **range** + **scan_ratio** の 4 アルゴリズム
- [x] multi-resolution consensus（`resolutions=[2.5, 4.0]` で precision 0.68 → 0.78）
- [x] **PyPI 公開** (v0.2.0, Trusted Publishing で release 自動化)
- [x] 再現可能ベンチマーク 2 本: AV2 (64-beam) + nuScenes mini (32-beam) — どちらも登録不要・1 コマンド
- [x] ブラウザ Playground (Pyodide): Box / Range / Temporal の 3 モード、ユーザー自身の PCD ドロップ対応
- [x] README: How It Compares (ERASOR / Removert とのポジショニング表) + 実測値テーブル
- [x] GitHub Pages デモ群、hero image、social card
- [x] テスト 84 件、CI (`test.yml`)、publish workflow (`publish.yml`)
- [x] GitHub About / topics 設定済み (lidar, slam, ros2, …)

### 未完了

- [ ] ★100 ロードマップ（本ファイルの主題、下の Roadmap 参照）

---

## ★100 Roadmap — 残り 26★ を取りに行く

優先順に 3 本。**Step A (lidarslam_ros2 連携) → Step B (SemanticKITTI / DynamicMap_Benchmark) → Step C (Playground 共有性 + Show HN)**。
A は即効性（自分の既存オーディエンスからの導線）、B は持続性（この問題を探している層への恒久導線）、C は単発スパイク狙い。

それぞれ独立して出荷できる。1 本ずつ完結させてから次に進む。

---

### Step A. lidarslam_ros2 連携例（最優先・即効性）

#### ゴール

`rsasaki0109/lidarslam_ros2`（**820★**）のユーザーに「SLAM の前段に
`dynamic-object-removal-realtime` を挟むと地図から動的物体の ghost が消える」を
launch 一発 + before/after 画像で見せ、lidarslam_ros2 README からリンクする。

#### なぜ星になるか

- 820★ のリポジトリの README に貼られる導線は、この repo にとって最も確度の高い流入経路
- 「SLAM 利用者が地図のゴーストに困る」はまさに本プロジェクトの想定ユースケースで、転換率が高い
- 開発コストが最小（launch ファイル + 検証 + 画像 1 枚 + README 2 箇所）

#### 成果物

1. `examples/lidarslam_ros2/README.md` — 手順書（英語）。bag 取得 → 連携 launch → 地図比較まで
2. `examples/lidarslam_ros2/dor_lidarslam.launch.py` — `dynamic-object-removal-realtime` を
   前段に挟む composable な launch（lidarslam 側はユーザー環境の launch をそのまま include or
   トピック remap で接続）
3. `examples/lidarslam_ros2/map_comparison.png` — 除去なし/ありの最終地図 before/after（hero 画像と同じ 2-panel 様式）
4. README (this repo): `## Works with your SLAM` セクション新設、画像 + 3 行
5. lidarslam_ros2 側: README に 1 ブロック（画像 + リンク）を追加する commit

#### 設計

- **データ**: lidarslam_ros2 の quickstart と同じ **NTU VIRAL `tnp_01`**（Ouster OS1-16, 約 580 秒, 屋外）を使う。
  ユーザーが既に持っている bag なので追加ダウンロード不要、再現条件も lidarslam_ros2 側と完全一致
- **トピック接続**:
  ```
  /os_cloud_node/points ──> dynamic-object-removal-realtime ──> /cleaned_points ──> (lidarslam input に remap)
  ```
  lidarslam 側はデフォルト入力 `/os_cloud_node/points` を `/cleaned_points` に remap するだけ
- **アルゴリズム選定**: 検出器なし前提なので `temporal` が第一候補
  （`--voxel-size 0.10 --temporal-window 5 --temporal-min-hits 3` を起点にチューニング）。
  OS1-16 は 16-beam と疎なので、nuScenes で得た「ビーム密度に解像度を合わせる」知見から
  `range` を使うなら粗い解像度が必要 — まず temporal で出し、range は発展項目として README に一言
- **比較プロトコル**: 同じ bag・同じ SLAM パラメータで 2 回走らせ、`/map_save` で保存した
  地図 PCD を同一視点でレンダリングして 2-panel に。可能なら除去点数 / 地図点数の差も数字で併記

#### 作業手順

1. ローカルに ROS2 + lidarslam_ros2 環境を用意し、NTU VIRAL `tnp_01` で素の SLAM を再現（baseline 地図保存）
2. `dynamic-object-removal-realtime` を挟んだ構成で同一 bag を処理（cleaned 地図保存）
3. パラメータ調整: 歩行者・車両 ghost が消えつつ静的構造が保たれる点を目視確認
4. 2-panel 画像生成（既存 hero 画像と同じ matplotlib トーンで）
5. `examples/lidarslam_ros2/` 一式 + README セクションを commit
6. lidarslam_ros2 側 README 更新（別リポジトリで commit & push）

#### 受け入れ条件

- [ ] 手順書に従って fresh 環境で再現できる（コマンドのコピペで完走）
- [ ] before/after で動的 ghost の減少が一目でわかる画像
- [ ] 静的構造（建物・地面）が目視で劣化していない
- [ ] lidarslam_ros2 README からのリンクが生きている

#### リスク / 確認事項

- NTU VIRAL tnp_01 に十分な動的物体（歩行者等）が映っているか **要確認**。
  乏しければ動的物体の多い別の公開 bag（例: 都市部の Ouster/Velodyne bag）に切り替える。
  その場合も「lidarslam_ros2 で動く公開データ」であることを優先
- OS1-16 の疎さで temporal の voxel パラメータが合わない可能性 → voxel-size を粗めに振る
- lidarslam_ros2 の現行 frontend は RKO-LIO（IMU 併用）。点群だけ差し替えても IMU 系は素通しで OK のはずだが、TF/タイムスタンプの整合を実機で確認

---

### Step B. SemanticKITTI 対応 + DynamicMap_Benchmark 接続（持続性）

#### ゴール

動的物体除去コミュニティの標準土俵 **KTH-RPL/DynamicMap_Benchmark** で
本プロジェクトの検出器不要 3 手法（range / scan_ratio / temporal）を評価し、
(a) 本 repo に再現スクリプト + 結果表を載せ、
(b) 先方 repo の `methods/` に numpy-only アダプタを PR する。

#### なぜ星になるか

- ERASOR / Removert / DUFOMap / BeautyMap / dynablox が載る同一ベンチに
  「**pip install 一発・numpy-only**」で参加する初の実装になる — 差別化が数字で立つ
- README の比較表が「positioning guide（再計測ではない）」という断り書きから
  「**同一データ・同一指標の実測**」に格上げされる
- 先方 README に method として載れば、この問題を能動的に探している層
  （研究者・SLAM エンジニア）への恒久的な被リンクになる
- plan の「benchmark 寄せ集めにしない」方針と矛盾しない:
  third-party 実装は一切取り込まず、**自分の手法を標準データで走らせるだけ**

#### DynamicMap_Benchmark の仕様（2026-06-10 時点の調査メモ）

- データ: Zenodo (record `10886629`) から DL。Semantic-KITTI (VLP-64) / Argoverse 2 /
  UDI-Plane (VLP-16) / KTH-Campus / Indoor-Floor (Livox mid-360) の 5 種
- 入力形式: **pose-attached な per-scan PCD**（pose は PCD の VIEWPOINT フィールドと推定 — **要確認**）
- 出力形式: 動的点を除去した cleaned map PCD
- 評価: `scripts/py/eval` のスクリプトが ground truth と比較して定量表を出す
  （指標は SA / DA / AA 系のはず — **先方論文 (ITSC 2023) で要確認**）
- メソッド追加: `methods/` フォルダに実行可能な `main.py` を置く構造。PR 歓迎と明記あり

#### 成果物

1. `scripts/run_dynamicmap_benchmark.py` — Zenodo からデータ DL → pose-attached PCD 読込 →
   累積 map 構築 → `clean_map_by_visibility` / `clean_map_by_scan_ratio` / temporal で除去 →
   先方の期待形式で cleaned map PCD を出力 → （先方 eval が pure-python なら）そのまま指標まで出す
2. `dynamic_object_removal.py` への最小追加:
   - PCD **VIEWPOINT 読み取り**（pose-attached PCD 対応）— `load_points` の拡張 or 専用ローダ
   - 既存 PCD ローダで不足があれば binary PCD の堅牢化
3. README: 「Measured on SemanticKITTI (DynamicMap_Benchmark)」セクション + 結果表
   （AV2 / nuScenes 表と同じ様式: precision/recall/F1 or 先方指標）
4. 先方への PR: `methods/dor_numpy/`（`main.py` + 依存は `pip install dynamic-object-removal` のみ）
   + 先方 README の手法表への追記

#### 設計方針

- まず **Semantic-KITTI seq 00 / 05** だけに絞る（先方の主要シーケンス）。5 データセット全部は追わない
- スクリプトは AV2 / nuScenes ベンチと同じ流儀:
  1 コマンド・登録不要・進捗ログ・最後に markdown 表を print
- 解像度はビーム密度に合わせる既存知見を適用（VLP-64 → AV2 並みの細かさ、
  UDI-Plane をやる場合は nuScenes の教訓で粗く — ただし初回スコープ外）
- 数字が ERASOR / DUFOMap に負ける項目は**隠さずそのまま載せる**。
  売りは「同等〜健闘する数字を numpy だけ・pip 一発で」であり、SOTA 主張ではない。
  README の文言もそのトーンで書く（nuScenes の scan_ratio の cautionary case と同じ誠実路線）

#### 作業手順

1. 先方 repo を clone し、データ形式・eval スクリプト・既存 method の `main.py` を精読
   （VIEWPOINT 仕様・指標定義・map/scan の座標系を確定 → 本ファイルの調査メモを更新）
2. Zenodo から Semantic-KITTI seq 00 を DL、pose-attached PCD ローダを実装（テスト付き）
3. range で end-to-end を通す → 先方 eval で数字を出す → scan_ratio / temporal を追加
4. パラメータを 1 セットに固定（過剰チューニングしない。AV2/nuScenes と同じ「sensor 密度で解像度を選ぶ」原則のみ）
5. `scripts/run_dynamicmap_benchmark.py` を仕上げ、README に結果セクション追加、commit
6. 先方フォーマットの `methods/dor_numpy/main.py` を書き、fork → PR
   （PR 本文に再現コマンドと数字、numpy-only という特徴を簡潔に）

#### 受け入れ条件

- [ ] `python3 scripts/run_dynamicmap_benchmark.py` 1 コマンドで DL から指標出力まで完走
- [ ] 3 手法 × seq 00（+05）の数字が README に載り、再現コマンドが併記されている
- [ ] 先方への PR が open される（merge は先方次第なので条件にしない）
- [ ] 本 repo に third-party 実装・重い依存が増えていない（`open3d` 等を足さない。
      PCD I/O は自前 numpy 実装を貫く）

#### リスク / 確認事項

- 先方 eval スクリプトが C++ / Open3D 依存の場合: 本 repo のスクリプトは
  「先方期待形式の cleaned map PCD を出力するところまで」を担当し、
  指標計算は先方スクリプト実行手順を README で案内する形に倒す（依存を持ち込まない）
- ground truth が「cleaned map 基準で抽出」とあるため、ダウンサンプリングの扱いに罠がありうる —
  既存 method の出力仕様を必ず踏襲する
- 数字が大幅に負ける可能性: それでも出す。ただし `min_see_through` / `resolutions` の
  precision 重視設定など、既存ノブの範囲で 1 回だけ調整パスを置く

---

### Step C. Playground 共有性強化 + Show HN（スパイク）

#### ゴール

Playground に (1) **共有 URL**（モード・パラメータ・プリセットを URL に保存）と
(2) **シーンプリセット**を追加し、素材を整えてから
「LiDAR dynamic object removal running entirely in your browser (Pyodide, numpy-only)」
として Show HN / Reddit に出す。

#### なぜ星になるか

- Playground は本 repo 最大の資産。だが現状 URL に状態がなく（`location.hash` /
  `URLSearchParams` 不使用 — 2026-06-10 確認済み）、「この設定を見て」が共有できない
- HN は「実ライブラリがブラウザで動く」「サーバ不要」「numpy が Pyodide で」型の話に強い。
  共有 URL があると、コメント欄で「この設定だと壊れる/すごい」が拡散ループになる
- Step A/B の成果（820★ repo からの導線、標準ベンチの数字）が出てから投稿すると
  着地ページ (README) の説得力が最大になる — **投稿は A/B 完了後**

#### 成果物

1. `demo/playground.html` 改修:
   - **URL 状態同期**: `mode`（box/range/temporal）+ 主要パラメータ
     （range: `margin` / `see-through` / `surface-hits`、temporal: `voxel` / `window` / `min-hits`）+
     プリセット ID を `URLSearchParams`（`?mode=range&see=4&preset=av2`）で読み書き。
     ページロード時に復元、変更時に `history.replaceState` で更新
   - **Share ボタン**: 現在状態の URL をクリップボードへコピー（コピー成功のトースト表示）
   - **シーンプリセット切替**: 既存 AV2 シーンに加え **nuScenes 32-beam シーン**を 1 つ追加
     （疎なセンサーで解像度を合わせる話が Playground 上で体験できる = README の nuScenes 節の実演）
2. `demo/sample_nuscenes_range.npz` — nuScenes mini から生成したプリセットデータ。
   **サイズ予算: 既存 `sample_av2_range.npz` と同程度以下**（Pages の初期ロードを悪化させない）。
   生成スクリプトは `scripts/run_nuscenes_benchmark.py` 系から派生させ、再現手順をコメントに残す
3. README: Playground 節に「共有 URL 対応」と nuScenes プリセットの 1 行追記
4. 投稿素材:
   - Show HN タイトル案: *Show HN: Dynamic object removal for LiDAR point clouds, in the browser (numpy + Pyodide)*
   - 1 段落の説明文（no GPU / no upload / 実ライブラリそのまま / 検出器不要モードあり）
   - 投稿先: HN、r/computervision、r/robotics、ROS Discourse(Show & Tell 相当カテゴリ)。
     各 1 回・誇張なし・数字は README の実測のみ

#### 設計メモ

- 状態は **query param** 採用（`#hash` より共有時の見た目が素直、GitHub Pages で問題なし）
- パラメータ名は短く安定させる（一度共有された URL を壊さない — 以後 rename しない契約）
- プリセットデータの読込は現行の `sample_av2_range.npz` ロードパス（fetch → npz parse）を共通化して分岐
- Playground は依存追加なし・単一 HTML を維持（ビルドステップを入れない）

#### 作業手順

1. URL 同期 + Share ボタン実装（mode 3 種 × 主要パラメータ。全パラメータは追わない）
2. nuScenes プリセットデータ生成 → サイズ確認 → 組み込み → モード切替 UI に追加
3. Playwright スクリーンショットで 3 モード × 2 プリセットの表示確認
   （既存の visual verification 手順を流用）
4. 共有 URL を別ブラウザ/シークレットウィンドウで開いて状態復元を確認
5. README 追記、playground_demo.gif を必要なら撮り直し
6. （A/B 完了後）投稿。投稿日の Pages 死活確認、当日はコメント返信に張り付く

#### 受け入れ条件

- [ ] 任意の設定で Share → 新規タブで開くと同じモード・パラメータ・プリセットが復元される
- [ ] nuScenes プリセットが追加され、初期ロードサイズの増分が +1 npz 以内
- [ ] パラメータなし URL（既存リンク）の挙動が完全に後方互換
- [ ] 単一 HTML・依存ゼロのまま

#### リスク

- Pyodide の初期ロードが遅い環境で HN コメントが「重い」に流れる →
  ロード中プログレス表示が貧弱なら先に直す（要現状確認）
- 投稿は各プラットフォーム 1 回きり。伸びなくても再投稿しない（スパム判定リスク）

---

## 実行順序と完了の定義

```
Step A (lidarslam_ros2 連携)   … 数時間〜1日。出荷したら即 lidarslam_ros2 README 更新
  ↓
Step B (DynamicMap_Benchmark)  … 数日。先方仕様の精読が先。PR open まで
  ↓
Step C (Playground + Show HN)  … 実装は半日〜1日。投稿は A/B の成果が README に載ってから
```

- 各 Step は独立に merge 可能。Step をまたぐ WIP を作らない
- ★100 到達が目的であって手段の完遂ではない — 途中で到達したら C の投稿タイミングだけ柔軟に

---

## Design decisions & rationale

### なぜ deep learning を使わないか

- LiDAR SLAM の後処理として使う想定。検出器は別にある（or 3D box annotation が既にある）
- 除去自体は幾何で十分 — 高価な GPU 推論は不要
- numpy only → pip install して即使える。Docker も GPU もいらない
- ベンチマーク: 24k 点で 1.5ms (box crop), CPU のみ
- **リアルタイム処理が可能**: ROS2 ノードとして PointCloud2 を受けて即座に filter → publish

### ポジショニング

詳細は README「How It Compares」参照。要点:

- ERASOR / Removert はオフラインの map cleaning 専用・C++/PCL — 本プロジェクトは
  per-scan / realtime + map cleaning 両対応・numpy-only
- 検出器不要モード（temporal / range / scan_ratio）は AV2 で F1 ≈ 0.60、
  multi-resolution consensus で precision 0.78 まで引ける
- nuScenes (32-beam) でも range は解像度をビーム密度に合わせれば汎化（F1 0.63）

### なぜ Argoverse 2 を選んだか

- **登録不要** で S3 から直接ダウンロードできる大規模 LiDAR データセット
- 64-beam, ~95k 点/フレーム, 3D cuboid annotation 付き、CC BY-NC-SA 4.0
- nuScenes mini も匿名 HTTPS で取得でき、第二データセットとして採用済み

### hero image のシーン選定

- scene `04994d08-156c-3018-9717-ba0e29be8153`: 平均 99 objects/frame
- 20 フレーム accumulated → 233k ghost points (11.9%) で十分な視覚的インパクト

---

## Confirmed facts

### Checked-in sequence source

`demo/index_3d_sequence_standalone.html` は以下の 12 フレームから再生成すると一致する:

```bash
/workspace/rosbag/GT/2025-05-28-12-48-29/verify_1_16_5_final/graph/*/cloud.pcd
```

再生成パラメータ: `--frame-count 12 --stride 1 --max-render-points 9000 --fps 4 --voxel-size 0.35 --window-size 5 --min-hits 3`

### Per-frame box JSON (2026-03-23 確定)

`verify_1_16_5_final` 配下に per-frame detection / box JSON は存在しない。
checked-in sequence の cleaned 側は temporal consistency ベースで確定。

### AV2 hero image source (2026-03-26 確定)

scene: `04994d08-156c-3018-9717-ba0e29be8153` (val split)
20 フレーム, 1,957,497 raw points, 233,123 ghost points removed (11.9%)

```bash
export SCENE=04994d08-156c-3018-9717-ba0e29be8153
aws s3 cp --no-sign-request --recursive s3://argoverse/datasets/av2/sensor/val/${SCENE}/sensors/lidar/ /tmp/av2_dense/lidar/ # 最初の20件
aws s3 cp --no-sign-request s3://argoverse/datasets/av2/sensor/val/${SCENE}/annotations.feather /tmp/av2_dense/
aws s3 cp --no-sign-request s3://argoverse/datasets/av2/sensor/val/${SCENE}/city_SE3_egovehicle.feather /tmp/av2_dense/
```

### lidarslam_ros2 (2026-06-10 調査)

- 820★。frontend は RKO-LIO（LiDAR-inertial odometry）+ backend `graph_based_slam`
- 入力トピック既定: 点群 `/os_cloud_node/points`, IMU `/os_cloud_node/imu`
- quickstart データ: NTU VIRAL `tnp_01`（Ouster OS1-16, ~580 s, 屋外）
- 地図保存: `/map_save` サービス。出力は Autoware-ready bundle

### DynamicMap_Benchmark (2026-06-10 調査)

- KTH-RPL/DynamicMap_Benchmark。データは Zenodo record `10886629`
- 5 データセット: Semantic-KITTI / Argoverse 2 / UDI-Plane / KTH-Campus / Indoor-Floor
- 入力: pose-attached per-scan PCD → 出力: cleaned map PCD
- 評価スクリプト: `scripts/py/eval`。メソッドは `methods/<name>/main.py` で追加、PR 歓迎
- 収録 8 手法: DUFOMap, Octomap w/ GF, dynablox, Octomap, DeFlow, BeautyMap, ERASOR, Removert
- **要確認**: pose の格納方式（VIEWPOINT?）、指標の正確な定義、eval の依存関係

### Playground 現状 (2026-06-10 確認)

- `demo/playground.html` 704 行、単一 HTML、依存追加なし
- URL 状態管理なし（`location.hash` / `URLSearchParams` 不使用）→ Step C で追加
- プリセットデータ: `sample_av2_cloud.npy` / `sample_av2_objects.json` / `sample_av2_range.npz`

---

## Do not do

- generic viewer controls を増やす
- panel を増やして同じ話を繰り返す
- `real detections` が無いのにそう読める表現にする
- `__pycache__` / `data/` / `*.egg-info` を commit する (.gitignore 設定済み)
- deep learning の依存を追加する（差別化ポイントは DL 不要であること）
- 比較検証の名目で、この repo を third-party 手法の寄せ集めにしない
  （Step B でも先方実装は取り込まない — 自手法を標準データで走らせるだけ）
- Playground にビルドステップ・外部 JS 依存を足す
- ベンチ数字が負ける項目を隠す・チューニングで盛る（誠実な cautionary case 路線を維持）
- C++/Rust ポート・新規手法の追加（★100 目的には寄与せず scope を薄める）
- HN / Reddit への再投稿・複数アカウント投稿

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

### ベンチマーク再現

```bash
python3 scripts/run_av2_benchmark.py --frames 12      # AV2 (64-beam)
python3 scripts/run_nuscenes_benchmark.py             # nuScenes mini (32-beam)
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

### ROS2 realtime（Step A で使う形）

```bash
dynamic-object-removal-realtime \
  --pointcloud-topic /os_cloud_node/points \
  --output-topic /cleaned_points \
  --algorithm temporal \
  --voxel-size 0.10 --temporal-window 5 --temporal-min-hits 3
```

### テスト

```bash
python3 -m pytest tests/ -v
```

### Visual verification

```bash
python3 -m http.server 8765
npx playwright screenshot --device="Desktop Chrome" --wait-for-timeout=2200 --full-page \
  http://127.0.0.1:8765/demo/playground.html /tmp/playground_screenshot.png
```
