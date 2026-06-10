# dynamic-3d-object-removal plan

Last updated: 2026-06-11 (Asia/Tokyo)
Repo: `rsasaki0109/dynamic-3d-object-removal`
Branch: `master`
Latest pushed commit: `8af8504` (library v0.5.0)
Stars: **74 / 100 (目標)** — fork 4, created 2026-03-05

---

## What this project is

LiDAR 点群から動的物体（車両・歩行者・自転車など）を除去するライブラリ。
**deep learning を使わない** — 幾何ベースのみ。依存は numpy だけ
（pyarrow は Argoverse 2 形式を読む場合のみ必要）。

アルゴリズムは 5 つ、すべて numpy:

1. **box** — 検出 3D box による per-scan crop（検出器 or annotation が必要）
2. **temporal** — voxel hit-count の時系列一貫性（検出器不要、最も単純・高 recall）
3. **range** — range-image 可視性（Removert 系 remove + revert、検出器不要、multi-resolution consensus 対応）
4. **scan_ratio** — 極座標カラムの擬似 occupancy（ERASOR 系 scan-ratio + ground revert、検出器不要。
   v0.4.0 から votes をカラム再訪数で正規化）
5. **fusion** (v0.5.0) — 検出器不要 3 チャネルの OR 合成: ray-sampled free-space carving
   （per-scan hit precedence 付き）+ DUFOMap 系 eroded void 確認（hit inflation +
   full-26-neighborhood erosion）+ scan-ratio votes（より厳しい fraction）。
   **Semantic-KITTI で DUFOMap 級（AA 98.6 / 98.0）** — 本 repo の看板手法

3 つの形態で提供:

1. **Python ライブラリ** (`dynamic_object_removal.py`, 1937 行)
2. **CLI** (`dynamic-object-removal`) — packaging 済みだが **PyPI 未公開（下記 Step 0 参照）**
3. **ROS2 リアルタイムノード** (`realtime.py`, 859 行) — box / temporal / range 対応

ベンチマーク 3 本（AV2 / nuScenes mini / Semantic-KITTI(DynamicMap_Benchmark)）と
テスト（89 passed + 1 skipped）付き。
ブラウザ Playground（Pyodide、Box / Range / Temporal の 3 モード、**共有 URL +
AV2/nuScenes プリセット切替対応**）が GitHub Pages にある。

---

## Headline numbers (2026-06-11 時点、全て再現スクリプト付き)

| ベンチ | センサー | ベスト手法 | 数字 | 次点 |
|---|---|---|---|---|
| Semantic-KITTI seq 00 / 05 (DynamicMap_Benchmark) | VLP-64, 141/321 scans | **fusion** | AA **98.6 / 98.0**（リーダーボード首位 DUFOMap は 98.6 / 96.3） | scan_ratio 95.4 / 96.9 |
| Argoverse 2 (12 sweeps 短窓) | 64-beam | **fusion**（short-window 閾値 0.7/3/4） | F1 **0.657** / static 0.974 | range 0.600 |
| nuScenes mini scene-0757 (12 keyframes) | 32-beam（疎） | **range ∧ scan_ratio**（マスク積） | F1 **0.642** / static 0.842 | range 単独 0.628 |

転移の教訓（README に全部明記済みの誠実路線）:

- **fusion は密センサー向け**。短窓ではデフォルト閾値（0.9/2/11）が構造的に発火しない →
  12 スキャンなら `free_votes_fraction=0.7, free_votes_floor=3, void_min_scans=4`（F1 0.39 → 0.66）
- **疎 32-beam では fusion は不適**: ~13 m 以遠で垂直ビーム間隔が carving voxel を超え、
  自スキャンのヒットが静的構造を守れずビーム間が彫り抜かれる。voxel を粗くしても回復しない
  （チャネル単離まで実測して F1 < 0.3 を確認）。疎センサーは range（解像度をビーム密度に合わせる）
- **range と scan_ratio の誤検出源は独立**（距離画像の自己遮蔽 vs 極座標カラムの空虚判定）なので、
  疎センサーでは dynamic マスクの積が両指標を同時改善する（追加コストゼロ）

---

## Architecture

```
dynamic_object_removal.py   # コアライブラリ + CLI (v0.5.0, 1937 行)
├── load_points()           # PCD(VIEWPOINT 対応), CSV, TXT, XYZ, NPY, BIN(KITTI), Feather(AV2)
├── load_boxes()            # JSON, CSV, KITTI label_2, Feather(AV2)
├── remove_points_in_boxes()
├── TemporalConsistencyFilter
├── remove_ghost_by_range_image() / clean_map_by_visibility()   # range (multi-resolution consensus)
├── remove_dynamic_by_scan_ratio() / clean_map_by_scan_ratio()  # scan_ratio (votes 正規化)
├── clean_map_by_fusion()   # fusion: free-space carving + eroded voids + scan-ratio votes (OR)
├── RangeImageGhostFilter   # ROS2 用ストリーミング range filter
└── save_points()

realtime.py                 # ROS2 PointCloud2 subscriber/publisher ノード (box/temporal/range)
bench.py                    # 速度 + 精度指標 (compute_accuracy_metrics, dynamic_gt_mask)

demo/
├── playground.html         # Pyodide Playground (925 行)。共有 URL (?mode=&preset=…) + Share ボタン
├── sample_av2_cloud.npy / sample_av2_range.npz / sample_nuscenes_range.npz  # 2 プリセット
├── run_scan_demo.py / run_scan_sequence_demo.py / index_3d_*.html
└── av2_before_after.png / av2_zoom.png / playground_demo.gif / story_mode.gif

scripts/
├── download_av2_sample.py / download_kitti_sample.py
├── build_playground_nuscenes_sample.py   # nuScenes プリセット npz の再現生成
├── run_av2_benchmark.py          # AV2 12 sweeps (P/R/F1/static)。fusion 行 + short-window フラグ付き
├── run_nuscenes_benchmark.py     # nuScenes mini。range∧scan_ratio 行 + fusion 行付き
└── run_dynamicmap_benchmark.py   # Zenodo teaser DL → 除去 → SA/DA/AA/HA 評価まで 1 コマンド

examples/
└── dynamicmap_benchmark/   # KTH-RPL/DynamicMap_Benchmark への methods/dor_numpy アダプタ (PR #28 の中身)
```

---

## Current state (2026-06-11)

### 完了済み（前回 plan 2026-06-10 以降の追加分に ★）

- [x] コアライブラリ: box + temporal + range + scan_ratio の 4 アルゴリズム
- [x] ★ **fusion 追加 (v0.5.0)** — KITTI seq 00/05 で AA 98.6 / 98.0、リーダーボード水準
- [x] ★ scan_ratio 改良: min_votes のスキャン数スケール (v0.3.0) → カラム再訪数で正規化 (v0.4.0)
- [x] multi-resolution consensus（`resolutions=[2.5, 4.0]` で precision 0.68 → 0.78）
- [x] 再現可能ベンチマーク 3 本: AV2 (64-beam) + nuScenes mini (32-beam) + ★ Semantic-KITTI
      (DynamicMap_Benchmark teaser) — すべて登録不要・1 コマンド
- [x] ★ **Step B 完了**: `run_dynamicmap_benchmark.py` + PCD VIEWPOINT 対応 +
      KTH-RPL/DynamicMap_Benchmark へ **PR #28 open（draft）** — `methods/dor_numpy/`、
      本文に SA/DA/AA 実測表 + 再現コマンド
- [x] ★ **fusion の転移評価**: AV2 短窓で best-in-table（short-window 閾値、F1 0.657）、
      nuScenes 32-beam は不適と実測で確定（チャネル単離まで）→ README に「sized to the sensor」
      の使い分け指針として明文化
- [x] ★ **range ∧ scan_ratio 交差** — nuScenes のベスト数字を更新（F1 0.628 → 0.642、static 0.808 → 0.842）
- [x] ★ **Step C の実装部分完了**: Playground 共有 URL（`?mode=&preset=`、Share ボタン）+
      nuScenes 32-beam プリセット（`sample_nuscenes_range.npz`、生成スクリプト付き）
- [x] ブラウザ Playground (Pyodide): Box / Range / Temporal、ユーザー自身の PCD ドロップ対応
- [x] README: How It Compares + AV2 / nuScenes / KITTI 実測テーブル + fusion API sizing 指針
- [x] GitHub Pages デモ群、hero image、social card、About / topics 設定
- [x] テスト 89 件、CI (`test.yml`)、publish workflow (`publish.yml`)

### 未完了（→ 下の Roadmap）

- [ ] **Step 0: PyPI 公開が実は未完**（重要 — README は既に `pip install dynamic-object-removal`
      を案内しているが、PyPI は 404。git tag は v0.1.0 のみで `publish.yml` は一度も発火していない）
- [ ] Step A: lidarslam_ros2 連携（未着手 — `examples/lidarslam_ros2/` なし）
- [ ] PR #28 の draft 解除（ready for review）とメンテナ対応
- [ ] Step C の投稿部分（Show HN / Reddit）— 実装は済み、投稿タイミング待ち

---

## ★100 Roadmap — 残り 26★

優先順: **Step 0 (PyPI 実公開・即日) → PR #28 ready 化 → Step A (lidarslam_ros2 連携) → Step C 投稿 (Show HN)**。

前回 plan から の変更: Step B は完了（PR open まで）。代わりに「README が案内する
インストール手段が存在しない」という信用問題（Step 0）が見つかったので最優先に置く。

---

### Step 0. PyPI v0.5.0 実公開（最優先・~30 分）

#### 問題

- README の Installation は `pip install dynamic-object-removal` を第一手段として案内、
  extras (`[ros2]` / `[benchmarks]`) まで書いてある
- しかし `https://pypi.org/pypi/dynamic-object-removal/json` は **404**。
  `pip index versions` も "No matching distribution found"
- 原因: `publish.yml`（GitHub release トリガの Trusted Publishing）が一度も発火していない。
  git tag は `v0.1.0` のみで GitHub release を作っていない
- PR #28 / README / Playground から来た新規ユーザーの最初のコマンドが失敗する状態。
  星目的以前の信用問題なので他より先に直す

#### 手順

1. PyPI 側で Trusted Publishing の設定が済んでいるか確認（pending publisher 登録が必要なら先に）
2. `git tag v0.5.0 && git push --tags` → GitHub release v0.5.0 作成（リリースノートは
   v0.2.0 以降の CHANGELOG 相当: scan_ratio 正規化 / fusion / ベンチ 3 本 / Playground 共有 URL）
3. `publish.yml` の実行を確認 → 失敗したら fallback: ローカルで
   `python3 -m build && /tmp/venv-twine/bin/twine upload dist/*`
4. fresh venv で `pip install dynamic-object-removal && dynamic-object-removal --version` を検証
5. PR #28 の Reproduce 節は `pip install git+https://…` なのでそのままでも動くが、
   PyPI 公開後は `pip install dynamic-object-removal` に揃えると一貫する（任意）

#### 受け入れ条件

- [ ] `pip install dynamic-object-removal` が fresh venv で成功し v0.5.0 が入る
- [ ] README のインストール手順がコピペで完走する

---

### PR #28 フォローアップ（随時・軽量）

- 現状: KTH-RPL/DynamicMap_Benchmark に **open / draft=true**。
  タイトル "Add dor_numpy: numpy-only detector-free cleaner (AA 98.6 / 98.0 on seq 00 / 05)"
- 本文には KITTI 実測表 + fusion の設計説明 + 転移結果（AV2 best-in-table / nuScenes 不適）+
  再現コマンドまで記載済み。チューニング caveat も明記（誠実路線）
- メンテナ Kin-Zhang から 6/10 にコメント「Thanks for merging. Let me know once it's ready to test.」
  — ready 化待ちの状態
- ★ 2026-06-11 レビュー済み（self + codex gpt-5.5 xhigh）、修正 4 件を push 済み（`46a87e1`）:
  install 行を git+https に（PyPI 404 + `>=0.3` は fusion に対し低すぎた）、range に
  `min_see_through=3 / max_surface_hits=3` を明示（ライブラリ既定 2/2 では README 表の
  range 行が再現しなかった）、identity VIEWPOINT を有効な原点として受理、
  evaluate_all.py のハードコード `algorithms` リスト編集手順を README に明記。
  `examples/dynamicmap_benchmark/` ミラーも同期済み
- 残作業:
  1. **draft 解除**（Step 0 完了後すぐ。インストール手段が生きてから見てもらう）
  2. メンテナのレビュー対応（出力形式・フォルダ規約の指摘に即応できるよう
     fork `/tmp/DynamicMap_Benchmark_fork` は保持）
  3. merge されたら README の DynamicMap セクションに「upstream に merge 済み」を 1 行
- gh CLI 注意: `gh pr edit` は GraphQL 廃止フィールドで壊れる →
  `gh api -X PATCH repos/KTH-RPL/DynamicMap_Benchmark/pulls/28 -F body=@file.md` を使う

---

### Step A. lidarslam_ros2 連携例（PyPI 公開後の本命）

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
   前段に挟む launch（lidarslam 側はトピック remap で接続）
3. `examples/lidarslam_ros2/map_comparison.png` — 除去なし/ありの最終地図 before/after
4. README (this repo): `## Works with your SLAM` セクション新設、画像 + 3 行
5. lidarslam_ros2 側: README に 1 ブロック（画像 + リンク）を追加する commit

#### 設計

- **データ**: lidarslam_ros2 quickstart と同じ **NTU VIRAL `tnp_01`**（Ouster OS1-16, ~580 s, 屋外）
- **トピック接続**:
  ```
  /os_cloud_node/points ──> dynamic-object-removal-realtime ──> /cleaned_points ──> (lidarslam input に remap)
  ```
- **アルゴリズム選定**: 検出器なし・リアルタイムなので `temporal` 第一候補
  （`--voxel-size 0.10 --temporal-window 5 --temporal-min-hits 3` 起点）。
  OS1-16 は 16-beam と疎 — nuScenes での教訓（疎センサーは range を粗解像度で / fusion は不適）
  がそのまま効く。realtime ノードは fusion 非対応のままで良い（fusion はオフライン map cleaner）
- **比較プロトコル**: 同 bag・同 SLAM パラメータで 2 回、`/map_save` の地図 PCD を
  同一視点レンダリングで 2-panel。除去点数 / 地図点数の差も数字で併記

#### 受け入れ条件

- [ ] 手順書に従って fresh 環境で再現できる（コマンドのコピペで完走）
- [ ] before/after で動的 ghost の減少が一目でわかる画像
- [ ] 静的構造（建物・地面）が目視で劣化していない
- [ ] lidarslam_ros2 README からのリンクが生きている

#### リスク / 確認事項

- NTU VIRAL tnp_01 に十分な動的物体（歩行者等）が映っているか **要確認**。
  乏しければ動的物体の多い別の公開 bag に切り替える
- OS1-16 の疎さで temporal の voxel パラメータが合わない可能性 → voxel-size を粗めに振る
- lidarslam_ros2 の現行 frontend は RKO-LIO（IMU 併用）。TF/タイムスタンプの整合を実機で確認

---

### Step C 残り. Show HN / Reddit 投稿（実装済み・タイミング待ち）

実装（共有 URL + Share ボタン + nuScenes プリセット）は **完了済み**（commit `100b6bf`）。
残りは投稿のみ。

#### 投稿の前提条件（揃ってから出す）

- [x] 標準ベンチの数字が README にある（KITTI fusion AA 98.6 / 98.0 — DUFOMap 級）
- [ ] Step 0: `pip install dynamic-object-removal` が生きている（**必須** — HN の最初のコメントで試される）
- [ ] できれば Step A の SLAM 連携画像（なくても可）
- [ ] Pyodide 初期ロードのプログレス表示が貧弱でないか当日確認

#### 素材

- タイトル案: *Show HN: Dynamic object removal for LiDAR point clouds, in the browser (numpy + Pyodide)*
- 1 段落: no GPU / no upload / 実ライブラリそのまま / 検出器不要モード / KITTI ベンチで DUFOMap 級
- 投稿先: HN、r/computervision、r/robotics、ROS Discourse。各 1 回・誇張なし・数字は README の実測のみ
- 当日: Pages 死活確認、コメント返信に張り付く。再投稿はしない

---

## 実行順序と完了の定義

```
Step 0 (PyPI 実公開)            … ~30 分。即日。README の install が嘘でなくなる
  ↓
PR #28 draft 解除               … 5 分 + メンテナ対応は随時
  ↓
Step A (lidarslam_ros2 連携)    … 数時間〜1日。出荷したら即 lidarslam_ros2 README 更新
  ↓
Step C 投稿 (Show HN)           … 半日（投稿 + 当日対応）
```

- 各 Step は独立に merge 可能。Step をまたぐ WIP を作らない
- ★100 到達が目的であって手段の完遂ではない — 途中で到達したら C の投稿タイミングだけ柔軟に

---

## Design decisions & rationale

### なぜ deep learning を使わないか

- LiDAR SLAM の後処理として使う想定。検出器は別にある（or 3D box annotation が既にある）
- 除去自体は幾何で十分 — KITTI で fusion が学習なしに DUFOMap 級 AA を出すのがその実証
- numpy only → pip install して即使える。Docker も GPU もいらない
- **リアルタイム処理が可能**: ROS2 ノードとして PointCloud2 を受けて即座に filter → publish
  （box / temporal / range。fusion はオフライン map cleaner でリアルタイム非対象）

### 手法の使い分け（実測に基づく公式見解 — README と一致させること）

- **密センサー (64-beam+) のオフライン map cleaning** → `fusion`。
  長尺はデフォルト、~12 スキャンの短窓は `0.7 / 3 / 4`（README「Sizing to your data」）
- **疎センサー (32-beam 以下)** → `range`（解像度をビーム密度に合わせる: nuScenes は 2.5°）、
  さらに `scan_ratio` との dynamic マスク積で precision/static を上積み
- **リアルタイム** → `temporal`（最速・単純）か `range`
- **box annotation がある** → `box`

### ポジショニング

詳細は README「How It Compares」参照。要点:

- ERASOR / Removert はオフライン map cleaning 専用・C++/PCL — 本プロジェクトは
  per-scan / realtime + map cleaning 両対応・numpy-only
- DynamicMap_Benchmark の土俵で fusion が DUFOMap 級（AA 98.6 / 98.0 vs 98.6 / 96.3）。
  学習系 4dNDF (AA ≈ 99) は別クラスとして README に明記
- 負ける数字・不適な組み合わせ（nuScenes × fusion 等）も隠さず載せる誠実路線が差別化の一部

### なぜ Argoverse 2 / nuScenes mini / Zenodo teaser か

- すべて**登録不要**で匿名ダウンロードできる — 「1 コマンドで再現」を成立させる唯一の条件
- AV2: 64-beam 密、nuScenes: 32-beam 疎（汎化の検証）、Zenodo teaser: コミュニティ標準指標 (SA/DA/AA)

---

## Confirmed facts

### DynamicMap_Benchmark（2026-06-10〜11 実装で確定）

- pose は PCD の **VIEWPOINT フィールド**に格納（`load_points` で対応済み）
- 指標は SA / DA / AA / HA。評価プロトコルは KDTree 半径 0.05 m の対応付け
  （`run_dynamicmap_benchmark.py` に pure-python 実装、scipy があれば高速化）
- メソッド追加は `methods/<name>/main.py`。本 repo のアダプタは `examples/dynamicmap_benchmark/`
  → PR #28（fork: `rsasaki0109/DynamicMap_Benchmark`, branch `add-dor-numpy`,
  ローカル clone `/tmp/DynamicMap_Benchmark_fork`）
- teaser zip は各 ~385 MB。seq 00: 141 scans / 17.4 M 点 / 96 k 動的 GT 点、
  seq 05: 321 scans / 39.9 M 点 / 684 k 動的 GT 点
- fusion 実行時間: seq 00 654 s / seq 05 1728 s（workers=6）。kept 点数は
  experiments / library / fork アダプタ間で bit-exact を確認済み

### fusion 転移評価（2026-06-10〜11 確定）

- AV2 12 sweeps: デフォルト閾値 F1 0.391（recall 0.26 — 単発ヒットの拒否権 + void 11 が貯まらない）
  → `0.7 / 3 / 4` で F1 0.657 / static 0.974（best-in-table）
- nuScenes 32-beam: ~13 m 以遠で垂直ビーム間隔 > carving voxel (0.3 m) が構造要因。
  粗 voxel (1.0 m) / 近距離限定 / チャネル単離すべて F1 < 0.3（free 単独 0.154、void 単独 R 0.007）
- nuScenes のベストは range ∧ scan_ratio の dynamic マスク積: F1 0.642 / static 0.842。
  AV2 では同種の合成は fusion 単独を超えない（fusion∨scan_ratio 0.658 は誤差）—
  fusion が既に 3 チャネル OR なので追加合成が効かない
- 実験ハーネス: `/tmp/fusion_xfer.py`（閾値スイープ）、`/tmp/combo_test.py`（マスク合成総当たり）

### PyPI / packaging（2026-06-11 確認）

- **PyPI に `dynamic-object-removal` は存在しない**（API 404）。README の install 節は先行して
  PyPI 前提の記述になっている → Step 0 で解消する
- `publish.yml` は GitHub release トリガ（Trusted Publishing 想定）。git tag は `v0.1.0` のみ
- ローカル twine 用 venv: `/tmp/venv-twine`（fallback 用）

### Playground 現状（2026-06-10 実装済み）

- `demo/playground.html` 925 行、単一 HTML、依存追加なし
- 共有 URL: `URLSearchParams`（mode / preset / 主要パラメータ）+ Share ボタン実装済み
- プリセット 2 種: AV2 64-beam (`sample_av2_range.npz`) / nuScenes 32-beam
  (`sample_nuscenes_range.npz`、`scripts/build_playground_nuscenes_sample.py` で再現生成)

### lidarslam_ros2（2026-06-10 調査）

- 820★。frontend は RKO-LIO（LiDAR-inertial odometry）+ backend `graph_based_slam`
- 入力トピック既定: 点群 `/os_cloud_node/points`, IMU `/os_cloud_node/imu`
- quickstart データ: NTU VIRAL `tnp_01`（Ouster OS1-16, ~580 s, 屋外）
- 地図保存: `/map_save` サービス。出力は Autoware-ready bundle

### Checked-in sequence source

`demo/index_3d_sequence_standalone.html` は以下の 12 フレームから再生成すると一致する:

```bash
/workspace/rosbag/GT/2025-05-28-12-48-29/verify_1_16_5_final/graph/*/cloud.pcd
```

再生成パラメータ: `--frame-count 12 --stride 1 --max-render-points 9000 --fps 4 --voxel-size 0.35 --window-size 5 --min-hits 3`

### AV2 hero image source (2026-03-26 確定)

scene: `04994d08-156c-3018-9717-ba0e29be8153` (val split)
20 フレーム, 1,957,497 raw points, 233,123 ghost points removed (11.9%)

### ローカルの大物 temp（消してよい候補 — 作業が落ち着いたら）

- `data/dynamicmap/00/dor_fusion_output.pcd`（~700 MB）ほか data/ 配下の生成物（untracked）
- `/tmp` のデータキャッシュ ~1 GB、`/tmp/venv-dor-test05`、`/tmp/venv-twine`（Step 0 まで保持）
- fork clone `/tmp/DynamicMap_Benchmark_fork`（PR #28 対応完了まで保持）

---

## Do not do

- generic viewer controls を増やす
- panel を増やして同じ話を繰り返す
- `real detections` が無いのにそう読める表現にする
- `__pycache__` / `data/` / `*.egg-info` を commit する (.gitignore 設定済み)
- deep learning の依存を追加する（差別化ポイントは DL 不要であること）
- 比較検証の名目で、この repo を third-party 手法の寄せ集めにしない
  （DynamicMap_Benchmark でも先方実装は取り込まなかった — 自手法を標準データで走らせるだけ）
- Playground にビルドステップ・外部 JS 依存を足す
- ベンチ数字が負ける項目を隠す・チューニングで盛る（nuScenes × fusion の不適も明記する誠実路線を維持）
- C++/Rust ポート・新規手法の追加（★100 目的には寄与せず scope を薄める）
- HN / Reddit への再投稿・複数アカウント投稿
- KITTI での追加チューニング（1 回 30 分級 + 2 シーケンス過適合のリスク。現数字で十分）

---

## Useful commands

### ベンチマーク再現（3 本）

```bash
python3 scripts/run_av2_benchmark.py --frames 12          # AV2 (64-beam, fusion 含む)
python3 scripts/run_nuscenes_benchmark.py                 # nuScenes mini (32-beam, 交差含む)
python3 scripts/run_dynamicmap_benchmark.py --sequences 00 05   # Semantic-KITTI (SA/DA/AA)
```

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

### ROS2 realtime（Step A で使う形）

```bash
dynamic-object-removal-realtime \
  --pointcloud-topic /os_cloud_node/points \
  --output-topic /cleaned_points \
  --algorithm temporal \
  --voxel-size 0.10 --temporal-window 5 --temporal-min-hits 3
```

### テスト / Visual verification

```bash
python3 -m pytest tests/ -v
python3 -m http.server 8765
npx playwright screenshot --device="Desktop Chrome" --wait-for-timeout=2200 --full-page \
  http://127.0.0.1:8765/demo/playground.html /tmp/playground_screenshot.png
```

### PR #28 の本文編集（gh pr edit は壊れている）

```bash
export PATH="$HOME/.local/bin:$PATH"
gh api -X PATCH repos/KTH-RPL/DynamicMap_Benchmark/pulls/28 -F body=@/tmp/pr28_body.md
```
