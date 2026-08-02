# dynamic-3d-object-removal plan

Last updated: 2026-08-02 (Asia/Tokyo)
Repo: `rsasaki0109/dynamic-3d-object-removal`
Branch: `master`
Latest release: **v0.5.0**（tag + GitHub release + PyPI、2026-06-11）
Stars: **74 / 100 (目標)** — fork 4, created 2026-03-05

---

## What this project is

LiDAR 点群から動的物体（車両・歩行者・自転車など）を除去するライブラリ。
**deep learning を使わない** — 幾何ベースのみ。依存は numpy だけ
（pyarrow は Argoverse 2 形式を読む場合のみ必要）。

アルゴリズムは 5 つ、すべて numpy:

1. **box** — 検出 3D box による per-scan crop（検出器 or annotation が必要）
2. **temporal** — voxel hit-count の時系列一貫性（検出器不要、最も単純・高 recall。visibility gate は opt-in）
3. **range** — range-image 可視性（Removert 系 remove + revert、検出器不要、multi-resolution consensus 対応）
4. **scan_ratio** — 極座標カラムの擬似 occupancy（ERASOR 系 scan-ratio + ground revert、検出器不要。
   v0.4.0 から votes をカラム再訪数で正規化）
5. **fusion** (v0.5.0) — 検出器不要 3 チャネルの OR 合成: ray-sampled free-space carving
   （per-scan hit precedence 付き）+ DUFOMap 系 eroded void 確認（hit inflation +
   full-26-neighborhood erosion）+ scan-ratio votes（より厳しい fraction）。
   **Semantic-KITTI で DUFOMap 級（AA 98.6 / 98.0）** — 本 repo の看板手法

3 つの形態で提供:

1. **Python ライブラリ** (`dynamic_object_removal.py`, 1937 行)
2. **CLI** (`dynamic-object-removal`) — **PyPI 公開済み（v0.5.0, 2026-06-11）**
3. **ROS2 リアルタイムノード** (`realtime.py`, 859 行) — box / temporal / range 対応

ベンチマーク 3 本（AV2 / nuScenes mini / Semantic-KITTI(DynamicMap_Benchmark)）と
テスト（merge後のフルスイートは **156 passed / 1 skipped**）付き。
ブラウザ Playground（Pyodide、Box / Range / Temporal の 3 モード、**共有 URL +
AV2/nuScenes プリセット切替対応**）が GitHub Pages にある。

---

## Headline numbers (2026-08-02 時点、全て再現スクリプト付き)

| ベンチ | センサー | ベスト手法 | 数字 | 次点 |
|---|---|---|---|---|
| Semantic-KITTI seq 00 / 05 (DynamicMap_Benchmark) | VLP-64, 141/321 scans | **fusion** | AA **98.6 / 98.0**（リーダーボード首位 DUFOMap は 98.6 / 96.3） | scan_ratio 95.4 / 96.9 |
| Argoverse 2 (3 logs, 12 sweeps mean) | 64-beam | **fusion**（short-window 閾値 0.7/3/4） | F1 **0.642** / static **0.964** | temporal (visibility-gated) F1 0.586 / static 0.968 |
| nuScenes mini (6 eligible scenes, 12 keyframes) | 32-beam（疎） | **range ∧ scan_ratio**（マスク積） | F1 **0.240** / static **0.931**（scene-0757 best-case: F1 0.642 / static 0.842） | temporal (visibility-gated) F1 0.236 / static 0.880 |

nuScenes は単一 scene の結果が transfer を過大評価していたため、5,000 GT dynamic points 以上の eligible 6 scenes の平均を headline にした（閾値未満の scene は一覧に残すが平均から除外）。

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

## Current state (2026-08-02)

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
- [x] ★ **Task B (2026-08-02) 完了**: AV2 3-scene mean は fusion F1 0.642 / static 0.964、nuScenes 6-eligible-scene mean は range ∧ scan_ratio F1 0.240 / static 0.931。scene-0757 の F1 0.642 は best-case で、単一 scene の headline は transfer を過大評価していた。追加 AV2 logs は annotation-only screening で選定
- [x] ★ **Task A (2026-08-02) 完了**: visibility-gated temporal を opt-in で追加。AV2 mean は ungated F1 0.254 / static 0.703 → gated F1 0.586 / static 0.968、nuScenes static 0.401 → 0.880。vectorized path は約 128–162 ms / 100k points
- [x] ★ **Step C の実装部分完了**: Playground 共有 URL（`?mode=&preset=`、Share ボタン）+
      nuScenes 32-beam プリセット（`sample_nuscenes_range.npz`、生成スクリプト付き）
- [x] ブラウザ Playground (Pyodide): Box / Range / Temporal、ユーザー自身の PCD ドロップ対応
- [x] README: How It Compares + AV2 / nuScenes / KITTI 実測テーブル + fusion API sizing 指針
- [x] GitHub Pages デモ群、hero image、social card、About / topics 設定
- [x] テスト **156 passed / 1 skipped**（merge後の `python -m pytest tests/ -q`、2026-08-02）、CI (`test.yml`)、publish workflow (`publish.yml`)
- [x] ★ **Gate 0 の実装部分**: detector-free realtime に timestamp 対応 `--fixed-frame` TF、
      TF unavailable/stale 時 fail-open、TF 統計、deskew 入力契約、移動センサー回帰 test を追加
- [x] ★ **R1 online benchmark**: pose 付き sequence の one-pass replay、point-wise 指標、static keep、
      warm-up / confirmation、filter latency、deadline miss / fail-open、pose-noise sweep、sensor profile を JSON 化
- [x] ★ AV2 12-sweep online baseline を実測。`range` は F1 0.406 / static keep 0.991 / p95 83.8 ms、
      `temporal` の精度優先候補 (0.20 m, min_hits=2) は F1 0.256 / static keep 0.723 / p95 50.7 ms

### 未完了（→ 下の Roadmap）

- [x] ★ **Step 0: PyPI v0.5.0 公開 — 完了（2026-06-11）**。tag v0.5.0 + GitHub release 作成、
      Trusted Publishing（pending publisher 登録は owner がブラウザで実施）経由で publish.yml 成功。
      fresh venv で `pip install dynamic-object-removal` → CLI 0.5.0 → import まで検証済み
- [x] Step A: lidarslam_ros2 連携の技術proof（TIERS same-frontend実map比較、別AV2 sequenceの
      detector-free moving-GT map proof、同一pose graphのdownstream GT画像）は完了。
      lidarslam_ros2側リンクとNTU実bagは追加validationとして未完了
- [ ] PR #28 の draft 解除（**owner の指示があるまで保留** — 2026-06-11 owner 判断）とメンテナ対応
- [ ] Step C の投稿部分（Show HN / Reddit）— 実装は済み、投稿タイミング待ち
- [ ] コアのリファクタリング（2026-06-11 着手、**途中・未コミット** — 詳細は次セクション）

### リファクタリング（2026-06-11 着手 — 途中で中断、作業ツリーに未コミット差分あり）

owner の「refactoring しよう」で着手。**挙動を変えない整理のみ**（公開 API・出力・数値は不変）を
方針とし、テストをセーフティネットに進めた。途中で owner 判断により中断。

#### 着手前に確認した制約（重要 — 次回も前提にすること）

- **`dynamic_object_removal.py` の単一ファイル構成は壊せない**。理由は 2 つ:
  1. Playground (`demo/playground.html` L659) が `fetch("../dynamic_object_removal.py")` で
     リポジトリ直下の 1 ファイルをそのまま Pyodide に読み込んでいる。ページ文言も
     「the exact `dynamic_object_removal.py` from this repo」と単一ファイルであること自体を売りにしている
  2. PyPI も `pyproject.toml` の `py-modules = ["dynamic_object_removal", "realtime", "bench"]`
     のフラット構成で v0.5.0 公開済み。パッケージ分割は import パスの破壊的変更になる
  - → **パッケージ化・ファイル分割はやらない**。ファイル内整理と重複排除に限定する
- 外部から使われている名前を確認済み（grep 済み、安全に変えられる範囲の根拠）:
  - tests は公開 API + `_parse_kitti_calib` のみ import
  - realtime.py / bench.py / demo / scripts / examples は `core.<公開名>` と `DEFAULT_*` 定数のみ。
    プライベート関数への外部依存は `_parse_kitti_calib`（テスト）の 1 件だけ
- ベースライン: 87 passed / 3 skipped（変更前後とも同じ。変更後も再実行してグリーン確認済み）

#### 適用済みの変更（`git diff` で 17+/17-、dynamic_object_removal.py のみ・未コミット）

1. **AV2 feather ローダーの yaw 計算を `_yaw_from_quaternion` に統一** —
   `_load_boxes_from_av2_feather` 内にクォータニオン→yaw の atan2 式が手書きで重複していた。
   `_yaw_from_quaternion((qx, qy, qz, qw))` 呼び出しに置換（順序は x,y,z,w。式は完全に同値）
2. **`_viewpoint_or_none(metadata)` ヘルパーを新設** — PCD VIEWPOINT が既定値
   （identity: `0 0 0 1 0 0 0`）なら None を返す判定が `_structured_pcd_to_scan` と
   `load_pcd_scan` の ascii 分岐の 2 か所に重複していたのを統合
3. **`_pixel_indices(col, row, n_cols, n_rows)` ヘルパーを新設** — レンジ画像の
   「clip して 1 次元 index に flatten」イディオムの共通化。
   `remove_ghost_by_range_image` には適用済み。**`_visibility_votes` への適用は
   owner が edit を reject したため未適用**（同じイディオムが inline のまま残っている。
   reject の意図が「この変更が嫌」なのか「作業を止めたかった」だけなのかは未確認 —
   再開時に owner に確認すること）

#### 洗い出し済み・未着手のリファクタ候補（優先度順）

1. **`main()` の range / scan_ratio 分岐の重複 ~70 行**（最大の獲物）。
   両分岐は「`--input-map` 検証 → map+query ロード → アルゴリズム呼び出し → 保存 →
   サマリ表示 → summary JSON」が 9 割同一。アルゴリズム名と filter 関数だけ差し替える
   `_run_map_cleaner(args, name=..., desc=..., fn=...)` ヘルパーに畳める
2. `_load_points_csv_or_txt` のデッドコード: `except Exception` の後の `except IndexError`
   は到達不能（IndexError は Exception のサブクラス）
3. `remove_points_in_boxes` の keep マスク更新が冗長
   （`mask_local` 経由の 4 行 → `keep[np.nonzero(keep)[0][inside]] = False` 相当に簡約可）
4. `load_points` / `load_boxes` の auto 判定の if 連鎖 → suffix→fmt の dict 化（軽微）
5. クロスファイル: realtime.py の `_eprint` / `_to_float` / `_as_vec3` は core と重複に見えるが
   **セマンティクスが違う**（core 版は raise、realtime 版は None 返し）。安易に統合しない。
   統合するなら方針を先に決めること
6. ベンチマークスクリプト 3 本（av2 / nuscenes / dynamicmap）の SA/DA/AA 系メトリクス計算の
   共有化 — scripts/ 配下に共通モジュールを置く案。効果はあるがスクリプトの自己完結性が
   下がるのでトレードオフ
7. **やらないと決めたもの**: `demo/run_scan_sequence_demo.py`（2433 行）は L1-1969 が
   ほぼ HTML/JS テンプレート文字列で、関数定義は L1970 以降のみ。テンプレートの分割は
   公開デモを壊すリスクに対して益が薄い。`if points.size == 0 or len(points) == 0` の
   冗長イディオム（~8 か所）も churn の割に益なしで見送り

#### 再開時の手順

1. `git diff dynamic_object_removal.py` で適用済み 3 件を確認（または先に commit してから続行）
2. 上の候補 1（main() の重複）から着手するのが費用対効果最大
3. 各編集後に `python3 -m pytest tests/ -q`（87 passed / 3 skipped が基準）
4. 仕上げに CLI スモーク（box / range / scan_ratio の 3 アルゴリズムを data/ のサンプルで 1 回ずつ）
   と Playground がローカルで読み込めることの確認（単一ファイル fetch が前提のため）

---

## 2026-07-13 research audit: realtime + post-processing

関連研究と現行コードを照合した結果、次の開発順をこの plan の最新方針とする。
結論は **SLAM 連携デモより前に realtime の pose contract を直す**、その後に
online MOS と offline map cleaning を別タスクとして評価する、である。

### 確認できた realtime の correctness gap

- `TemporalConsistencyFilter` は入力座標をそのまま voxel history に入れる。
- `RangeImageGhostFilter` は docstring で「全入力が shared frame」と要求している。
- 一方 `realtime.py` は LiDAR frame の点を TF 変換せず、毎フレーム sensor origin `(0, 0, 0)`
  として履歴へ渡している。したがって detector-free realtime 2 手法は、固定 LiDAR または
  外部で ego-motion compensation 済みの入力でのみ正しい。
- 2026-07-13 の合成再現（静的な壁 861 点、センサーを x 方向へ 1 m/frame 移動）:

  | input | temporal kept (frames 1..4) | range kept (frames 1..4) |
  |---|---:|---:|
  | raw sensor frame | `0, 0, 0, 0` | `861, 348, 208, 338` |
  | pose-aligned world frame | `0, 0, 861, 861` | `861, 861, 861, 861` |

  `temporal` の最初の 2 フレームが 0 なのは `min_hits=3` の仕様どおり。問題は warm-up 後も
  raw sensor frame では静的壁を保持できないこと。現行 unit tests は同一点群の反復だけ、
  `bench.py` も同一スキャンの反復だけなので、この failure mode を検出しない。

### 研究から採用する設計要素

| source | 採用候補 | この repo での位置づけ |
|---|---|---|
| [Dynablox (RA-L 2023)](https://arxiv.org/abs/2304.10049) | pose / sensing error を考慮した conservative free space | realtime fixed-frame history と uncertainty guard |
| [DynamicFilter (ICRA 2022)](https://arxiv.org/abs/2206.15102) | scan-to-map front-end + map-to-map back-end | 即時 scan filter と非同期 submap cleanup の分離 |
| [DUFOMap (RA-L 2024)](https://arxiv.org/abs/2403.01449) | eroded void confirmation | offline `fusion` に実装済み。online 化より回帰評価を優先 |
| [4DMOS (RA-L 2022)](https://arxiv.org/abs/2206.04129) | receding horizon + recursive Bayes evidence | hard hit-count を置換せず、optional probabilistic mode で検証 |
| [Learning-free MOS (IROS 2025)](https://mvp.in.tum.de/static/documents/Felix/IROS25.pdf) | 前後 range residual、cluster、Beta evidence | 1-frame delay の `range_prob` 実験候補 |
| [BeautyMap (RA-L 2024)](https://arxiv.org/abs/2405.07283) | binary height matrix + visibility restoration | sparse LiDAR 用 offline candidate generator 候補 |
| [HeLiMOS (IROS 2024)](https://arxiv.org/abs/2408.06328) | heterogeneous LiDAR / instance-aware tracking evaluation | optional large benchmark。通常 CI には入れない |
| [4D implicit mapping (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhong_3D_LiDAR_Mapping_in_Dynamic_Environments_using_a_4D_Implicit_CVPR_2024_paper.pdf) | 4D TSDF と static/dynamic 分解 | accuracy upper bound。GPU/optimization が thesis 外なので core には入れない |
| [ELite (2025)](https://arxiv.org/abs/2502.13452) | short/long-term ephemerality | multi-session 対応時の将来候補。短窓 proof より後 |

### 実行順と acceptance gates

#### Gate 0 — pose-aware realtime correctness（最優先）

実装状況（2026-07-13）:

- [x] `--fixed-frame` と cloud timestamp での TF lookup
- [x] fixed frame 内での temporal / range filtering と、元座標・元 header の出力保持
- [x] missing / stale / invalid TF の fail-open と統計
- [x] deskew 済み入力 contract、固定センサー時の例外を README に明記
- [x] identity / translation / yaw / missing / stale TF、移動する sensor と静的壁・transient の unit tests
- [x] 別の実ROS2 bagを実時間replayしend-to-end callbackを確認。window 3はp95 30.2 ms、
      decode 1.7 / filter 27.7 / publish 0.9 ms、92 aligned / 1 fail-open。NTU map比較はStep Aに残す

成果物:

1. `realtime.py` に `--fixed-frame` と timestamp 対応 TF lookup を追加する。
2. `temporal` / `range` の履歴は fixed frame で保持し、出力は元の `PointCloud2.header.frame_id` に戻す。
3. TF unavailable / stale の方針を fail-open（未除去の scan を publish）にし、回数を summary に出す。
4. README の「temporal は pose 不要」を「固定センサーまたは ego-motion compensated input」に訂正する。
5. deskew は本 step に内製しない。deskew 済み入力を contract とし、未対応を明記する。

合格条件:

- 上記の移動センサー静的壁 test が warm-up 後 100% keep。
- 同じ test に新規 transient object を加え、静的壁を保ったまま object を除去。
- identity / translation / yaw / missing TF / stale TF の unit tests。
- replay benchmark で callback p95 が入力周期未満（10 Hz 入力なら `< 100 ms`）かつ drop なし。
- Gate 0 完了までは Step A の SLAM before/after を公開成果として扱わない。

#### R1 — online evaluation を独立させる

`scripts/run_online_benchmark.py`（名称仮）を作り、pose 付き scan sequence を時刻順に一度だけ replay する。
同一 scan の反復速度測定は microbenchmark として残すが、accuracy の根拠にはしない。

必須出力:

- point-wise precision / recall / F1 or IoU
- static keep、warm-up frames、time-to-confirm
- callback p50 / p95 / max、dropped / fail-open frames
- pose noise sweep（translation / yaw）
- sensor profile、range resolution、voxel size を summary JSON に記録

実装・実測状況（2026-07-13）:

- [x] `scripts/run_online_benchmark.py` と AV2 manifest exporter を追加
- [x] pose、deskew contract、point label / dynamic box GT、missing-pose fail-open を test
- [x] AV2 の ego-motion-compensated 12 sweeps（stride 3、1,235,563 points）を一度だけ replay
- [x] `range` baseline: precision 0.694 / recall 0.287 / F1 0.406 / static keep 0.991 / p95 83.8 ms
- [x] bounded-cost `range` window 3: precision 0.650 / recall 0.301 / F1 0.412 /
      static keep 0.988 / p95 31–38 ms。window 5よりF1と速度を改善し、staticは0.3 point低下
- [x] `temporal` 0.20 m / min_hits 2: precision 0.157 / recall 0.701 / F1 0.256 /
      static keep 0.723 / p95 50.7 ms
- [x] pose-noise sweep: `range` は translation sigma 0.10 m で F1 0.375 / static 0.979、
      yaw sigma 1.0 deg で F1 0.374 / static 0.979。temporal は同条件で static 0.598 / 0.568
- [x] temporal voxel history を vectorize し、0.10 m baseline の p95 を約859 msから約48 msへ短縮

ここで報告するR1 latencyはfilter本体であり、ROS serialization / publishを含むcallback latencyではない。
R1は再現可能なalgorithm gateとして完了。Gate 0のend-to-end callbackは別の実bagで確認済みで、
Step Aの技術proofはTIERSとAV2 downstream GTで完了。NTU実bagは追加validationとして残す。

#### R2 — optional `range_prob` prototype

pose-aware baseline が成立してから、前・対象・後の 3 scan range residual、range-image clustering、
Beta evidence を別モードとして試す。1-frame delay を API と summary に明記する。

採用条件:

- pose-aware `range` に対し online IoU/F1 または static keep を改善。
- p95 が対象 sensor period 未満。
- SemanticKITTI だけでなく sparse / heterogeneous sensor でも改善または非劣化。
- 条件を満たさなければ core mode にせず実験 branch/script に留める。

実装・実測状況（2026-07-13）:

- [x] `scripts/run_range_prob_ablation.py` に前/対象/後 scan の residual、Beta posterior、
      range-image connected components、1-frame delay、境界frame、pose-noise、latencyを実装
- [x] AV2の比較対象を同じinterior frameへ揃えて再集計。`range` baselineは
      precision 0.626 / recall 0.311 / F1 0.415 / static keep 0.987
- [x] AV2 best候補はprecision 0.569 / recall 0.347 / F1 0.431 / static keep 0.981 /
      p95約42 ms。F1は+0.015だがstatic keepは-0.006で、pose translation 0.1 mでは
      F1 0.340 / static 0.949まで低下し、通常の`range`よりrobustnessが悪い
- [x] nuScenes scene-0757では同一interior baselineのF1 0.168 / static 0.703に対し、
      同候補はF1 0.083 / static 0.754。static改善と引き換えにF1が大幅低下し、入力も
      `deskewed: false`のため採用gateを満たさない
- [x] **R2判定: 不採用**。実験scriptは反証可能なablationとして残すが、core mode / public APIへは昇格しない

#### O1 — offline selector と evidence normalization

現在の `fusion` の OR を直ちに置換しない。まず全 channel について、全 scan 数ではなく
その点/voxelを実際に観測した回数、距離、推定 beam spacing を保存する。その上で:

- dense (64-beam+) は現行 `fusion` を baseline。
- sparse (32-beam 以下) は既に勝っている `range ∩ scan_ratio` を baseline。
- sensor metadata がない場合だけ point density から profile を推定。

非回帰条件（checked-in benchmark command の同一データ・同一 GT）:

- AV2: F1 `>= 0.657`、static keep `>= 0.974`
- nuScenes: F1 `>= 0.642`、static keep `>= 0.842`
- DynamicMap seq 00 / 05: AA が現行値から 0.2 point を超えて低下しない
- 精度が同等の場合のみ runtime / memory 改善を採用理由にできる

実装・実測状況（2026-07-13）:

- [x] private O1 evidenceとして各map pointのraw observation / raw see-through / raw surface、
      sensor originからの距離、native beam spacing、effective observation、weighted ratioを実装
- [x] explicit sensor metadata selectorを実装。64-beam / vertical spacing <=0.8 degは`fusion`、
      32-beam / >=1.0 degは`range ∩ scan_ratio`、metadata不明は推測せず`unknown`
- [x] AV2 stride 3を同条件再実行: selected `fusion` はF1 0.657091 / static 0.973881で
      公開3桁gateを維持。normalized visibility候補はF1 0.634315 / static 0.979513で不採用
- [x] nuScenes scene-0757を同条件再実行: baselineはF1 0.641749 / static 0.841699。
      raw floorsを不変にした80-config gridのbestはF1 0.642004 / static 0.842618だが、改善は
      surface ratio guard由来で距離・beam normalization由来ではなく、単一scene tuningのため不昇格
- [x] scene-0757のbest configをheld-out scene-0796へ固定適用すると、baselineとcandidateの
      全accuracy metricsが完全一致。distance/beam normalization由来の改善は再現しないため不昇格を維持
- [x] `scripts/check_sensor_aware_gates.py` を追加。READMEの丸め値を手入力せず、3 benchmarkの
      実summaryからselector routeとAV2 / nuScenes / DynamicMap非回帰gateを機械判定する
- [x] DynamicMap seq 00を公式MD5確認後に現worktreeで再生成。dense selectorの`fusion`は
      SA 98.88 / DA 98.27 / AA 98.58で、既存AA 98.6（公開丸め）を維持
- [x] DynamicMap seq 05も公式MD5確認後に同worktreeで実行。dense selectorの`fusion`は
      SA 97.95 / DA 98.10 / AA 98.03。seq 00 / 05ともAA非回帰gateを通過し、raw cacheは削除
- [x] 大規模gateで不要な手法を再計算しないprivate benchmark option `--methods`を追加。
      既定は従来どおり全手法で、seq 05 O1は結果不変の`--methods fusion`で実行
- [x] AV2 / nuScenes / DynamicMap 00 / 05の6非回帰checkは全pass。ただしnormalized候補は
      held-out改善を再現しないため、**O1判定: selector/evidenceはprivateのまま不昇格**

#### O2 — BeautyMap-style candidate generator

binary height signature と hierarchical z refinement を、最終判定器ではなく候補領域生成として試す。
候補は既存 visibility / surface revert で精査する。まず nuScenes 32-beam で ablation し、
O1 の gate を超えた場合だけ公開 API へ昇格する。

実装・実測状況（2026-07-13）:

- [x] `scripts/run_height_candidate_ablation.py` にXY cell、coarse/fine z-bin、scan visit/hit、
      height persistence、ground-adjacent revertを実装。候補単独で判定せず、sensor別の既存
      baseline dynamic maskとの積として評価する
- [x] nuScenes scene-0757: `range ∩ scan_ratio` baseline F1 0.642 / static 0.842に対し、
      2.0 m XY、0.5/0.25 m z-bin、persistence 0.75でF1 0.846 / static 0.966。
      unique-key集計後の全処理4.52 s（baseline 1.68 s、候補grid 2.84 s）
- [x] AV2 stride 3: `fusion` baseline F1 0.657 / static 0.974に対し、2.0 m XY、
      0.5/0.25 m z-bin、persistence 1.0でF1 0.712 / static 0.988。
      persistence 1.0でも改善するため、現時点の寄与は低永続性よりtall-cell候補、
      hierarchical z refinement、ground revertにある
- [x] mapの重複点ではなくunique XY/XYZ keyでvisit/hitを集計し、最終点へ展開する同値最適化。
      AV2候補gridは66.42 sから16.36 s（4.06倍）、全処理は144.55 sから85.60 sへ短縮し、
      best configと全accuracy metricsは変更前と完全一致
- [x] DynamicMap benchmarkにprivateな`--height-candidate-ablation`、3 datasetとdeskew契約を
      必須にする`scripts/check_height_candidate_gates.py`を追加。不足結果をREADME値で代用しない
- [x] scene別の再tuningをせず、固定nuScenes設定（2.0 m XY、0.5/0.25 m z、persistence 0.75）を
      held-outへ適用。scene-0061はF1 0.0111→0.0116 / static 0.787→0.811で非回帰だが絶対精度が低い。
      scene-0796はF1 0.0272→0.0114 / static 0.940→0.959でF1非回帰gateを違反
- [x] persistence 1.0の単一固定設定も確認。scene-0757では改善が消えてbaseline同等、scene-0796の
      F1低下は残るため、0.75/1.0のどちらもsensor横断の固定設定として成立しない
- [x] DynamicMap seq 00では`fusion` AA 98.58に対し`fusion_height_candidate` AA 98.60。
      +0.02 pointで同sequenceのAA非回帰gateは通過したが、held-out sparse failureを覆さない
- [x] **O2判定: public昇格は不採用**。AV2とscene-0757の改善、DynamicMap 00のAA非回帰は
      研究ablationとして残すが、held-out sparse非回帰とdeskew契約を満たさないためprivateに留める

#### O3 — two-pass SLAM / long-term map

Gate 0 と Step A の実測後に、Pass 1 clean → static-only pose refinement → Pass 2 clean を評価する。
短期 demo の主張を強めない限り、4D neural field や multi-session ephemerality は core に追加しない。

#### 2026-07-13 literature refresh after the ablations

O1/R2/O2とdownstream strict proofの結果を受け、2025–2026の一次資料を再確認した。
次の研究優先順位は「別のper-point閾値探索」ではなく、online front-endと非同期map back-endの
分離、およびtwo-pass trajectory評価とする。

- [FreeDOM (2025)](https://arxiv.org/abs/2504.11073) はmulti-resolutionのconservative free-space、
  scan-removal front-end、incremental map-refinement back-endを分け、異種LiDARを含む評価を行う。
  本repoのAV2 downstream `range`が14.1% moving-GT reductionに対してremoved precision 21.8%に
  留まったため、次の高価値候補は現在scanだけを強く削ることではなく、履歴free-spaceで残留点を
  非同期再判定するbounded back-endである
- [Clean and consistent MLS point clouds from the start (ISPRS JPRS 2026)]
  (https://doi.org/10.1016/j.isprsjprs.2026.05.039) はmoving-object removalとLIOをcoupleし、
  Pass 1のpose/filter結果からstatic-only Pass 2でtrajectoryをrefineする。これはO3を支持するが、
  本repoではまず同一poseのmap改善を確立済みなので、次段階はtrajectory ATE/RPEとmap GTを同時に
  改善するかを測る。poseを変えた結果を現在のsame-pose proofへ混ぜない
- [HDID (Applied Optics 2026)](https://doi.org/10.1364/AO.588185) はheight-density occupancy、
  density ratio、inner-loop detectionを組み合わせる。O2のheight persistence単独はheld-out
  nuScenesでF1を0.0272→0.0114へ悪化させたため、再挑戦するならheightだけでなくdensity ratioを
  独立channelとして実装し、同じheld-out gateを先に通す。現時点ではpublic APIへ追加しない
- [Learning-Free MOS (IROS 2025)]
  (https://www.research.visual-navigation.com/static/documents/Felix/IROS25.pdf) のrange residual、
  clustering、Beta evidenceはR2で独立再現したが、AV2で小改善・static低下、nuScenesでF1低下、
  pose-noise robustness低下だった。sensor横断の原因仮説なしにthreshold sweepを再開しない
- [Lifelong 3D Mapping (2025)](https://arxiv.org/abs/2501.18110) はmulti-session alignment、
  positive/negative change、map versioningまで扱う。これは有望だが、single-session短窓proofと
  online MOSの境界を越えるため、O3より後の独立roadmapに留める

O3のacceptance gate:

1. Pass 1とPass 2で使用frame/stamp集合を保存し、欠落0を確認する。
2. Pass 1 raw、Pass 1 cleaned、Pass 2 static-refinedを分離し、同一pose比較とpose改善比較を混ぜない。
3. GT trajectoryがあるdatasetでATE/RPE非回帰、moving-GT map contamination低下、static GT保持、
   callback/front-end latencyとback-end amortized costを同時に報告する。
4. AV2だけでなくsparse/heterogeneous sensorのheld-outを通るまで公開default/APIにしない。

### 評価タスクを混ぜない

1. **Online MOS**: 現在（または明示した1-frame delay）の scan に対する moving/static 分類。
2. **Online static mapping**: 時間とともに構築される map の ghost、static completeness、latency、memory。
3. **Offline map cleaning**: 完成済み pose-aligned map の SA / DA / AA、runtime、pose-noise robustness。

README の数字は必ず上のどのタスクかを併記する。NTU VIRAL の Step A は downstream SLAM proof には使えるが、
point-wise dynamic GT がない限りアルゴリズム精度 benchmark には使わない。

---

## ★100 Roadmap — 残り 26★

実装優先順: **Gate 0 (pose-aware realtime) → R1 (online benchmark) → Step A
(lidarslam_ros2 連携)** は技術proofまで完了。残りは **O1 (offline selector) → R2/O2 の採否判断**。
PR #28 ready 化と Step C 投稿は owner の GO / 投稿タイミング待ちで、実装順とは別管理とする。

Step 0 (PyPI) と Step B (DynamicMap PR) は完了。なお owner は一度「PyPI 公開しない」に
傾いた後、公開する方針に確定した（2026-06-11）。

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
- ★ PyPI 公開後、install 行を `pip install "dynamic-object-removal>=0.5"` に統一（`ab0a2b9`）
- 残作業:
  1. **draft 解除 — owner の指示があるまでしない**（2026-06-11 owner 判断。技術的ブロッカーは
     すべて解消済みで、GO が出ればすぐ ready 化できる）
  2. メンテナのレビュー対応（出力形式・フォルダ規約の指摘に即応できるよう
     fork `/tmp/DynamicMap_Benchmark_fork` は保持）
  3. merge されたら README の DynamicMap セクションに「upstream に merge 済み」を 1 行
- gh CLI 注意: `gh pr edit` は GraphQL 廃止フィールドで壊れる →
  `gh api -X PATCH repos/KTH-RPL/DynamicMap_Benchmark/pulls/28 -F body=@file.md` を使う

---

### Step A. lidarslam_ros2 連携例（PyPI 公開後の本命）

実装・調査状況（2026-07-13）:

- [x] 隣接 clone の現行 RKO-LIO が deskewed `/rko_lio/frame`、同 timestamp の odom TF、
      `/rko_lio/odometry` を publish することを source で確認
- [x] 循環依存を避ける接続を `RKO-LIO -> DOR -> graph/map backend` と確定。frontend odometry は
      baseline / filtered で同一とし、これは online static mapping の比較であると明記
- [x] `examples/lidarslam_ros2/` に experimental launch、NTU 起点パラメータ、受け入れ手順、test を追加
- [x] 別の実 ROS2 bag で CLI lifecycle と TF fail-open を smoke。window 5 は callback p95 253.7 msで
      gate未達。window 3は実時間 replayで p95 30.2 ms（decode 1.7 / filter 27.7 / publish 0.9 ms）
- [x] window 3をAV2でも再評価: F1 0.412 / static 0.988 / filter p95 31–38 ms。pose noise 0.10 mで
      F1 0.372 / static 0.973、1.0 degで F1 0.362 / static 0.970
- [ ] NTU VIRAL bag はローカル未配置。公式 zip は8,736,253,605 bytes、2026-07-13時点の
      `/home` 空きは2.8 GBのため安全に取得・展開・ROS2変換できない。容量確保後に同条件2-run・
      map比較を追加validationとして実施（callback profilingとTIERS map proofは完了）
- [x] 追加取得不要のStep A候補として、ローカルTIERS Indoor02 Ouster bagを発見。42.3 s、
      PointCloud2 423 frame、IMU 4,231件で、cloudは`t` uint32 timestampを持ちRKO-LIO reader対応。
      Ouster points/IMUは同一clockへrestamp済みで、identity近似extrinsicの限界もconfigへ明記
- [x] TIERSで一回のRKO-LIO frontendをsame-stamp raw/cleaned relayへ分岐。377 paired frame、
      TF failure / fail-open 0、両pose graphはSHA-256まで同一。mapは13,272→13,111点
- [x] 0.2 m空間比較でdense構造proxy 99.74%保持、filtered点のbaseline支持率100%。
      baseline-only候補70点の0.5 m近傍中央値は6（保持点29）で疎な候補へ偏る。ただしTIERSに
      dynamic GTがないため「ghost確定」とは呼ばず、NTUをStep A必須条件から追加validationへ変更
- [x] 隣接 clone の save-time dynamic filterは本repoのonline filterと分離し、結果を流用していない
- [x] offline reader queue deadlockを再現し、header/payload不変でstorage順のみ調整する
      `prepare_offline_lio_bag.py`、bag timing診断、same-pose map比較scriptを追加
- [x] realtime nodeにexact-stamp odometry relay、odom pose + lidar extrinsic合成、同stamp baseline relay、
      cache/drop統計を追加。map proofは5 Hz replayで、別bagの10 Hz latency gateとは分離
- [x] GTがないTIERS結果を補完するstrict offline proofをAV2 scene `0b5142c1…`で追加。
      同一pose・同一12 sweeps・1,235,563点をmoving-track GT 84,471点で全点評価し、detector-free
      `fusion`はGT dynamic recall 66.35%、static GT 97.39%保持、precision 65.08%、F1 65.71%。
      `demo/av2_gt_map_proof.{png,json}`を同じbenchmark commandから再生成可能にし、旧20-frame
      box demoの233,460点は全注釈物体cropでありmoving-GT値ではないことをREADME/landingで明記
- [x] AV2 manifestをROS2 PointCloud2/Odometry bagへ変換する
      `prepare_online_manifest_rosbag.py`とlaunchの`frontend_mode:=external`を追加。整数ns、first-pose
      rebase、GT非公開を保証し、DOR固定出力bagでbaseline/cleaned/odom各11件を確認
- [x] 初期AV2 live-backend比較は**不採用**。同時runはgraph size 2対3、初期sequential DDS
      replayもheavy callbackが同一message集合を消費できずbaseline 4対cleaned 7となった。
      このrunのmap点数はghost根拠に使わない
- [x] 固定DOR出力bagを隣接repo既存のdeterministic `graph_slam_offline_runner`で再実行。
      両branchがexact-stamp cloud/odom 11/11 pair、11 submap、unpaired 0、loop edge 0を記録し、
      raw/optimized TUM trajectoryとloop-edge CSVがbyte-identicalになった
- [x] `compare_downstream_gt_maps.py`と`demo/av2_downstream_gt_map_proof.png`を追加。
      同一poseの`map_optimized.pcd`でraw 1,132,807→cleaned 1,081,968点、moving GT
      78,270→67,212（14.13% reduction）、static GT 1,054,537→1,014,756（96.23% preserve）、
      removed-point precision 21.75%。全map pointがsource GTに0.01 m以内（最大7.7 µm）で対応し、
      labelsは評価時だけ使用

#### ゴール

`rsasaki0109/lidarslam_ros2`（**820★**）のユーザーに「odometry と map backend の間に
`dynamic-object-removal-realtime` を挟むと地図から動的物体の ghost が消える」を
launch 一発 + before/after 画像で見せ、lidarslam_ros2 README からリンクする。

#### なぜ星になるか

- 820★ のリポジトリの README に貼られる導線は、この repo にとって最も確度の高い流入経路
- 「SLAM 利用者が地図のゴーストに困る」はまさに本プロジェクトの想定ユースケースで、転換率が高い
- 開発コストが最小（launch ファイル + 検証 + 画像 1 枚 + README 2 箇所）

#### 成果物

1. `examples/lidarslam_ros2/README.md` — 手順書（英語）。bag 取得 → 連携 launch → 地図比較まで
2. `examples/lidarslam_ros2/dor_lidarslam.launch.py` — `dynamic-object-removal-realtime` を
   RKO-LIO と graph backend の間に挟む launch（cleaned topic remap で接続）
3. `examples/lidarslam_ros2/map_comparison.png` — 除去なし/ありの最終地図 before/after
4. README (this repo): `## Works with your SLAM` セクション新設、画像 + 3 行
5. lidarslam_ros2 側: README に 1 ブロック（画像 + リンク）を追加する commit

#### 設計

- **データ**: lidarslam_ros2 quickstart と同じ **NTU VIRAL `tnp_01`**（Ouster OS1-16, ~580 s, 屋外）
- **トピック接続**:
  ```
  /os_cloud_node/points ──> dynamic-object-removal-realtime ──> /cleaned_points ──> (lidarslam input に remap)
  ```
- **前提**: 上記 Gate 0 の pose-aware realtime が完了していること。raw LiDAR frame の履歴比較では開始しない。
- **アルゴリズム選定**: AV2 online 実測で static keep と F1 が明確に高かった pose-aligned
  `range` を第一候補、`temporal` を比較対象にする。OS1-16 は AV2 よりさらに疎いため、実データで
  vertical resolution と margin を sensor に合わせて探索する。`temporal` は static keep の劣化を解消できる
  設定が得られるまで公開 default にしない。realtime ノードは fusion 非対応のままで良い
  （fusion はオフライン map cleaner）。
- **比較プロトコル**: 一回のfrontend出力をexact-stamp raw/cleaned branchへ分岐し、同じodom・
  frame集合・SLAMパラメータで2 backendを同時実行。個別`/map_save`、同一視点、空間対応率を記録

#### 受け入れ条件

- [ ] 手順書に従って fresh 環境で再現できる（コマンドのコピペで完走）
- [x] detector-free offline mapについて、moving-track GTを重ねたraw/cleaned/TP-FP画像を追加
- [x] 同画像と全点指標でstatic GT 97.39%保持を確認
- [x] lidarslam_ros2 downstream map自体のGT付きbefore/after画像
- [ ] lidarslam_ros2 README からのリンクが生きている

#### リスク / 確認事項

- NTU VIRAL tnp_01 に十分な動的物体（歩行者等）が映っているか **要確認**。
  乏しければ動的物体の多い別の公開 bag に切り替える
- point-wise dynamic GT はないため、ここで得る before/after は downstream proof であり精度 benchmark ではない。
- OS1-16 の疎さで temporal の voxel パラメータが合わない可能性 → voxel-size を粗めに振る
- lidarslam_ros2 の現行 frontend は RKO-LIO（IMU 併用）。TF/タイムスタンプの整合を実機で確認

---

### Step C 残り. Show HN / Reddit 投稿（実装済み・タイミング待ち）

実装（共有 URL + Share ボタン + nuScenes プリセット）は **完了済み**（commit `100b6bf`）。
残りは投稿のみ。

#### 投稿の前提条件（揃ってから出す）

- [x] 標準ベンチの数字が README にある（KITTI fusion AA 98.6 / 98.0 — DUFOMap 級）
- [x] `pip install dynamic-object-removal` が生きている（v0.5.0、fresh venv 検証済み 2026-06-11）
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
Gate 0 (pose-aware realtime)   … shipped
  ↓
R1 (online sequence benchmark) … shipped: correctness / latency / pose-noise を再現可能にする
  ↓
Step A (lidarslam_ros2 連携)    … shipped: downstream SLAM proof（NTUは追加validation）
  ↓
O1 (offline selector)           … 既存 3 benchmark の非回帰 gate
  ↓
R2 / O2                         … ablation を通った候補だけ public API 化
```

- PR #28 draft 解除は owner の GO 待ち、Step C は投稿タイミング待ち。上の技術実装と並行管理する
- 各 Step は独立に merge 可能。Step をまたぐ WIP を作らない
- ★100 到達が目的であって手段の完遂ではない — 途中で到達したら C の投稿タイミングだけ柔軟に

---

## Design decisions & rationale

### なぜ deep learning を使わないか

- LiDAR SLAM の後処理として使う想定。検出器は別にある（or 3D box annotation が既にある）
- 除去自体は幾何で十分 — KITTI で fusion が学習なしに DUFOMap 級 AA を出すのがその実証
- numpy only → pip install して即使える。Docker も GPU もいらない
- **リアルタイム処理が可能**: ROS2 ノードとして PointCloud2 を受けて即座に filter → publish。
  `box` は scan-local、`temporal` / `range` は固定 LiDAR または pose-aligned input が前提。
  移動プラットフォーム対応は Gate 0、`fusion` はオフライン map cleaner でリアルタイム非対象

### 手法の使い分け（実測に基づく公式見解 — README と一致させること）

- **密センサー (64-beam+) のオフライン map cleaning** → `fusion`。
  長尺はデフォルト、~12 スキャンの短窓は `0.7 / 3 / 4`（README「Sizing to your data」）
- **疎センサー (32-beam 以下)** → `range`（解像度をビーム密度に合わせる: nuScenes は 2.5°）、
  さらに `scan_ratio` との dynamic マスク積で precision/static を上積み
- **リアルタイム** → fixed frame に pose-align 済みの `temporal`（最速・単純）か `range`。
  固定 LiDAR でない raw sensor-frame sequence に detector-free history filter を直接使わない
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
- Task E runtime (2026-08-03, this Windows machine, cached AV2 and nuScenes data):
  AV2 scene `0b5142c1…` measured 292.206 s / 242.168 s on the preserved HEAD
  implementation (`workers=1/6`, with the Windows sequential fallback) and
  291.429 s / 72.854 s on the new implementation (`workers=1/6`, with spawn);
  nuScenes scene-0757 measured 31.031 s / 31.702 s before and 48.158 s / 12.872 s
  after. The new `workers=6` keep masks are bit-exact with both old worker-count
  masks on both cached inputs. The historical KITTI seq 00/05 runtime numbers
  are intentionally not repeated here because they were not measured on this
  machine during Task E.

### fusion 転移評価（2026-06-10〜11 確定）

- AV2 12 sweeps: デフォルト閾値 F1 0.391（recall 0.26 — 単発ヒットの拒否権 + void 11 が貯まらない）
  → `0.7 / 3 / 4` で F1 0.657 / static 0.974（best-in-table）
- nuScenes 32-beam: ~13 m 以遠で垂直ビーム間隔 > carving voxel (0.3 m) が構造要因。
  粗 voxel (1.0 m) / 近距離限定 / チャネル単離すべて F1 < 0.3（free 単独 0.154、void 単独 R 0.007）
- nuScenes のベストは range ∧ scan_ratio の dynamic マスク積: F1 0.642 / static 0.842。
  AV2 では同種の合成は fusion 単独を超えない（fusion∨scan_ratio 0.658 は誤差）—
  fusion が既に 3 チャネル OR なので追加合成が効かない
- 実験ハーネス: `/tmp/fusion_xfer.py`（閾値スイープ）、`/tmp/combo_test.py`（マスク合成総当たり）

### Multi-scene benchmark + temporal visibility（2026-08-02 確定）

- nuScenes は 10 scenes を走査し、GT dynamic points が 5,000 未満の scene は一覧に残して mean から除外。eligible 6-scene mean は range ∧ scan_ratio F1 0.240 / static 0.931、scene-0757 は busy-scene の best-case（F1 0.642 / static 0.842）で、単一 scene の数字は transfer を過大評価する
- AV2 は annotation-only screening で moving content の多い 3 logs を選定。全 scene の fusion mean は F1 0.642 / static 0.964。追加 logs は `04994d08…` / `05fa5048…`（default は `0b5142c1…`）
- visibility-gated temporal は opt-in。AV2 mean は F1 0.254 → 0.586、static 0.703 → 0.968、nuScenes static 0.401 → 0.880。vectorized timing は old Counter 515.873 ms → ungated 127.715 ms / gated 162.429 ms（100k points）
- merge sanity: cached `python scripts/run_nuscenes_benchmark.py --scene scene-0757` は temporal ungated P/R/F1/static `0.073 / 0.217 / 0.109 / 0.473`、visibility-gated `0.277 / 0.228 / 0.250 / 0.887` を再現
- 詳細な per-scene 表・設定・再現ログは `data/benchmark_results/multiscene.md` が source of truth。ベンチマークのデータキャッシュは gitignored の `data/` 配下

### PyPI / packaging（2026-06-11 確認）

- **PyPI に `dynamic-object-removal` v0.5.0 公開済み**（2026-06-11、wheel + sdist）。
  Trusted Publishing 設定済み（owner: rsasaki0109 / repo: dynamic-3d-object-removal /
  workflow: publish.yml / environment: pypi）— 以後は tag push だけで公開される
- `publish.yml` のトリガは **tag push (`v*`)**（release 作成ではない）+ workflow_dispatch
- fresh venv 検証済み: `pip install dynamic-object-removal` → CLI/--version/import OK

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

### AV2 visual proof sources

- strict detector-free proof（2026-07-13）: scene `0b5142c1-420b-3fea-9e98-b87327ae22c6`、
  12 sweeps、1,235,563点、moving-track GT 84,471点。fusion recall 66.35%、static GT 97.39%保持
- legacy interactive box preview（2026-03-26）: scene `04994d08-156c-3018-9717-ba0e29be8153`、
  20フレーム、1,957,497 raw points、233,460 annotated-object points cropped。
  parked objectも含むためこの値をghost GTやdetector-free精度とは呼ばない

### ローカルの大物 temp（消してよい候補 — 作業が落ち着いたら）

- `data/dynamicmap/00/dor_fusion_output.pcd`（~700 MB）ほか data/ 配下の生成物（untracked）
- `/tmp` のデータキャッシュ ~1 GB、`/tmp/venv-dor-test05`、`/tmp/venv-twine`（PyPI 公開済みのため不要）
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
python3 scripts/run_av2_benchmark.py --scenes 0b5142c1-420b-3fea-9e98-b87327ae22c6 04994d08-156c-3018-9717-ba0e29be8153 05fa5048-f355-3274-b565-c0ddc547b315   # AV2 (64-beam, 3-scene mean)
python3 scripts/run_nuscenes_benchmark.py --scenes all    # nuScenes mini (32-beam, 6-scene eligible mean)
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
