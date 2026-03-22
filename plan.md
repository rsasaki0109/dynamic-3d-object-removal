# dynamic-3d-object-removal next-step plan

Last updated: 2026-03-23 (Asia/Tokyo)
Repo: `/workspace/ai_coding_ws/dynamic-3d-object-removal`
Branch: `master`
Latest known pushed commit: `2c74440`

## Current state

- GitHub Pages の sequence demo は proof demo としてかなり整っている
- README は短く整理済み
- sequence demo のメイン見出しは `動的物体のせいで地図に残るノイズを除去する` に更新済み
- 無闇な UI 追加より、残りはデータ側の詰めが重要

## Confirmed facts

### Checked-in sequence source

現在の `demo/index_3d_sequence_standalone.html` は、次の 12 フレームから再生成すると一致する。

```bash
/workspace/rosbag/GT/2025-05-28-12-48-29/verify_1_16_5_final/graph/*/cloud.pcd
```

再生成条件:

- `--frame-count 12`
- `--stride 1`
- `--max-render-points 9000`
- `--fps 4`
- `--voxel-size 0.35`
- `--window-size 5`
- `--min-hits 3`

### Per-frame box JSON 探索結果 (2026-03-23 確定)

- `verify_1_16_5_final` 配下を網羅的に探索済み
- JSON ファイルは `execution_time.json` 等のメタデータのみ。detection / box データは存在しない
- `graph/*/data.txt` はカメラ姿勢・変換行列であり、検出結果ではない
- **結論: per-frame box JSON はこのデータセットに存在しない**
- checked-in sequence の cleaned 側は `temporal consistency` ベースで確定

### 第2 sequence 候補 (2026-03-23 調査)

- `verify_1_16_5_final3` (133 フレーム, 同形式) — detection box なし
- `/workspace/rosbag/id/2025-12-10-09-54-25/pointcloud_pcd_5` (222 フレーム, 22GB) — 大規模だが detection box なし
- `/workspace/rosbag/202601_pmo/2_PMO_FUKUSHIMA/` (43 フレーム) — 鉄道系、detection box なし
- いずれも transient clutter の有無は未検証

## Priority

### ~~1. Find real per-frame boxes for the checked-in sequence~~ (完了 — 存在しない)

2026-03-23 に網羅探索し、per-frame box JSON は存在しないことを確定。

### 2. 第二の strong sequence を追加する

per-frame box が無いため、次の高価値は第二の strong sequence を足すこと。

候補データの transient clutter 有無を検証する必要あり。

条件:

- transient clutter が十分ある
- cleaned 側で静的構造が残る
- 今の主張を弱めない

成功条件:

- cherry-pick に見えにくい第二例を出せる

### ~~3. Keep messaging aligned~~ (完了)

2026-03-23 に `index.html` と `demo/index.html` の文言を README に合わせて更新。
temporal consistency ベースであることを明示し、"auto transient boxes" の誤解を招く表現を修正。

## Do not do

- generic viewer controls を増やす
- panel を増やして同じ話を繰り返す
- `real detections` が無いのにそう読める表現にする
- `__pycache__` を commit する

## Useful commands

### Sequence repro

```bash
python3 demo/run_scan_sequence_demo.py \
  --input-glob "/workspace/rosbag/GT/2025-05-28-12-48-29/verify_1_16_5_final/graph/*/cloud.pcd" \
  --frame-count 12 \
  --stride 1 \
  --max-render-points 9000 \
  --fps 4 \
  --voxel-size 0.35 \
  --window-size 5 \
  --min-hits 3 \
  --output-html demo/index_3d_sequence_standalone.html
```

### Sequence repro with real boxes

```bash
python3 demo/run_scan_sequence_demo.py \
  --input-glob "/workspace/rosbag/GT/2025-05-28-12-48-29/verify_1_16_5_final/graph/*/cloud.pcd" \
  --input-objects /path/to/objects.json \
  --frame-count 12 \
  --stride 1 \
  --max-render-points 9000 \
  --fps 4 \
  --voxel-size 0.35 \
  --output-html demo/index_3d_sequence_standalone.html
```

### Visual verification

```bash
python3 -m http.server 8765
```

```bash
npx playwright screenshot --device="Desktop Chrome" --wait-for-timeout=2200 --full-page http://127.0.0.1:8765/demo/index_3d_sequence_standalone.html /tmp/local_sequence.png
```

## Stop rule

次のどれかで demo 作業は一旦止めてよい。

- real per-frame boxes を入れて checked-in sequence を更新できた
- per-frame boxes は無く、第二 sequence も弱いので、今の demo を完成版として据える
- 第二の strong sequence を追加できた
