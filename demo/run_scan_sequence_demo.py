#!/usr/bin/env python3
"""Build a self-contained accumulation split demo from multiple point-cloud frames."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_object_removal import (  # noqa: E402
    DEFAULT_BOX_MARGIN,
    TemporalConsistencyFilter,
    load_points,
    parse_boxes_payload,
    remove_points_in_boxes,
)

DEFAULT_FPS = 4.0
DEFAULT_FRAME_COUNT = 12
DEFAULT_MAX_RENDER_POINTS = 9000
DEFAULT_VOXEL_SIZE = 0.35
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MIN_HITS = 3

HTML_TEMPLATE = r'''<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Dynamic 3D Object Removal | Accumulation Split Demo</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f4f8fb;
        --panel: rgba(255, 255, 255, 0.9);
        --panel-strong: rgba(255, 255, 255, 0.97);
        --line: rgba(148, 163, 184, 0.28);
        --text: #0f172a;
        --muted: #475569;
        --raw: #c2410c;
        --raw-soft: #fb7185;
        --clean: #0f766e;
        --clean-soft: #2dd4bf;
        --path: #f59e0b;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Trebuchet MS", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at 10% 12%, rgba(194, 65, 12, 0.1), transparent 22%),
          radial-gradient(circle at 84% 10%, rgba(15, 118, 110, 0.13), transparent 24%),
          linear-gradient(180deg, #fbfdff 0%, #eef5fb 48%, #f5f8fc 100%);
      }
      .shell {
        display: grid;
        grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
        gap: 18px;
        min-height: 100vh;
        padding: 18px;
      }
      .panel, .compare-card {
        border-radius: 24px;
        border: 1px solid var(--line);
        background: var(--panel);
        box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(12px);
      }
      .panel {
        padding: 22px;
        display: flex;
        flex-direction: column;
        gap: 18px;
      }
      .eyebrow {
        display: inline-flex;
        width: fit-content;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.12);
        color: #115e59;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      h1 {
        margin: 0;
        font-size: clamp(28px, 4vw, 38px);
        line-height: 1.05;
      }
      .lead {
        margin: 0;
        color: var(--muted);
        line-height: 1.7;
      }
      .section {
        display: grid;
        gap: 10px;
      }
      .section h2 {
        margin: 0;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .thesis {
        display: grid;
        gap: 10px;
      }
      .thesis-card {
        padding: 12px 14px;
        border-radius: 16px;
        background: var(--panel-strong);
        border: 1px solid rgba(148, 163, 184, 0.22);
      }
      .thesis-card strong {
        display: block;
        font-size: 14px;
        margin-bottom: 5px;
      }
      .thesis-card span {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.55;
      }
      .stats {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }
      .stat {
        padding: 12px 14px;
        border-radius: 16px;
        background: var(--panel-strong);
        border: 1px solid rgba(148, 163, 184, 0.22);
      }
      .stat strong {
        display: block;
        font-size: 20px;
        line-height: 1.1;
      }
      .stat span {
        display: block;
        margin-top: 4px;
        font-size: 13px;
        color: var(--muted);
      }
      .controls {
        display: grid;
        gap: 12px;
      }
      .buttons {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
      }
      button, select, input[type="range"] { font: inherit; }
      button, select {
        border: 1px solid rgba(15, 118, 110, 0.18);
        background: #fff;
        color: var(--text);
        border-radius: 12px;
        padding: 10px 12px;
      }
      button.primary {
        background: linear-gradient(135deg, var(--clean) 0%, #115e59 100%);
        color: #fff;
        border: none;
      }
      .control-row {
        display: grid;
        gap: 8px;
      }
      .control-row label {
        font-size: 13px;
        font-weight: 700;
        color: var(--muted);
      }
      .range-meta, .small-meta {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 13px;
        color: var(--muted);
      }
      .switches {
        border: 0;
        padding: 0;
        margin: 0;
        display: grid;
        gap: 8px;
      }
      .switches label {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 14px;
      }
      .dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
      }
      .bev-card {
        padding: 12px;
        border-radius: 18px;
        background: #08111f;
        border: 1px solid rgba(15, 23, 42, 0.4);
      }
      #timeline-canvas {
        display: block;
        width: 100%;
        height: 120px;
        border-radius: 12px;
        background: linear-gradient(180deg, #08111f 0%, #0b1728 100%);
      }
      #ghost-bev {
        display: block;
        width: 100%;
        height: 220px;
        border-radius: 12px;
        background: linear-gradient(180deg, #08111f 0%, #0b1728 100%);
      }
      .bev-meta {
        margin-top: 8px;
        color: #cbd5e1;
        font-size: 12px;
        line-height: 1.6;
      }
      .compare-card {
        position: relative;
        padding: 16px;
        display: grid;
        gap: 12px;
        min-height: min(88vh, 940px);
      }
      .story-chip {
        position: absolute;
        left: 50%;
        top: 14px;
        transform: translateX(-50%);
        max-width: min(70%, 720px);
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.3);
        color: #e2e8f0;
        font-size: 13px;
        line-height: 1.4;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.22);
        pointer-events: none;
      }
      .compare-head {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .compare-label {
        padding: 14px 16px;
        border-radius: 18px;
        background: var(--panel-strong);
        border: 1px solid rgba(148, 163, 184, 0.22);
      }
      .compare-label strong {
        display: block;
        font-size: 18px;
      }
      .compare-label span {
        display: block;
        margin-top: 6px;
        font-size: 13px;
        color: var(--muted);
        line-height: 1.55;
      }
      .compare-label.raw strong { color: var(--raw); }
      .compare-label.clean strong { color: var(--clean); }
      #compare-stage {
        position: relative;
        min-height: min(72vh, 760px);
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.96) 0%, rgba(15, 23, 42, 0.96) 50%, rgba(3, 18, 16, 0.98) 50%, rgba(3, 18, 16, 0.98) 100%);
      }
      #compare-stage canvas {
        display: block;
        width: 100%;
        height: 100%;
      }
      .split-line {
        position: absolute;
        top: 0;
        bottom: 0;
        left: 50%;
        width: 1px;
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.32), rgba(255,255,255,0.06));
        pointer-events: none;
      }
      .canvas-hud {
        position: absolute;
        top: 14px;
        left: 14px;
        right: 14px;
        display: flex;
        justify-content: space-between;
        gap: 12px;
        pointer-events: none;
      }
      .canvas-chip {
        padding: 8px 10px;
        border-radius: 12px;
        font-size: 12px;
        color: #e2e8f0;
        background: rgba(15, 23, 42, 0.58);
        border: 1px solid rgba(226, 232, 240, 0.12);
      }
      .compare-footer {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
      }
      .footer-card {
        padding: 12px 14px;
        border-radius: 16px;
        background: var(--panel-strong);
        border: 1px solid rgba(148, 163, 184, 0.22);
      }
      .footer-card strong {
        display: block;
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      .footer-card span {
        font-size: 18px;
        font-weight: 700;
      }
      .evidence-strip {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .evidence-card {
        padding: 14px;
        border-radius: 18px;
        background: var(--panel-strong);
        border: 1px solid rgba(148, 163, 184, 0.22);
        display: grid;
        gap: 10px;
      }
      .evidence-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 12px;
      }
      .evidence-head strong {
        font-size: 15px;
      }
      .evidence-head span {
        color: var(--muted);
        font-size: 12px;
      }
      .evidence-copy {
        margin: 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
      }
      .mini-compare {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }
      .mini-pane {
        display: grid;
        gap: 8px;
      }
      .mini-pane span {
        font-size: 12px;
        font-weight: 700;
        color: var(--muted);
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      .mini-pane canvas {
        display: block;
        width: 100%;
        height: 148px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: linear-gradient(180deg, #08111f 0%, #0b1728 100%);
      }
      .note-box {
        padding: 12px 14px;
        border-radius: 16px;
        background: rgba(15, 118, 110, 0.08);
        color: #134e4a;
        line-height: 1.6;
        font-size: 13px;
      }
      a { color: #115e59; }
      @media (max-width: 1100px) {
        .shell { grid-template-columns: 1fr; }
      }
      @media (max-width: 720px) {
        .compare-head, .compare-footer, .stats, .evidence-strip, .mini-compare { grid-template-columns: 1fr; }
        #compare-stage { min-height: 64vh; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <aside class="panel">
        <div class="section">
          <span class="eyebrow">Impact Demo</span>
          <h1>動的点を見せるのではなく、地図の汚染を見せる</h1>
          <p class="lead">
            初期表示は build-up 後の最終状態です。左は観測した点をそのまま積算、右は cleaned 点だけを積算します。主役は current frame ではなく、最後に残る ghost の差です。
          </p>
        </div>

        <div class="section thesis">
          <div class="thesis-card">
            <strong>Raw accumulation</strong>
            <span>その時々に見えた点を全部残すので、移動体や transient clutter がゴーストとして積み上がります。</span>
          </div>
          <div class="thesis-card">
            <strong>Cleaned accumulation</strong>
            <span>継続して観測された点だけを積むので、静的構造の輪郭が先に残ります。checked-in 版は temporal consistency で cleaned map を作っています。</span>
          </div>
        </div>

        <div class="section">
          <h2>Headline</h2>
          <div class="stats">
            <div class="stat"><strong id="stat-frame-count">0</strong><span>frames</span></div>
            <div class="stat"><strong id="stat-ghost-ratio">0%</strong><span>final ghost ratio</span></div>
            <div class="stat"><strong id="stat-ghost-voxels">0</strong><span>ghost voxels</span></div>
            <div class="stat"><strong id="stat-stable-voxels">0</strong><span>stable voxels</span></div>
          </div>
        </div>

        <div class="section controls">
          <h2>Playback</h2>
          <div class="buttons">
            <button class="primary" id="play-toggle">Replay build-up</button>
            <button id="story-mode">Story mode</button>
            <button id="fit-view">Fit view</button>
            <button id="focus-ghost">Focus ghost hotspot</button>
            <button id="focus-stable">Focus static keep</button>
            <button id="download-shot">Screenshot</button>
          </div>
          <div class="control-row">
            <label for="frame-slider">Frame</label>
            <input id="frame-slider" type="range" min="0" max="0" value="0" step="1" />
            <div class="range-meta">
              <span id="frame-label">frame 0 / 0</span>
              <span id="frame-source">-</span>
            </div>
          </div>
          <div class="control-row">
            <label for="speed-select">Speed</label>
            <select id="speed-select">
              <option value="0.5">0.5x</option>
              <option value="1" selected>1x</option>
              <option value="2">2x</option>
              <option value="4">4x</option>
            </select>
          </div>
          <div class="control-row">
            <label for="point-size">Point size</label>
            <input id="point-size" type="range" min="0.6" max="3.2" step="0.1" value="1.2" />
            <div class="small-meta"><span>small</span><span id="point-size-value">1.2 px</span><span>large</span></div>
          </div>
        </div>

        <div class="section">
          <h2>Layers</h2>
          <fieldset class="switches">
            <label><input id="toggle-current" type="checkbox" checked /> <span class="dot" style="background: #e2e8f0"></span>current frame highlight</label>
            <label><input id="toggle-transient" type="checkbox" checked /> <span class="dot" style="background: var(--raw-soft)"></span>transient / removed points</label>
            <label><input id="toggle-path" type="checkbox" checked /> <span class="dot" style="background: var(--path)"></span>frame path</label>
            <label id="toggle-boxes-wrap"><input id="toggle-boxes" type="checkbox" checked /> <span class="dot" style="background: #f97316"></span>3D boxes</label>
          </fieldset>
        </div>

        <div class="section">
          <h2>Current frame</h2>
          <div class="small-meta"><span>frame points</span><span id="meta-input">0</span></div>
          <div class="small-meta"><span>cleaned frame points</span><span id="meta-kept">0</span></div>
          <div class="small-meta"><span>accum ghost voxels</span><span id="meta-ghost">0</span></div>
          <div class="small-meta"><span>accum ghost ratio</span><span id="meta-ratio">0%</span></div>
        </div>

        <div class="section">
          <h2>Contamination timeline</h2>
          <div class="bev-card">
            <canvas id="timeline-canvas" width="320" height="120"></canvas>
            <div class="bev-meta">ghost ratio per frame. the blue cursor tracks the current frame, and the amber mark shows the peak contamination moment in the sequence.</div>
          </div>
        </div>

        <div class="section">
          <h2>Ghost heatmap</h2>
          <div class="bev-card">
            <canvas id="ghost-bev" width="320" height="220"></canvas>
            <div class="bev-meta">teal = final cleaned footprint / pink-yellow = raw-only occupancy. hotter cells indicate where transient structure remains if you keep every observation.</div>
          </div>
        </div>

        <div class="section">
          <h2>Source</h2>
          <p class="lead" id="source-note"></p>
        </div>
      </aside>

      <section class="compare-card">
        <div class="compare-head">
          <div class="compare-label raw">
            <strong>Raw accumulation</strong>
            <span>左は見えた点をそのまま積む。current frame の transient / removed は強いピンクで重ねています。</span>
          </div>
          <div class="compare-label clean">
            <strong>Cleaned accumulation</strong>
            <span>右は cleaned 点だけを積む。current frame の cleaned 点を明るく重ねて、静的構造がどこから立ち上がるかを見せます。</span>
          </div>
        </div>

        <div id="compare-stage">
          <div class="split-line"></div>
          <div class="story-chip" id="story-chip">Final map impact: ghost voxels shrink once transient clutter is cropped.</div>
          <div class="canvas-hud">
            <div class="canvas-chip" id="hud-left">raw</div>
            <div class="canvas-chip" id="hud-right">cleaned</div>
          </div>
        </div>

        <div class="compare-footer">
          <div class="footer-card"><strong>Accumulated raw voxels</strong><span id="footer-raw">0</span></div>
          <div class="footer-card"><strong>Accumulated cleaned voxels</strong><span id="footer-clean">0</span></div>
          <div class="footer-card"><strong>Accumulated ghost voxels</strong><span id="footer-ghost">0</span></div>
        </div>

        <div class="evidence-strip">
          <div class="evidence-card">
            <div class="evidence-head">
              <strong>Ghost hotspot</strong>
              <span id="hotspot-metric">0 raw-only cells in crop</span>
            </div>
            <div class="mini-compare">
              <div class="mini-pane">
                <span>raw crop</span>
                <canvas id="hotspot-raw" width="220" height="148"></canvas>
              </div>
              <div class="mini-pane">
                <span>cleaned crop</span>
                <canvas id="hotspot-clean" width="220" height="148"></canvas>
              </div>
            </div>
            <p class="evidence-copy" id="hotspot-copy">largest residual contamination pocket. raw keeps the transient footprint; cleaned removes it from the final map.</p>
          </div>
          <div class="evidence-card">
            <div class="evidence-head">
              <strong>Static structure preserved</strong>
              <span id="preserve-metric">0% overlap</span>
            </div>
            <div class="mini-compare">
              <div class="mini-pane">
                <span>raw crop</span>
                <canvas id="preserve-raw" width="220" height="148"></canvas>
              </div>
              <div class="mini-pane">
                <span>cleaned crop</span>
                <canvas id="preserve-clean" width="220" height="148"></canvas>
              </div>
            </div>
            <p class="evidence-copy" id="preserve-copy">dense stable footprint with near-zero ghost leakage. this is the counter-example to “cleaning just erases structure”.</p>
          </div>
        </div>

        <div class="note-box">
          drag: orbit / wheel: zoom / right drag: pan. the page opens at the final accumulation because that is where the map contamination difference is clearest. use <code>Replay build-up</code> to watch how the ghost grows over time. when detections are absent, the checked-in page bootstraps transient boxes and uses them for a sampled box-removal preview.
        </div>
      </section>
    </div>

    <script type="importmap">
      {
        "imports": {
          "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
          "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
      }
    </script>
    <script type="module">
      import * as THREE from "three";
      import { OrbitControls } from "three/addons/controls/OrbitControls.js";

      const DEMO_DATA = __DEMO_DATA__;
      const AUTO_BOX_GRID = 0.85;
      const AUTO_BOX_MIN_POINTS = 18;
      const AUTO_BOX_MAX_BOXES = 3;
      const AUTO_BOX_PADDING = [0.8, 0.8, 0.6];

      function deriveTransientBoxes(flat) {
        if (!flat || flat.length < AUTO_BOX_MIN_POINTS * 3) return [];
        const cells = new Map();
        for (let i = 0; i < flat.length; i += 3) {
          const gx = Math.floor(flat[i] / AUTO_BOX_GRID);
          const gy = Math.floor(flat[i + 1] / AUTO_BOX_GRID);
          const key = `${gx},${gy}`;
          let cell = cells.get(key);
          if (!cell) {
            cell = { gx, gy, indices: [] };
            cells.set(key, cell);
          }
          cell.indices.push(i);
        }

        const visited = new Set();
        const boxes = [];
        for (const cell of cells.values()) {
          const seedKey = `${cell.gx},${cell.gy}`;
          if (visited.has(seedKey)) continue;
          const queue = [cell];
          visited.add(seedKey);
          const indices = [];
          while (queue.length > 0) {
            const current = queue.pop();
            indices.push(...current.indices);
            for (let dx = -1; dx <= 1; dx += 1) {
              for (let dy = -1; dy <= 1; dy += 1) {
                const key = `${current.gx + dx},${current.gy + dy}`;
                if (visited.has(key) || !cells.has(key)) continue;
                visited.add(key);
                queue.push(cells.get(key));
              }
            }
          }

          if (indices.length < AUTO_BOX_MIN_POINTS) continue;
          let xmin = Infinity;
          let ymin = Infinity;
          let zmin = Infinity;
          let xmax = -Infinity;
          let ymax = -Infinity;
          let zmax = -Infinity;
          for (const idx of indices) {
            const x = flat[idx];
            const y = flat[idx + 1];
            const z = flat[idx + 2];
            xmin = Math.min(xmin, x);
            ymin = Math.min(ymin, y);
            zmin = Math.min(zmin, z);
            xmax = Math.max(xmax, x);
            ymax = Math.max(ymax, y);
            zmax = Math.max(zmax, z);
          }
          const size = [
            Math.max(xmax - xmin + AUTO_BOX_PADDING[0], AUTO_BOX_GRID * 1.5),
            Math.max(ymax - ymin + AUTO_BOX_PADDING[1], AUTO_BOX_GRID * 1.5),
            Math.max(zmax - zmin + AUTO_BOX_PADDING[2], 1.2),
          ];
          if (size[0] * size[1] > 90 || size[2] > 8) continue;
          boxes.push({
            center: [
              Number(((xmin + xmax) * 0.5).toFixed(3)),
              Number(((ymin + ymax) * 0.5).toFixed(3)),
              Number(((zmin + zmax) * 0.5).toFixed(3)),
            ],
            size: size.map((value) => Number(value.toFixed(3))),
            yaw: 0,
            label: `auto transient box ${boxes.length + 1}`,
            points: indices.length,
          });
        }
        boxes.sort((a, b) => b.points - a.points);
        return boxes.slice(0, AUTO_BOX_MAX_BOXES);
      }

      function rotateToBoxFrame(dx, dy, yaw = 0) {
        const c = Math.cos(yaw);
        const s = Math.sin(yaw);
        return [dx * c + dy * s, -dx * s + dy * c];
      }

      function splitPointsByBoxes(flat, objects) {
        if (!flat || flat.length === 0 || !objects || objects.length === 0) {
          return { kept: flat ? new Float32Array(flat) : new Float32Array(), removed: new Float32Array() };
        }
        const kept = [];
        const removed = [];
        pointLoop:
        for (let i = 0; i < flat.length; i += 3) {
          const x = flat[i];
          const y = flat[i + 1];
          const z = flat[i + 2];
          for (const object of objects) {
            const dx = x - object.center[0];
            const dy = y - object.center[1];
            const dz = z - object.center[2];
            const [lx, ly] = rotateToBoxFrame(dx, dy, object.yaw || 0);
            if (
              Math.abs(lx) <= object.size[0] * 0.5 &&
              Math.abs(ly) <= object.size[1] * 0.5 &&
              Math.abs(dz) <= object.size[2] * 0.5
            ) {
              removed.push(x, y, z);
              continue pointLoop;
            }
          }
          kept.push(x, y, z);
        }
        return {
          kept: new Float32Array(kept),
          removed: new Float32Array(removed),
        };
      }

      function voxelSetFromFlat(flat, voxelSize) {
        const voxels = new Set();
        if (!flat || flat.length === 0) return voxels;
        for (let i = 0; i < flat.length; i += 3) {
          const vx = Math.floor(flat[i] / voxelSize);
          const vy = Math.floor(flat[i + 1] / voxelSize);
          const vz = Math.floor(flat[i + 2] / voxelSize);
          voxels.add(`${vx},${vy},${vz}`);
        }
        return voxels;
      }

      function addVoxels(target, source) {
        for (const item of source) target.add(item);
      }

      function differenceVoxels(a, b) {
        const diff = new Set();
        for (const item of a) if (!b.has(item)) diff.add(item);
        return diff;
      }

      function projectBevFromVoxelSet(voxelSet, voxelSize) {
        const counts = new Map();
        for (const key of voxelSet) {
          const [sx, sy] = key.split(",", 2);
          const xyKey = `${sx},${sy}`;
          counts.set(xyKey, (counts.get(xyKey) || 0) + 1);
        }
        if (counts.size === 0) {
          return { flat: [], maxCount: 0, bounds: { xmin: -1, xmax: 1, ymin: -1, ymax: 1 } };
        }
        const flat = [];
        let xmin = Infinity;
        let xmax = -Infinity;
        let ymin = Infinity;
        let ymax = -Infinity;
        let maxCount = 0;
        for (const [xyKey, count] of Array.from(counts.entries()).sort()) {
          const [sx, sy] = xyKey.split(",").map(Number);
          const x = (sx + 0.5) * voxelSize;
          const y = (sy + 0.5) * voxelSize;
          flat.push(Number(x.toFixed(3)), Number(y.toFixed(3)), count);
          xmin = Math.min(xmin, x);
          xmax = Math.max(xmax, x);
          ymin = Math.min(ymin, y);
          ymax = Math.max(ymax, y);
          maxCount = Math.max(maxCount, count);
        }
        return {
          flat,
          maxCount,
          bounds: {
            xmin: Number(xmin.toFixed(3)),
            xmax: Number(xmax.toFixed(3)),
            ymin: Number(ymin.toFixed(3)),
            ymax: Number(ymax.toFixed(3)),
          },
        };
      }

      const baseFrames = (DEMO_DATA.frames || []).map((frame) => ({
        ...frame,
        objects: Array.isArray(frame.objects) ? frame.objects : [],
        input: new Float32Array(frame.input || []),
        kept: new Float32Array(frame.kept || []),
        removed: new Float32Array(frame.removed || []),
      }));
      const frames = baseFrames.map((frame) => {
        const derivedObjects = frame.objects.length > 0 ? frame.objects : deriveTransientBoxes(frame.removed);
        const objectsDerived = frame.objects.length === 0 && derivedObjects.length > 0;
        const boxed = derivedObjects.length > 0 ? splitPointsByBoxes(frame.input, derivedObjects) : null;
        return {
          ...frame,
          objects: derivedObjects,
          objectsDerived,
          kept: boxed ? boxed.kept : frame.kept,
          removed: boxed ? boxed.removed : frame.removed,
          render_kept_points: boxed ? boxed.kept.length / 3 : frame.render_kept_points,
          render_removed_points: boxed ? boxed.removed.length / 3 : frame.render_removed_points,
        };
      });
      const pathPoints = (DEMO_DATA.path || []).map((point) => point.map(Number));
      const usesDerivedBoxes = frames.some((frame) => frame.objectsDerived);
      const hasBoxes = frames.some((frame) => Array.isArray(frame.objects) && frame.objects.length > 0);
      const voxelSize = Math.max(Number((DEMO_DATA.bev || {}).voxel_size) || 1, 1e-3);

      if (hasBoxes) {
        const rawAccum = new Set();
        const cleanAccum = new Set();
        for (const frame of frames) {
          addVoxels(rawAccum, voxelSetFromFlat(frame.input, voxelSize));
          addVoxels(cleanAccum, voxelSetFromFlat(frame.kept, voxelSize));
          const ghostAccum = differenceVoxels(rawAccum, cleanAccum);
          frame.raw_voxels = rawAccum.size;
          frame.clean_voxels = cleanAccum.size;
          frame.ghost_voxels = ghostAccum.size;
          frame.ghost_ratio_pct = Number((100 * ghostAccum.size / Math.max(1, rawAccum.size)).toFixed(2));
        }
        const finalGhost = differenceVoxels(rawAccum, cleanAccum);
        const cleanBev = projectBevFromVoxelSet(cleanAccum, voxelSize);
        const ghostBev = projectBevFromVoxelSet(finalGhost, voxelSize);
        DEMO_DATA.meta.final_clean_voxels = cleanAccum.size;
        DEMO_DATA.meta.final_ghost_voxels = finalGhost.size;
        DEMO_DATA.meta.final_ghost_ratio_pct = Number((100 * finalGhost.size / Math.max(1, rawAccum.size)).toFixed(2));
        DEMO_DATA.bev = {
          voxel_size: voxelSize,
          clean: cleanBev.flat,
          ghost: ghostBev.flat,
          max_clean_count: cleanBev.maxCount,
          max_ghost_count: ghostBev.maxCount,
          bounds: {
            xmin: Math.min(cleanBev.bounds.xmin, ghostBev.bounds.xmin),
            xmax: Math.max(cleanBev.bounds.xmax, ghostBev.bounds.xmax),
            ymin: Math.min(cleanBev.bounds.ymin, ghostBev.bounds.ymin),
            ymax: Math.max(cleanBev.bounds.ymax, ghostBev.bounds.ymax),
          },
        };
      }

      const bev = DEMO_DATA.bev || { bounds: { xmin: -1, xmax: 1, ymin: -1, ymax: 1 }, clean: [], ghost: [], voxel_size: 1, max_clean_count: 1, max_ghost_count: 1 };

      function flatTriplesToCells(flat) {
        const cells = [];
        for (let i = 0; i < flat.length; i += 3) {
          const x = Number(flat[i]);
          const y = Number(flat[i + 1]);
          const count = Number(flat[i + 2]) || 0;
          const ix = Math.round((x / voxelSize) - 0.5);
          const iy = Math.round((y / voxelSize) - 0.5);
          cells.push({ x, y, count, ix, iy });
        }
        return cells;
      }

      const cleanCells = flatTriplesToCells(bev.clean || []);
      const ghostCells = flatTriplesToCells(bev.ghost || []);
      const rawCellMap = new Map();
      for (const cell of cleanCells) {
        rawCellMap.set(`${cell.ix},${cell.iy}`, { ...cell, cleanCount: cell.count, ghostCount: 0, totalCount: cell.count });
      }
      for (const cell of ghostCells) {
        const key = `${cell.ix},${cell.iy}`;
        const existing = rawCellMap.get(key);
        if (existing) {
          existing.ghostCount += cell.count;
          existing.totalCount = existing.cleanCount + existing.ghostCount;
        } else {
          rawCellMap.set(key, { ...cell, cleanCount: 0, ghostCount: cell.count, totalCount: cell.count });
        }
      }
      const rawCells = Array.from(rawCellMap.values());
      const cropRadiusWorld = voxelSize * 5.5;

      function centroidFromFlat(flat, fallback) {
        if (!flat || flat.length < 3) return fallback;
        let sx = 0;
        let sy = 0;
        let sz = 0;
        const n = flat.length / 3;
        for (let i = 0; i < flat.length; i += 3) {
          sx += flat[i];
          sy += flat[i + 1];
          sz += flat[i + 2];
        }
        return [sx / n, sy / n, sz / n];
      }

      function neighborhoodGhost(ix, iy) {
        let total = 0;
        for (const cell of ghostCells) {
          if (Math.abs(cell.ix - ix) <= 1 && Math.abs(cell.iy - iy) <= 1) total += cell.count;
        }
        return total;
      }

      function pickGhostHotspot() {
        if (!ghostCells.length) return null;
        return ghostCells.reduce((best, cell) => {
          if (!best) return cell;
          if (cell.count > best.count) return cell;
          return best;
        }, null);
      }

      function pickPreservedSpot() {
        if (!cleanCells.length) return null;
        const zeroGhost = cleanCells.filter((cell) => neighborhoodGhost(cell.ix, cell.iy) === 0);
        const candidates = zeroGhost.length ? zeroGhost : cleanCells;
        return candidates.reduce((best, cell) => {
          const penalty = neighborhoodGhost(cell.ix, cell.iy);
          const score = cell.count * 1000 - penalty;
          if (!best || score > best.score) return { ...cell, score };
          return best;
        }, null);
      }

      function cellsInCrop(cells, focus) {
        if (!focus) return [];
        return cells.filter((cell) => Math.abs(cell.x - focus.x) <= cropRadiusWorld && Math.abs(cell.y - focus.y) <= cropRadiusWorld);
      }

      function cropStats(focus) {
        const rawCrop = cellsInCrop(rawCells, focus);
        const cleanCrop = cellsInCrop(cleanCells, focus);
        const ghostCrop = cellsInCrop(ghostCells, focus);
        const rawCellCount = rawCrop.length;
        const cleanCellCount = cleanCrop.length;
        const ghostCellCount = ghostCrop.length;
        const overlapPct = Math.round(100 * cleanCellCount / Math.max(1, rawCellCount));
        return { rawCrop, cleanCrop, ghostCrop, rawCellCount, cleanCellCount, ghostCellCount, overlapPct };
      }

      const proofBounds = DEMO_DATA.limits || { xmin: -1, xmax: 1, ymin: -1, ymax: 1, zmin: -1, zmax: 1 };
      const proofFallback = {
        x: (Number(proofBounds.xmin) + Number(proofBounds.xmax)) * 0.5,
        y: (Number(proofBounds.ymin) + Number(proofBounds.ymax)) * 0.5,
        z: (Number(proofBounds.zmin) + Number(proofBounds.zmax)) * 0.5,
      };
      const proof = (() => {
        const ghostFocus2D = pickGhostHotspot() || { x: proofFallback.x, y: proofFallback.y, ix: 0, iy: 0, count: 0 };
        const preserveFocus2D = pickPreservedSpot() || ghostFocus2D;
        const ghostFrame = frames.reduce((best, frame) => (!best || frame.removed_points > best.removed_points ? frame : best), null);
        const ghostObjects = ghostFrame && ghostFrame.objects ? ghostFrame.objects : [];
        const ghostFocus3D = ghostObjects.length > 0
          ? ghostObjects[0].center
          : centroidFromFlat(ghostFrame ? ghostFrame.removed : null, [ghostFocus2D.x, ghostFocus2D.y, proofFallback.z]);
        return {
          ghostFocus2D,
          preserveFocus2D,
          ghostFocus3D,
          preserveFocus3D: [preserveFocus2D.x, preserveFocus2D.y, proofFallback.z],
          ghostCrop: cropStats(ghostFocus2D),
          preserveCrop: cropStats(preserveFocus2D),
          ghostFrame,
          ghostObjects,
        };
      })();

      const stage = document.getElementById("compare-stage");
      const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      stage.appendChild(renderer.domElement);

      const camera = new THREE.PerspectiveCamera(52, 1, 0.05, 4000);
      camera.up.set(0, 0, 1);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.zoomSpeed = 0.35;
      controls.enablePan = true;

      const limits = DEMO_DATA.limits;
      const dx = Math.max(1, limits.xmax - limits.xmin);
      const dy = Math.max(1, limits.ymax - limits.ymin);
      const dz = Math.max(1, limits.zmax - limits.zmin);
      const center = new THREE.Vector3(
        (limits.xmin + limits.xmax) * 0.5,
        (limits.ymin + limits.ymax) * 0.5,
        (limits.zmin + limits.zmax) * 0.5,
      );
      const extent = Math.max(dx, dy, dz, 10);

      const clampFrame = (index) => Math.max(0, Math.min(frames.length - 1, index));
      const initialFrame = Math.max(frames.length - 1, 0);

      const state = {
        frameIndex: initialFrame,
        playing: false,
        speed: 1,
        pointSize: 1.2,
        showCurrent: true,
        showTransient: true,
        showPath: true,
        showBoxes: hasBoxes,
      };
      let storyTimers = [];
      let storyAutoRan = false;

      function createScene(background, gridColorA, gridColorB) {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(background);
        const ambient = new THREE.AmbientLight(0xffffff, 0.9);
        const key = new THREE.DirectionalLight(0xffffff, 0.8);
        key.position.set(25, -20, 42);
        const grid = new THREE.GridHelper(extent * 1.3, 28, gridColorA, gridColorB);
        grid.rotation.x = Math.PI / 2;
        grid.position.z = limits.zmin;
        grid.material.transparent = true;
        grid.material.opacity = 0.22;
        scene.add(ambient, key, grid);
        return {
          scene,
          history: new THREE.Group(),
          current: new THREE.Group(),
          path: new THREE.Group(),
        };
      }

      const raw = createScene(0x0f172a, 0x334155, 0x1e293b);
      const clean = createScene(0x041712, 0x134e4a, 0x0f2f2b);
      raw.scene.add(raw.history, raw.current, raw.path);
      clean.scene.add(clean.history, clean.current, clean.path);

      function disposeObject(object) {
        if (!object) return;
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
          else object.material.dispose();
        }
      }

      function clearGroup(group) {
        while (group.children.length > 0) {
          const child = group.children.pop();
          if (child.children && child.children.length) {
            clearGroup(child);
          }
          disposeObject(child);
        }
      }

      function makePoints(flat, color, size, opacity, depthTest = true) {
        if (!flat || flat.length === 0) return null;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
        const material = new THREE.PointsMaterial({
          color,
          size,
          sizeAttenuation: true,
          transparent: opacity < 1,
          opacity,
          depthWrite: false,
          depthTest,
        });
        const points = new THREE.Points(geometry, material);
        return points;
      }

      function makeBoxes(objects, color) {
        if (!objects || objects.length === 0) return null;
        const group = new THREE.Group();
        for (const object of objects) {
          const geometry = new THREE.BoxGeometry(object.size[0], object.size[1], object.size[2]);
          const edges = new THREE.EdgesGeometry(geometry);
          const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
          const lines = new THREE.LineSegments(edges, material);
          lines.position.set(object.center[0], object.center[1], object.center[2]);
          lines.rotation.z = object.yaw || 0;
          group.add(lines);
        }
        return group;
      }

      function makePath(flat, color) {
        if (!flat || flat.length < 6) return null;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
        const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 });
        return new THREE.Line(geometry, material);
      }

      function makeMarker(centerPoint, color) {
        if (!centerPoint || centerPoint.length !== 3) return null;
        const geometry = new THREE.SphereGeometry(Math.max(extent * 0.012, 0.18), 16, 16);
        const material = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.18 });
        const marker = new THREE.Mesh(geometry, material);
        marker.position.set(centerPoint[0], centerPoint[1], centerPoint[2]);
        return marker;
      }

      const prefixCache = { input: [], kept: [], path: [] };

      function buildPrefix(index, key) {
        if (prefixCache[key][index]) return prefixCache[key][index];
        let total = 0;
        for (let i = 0; i <= index; i += 1) total += frames[i][key].length;
        const merged = new Float32Array(total);
        let offset = 0;
        for (let i = 0; i <= index; i += 1) {
          merged.set(frames[i][key], offset);
          offset += frames[i][key].length;
        }
        prefixCache[key][index] = merged;
        return merged;
      }

      function buildPathPrefix(index) {
        if (prefixCache.path[index]) return prefixCache.path[index];
        const slice = pathPoints.slice(0, index + 1);
        const merged = new Float32Array(slice.length * 3);
        let offset = 0;
        for (const point of slice) {
          merged[offset++] = point[0];
          merged[offset++] = point[1];
          merged[offset++] = point[2];
        }
        prefixCache.path[index] = merged;
        return merged;
      }

      function updateGroup(group, nodes) {
        clearGroup(group);
        for (const node of nodes) {
          if (node) group.add(node);
        }
      }

      function setStoryText(message) {
        document.getElementById("story-chip").textContent = message;
      }

      function stopStory() {
        for (const timer of storyTimers) clearTimeout(timer);
        storyTimers = [];
      }

      function updatePathGroups(frame) {
        const pathFlat = state.showPath ? buildPathPrefix(state.frameIndex) : null;
        const pathColor = 0xf59e0b;
        updateGroup(raw.path, [makePath(pathFlat, pathColor), makeMarker(frame.center, pathColor)]);
        updateGroup(clean.path, [makePath(pathFlat, pathColor), makeMarker(frame.center, pathColor)]);
      }

      function drawTimeline() {
        const canvas = document.getElementById("timeline-canvas");
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.max(260, Math.floor(rect.width));
        const height = Math.max(120, Math.floor(rect.height));
        if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
          canvas.width = Math.floor(width * dpr);
          canvas.height = Math.floor(height * dpr);
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#08111f";
        ctx.fillRect(0, 0, width, height);

        const margin = { left: 18, right: 14, top: 14, bottom: 22 };
        const chartW = width - margin.left - margin.right;
        const chartH = height - margin.top - margin.bottom;
        const ratios = frames.map((frame) => Number(frame.ghost_ratio_pct) || 0);
        const maxRatio = Math.max(1, ...ratios);
        const peakIndex = ratios.reduce((best, value, index) => (value > ratios[best] ? index : best), 0);

        ctx.strokeStyle = "rgba(255,255,255,0.10)";
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top + chartH);
        ctx.lineTo(margin.left + chartW, margin.top + chartH);
        ctx.strokeStyle = "rgba(148,163,184,0.35)";
        ctx.stroke();

        if (ratios.length > 0) {
          ctx.beginPath();
          ratios.forEach((ratio, index) => {
            const x = margin.left + (chartW * index) / Math.max(1, ratios.length - 1);
            const y = margin.top + chartH - (chartH * ratio) / maxRatio;
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          const gradient = ctx.createLinearGradient(0, margin.top, 0, margin.top + chartH);
          gradient.addColorStop(0, "rgba(248, 113, 113, 0.95)");
          gradient.addColorStop(1, "rgba(45, 212, 191, 0.9)");
          ctx.strokeStyle = gradient;
          ctx.lineWidth = 2.2;
          ctx.stroke();

          const peakX = margin.left + (chartW * peakIndex) / Math.max(1, ratios.length - 1);
          const peakY = margin.top + chartH - (chartH * ratios[peakIndex]) / maxRatio;
          ctx.fillStyle = "rgba(251,191,36,0.95)";
          ctx.beginPath();
          ctx.arc(peakX, peakY, 4, 0, Math.PI * 2);
          ctx.fill();

          const currentX = margin.left + (chartW * state.frameIndex) / Math.max(1, ratios.length - 1);
          ctx.strokeStyle = "rgba(96,165,250,0.95)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(currentX, margin.top);
          ctx.lineTo(currentX, margin.top + chartH);
          ctx.stroke();
        }

        ctx.fillStyle = "rgba(226,232,240,0.92)";
        ctx.font = "12px Trebuchet MS, sans-serif";
        ctx.fillText(`peak ${ratios[peakIndex]?.toFixed(1) || '0.0'}%`, margin.left, 14);
        ctx.fillText(`final ${ratios[ratios.length - 1]?.toFixed(1) || '0.0'}%`, Math.max(margin.left, width - 86), 14);
      }

      function drawGhostBEV() {
        const canvas = document.getElementById("ghost-bev");
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.max(260, Math.floor(rect.width));
        const height = Math.max(220, Math.floor(rect.height));
        if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
          canvas.width = Math.floor(width * dpr);
          canvas.height = Math.floor(height * dpr);
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#08111f";
        ctx.fillRect(0, 0, width, height);

        const margin = 16;
        const bounds = bev.bounds || { xmin: -1, xmax: 1, ymin: -1, ymax: 1 };
        const rangeX = Math.max(1e-6, bounds.xmax - bounds.xmin);
        const rangeY = Math.max(1e-6, bounds.ymax - bounds.ymin);
        const scaleX = (width - margin * 2) / rangeX;
        const scaleY = (height - margin * 2) / rangeY;
        const scale = Math.min(scaleX, scaleY);
        const cell = Math.max(2, Math.min(8, Math.floor(scale * (bev.voxel_size || 1) * 0.9)));

        function project(x, y) {
          const px = margin + (x - bounds.xmin) * scale;
          const py = height - margin - (y - bounds.ymin) * scale;
          return [px, py];
        }

        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

        const cleanFlat = bev.clean || [];
        const ghostFlat = bev.ghost || [];
        const maxClean = Math.max(1, Number(bev.max_clean_count) || 1);
        const maxGhost = Math.max(1, Number(bev.max_ghost_count) || 1);

        for (let i = 0; i < cleanFlat.length; i += 3) {
          const [px, py] = project(cleanFlat[i], cleanFlat[i + 1]);
          const alpha = 0.06 + 0.26 * Math.min(1, cleanFlat[i + 2] / maxClean);
          ctx.fillStyle = `rgba(45, 212, 191, ${alpha.toFixed(3)})`;
          ctx.fillRect(px - cell * 0.5, py - cell * 0.5, cell, cell);
        }

        for (let i = 0; i < ghostFlat.length; i += 3) {
          const [px, py] = project(ghostFlat[i], ghostFlat[i + 1]);
          const t = Math.min(1, ghostFlat[i + 2] / maxGhost);
          const r = Math.round(255 - 10 * t);
          const g = Math.round(110 + 70 * (1 - t));
          const b = Math.round(100 + 45 * t);
          const alpha = 0.34 + 0.58 * t;
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
          ctx.fillRect(px - cell * 0.6, py - cell * 0.6, cell * 1.2, cell * 1.2);
        }

        ctx.fillStyle = "rgba(226,232,240,0.92)";
        ctx.font = "12px Trebuchet MS, sans-serif";
        ctx.fillText("final accumulation footprint", 14, 18);
        ctx.fillStyle = "rgba(45,212,191,0.92)";
        ctx.fillRect(14, 28, 10, 10);
        ctx.fillStyle = "rgba(226,232,240,0.82)";
        ctx.fillText("cleaned", 30, 37);
        ctx.fillStyle = "rgba(255,120,130,0.95)";
        ctx.fillRect(92, 28, 10, 10);
        ctx.fillStyle = "rgba(226,232,240,0.82)";
        ctx.fillText("ghost / raw-only", 108, 37);
      }

      function drawEvidenceCanvas(canvasId, focus, mode, emphasis) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !focus) return;
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.max(180, Math.floor(rect.width || canvas.width));
        const height = Math.max(140, Math.floor(rect.height || canvas.height));
        if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
          canvas.width = Math.floor(width * dpr);
          canvas.height = Math.floor(height * dpr);
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#08111f";
        ctx.fillRect(0, 0, width, height);

        const bounds = {
          xmin: focus.x - cropRadiusWorld,
          xmax: focus.x + cropRadiusWorld,
          ymin: focus.y - cropRadiusWorld,
          ymax: focus.y + cropRadiusWorld,
        };
        const margin = 12;
        const rangeX = Math.max(1e-6, bounds.xmax - bounds.xmin);
        const rangeY = Math.max(1e-6, bounds.ymax - bounds.ymin);
        const scale = Math.min((width - margin * 2) / rangeX, (height - margin * 2) / rangeY);
        const cell = Math.max(4, Math.min(16, Math.floor(scale * voxelSize * 0.9)));
        const maxClean = Math.max(1, Number(bev.max_clean_count) || 1);
        const maxGhost = Math.max(1, Number(bev.max_ghost_count) || 1);

        function project(x, y) {
          return [
            margin + (x - bounds.xmin) * scale,
            height - margin - (y - bounds.ymin) * scale,
          ];
        }

        const rawCrop = cellsInCrop(rawCells, focus);
        const cleanCrop = cellsInCrop(cleanCells, focus);
        const ghostCrop = cellsInCrop(ghostCells, focus);

        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

        if (mode === "raw") {
          for (const cellData of rawCrop) {
            if (cellData.cleanCount > 0) {
              const [px, py] = project(cellData.x, cellData.y);
              const alpha = 0.08 + 0.28 * Math.min(1, cellData.cleanCount / maxClean);
              ctx.fillStyle = `rgba(45, 212, 191, ${alpha.toFixed(3)})`;
              ctx.fillRect(px - cell * 0.5, py - cell * 0.5, cell, cell);
            }
            if (cellData.ghostCount > 0) {
              const [px, py] = project(cellData.x, cellData.y);
              const t = Math.min(1, cellData.ghostCount / maxGhost);
              const alpha = 0.36 + 0.52 * t;
              ctx.fillStyle = `rgba(255, 110, 130, ${alpha.toFixed(3)})`;
              ctx.fillRect(px - cell * 0.62, py - cell * 0.62, cell * 1.24, cell * 1.24);
            }
          }
        } else {
          for (const cellData of cleanCrop) {
            const [px, py] = project(cellData.x, cellData.y);
            const alpha = 0.1 + 0.34 * Math.min(1, cellData.count / maxClean);
            ctx.fillStyle = `rgba(45, 212, 191, ${alpha.toFixed(3)})`;
            ctx.fillRect(px - cell * 0.5, py - cell * 0.5, cell, cell);
          }
        }

        if (emphasis === "ghost") {
          for (const cellData of ghostCrop) {
            const [px, py] = project(cellData.x, cellData.y);
            ctx.strokeStyle = "rgba(255, 236, 153, 0.88)";
            ctx.lineWidth = 1.2;
            ctx.strokeRect(px - cell * 0.8, py - cell * 0.8, cell * 1.6, cell * 1.6);
          }
          for (const object of proof.ghostObjects || []) {
            const ox0 = object.center[0] - object.size[0] * 0.5;
            const ox1 = object.center[0] + object.size[0] * 0.5;
            const oy0 = object.center[1] - object.size[1] * 0.5;
            const oy1 = object.center[1] + object.size[1] * 0.5;
            if (ox1 < bounds.xmin || ox0 > bounds.xmax || oy1 < bounds.ymin || oy0 > bounds.ymax) continue;
            const [px0, py0] = project(ox0, oy0);
            const [px1, py1] = project(ox1, oy1);
            ctx.strokeStyle = "rgba(255, 204, 128, 0.95)";
            ctx.lineWidth = 1.6;
            ctx.strokeRect(px0, py1, Math.max(2, px1 - px0), Math.max(2, py0 - py1));
          }
        }

        const [cx, cy] = project(focus.x, focus.y);
        ctx.strokeStyle = emphasis === "ghost" ? "rgba(255,236,153,0.95)" : "rgba(191,219,254,0.92)";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(6, cell * 0.75), 0, Math.PI * 2);
        ctx.stroke();
      }

      function drawEvidencePanels() {
        drawEvidenceCanvas("hotspot-raw", proof.ghostFocus2D, "raw", "ghost");
        drawEvidenceCanvas("hotspot-clean", proof.ghostFocus2D, "clean", "ghost");
        drawEvidenceCanvas("preserve-raw", proof.preserveFocus2D, "raw", "preserve");
        drawEvidenceCanvas("preserve-clean", proof.preserveFocus2D, "clean", "preserve");
      }

      function updateFrame(index) {
        if (frames.length === 0) return;
        state.frameIndex = clampFrame(index);
        const frame = frames[state.frameIndex];

        const rawAccum = buildPrefix(state.frameIndex, "input");
        const cleanAccum = buildPrefix(state.frameIndex, "kept");

        updateGroup(raw.history, [
          makePoints(rawAccum, 0x94a3b8, state.pointSize * 0.92, 0.34, true),
        ]);
        updateGroup(raw.current, [
          state.showCurrent ? makePoints(frame.input, 0xe2e8f0, state.pointSize * 1.05, 0.24, true) : null,
          state.showTransient ? makePoints(frame.removed, 0xff4d6d, state.pointSize * 2.45, 0.98, false) : null,
          state.showBoxes ? makeBoxes(frame.objects, 0xf97316) : null,
        ]);
        updateGroup(clean.history, [
          makePoints(cleanAccum, 0x14b8a6, state.pointSize * 1.02, 0.9, true),
        ]);
        updateGroup(clean.current, [
          state.showCurrent ? makePoints(frame.kept, 0x99f6e4, state.pointSize * 1.35, 1.0, false) : null,
          state.showBoxes ? makeBoxes(frame.objects, 0xfbbf24) : null,
        ]);
        updatePathGroups(frame);

        document.getElementById("frame-slider").value = String(state.frameIndex);
        document.getElementById("frame-label").textContent = `frame ${state.frameIndex + 1} / ${frames.length}`;
        document.getElementById("frame-source").textContent = frame.source;
        document.getElementById("meta-input").textContent = frame.input_points.toLocaleString();
        document.getElementById("meta-kept").textContent = Math.round(frame.render_kept_points || frame.kept_points).toLocaleString();
        document.getElementById("meta-ghost").textContent = frame.ghost_voxels.toLocaleString();
        document.getElementById("meta-ratio").textContent = `${frame.ghost_ratio_pct.toFixed(1)}%`;

        document.getElementById("footer-raw").textContent = frame.raw_voxels.toLocaleString();
        document.getElementById("footer-clean").textContent = frame.clean_voxels.toLocaleString();
        document.getElementById("footer-ghost").textContent = frame.ghost_voxels.toLocaleString();
        document.getElementById("hud-left").textContent = `raw: ${frame.raw_voxels.toLocaleString()} voxels`;
        document.getElementById("hud-right").textContent = `cleaned: ${frame.clean_voxels.toLocaleString()} voxels`;
        drawTimeline();
      }

      function moveCamera(targetPoint, sceneExtent) {
        const fov = camera.fov * Math.PI / 180;
        const dist = sceneExtent / Math.tan(fov * 0.5);
        camera.near = Math.max(sceneExtent * 1e-4, 0.05);
        camera.far = Math.max(dist * 6, extent * 10, 6000);
        camera.position.set(targetPoint.x + dist * 0.88, targetPoint.y - dist * 0.78, targetPoint.z + dist * 0.76);
        controls.target.copy(targetPoint);
        camera.updateProjectionMatrix();
        controls.update();
      }

      function fitView() {
        moveCamera(center, extent * 1.15);
      }

      function focusProof(point) {
        moveCamera(new THREE.Vector3(point[0], point[1], point[2]), Math.max(extent * 0.28, voxelSize * 18, 10));
      }

      function resizeRenderer() {
        const width = stage.clientWidth;
        const height = stage.clientHeight;
        renderer.setSize(width, height, false);
      }

      function renderSplit() {
        resizeRenderer();
        const width = stage.clientWidth;
        const height = stage.clientHeight;
        const leftWidth = Math.floor(width / 2);
        const rightWidth = width - leftWidth;
        const aspect = Math.max(leftWidth, 1) / Math.max(height, 1);
        camera.aspect = aspect;
        camera.updateProjectionMatrix();

        renderer.setScissorTest(true);
        renderer.setViewport(0, 0, leftWidth, height);
        renderer.setScissor(0, 0, leftWidth, height);
        renderer.render(raw.scene, camera);

        renderer.setViewport(leftWidth, 0, rightWidth, height);
        renderer.setScissor(leftWidth, 0, rightWidth, height);
        renderer.render(clean.scene, camera);
        renderer.setScissorTest(false);
      }

      let playbackTimer = null;
      function stopPlayback() {
        state.playing = false;
        document.getElementById("play-toggle").textContent = "Replay build-up";
        if (playbackTimer) {
          clearInterval(playbackTimer);
          playbackTimer = null;
        }
      }

      function storyMessages() {
        const ghost = DEMO_DATA.meta.final_ghost_voxels.toLocaleString();
        const stable = DEMO_DATA.meta.final_clean_voxels.toLocaleString();
        return {
          overview: `Final map impact: raw leaves ${ghost} ghost voxels while cleaned keeps ${stable} stable voxels.`,
          hotspot: `Ghost hotspot: this crop is where raw-only occupancy persists if transient clutter is accumulated.`,
          stable: `Static preserved: this crop stays dense after cleaning, so the preview is not just erasing structure.`,
          outro: `Story complete. Replay build-up or scrub frames to inspect the sampled box-removal preview.`
        };
      }

      function runStoryMode() {
        stopPlayback();
        stopStory();
        updateFrame(initialFrame);
        fitView();
        renderSplit();
        const messages = storyMessages();
        setStoryText(messages.overview);
        storyTimers.push(setTimeout(() => {
          focusProof(proof.ghostFocus3D);
          renderSplit();
          setStoryText(messages.hotspot);
        }, 1600));
        storyTimers.push(setTimeout(() => {
          focusProof(proof.preserveFocus3D);
          renderSplit();
          setStoryText(messages.stable);
        }, 3600));
        storyTimers.push(setTimeout(() => {
          fitView();
          renderSplit();
          setStoryText(messages.outro);
        }, 5600));
      }

      function restartPlayback() {
        if (playbackTimer) {
          clearInterval(playbackTimer);
          playbackTimer = null;
        }
        if (!state.playing || frames.length <= 1) return;
        const fps = Math.max(0.5, Number(DEMO_DATA.meta.default_fps) * state.speed);
        playbackTimer = setInterval(() => {
          if (state.frameIndex >= frames.length - 1) {
            stopPlayback();
            return;
          }
          updateFrame(state.frameIndex + 1);
          renderSplit();
        }, 1000 / fps);
      }

      document.getElementById("play-toggle").addEventListener("click", () => {
        stopStory();
        if (state.playing) {
          stopPlayback();
          setStoryText(storyMessages().outro);
          return;
        }
        if (state.frameIndex >= frames.length - 1) {
          updateFrame(0);
        }
        state.playing = true;
        document.getElementById("play-toggle").textContent = "Pause";
        setStoryText("Build-up replay: watch ghost occupancy grow on the raw side while the cleaned side stays tighter.");
        restartPlayback();
        renderSplit();
      });
      document.getElementById("story-mode").addEventListener("click", () => {
        runStoryMode();
      });
      document.getElementById("fit-view").addEventListener("click", () => {
        stopStory();
        fitView();
        setStoryText(storyMessages().overview);
        renderSplit();
      });
      document.getElementById("focus-ghost").addEventListener("click", () => {
        stopStory();
        focusProof(proof.ghostFocus3D);
        setStoryText(storyMessages().hotspot);
        renderSplit();
      });
      document.getElementById("focus-stable").addEventListener("click", () => {
        stopStory();
        focusProof(proof.preserveFocus3D);
        setStoryText(storyMessages().stable);
        renderSplit();
      });
      document.getElementById("download-shot").addEventListener("click", () => {
        renderSplit();
        const link = document.createElement("a");
        link.href = renderer.domElement.toDataURL("image/png");
        link.download = `accumulation_split_${String(state.frameIndex).padStart(3, "0")}.png`;
        link.click();
      });
      document.getElementById("frame-slider").addEventListener("input", (event) => {
        stopPlayback();
        stopStory();
        updateFrame(Number(event.target.value));
        setStoryText(`Manual scrub: frame ${Number(event.target.value) + 1} / ${frames.length}`);
        renderSplit();
      });
      document.getElementById("speed-select").addEventListener("change", (event) => {
        state.speed = Number(event.target.value);
        stopStory();
        restartPlayback();
      });
      document.getElementById("point-size").addEventListener("input", (event) => {
        state.pointSize = Number(event.target.value);
        document.getElementById("point-size-value").textContent = `${state.pointSize.toFixed(1)} px`;
        stopStory();
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-current").addEventListener("change", (event) => {
        state.showCurrent = event.target.checked;
        stopStory();
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-transient").addEventListener("change", (event) => {
        state.showTransient = event.target.checked;
        stopStory();
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-path").addEventListener("change", (event) => {
        state.showPath = event.target.checked;
        stopStory();
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-boxes").addEventListener("change", (event) => {
        state.showBoxes = event.target.checked;
        stopStory();
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-boxes-wrap").hidden = !hasBoxes;

      document.getElementById("stat-frame-count").textContent = String(frames.length);
      document.getElementById("stat-ghost-ratio").textContent = `${DEMO_DATA.meta.final_ghost_ratio_pct.toFixed(1)}%`;
      document.getElementById("stat-ghost-voxels").textContent = DEMO_DATA.meta.final_ghost_voxels.toLocaleString();
      document.getElementById("stat-stable-voxels").textContent = DEMO_DATA.meta.final_clean_voxels.toLocaleString();
      const sourceSuffix = hasBoxes
        ? (usesDerivedBoxes
            ? " The checked-in sequence has no shipped detections, so this page bootstraps auto transient boxes from temporal outliers and uses those boxes for a sampled box-removal preview."
            : " This page is rendering the sampled box-removal preview driven by the provided detections.")
        : "";
      document.getElementById("source-note").textContent = `${DEMO_DATA.meta.source_note}${sourceSuffix}`;
      document.getElementById("frame-slider").max = String(Math.max(0, frames.length - 1));
      document.getElementById("frame-slider").value = String(initialFrame);
      document.getElementById("hotspot-metric").textContent = `${proof.ghostCrop.ghostCellCount.toLocaleString()} raw-only cells in crop`;
      document.getElementById("preserve-metric").textContent = `${proof.preserveCrop.overlapPct}% footprint overlap`;
      document.getElementById("hotspot-copy").textContent = hasBoxes
        ? `largest residual contamination pocket: ${proof.ghostCrop.ghostCellCount.toLocaleString()} raw-only cells survive here, and ${proof.ghostObjects.length} box candidates summarize the current-frame clutter that the box-removal preview crops from the sampled sequence.`
        : `largest residual contamination pocket: ${proof.ghostCrop.ghostCellCount.toLocaleString()} raw-only cells survive inside this crop if every observation is accumulated.`;
      document.getElementById("preserve-copy").textContent = `this dense static crop keeps ${proof.preserveCrop.overlapPct}% of its footprint after cleaning, with ${proof.preserveCrop.ghostCellCount.toLocaleString()} raw-only cells leaking into the same area.`;

      window.addEventListener("resize", () => {
        drawTimeline();
        drawGhostBEV();
        drawEvidencePanels();
        renderSplit();
      });
      controls.addEventListener("change", renderSplit);

      fitView();
      drawTimeline();
      drawGhostBEV();
      drawEvidencePanels();
      updateFrame(initialFrame);
      setStoryText(storyMessages().overview);
      renderSplit();
      if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        storyTimers.push(setTimeout(() => {
          if (!storyAutoRan) {
            storyAutoRan = true;
            runStoryMode();
          }
        }, 900));
      }

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
      }
      animate();
    </script>
  </body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True, help="glob for frame clouds, e.g. /path/to/graph/*/cloud.pcd")
    parser.add_argument("--input-objects", help="optional JSON file. Either one global box payload or a frame-name -> payload map")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-render-points", type=int, default=DEFAULT_MAX_RENDER_POINTS)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--box-margin", type=float, nargs=3, default=DEFAULT_BOX_MARGIN)
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    parser.add_argument("--title", default="Raw vs cleaned accumulation")
    parser.add_argument("--output-scene", type=Path)
    parser.add_argument("--output-html", type=Path, default=ROOT / "demo" / "index_3d_sequence_standalone.html")
    return parser.parse_args()


def _sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("points must have shape (N,3+)")
    xyz = xyz[:, :3]
    if max_points <= 0 or len(xyz) <= max_points:
        return xyz.copy()
    indices = np.sort(rng.choice(len(xyz), size=max_points, replace=False))
    return xyz[indices]


def _round_points(points: np.ndarray) -> list[float]:
    if points.size == 0:
        return []
    return np.round(points.astype(np.float32), 3).reshape(-1).tolist()


def _round_vec(vec: np.ndarray, decimals: int = 3) -> list[float]:
    return [round(float(v), decimals) for v in np.asarray(vec, dtype=np.float64).reshape(-1)]


def _new_limits() -> dict[str, float]:
    inf = float("inf")
    return {"xmin": inf, "xmax": -inf, "ymin": inf, "ymax": -inf, "zmin": inf, "zmax": -inf}


def _update_limits(limits: dict[str, float], points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    limits["xmin"] = min(limits["xmin"], float(mins[0]))
    limits["xmax"] = max(limits["xmax"], float(maxs[0]))
    limits["ymin"] = min(limits["ymin"], float(mins[1]))
    limits["ymax"] = max(limits["ymax"], float(maxs[1]))
    limits["zmin"] = min(limits["zmin"], float(mins[2]))
    limits["zmax"] = max(limits["zmax"], float(maxs[2]))


def _finalize_limits(limits: dict[str, float]) -> dict[str, float]:
    if not np.isfinite(list(limits.values())).all():
        return {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0, "zmin": -1.0, "zmax": 1.0}
    return {key: round(value, 3) for key, value in limits.items()}


def _is_global_box_payload(raw: Any) -> bool:
    if isinstance(raw, list):
        return True
    if not isinstance(raw, dict):
        return False
    keys = set(raw.keys())
    global_keys = {
        "objects",
        "detections",
        "boxes",
        "center",
        "x",
        "position",
        "pose",
        "size",
        "dimensions",
        "extent",
        "bbox",
        "box",
        "length",
        "width",
        "height",
        "l",
        "w",
        "h",
        "shape",
    }
    return bool(keys & global_keys)


def _load_boxes_spec(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_boxes(spec: Any, frame_path: Path) -> list[Any]:
    if spec is None:
        return []
    if _is_global_box_payload(spec):
        return parse_boxes_payload(spec, skip_invalid=True)
    if not isinstance(spec, dict):
        return []
    candidates = [
        frame_path.parent.name,
        frame_path.name,
        frame_path.stem,
        f"{frame_path.parent.name}/{frame_path.name}",
        frame_path.as_posix(),
    ]
    for key in candidates:
        if key in spec:
            return parse_boxes_payload(spec[key], skip_invalid=True)
    return []


def _voxel_set(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    if points.size == 0:
        return set()
    voxels = np.floor(np.asarray(points, dtype=np.float64)[:, :3] / voxel_size).astype(np.int32)
    if len(voxels) == 0:
        return set()
    unique = np.unique(voxels, axis=0)
    return {tuple(int(v) for v in row) for row in unique}


def _project_bev(voxel_set: set[tuple[int, int, int]], voxel_size: float) -> tuple[list[float], int, dict[str, float]]:
    counts: dict[tuple[int, int], int] = {}
    for vx, vy, vz in voxel_set:
        key = (vx, vy)
        counts[key] = counts.get(key, 0) + 1
    flat: list[float] = []
    if not counts:
        return flat, 0, {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0}
    xs = []
    ys = []
    max_count = 0
    for (vx, vy), count in sorted(counts.items()):
        x = (vx + 0.5) * voxel_size
        y = (vy + 0.5) * voxel_size
        flat.extend([round(x, 3), round(y, 3), int(count)])
        xs.append(x)
        ys.append(y)
        max_count = max(max_count, count)
    bounds = {
        "xmin": round(min(xs), 3),
        "xmax": round(max(xs), 3),
        "ymin": round(min(ys), 3),
        "ymax": round(max(ys), 3),
    }
    return flat, max_count, bounds


def _merge_bounds(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {
        "xmin": min(a["xmin"], b["xmin"]),
        "xmax": max(a["xmax"], b["xmax"]),
        "ymin": min(a["ymin"], b["ymin"]),
        "ymax": max(a["ymax"], b["ymax"]),
    }


def main() -> None:
    args = parse_args()
    frame_paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not frame_paths:
        raise SystemExit(f"no frames matched: {args.input_glob}")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.frame_count <= 0:
        raise SystemExit("--frame-count must be positive")

    selected = frame_paths[args.start_index : args.start_index + args.frame_count * args.stride : args.stride]
    if not selected:
        raise SystemExit("selection is empty; adjust --start-index / --frame-count / --stride")

    boxes_spec = _load_boxes_spec(Path(args.input_objects)) if args.input_objects else None
    mode = "boxes" if boxes_spec is not None else "temporal_consistency"
    temporal_filter = None if mode == "boxes" else TemporalConsistencyFilter(
        voxel_size=args.voxel_size,
        window_size=args.window_size,
        min_hits=args.min_hits,
    )
    rng = np.random.default_rng(args.seed)

    origin: np.ndarray | None = None
    limits = _new_limits()
    path_points: list[list[float]] = []
    frames: list[dict[str, Any]] = []

    raw_voxels_accum: set[tuple[int, int, int]] = set()
    clean_voxels_accum: set[tuple[int, int, int]] = set()

    total_input_points = 0
    total_kept_points = 0
    total_removed_points = 0

    for frame_index, path in enumerate(selected):
        points = load_points(path, fmt="auto")
        if points.size == 0:
            continue

        center = np.asarray(points[:, :3].mean(axis=0), dtype=np.float64)
        if origin is None:
            origin = center.copy()

        shifted_full = np.asarray(points[:, :3], dtype=np.float64) - origin
        boxes = _resolve_boxes(boxes_spec, path) if boxes_spec is not None else []
        shifted_boxes = []
        for box in boxes:
            shifted_center = np.asarray(box.center, dtype=np.float64) - origin
            shifted_boxes.append(type(box)(center=shifted_center, size=box.size, yaw=box.yaw, label=box.label))

        if mode == "boxes":
            if shifted_boxes:
                kept_full, keep_mask = remove_points_in_boxes(shifted_full, shifted_boxes, margin=args.box_margin)
            else:
                keep_mask = np.ones(shifted_full.shape[0], dtype=bool)
                kept_full = shifted_full
        else:
            assert temporal_filter is not None
            kept_full, keep_mask = temporal_filter.filter(shifted_full)

        removed_full = shifted_full[~keep_mask]
        input_sample = _sample_points(shifted_full, args.max_render_points, rng)
        kept_sample = _sample_points(kept_full, args.max_render_points, rng)
        removed_sample = _sample_points(removed_full, args.max_render_points, rng)

        raw_voxels_accum |= _voxel_set(shifted_full, args.voxel_size)
        clean_voxels_accum |= _voxel_set(kept_full, args.voxel_size)
        ghost_voxels_accum = raw_voxels_accum - clean_voxels_accum

        frame_center = input_sample.mean(axis=0) if len(input_sample) else center - origin
        _update_limits(limits, input_sample)
        _update_limits(limits, kept_sample)
        _update_limits(limits, removed_sample)
        _update_limits(limits, frame_center.reshape(1, 3))
        path_points.append(_round_vec(frame_center))

        frame_objects = []
        for box in shifted_boxes:
            frame_objects.append(
                {
                    "center": _round_vec(box.center),
                    "size": _round_vec(box.size),
                    "yaw": round(float(box.yaw), 6),
                    "label": box.label or "object",
                }
            )

        total_input_points += int(len(shifted_full))
        total_kept_points += int(len(kept_full))
        total_removed_points += int(len(removed_full))

        raw_voxel_count = len(raw_voxels_accum)
        clean_voxel_count = len(clean_voxels_accum)
        ghost_voxel_count = len(ghost_voxels_accum)

        frames.append(
            {
                "index": frame_index,
                "name": path.parent.name,
                "source": f"{path.parent.name}/{path.name}",
                "input_points": int(len(shifted_full)),
                "kept_points": int(len(kept_full)),
                "removed_points": int(len(removed_full)),
                "render_input_points": int(len(input_sample)),
                "render_kept_points": int(len(kept_sample)),
                "render_removed_points": int(len(removed_sample)),
                "raw_voxels": raw_voxel_count,
                "clean_voxels": clean_voxel_count,
                "ghost_voxels": ghost_voxel_count,
                "ghost_ratio_pct": round(100.0 * ghost_voxel_count / max(1, raw_voxel_count), 2),
                "input": _round_points(input_sample),
                "kept": _round_points(kept_sample),
                "removed": _round_points(removed_sample),
                "center": _round_vec(frame_center),
                "objects": frame_objects,
            }
        )

    if origin is None or not frames:
        raise SystemExit("no valid frames were generated")

    final_ghost_voxels = raw_voxels_accum - clean_voxels_accum
    clean_bev, max_clean_count, clean_bounds = _project_bev(clean_voxels_accum, args.voxel_size)
    ghost_bev, max_ghost_count, ghost_bounds = _project_bev(final_ghost_voxels, args.voxel_size)
    bev_bounds = _merge_bounds(clean_bounds, ghost_bounds)

    if mode == "boxes":
        source_note = (
            "checked-in sequence demo uses per-frame boxes for cleaned accumulation. "
            "raw keeps all observations, cleaned keeps points after box removal."
        )
    else:
        source_note = (
            f"checked-in sequence demo uses a real local multi-frame sequence and temporal consistency ({args.voxel_size:.2f}m / {args.window_size} / {args.min_hits}) "
            "for the cleaned side. raw keeps everything that was observed; cleaned keeps only persistent structure."
        )

    scene = {
        "meta": {
            "title": args.title,
            "frame_count": len(frames),
            "default_fps": round(float(args.fps), 2),
            "max_render_points": int(args.max_render_points),
            "origin": _round_vec(origin),
            "mode": mode,
            "total_input_points": total_input_points,
            "total_kept_points": total_kept_points,
            "total_removed_points": total_removed_points,
            "final_raw_voxels": len(raw_voxels_accum),
            "final_clean_voxels": len(clean_voxels_accum),
            "final_ghost_voxels": len(final_ghost_voxels),
            "final_ghost_ratio_pct": round(100.0 * len(final_ghost_voxels) / max(1, len(raw_voxels_accum)), 2),
            "source_note": source_note,
        },
        "limits": _finalize_limits(limits),
        "path": path_points,
        "bev": {
            "voxel_size": round(float(args.voxel_size), 3),
            "bounds": bev_bounds,
            "clean": clean_bev,
            "ghost": ghost_bev,
            "max_clean_count": max_clean_count,
            "max_ghost_count": max_ghost_count,
        },
        "frames": frames,
    }

    if args.output_scene:
        args.output_scene.write_text(json.dumps(scene, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    html = HTML_TEMPLATE.replace("__DEMO_DATA__", json.dumps(scene, ensure_ascii=False, separators=(",", ":")))
    args.output_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
