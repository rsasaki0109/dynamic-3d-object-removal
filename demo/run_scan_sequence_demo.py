#!/usr/bin/env python3
"""Build a self-contained split-view accumulation demo from multiple point-cloud frames."""

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
      button, select, input[type="range"] {
        font: inherit;
      }
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
      .compare-card {
        position: relative;
        padding: 16px;
        display: grid;
        gap: 12px;
        min-height: min(88vh, 940px);
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
        .compare-head, .compare-footer, .stats { grid-template-columns: 1fr; }
        #compare-stage { min-height: 64vh; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <aside class="panel">
        <div class="section">
          <span class="eyebrow">Impact Demo</span>
          <h1>動的点を見せるのではなく、地図が締まることを見せる</h1>
          <p class="lead">
            左は観測した点をそのまま積算、右は各フレームから安定点だけを積算します。比較の主役は current frame ではなく、時間方向に積んだ後の差です。
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
          <h2>Overview</h2>
          <div class="stats">
            <div class="stat"><strong id="stat-frame-count">0</strong><span>frames</span></div>
            <div class="stat"><strong id="stat-fps">0</strong><span>default fps</span></div>
            <div class="stat"><strong id="stat-max-render">0</strong><span>max rendered / frame</span></div>
            <div class="stat"><strong id="stat-mode">-</strong><span>cleaning mode</span></div>
          </div>
        </div>

        <div class="section controls">
          <h2>Playback</h2>
          <div class="buttons">
            <button class="primary" id="play-toggle">Play</button>
            <button id="fit-view">Fit view</button>
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
          <div class="small-meta"><span>original points</span><span id="meta-input">0</span></div>
          <div class="small-meta"><span>cleaned points</span><span id="meta-kept">0</span></div>
          <div class="small-meta"><span>transient / removed</span><span id="meta-removed">0</span></div>
          <div class="small-meta"><span>clean ratio</span><span id="meta-ratio">0%</span></div>
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
          <div class="canvas-hud">
            <div class="canvas-chip" id="hud-left">raw</div>
            <div class="canvas-chip" id="hud-right">cleaned</div>
          </div>
        </div>

        <div class="compare-footer">
          <div class="footer-card"><strong>Accumulated raw</strong><span id="footer-raw">0</span></div>
          <div class="footer-card"><strong>Accumulated cleaned</strong><span id="footer-clean">0</span></div>
          <div class="footer-card"><strong>Current transient / removed</strong><span id="footer-removed">0</span></div>
        </div>

        <div class="note-box">
          drag: orbit / wheel: zoom / right drag: pan. checked-in continuous demo is generated from a real local multi-frame sequence and uses temporal consistency for the cleaned side unless per-frame object boxes are supplied.
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
      const frames = (DEMO_DATA.frames || []).map((frame) => ({
        ...frame,
        input: new Float32Array(frame.input || []),
        kept: new Float32Array(frame.kept || []),
        removed: new Float32Array(frame.removed || []),
      }));
      const pathPoints = (DEMO_DATA.path || []).map((point) => point.map(Number));
      const hasBoxes = frames.some((frame) => Array.isArray(frame.objects) && frame.objects.length > 0);

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

      const state = {
        frameIndex: 0,
        playing: false,
        speed: 1,
        pointSize: 1.2,
        showCurrent: true,
        showTransient: true,
        showPath: true,
        showBoxes: hasBoxes,
      };

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
        return new THREE.Points(geometry, material);
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

      function updatePathGroups(frame) {
        const pathFlat = state.showPath ? buildPathPrefix(state.frameIndex) : null;
        const pathColor = 0xf59e0b;
        updateGroup(raw.path, [makePath(pathFlat, pathColor), makeMarker(frame.center, pathColor)]);
        updateGroup(clean.path, [makePath(pathFlat, pathColor), makeMarker(frame.center, pathColor)]);
      }

      function updateFrame(index) {
        if (frames.length === 0) return;
        state.frameIndex = ((index % frames.length) + frames.length) % frames.length;
        const frame = frames[state.frameIndex];

        const rawAccum = buildPrefix(state.frameIndex, "input");
        const cleanAccum = buildPrefix(state.frameIndex, "kept");

        const rawNodes = [
          makePoints(rawAccum, 0x94a3b8, state.pointSize * 0.92, 0.34, true),
          state.showCurrent ? makePoints(frame.input, 0xe2e8f0, state.pointSize * 1.05, 0.26, true) : null,
          state.showTransient ? makePoints(frame.removed, 0xff4d6d, state.pointSize * 2.45, 0.98, false) : null,
          state.showBoxes ? makeBoxes(frame.objects, 0xf97316) : null,
        ];
        const cleanNodes = [
          makePoints(cleanAccum, 0x14b8a6, state.pointSize * 1.02, 0.9, true),
          state.showCurrent ? makePoints(frame.kept, 0x99f6e4, state.pointSize * 1.35, 1.0, false) : null,
          state.showBoxes ? makeBoxes(frame.objects, 0xfbbf24) : null,
        ];

        updateGroup(raw.history, rawNodes.slice(0, 1));
        updateGroup(raw.current, rawNodes.slice(1));
        updateGroup(clean.history, cleanNodes.slice(0, 1));
        updateGroup(clean.current, cleanNodes.slice(1));
        updatePathGroups(frame);

        document.getElementById("frame-slider").value = String(state.frameIndex);
        document.getElementById("frame-label").textContent = `frame ${state.frameIndex + 1} / ${frames.length}`;
        document.getElementById("frame-source").textContent = frame.source;
        document.getElementById("meta-input").textContent = frame.input_points.toLocaleString();
        document.getElementById("meta-kept").textContent = frame.kept_points.toLocaleString();
        document.getElementById("meta-removed").textContent = frame.removed_points.toLocaleString();
        document.getElementById("meta-ratio").textContent = `${((frame.kept_points / Math.max(1, frame.input_points)) * 100).toFixed(1)}%`;
        document.getElementById("footer-raw").textContent = `${(rawAccum.length / 3).toLocaleString()} pts`;
        document.getElementById("footer-clean").textContent = `${(cleanAccum.length / 3).toLocaleString()} pts`;
        document.getElementById("footer-removed").textContent = `${frame.removed_points.toLocaleString()} pts`;
        document.getElementById("hud-left").textContent = `raw: ${(rawAccum.length / 3).toLocaleString()} accumulated`; 
        document.getElementById("hud-right").textContent = `cleaned: ${(cleanAccum.length / 3).toLocaleString()} accumulated`;
      }

      function fitView() {
        const fov = camera.fov * Math.PI / 180;
        const dist = (extent * 1.15) / Math.tan(fov * 0.5);
        camera.near = Math.max(extent * 1e-4, 0.05);
        camera.far = Math.max(dist * 6, extent * 10, 6000);
        camera.position.set(center.x + dist * 0.88, center.y - dist * 0.78, center.z + dist * 0.76);
        controls.target.copy(center);
        camera.updateProjectionMatrix();
        controls.update();
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
      function restartPlayback() {
        if (playbackTimer) {
          clearInterval(playbackTimer);
          playbackTimer = null;
        }
        if (!state.playing || frames.length <= 1) return;
        const fps = Math.max(0.5, Number(DEMO_DATA.meta.default_fps) * state.speed);
        playbackTimer = setInterval(() => {
          updateFrame(state.frameIndex + 1);
          renderSplit();
        }, 1000 / fps);
      }

      document.getElementById("play-toggle").addEventListener("click", () => {
        state.playing = !state.playing;
        document.getElementById("play-toggle").textContent = state.playing ? "Pause" : "Play";
        restartPlayback();
      });
      document.getElementById("fit-view").addEventListener("click", () => {
        fitView();
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
        updateFrame(Number(event.target.value));
        renderSplit();
      });
      document.getElementById("speed-select").addEventListener("change", (event) => {
        state.speed = Number(event.target.value);
        restartPlayback();
      });
      document.getElementById("point-size").addEventListener("input", (event) => {
        state.pointSize = Number(event.target.value);
        document.getElementById("point-size-value").textContent = `${state.pointSize.toFixed(1)} px`;
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-current").addEventListener("change", (event) => {
        state.showCurrent = event.target.checked;
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-transient").addEventListener("change", (event) => {
        state.showTransient = event.target.checked;
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-path").addEventListener("change", (event) => {
        state.showPath = event.target.checked;
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-boxes").addEventListener("change", (event) => {
        state.showBoxes = event.target.checked;
        updateFrame(state.frameIndex);
        renderSplit();
      });
      document.getElementById("toggle-boxes-wrap").hidden = !hasBoxes;

      document.getElementById("stat-frame-count").textContent = String(frames.length);
      document.getElementById("stat-fps").textContent = String(DEMO_DATA.meta.default_fps);
      document.getElementById("stat-max-render").textContent = String(DEMO_DATA.meta.max_render_points);
      document.getElementById("stat-mode").textContent = DEMO_DATA.meta.mode_label;
      document.getElementById("source-note").textContent = DEMO_DATA.meta.source_note;
      document.getElementById("frame-slider").max = String(Math.max(0, frames.length - 1));

      window.addEventListener("resize", renderSplit);
      controls.addEventListener("change", renderSplit);

      fitView();
      updateFrame(0);
      renderSplit();

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

    total_input_points = 0
    total_kept_points = 0
    total_removed_points = 0

    for frame_index, path in enumerate(selected):
        points = load_points(path, fmt="auto")
        boxes = _resolve_boxes(boxes_spec, path) if boxes_spec is not None else []

        if mode == "boxes":
            if boxes:
                kept_full, keep_mask = remove_points_in_boxes(points, boxes, margin=args.box_margin)
            else:
                keep_mask = np.ones(points.shape[0], dtype=bool)
                kept_full = points
        else:
            assert temporal_filter is not None
            kept_full, keep_mask = temporal_filter.filter(points)

        removed_full = points[~keep_mask]
        input_sample = _sample_points(points, args.max_render_points, rng)
        kept_sample = _sample_points(kept_full, args.max_render_points, rng)
        removed_sample = _sample_points(removed_full, args.max_render_points, rng)

        center = input_sample.mean(axis=0) if len(input_sample) else np.zeros(3, dtype=np.float64)
        if origin is None:
            origin = center.copy()

        shifted_input = input_sample - origin
        shifted_kept = kept_sample - origin
        shifted_removed = removed_sample - origin
        shifted_center = center - origin

        _update_limits(limits, shifted_input)
        _update_limits(limits, shifted_kept)
        _update_limits(limits, shifted_removed)
        _update_limits(limits, shifted_center.reshape(1, 3))
        path_points.append(_round_vec(shifted_center))

        frame_objects = []
        for box in boxes:
            shifted_box_center = np.asarray(box.center, dtype=np.float64) - origin
            frame_objects.append(
                {
                    "center": _round_vec(shifted_box_center),
                    "size": _round_vec(box.size),
                    "yaw": round(float(box.yaw), 6),
                    "label": box.label or "object",
                }
            )

        total_input_points += int(len(points))
        total_kept_points += int(len(kept_full))
        total_removed_points += int(len(removed_full))

        frames.append(
            {
                "index": frame_index,
                "name": path.parent.name,
                "source": f"{path.parent.name}/{path.name}",
                "input_points": int(len(points)),
                "kept_points": int(len(kept_full)),
                "removed_points": int(len(removed_full)),
                "render_input_points": int(len(input_sample)),
                "render_kept_points": int(len(kept_sample)),
                "render_removed_points": int(len(removed_sample)),
                "input": _round_points(shifted_input),
                "kept": _round_points(shifted_kept),
                "removed": _round_points(shifted_removed),
                "center": _round_vec(shifted_center),
                "objects": frame_objects,
            }
        )

    assert origin is not None

    if mode == "boxes":
        mode_label = "bounding boxes"
        source_note = (
            "checked-in sequence demo is using per-frame boxes for cleaned accumulation."
            " raw keeps all accumulated observations, cleaned keeps points after box removal."
        )
    else:
        mode_label = f"temporal consistency ({args.voxel_size:.2f}m / {args.window_size} / {args.min_hits})"
        source_note = (
            "checked-in sequence demo uses a real local multi-frame sequence and a temporal-consistency filter."
            " cleaned accumulation keeps only voxels that persist across frames; raw keeps everything that was observed."
        )

    scene = {
        "meta": {
            "title": args.title,
            "frame_count": len(frames),
            "default_fps": round(float(args.fps), 2),
            "max_render_points": int(args.max_render_points),
            "origin": _round_vec(origin),
            "mode": mode,
            "mode_label": mode_label,
            "total_input_points": total_input_points,
            "total_kept_points": total_kept_points,
            "total_removed_points": total_removed_points,
            "source_note": source_note,
        },
        "limits": _finalize_limits(limits),
        "path": path_points,
        "frames": frames,
    }

    if args.output_scene:
        args.output_scene.write_text(json.dumps(scene, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    html = HTML_TEMPLATE.replace("__DEMO_DATA__", json.dumps(scene, ensure_ascii=False, separators=(",", ":")))
    args.output_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
