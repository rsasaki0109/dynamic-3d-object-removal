#!/usr/bin/env python3
"""Build a self-contained sequence demo page from multiple point-cloud frames."""

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
    load_points,
    parse_boxes_payload,
    remove_points_in_boxes,
)

DEFAULT_FPS = 4.0
DEFAULT_FRAME_COUNT = 12
DEFAULT_MAX_RENDER_POINTS = 12000

HTML_TEMPLATE = r'''<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Dynamic 3D Object Removal | Sequence Demo</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f3f7fb;
        --panel: rgba(255, 255, 255, 0.88);
        --panel-strong: rgba(255, 255, 255, 0.96);
        --line: rgba(148, 163, 184, 0.32);
        --text: #0f172a;
        --muted: #475569;
        --accent: #0f766e;
        --accent-strong: #115e59;
        --input: #2563eb;
        --kept: #0f766e;
        --removed: #dc2626;
        --path: #f59e0b;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Trebuchet MS", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
        background:
          radial-gradient(circle at 15% 20%, rgba(37, 99, 235, 0.14), transparent 24%),
          radial-gradient(circle at 80% 10%, rgba(15, 118, 110, 0.16), transparent 22%),
          linear-gradient(180deg, #f8fbff 0%, #edf4fb 55%, #f7fafc 100%);
        color: var(--text);
      }
      .shell {
        display: grid;
        grid-template-columns: minmax(320px, 380px) minmax(0, 1fr);
        gap: 18px;
        min-height: 100vh;
        padding: 18px;
      }
      .panel,
      .viewer-wrap {
        border: 1px solid var(--line);
        border-radius: 22px;
        background: var(--panel);
        box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(12px);
      }
      .panel {
        padding: 22px 22px 18px;
        display: flex;
        flex-direction: column;
        gap: 18px;
      }
      .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        width: fit-content;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.12);
        color: var(--accent-strong);
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
        font-size: 14px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--muted);
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
        color: var(--muted);
        font-size: 13px;
      }
      .controls {
        display: grid;
        gap: 12px;
      }
      .control-row {
        display: grid;
        gap: 8px;
      }
      .control-row label,
      .switches legend {
        margin: 0;
        font-size: 13px;
        font-weight: 700;
        color: var(--muted);
      }
      .buttons {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
      }
      button,
      select,
      input[type="range"] {
        font: inherit;
      }
      button,
      select {
        border: 1px solid rgba(15, 118, 110, 0.18);
        background: #fff;
        color: var(--text);
        border-radius: 12px;
        padding: 10px 12px;
      }
      button.primary {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
        color: #fff;
        border: none;
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
        color: var(--text);
      }
      .dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
      }
      .range-meta,
      .small-meta {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 13px;
        color: var(--muted);
      }
      input[type="range"] {
        width: 100%;
      }
      .viewer-wrap {
        position: relative;
        overflow: hidden;
        min-height: min(82vh, 860px);
      }
      #viewer {
        position: absolute;
        inset: 0;
      }
      .hud {
        position: absolute;
        top: 16px;
        right: 16px;
        display: grid;
        gap: 8px;
        pointer-events: none;
      }
      .hud-card {
        min-width: 220px;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.24);
        color: var(--text);
      }
      .hud-card strong {
        display: block;
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 4px;
      }
      .viewer-note {
        position: absolute;
        left: 16px;
        bottom: 16px;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.66);
        color: #e2e8f0;
        font-size: 12px;
        line-height: 1.5;
      }
      a {
        color: var(--accent-strong);
      }
      @media (max-width: 980px) {
        .shell {
          grid-template-columns: 1fr;
        }
        .viewer-wrap {
          min-height: 64vh;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <aside class="panel">
        <div class="section">
          <span class="eyebrow">Sequence Demo</span>
          <h1>連続点群を Pages 上でそのまま再生</h1>
          <p class="lead">
            実フレーム列を sampled 埋め込みした self-contained ページです。GitHub Pages 上でも play / pause / scrub がそのまま動きます。
          </p>
        </div>

        <div class="section">
          <h2>Overview</h2>
          <div class="stats">
            <div class="stat">
              <strong id="stat-frame-count">0</strong>
              <span>frames</span>
            </div>
            <div class="stat">
              <strong id="stat-fps">0</strong>
              <span>default fps</span>
            </div>
            <div class="stat">
              <strong id="stat-render-cap">0</strong>
              <span>max rendered / frame</span>
            </div>
            <div class="stat">
              <strong id="stat-layer-mode">input</strong>
              <span>active layer</span>
            </div>
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
            <input id="point-size" type="range" min="0.6" max="3.5" step="0.1" value="1.4" />
            <div class="small-meta">
              <span>small</span>
              <span id="point-size-value">1.4 px</span>
              <span>large</span>
            </div>
          </div>
        </div>

        <div class="section">
          <h2>Layers</h2>
          <fieldset class="switches">
            <label id="toggle-input-wrap"><input id="toggle-input" type="checkbox" checked /> <span class="dot" style="background: var(--input)"></span>入力点群</label>
            <label id="toggle-kept-wrap"><input id="toggle-kept" type="checkbox" /> <span class="dot" style="background: var(--kept)"></span>除去後</label>
            <label id="toggle-removed-wrap"><input id="toggle-removed" type="checkbox" /> <span class="dot" style="background: var(--removed)"></span>除去点</label>
            <label id="toggle-boxes-wrap"><input id="toggle-boxes" type="checkbox" /> <span class="dot" style="background: #f97316"></span>3D box</label>
            <label id="toggle-path-wrap"><input id="toggle-path" type="checkbox" checked /> <span class="dot" style="background: var(--path)"></span>フレーム軌跡</label>
          </fieldset>
        </div>

        <div class="section">
          <h2>Current frame</h2>
          <div class="small-meta"><span>original points</span><span id="meta-original">0</span></div>
          <div class="small-meta"><span>rendered input</span><span id="meta-rendered-input">0</span></div>
          <div class="small-meta"><span>rendered kept</span><span id="meta-rendered-kept">0</span></div>
          <div class="small-meta"><span>rendered removed</span><span id="meta-rendered-removed">0</span></div>
          <div class="small-meta"><span>objects</span><span id="meta-objects">0</span></div>
        </div>

        <div class="section">
          <h2>Source</h2>
          <p class="lead" id="source-note"></p>
        </div>
      </aside>

      <section class="viewer-wrap">
        <div id="viewer"></div>
        <div class="hud">
          <div class="hud-card">
            <strong>Current frame</strong>
            <div id="hud-frame">-</div>
          </div>
          <div class="hud-card">
            <strong>Counts</strong>
            <div id="hud-counts">-</div>
          </div>
        </div>
        <div class="viewer-note">
          drag: orbit / wheel: zoom / right drag: pan<br />
          sequence is recentered around the first frame for stable WebGL rendering
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
      const frames = DEMO_DATA.frames || [];
      const hasKept = frames.some((frame) => Array.isArray(frame.kept) && frame.kept.length > 0);
      const hasRemoved = frames.some((frame) => Array.isArray(frame.removed) && frame.removed.length > 0);
      const hasBoxes = frames.some((frame) => Array.isArray(frame.objects) && frame.objects.length > 0);

      const viewer = document.getElementById("viewer");
      const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(viewer.clientWidth, viewer.clientHeight);
      viewer.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf6f9fc);

      const camera = new THREE.PerspectiveCamera(52, viewer.clientWidth / viewer.clientHeight, 0.1, 4000);
      camera.up.set(0, 0, 1);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(0, 0, 0);

      const ambient = new THREE.AmbientLight(0xffffff, 0.9);
      const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
      keyLight.position.set(25, -30, 40);
      scene.add(ambient, keyLight);

      const limits = DEMO_DATA.limits;
      const dx = Math.max(1, limits.xmax - limits.xmin);
      const dy = Math.max(1, limits.ymax - limits.ymin);
      const dz = Math.max(1, limits.zmax - limits.zmin);
      const extent = Math.max(dx, dy, dz, 8);
      const grid = new THREE.GridHelper(extent * 1.25, 28, 0x94a3b8, 0xd9e2ec);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = limits.zmin;
      scene.add(grid);

      const state = {
        frameIndex: 0,
        pointSize: 1.4,
        speed: 1,
        playing: false,
        showInput: true,
        showKept: hasKept,
        showRemoved: false,
        showBoxes: hasBoxes,
        showPath: true,
      };

      const groups = {
        frame: new THREE.Group(),
        path: null,
        marker: null,
      };
      scene.add(groups.frame);

      function flatToArray(flat) {
        return new Float32Array(flat || []);
      }

      function makePointSet(flat, color, alpha) {
        if (!flat || flat.length === 0) {
          return null;
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(flatToArray(flat), 3));
        const material = new THREE.PointsMaterial({
          color,
          size: state.pointSize,
          sizeAttenuation: true,
          transparent: alpha < 1,
          opacity: alpha,
        });
        return new THREE.Points(geometry, material);
      }

      function makeBoxes(objects) {
        if (!objects || objects.length === 0) {
          return null;
        }
        const group = new THREE.Group();
        for (const object of objects) {
          const geometry = new THREE.BoxGeometry(object.size[0], object.size[1], object.size[2]);
          const edges = new THREE.EdgesGeometry(geometry);
          const material = new THREE.LineBasicMaterial({ color: 0xf97316 });
          const lines = new THREE.LineSegments(edges, material);
          lines.position.set(object.center[0], object.center[1], object.center[2]);
          lines.rotation.z = object.yaw || 0;
          group.add(lines);
        }
        return group;
      }

      function clearGroup(group) {
        while (group.children.length > 0) {
          const child = group.children.pop();
          if (child.geometry) child.geometry.dispose();
          if (child.material) {
            if (Array.isArray(child.material)) {
              child.material.forEach((material) => material.dispose());
            } else {
              child.material.dispose();
            }
          }
        }
      }

      function updatePathLayer() {
        if (groups.path) {
          scene.remove(groups.path);
          groups.path.geometry.dispose();
          groups.path.material.dispose();
          groups.path = null;
        }
        if (groups.marker) {
          scene.remove(groups.marker);
          groups.marker.geometry.dispose();
          groups.marker.material.dispose();
          groups.marker = null;
        }
        if (!state.showPath || !Array.isArray(DEMO_DATA.path) || DEMO_DATA.path.length === 0) {
          return;
        }
        const pathArray = new Float32Array(DEMO_DATA.path.flat());
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(pathArray, 3));
        const material = new THREE.LineBasicMaterial({ color: 0xf59e0b, linewidth: 2 });
        groups.path = new THREE.Line(geometry, material);
        scene.add(groups.path);

        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(Math.max(extent * 0.012, 0.15), 16, 16),
          new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0x7c2d12, emissiveIntensity: 0.3 })
        );
        groups.marker = marker;
        scene.add(marker);
      }

      function updateFrame(index) {
        if (frames.length === 0) {
          return;
        }
        state.frameIndex = ((index % frames.length) + frames.length) % frames.length;
        const frame = frames[state.frameIndex];
        document.getElementById("frame-slider").value = String(state.frameIndex);
        document.getElementById("frame-label").textContent = `frame ${state.frameIndex + 1} / ${frames.length}`;
        document.getElementById("frame-source").textContent = frame.source;
        document.getElementById("hud-frame").textContent = `${frame.name} (${state.frameIndex + 1}/${frames.length})`;
        document.getElementById("hud-counts").textContent = `${frame.input_points.toLocaleString()} pts | render ${frame.render_input_points.toLocaleString()}`;
        document.getElementById("meta-original").textContent = frame.input_points.toLocaleString();
        document.getElementById("meta-rendered-input").textContent = frame.render_input_points.toLocaleString();
        document.getElementById("meta-rendered-kept").textContent = (frame.render_kept_points || 0).toLocaleString();
        document.getElementById("meta-rendered-removed").textContent = (frame.render_removed_points || 0).toLocaleString();
        document.getElementById("meta-objects").textContent = String((frame.objects || []).length);

        clearGroup(groups.frame);

        if (state.showInput && frame.input?.length) {
          const points = makePointSet(frame.input, 0x2563eb, hasKept || hasRemoved ? 0.32 : 0.78);
          if (points) groups.frame.add(points);
        }
        if (state.showKept && frame.kept?.length) {
          const points = makePointSet(frame.kept, 0x0f766e, 0.88);
          if (points) groups.frame.add(points);
        }
        if (state.showRemoved && frame.removed?.length) {
          const points = makePointSet(frame.removed, 0xdc2626, 0.95);
          if (points) groups.frame.add(points);
        }
        if (state.showBoxes && frame.objects?.length) {
          const boxes = makeBoxes(frame.objects);
          if (boxes) groups.frame.add(boxes);
        }

        if (groups.marker && frame.center) {
          groups.marker.position.set(frame.center[0], frame.center[1], frame.center[2]);
        }

        let layerMode = "input";
        if (state.showRemoved) layerMode = "removed";
        else if (state.showKept) layerMode = "kept";
        document.getElementById("stat-layer-mode").textContent = layerMode;
      }

      function fitView() {
        const center = new THREE.Vector3(
          (limits.xmin + limits.xmax) * 0.5,
          (limits.ymin + limits.ymax) * 0.5,
          (limits.zmin + limits.zmax) * 0.5
        );
        const radius = Math.max(dx, dy, dz, 8);
        camera.position.set(center.x + radius * 0.9, center.y - radius * 1.25, center.z + radius * 0.7);
        controls.target.copy(center);
        controls.update();
      }

      let playbackTimer = null;
      function restartPlayback() {
        if (playbackTimer) {
          clearInterval(playbackTimer);
          playbackTimer = null;
        }
        if (!state.playing || frames.length <= 1) {
          return;
        }
        const fps = Math.max(0.5, Number(DEMO_DATA.meta.default_fps) * state.speed);
        playbackTimer = setInterval(() => {
          updateFrame(state.frameIndex + 1);
        }, 1000 / fps);
      }

      document.getElementById("play-toggle").addEventListener("click", () => {
        state.playing = !state.playing;
        document.getElementById("play-toggle").textContent = state.playing ? "Pause" : "Play";
        restartPlayback();
      });
      document.getElementById("fit-view").addEventListener("click", fitView);
      document.getElementById("download-shot").addEventListener("click", () => {
        renderer.render(scene, camera);
        const link = document.createElement("a");
        link.href = renderer.domElement.toDataURL("image/png");
        link.download = `sequence_frame_${String(state.frameIndex).padStart(3, "0")}.png`;
        link.click();
      });
      document.getElementById("frame-slider").addEventListener("input", (event) => {
        updateFrame(Number(event.target.value));
      });
      document.getElementById("speed-select").addEventListener("change", (event) => {
        state.speed = Number(event.target.value);
        restartPlayback();
      });
      document.getElementById("point-size").addEventListener("input", (event) => {
        state.pointSize = Number(event.target.value);
        document.getElementById("point-size-value").textContent = `${state.pointSize.toFixed(1)} px`;
        updateFrame(state.frameIndex);
      });
      document.getElementById("toggle-input").addEventListener("change", (event) => {
        state.showInput = event.target.checked;
        updateFrame(state.frameIndex);
      });
      document.getElementById("toggle-kept").addEventListener("change", (event) => {
        state.showKept = event.target.checked;
        updateFrame(state.frameIndex);
      });
      document.getElementById("toggle-removed").addEventListener("change", (event) => {
        state.showRemoved = event.target.checked;
        updateFrame(state.frameIndex);
      });
      document.getElementById("toggle-boxes").addEventListener("change", (event) => {
        state.showBoxes = event.target.checked;
        updateFrame(state.frameIndex);
      });
      document.getElementById("toggle-path").addEventListener("change", (event) => {
        state.showPath = event.target.checked;
        updatePathLayer();
        updateFrame(state.frameIndex);
      });

      document.getElementById("toggle-kept-wrap").hidden = !hasKept;
      document.getElementById("toggle-removed-wrap").hidden = !hasRemoved;
      document.getElementById("toggle-boxes-wrap").hidden = !hasBoxes;

      document.getElementById("stat-frame-count").textContent = String(frames.length);
      document.getElementById("stat-fps").textContent = String(DEMO_DATA.meta.default_fps);
      document.getElementById("stat-render-cap").textContent = String(DEMO_DATA.meta.max_render_points);
      document.getElementById("source-note").textContent = DEMO_DATA.meta.source_note;
      document.getElementById("frame-slider").max = String(Math.max(0, frames.length - 1));

      fitView();
      updatePathLayer();
      updateFrame(0);

      function onResize() {
        const width = viewer.clientWidth;
        const height = viewer.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      }
      window.addEventListener("resize", onResize);

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
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
    parser.add_argument("--title", default="Continuous point-cloud sequence")
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
    compact = np.round(points.astype(np.float32), 3)
    return compact.reshape(-1).tolist()


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
    rng = np.random.default_rng(args.seed)

    raw_frames: list[dict[str, Any]] = []
    origin: np.ndarray | None = None

    for frame_index, path in enumerate(selected):
        points = load_points(path, fmt="auto")
        boxes = _resolve_boxes(boxes_spec, path)

        sampled_input = _sample_points(points, args.max_render_points, rng)
        kept_points: np.ndarray | None = None
        removed_points = np.zeros((0, 3), dtype=np.float64)
        if boxes:
            kept_full, keep_mask = remove_points_in_boxes(points, boxes, margin=args.box_margin)
            removed_full = points[~keep_mask]
            kept_points = _sample_points(kept_full, args.max_render_points, rng)
            removed_points = _sample_points(removed_full, args.max_render_points, rng)

        center = sampled_input.mean(axis=0) if len(sampled_input) else np.zeros(3, dtype=np.float64)
        if origin is None:
            origin = center.copy()

        raw_frames.append(
            {
                "index": frame_index,
                "name": path.parent.name,
                "source": f"{path.parent.name}/{path.name}",
                "input_points": int(len(points)),
                "render_input_points": int(len(sampled_input)),
                "kept_points": int(len(points) if kept_points is None else 0),
                "render_kept_points": int(0 if kept_points is None else len(kept_points)),
                "removed_points": int(len(removed_points) if boxes else 0),
                "render_removed_points": int(len(removed_points)),
                "input": sampled_input,
                "kept": kept_points,
                "removed": removed_points,
                "objects": boxes,
                "center": center,
            }
        )

    assert origin is not None
    limits = _new_limits()
    path_points: list[list[float]] = []
    frames: list[dict[str, Any]] = []

    for frame in raw_frames:
        input_points = frame["input"] - origin
        kept_points = None if frame["kept"] is None else frame["kept"] - origin
        removed_points = frame["removed"] - origin
        center = frame["center"] - origin

        _update_limits(limits, input_points)
        if kept_points is not None:
            _update_limits(limits, kept_points)
        _update_limits(limits, removed_points)
        path_points.append(_round_vec(center))

        objects = []
        for box in frame["objects"]:
            shifted_center = np.asarray(box.center, dtype=np.float64) - origin
            objects.append(
                {
                    "center": _round_vec(shifted_center),
                    "size": _round_vec(box.size),
                    "yaw": round(float(box.yaw), 6),
                    "label": box.label or "object",
                }
            )

        frames.append(
            {
                "index": frame["index"],
                "name": frame["name"],
                "source": frame["source"],
                "input_points": frame["input_points"],
                "render_input_points": frame["render_input_points"],
                "render_kept_points": frame["render_kept_points"],
                "render_removed_points": frame["render_removed_points"],
                "input": _round_points(input_points),
                "kept": None if kept_points is None else _round_points(kept_points),
                "removed": _round_points(removed_points),
                "objects": objects,
                "center": _round_vec(center),
            }
        )

    scene = {
        "meta": {
            "title": args.title,
            "frame_count": len(frames),
            "default_fps": round(float(args.fps), 2),
            "max_render_points": int(args.max_render_points),
            "origin": _round_vec(origin),
            "source_note": (
                "checked-in 版は local multi-frame sequence を sampled 埋め込みしています。"
                " per-frame boxes JSON を渡せば kept / removed / box layer の再生にもそのまま使えます。"
            ),
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
