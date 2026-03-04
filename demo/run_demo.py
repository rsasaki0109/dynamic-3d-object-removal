#!/usr/bin/env python3
"""Generate a small synthetic demo and comparison images for dynamic object removal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dynamic_object_removal as core

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "demo_input.xyz"
OBJECTS = ROOT / "demo_objects.json"
OUTPUT = ROOT / "demo_output.xyz"
OUT_BEFORE_AFTER = ROOT / "demo_before_after.png"
OUT_COMPARISON = ROOT / "demo_comparison.png"
OUT_BEFORE_AFTER_3D = ROOT / "demo_before_after_3d.png"
SCENE_JSON = ROOT / "demo_scene.json"
OUT_STANDALONE_3D = ROOT / "index_3d_standalone.html"


def generate_data() -> dict[str, int]:
    rng = np.random.default_rng(42)

    # Background points
    a = rng.uniform(0.0, 20.0, size=(7000, 1))
    b = rng.uniform(0.0, 12.0, size=(7000, 1))
    z1 = rng.normal(0.0, 0.02, size=(7000, 1))
    static = np.column_stack([a, b, z1])

    # Dynamic object area (vehicle-like box)
    car_center = np.array([8.0, 5.2, 0.55])
    car_size = np.array([2.8, 1.4, 1.2])
    a2 = rng.uniform(-car_size[0] / 2, car_size[0] / 2, size=(2200, 1)) + car_center[0]
    b2 = rng.uniform(-car_size[1] / 2, car_size[1] / 2, size=(2200, 1)) + car_center[1]
    z2 = rng.uniform(0.0, car_size[2], size=(2200, 1))
    dynamic = np.column_stack([a2, b2, z2])

    # Static objects to show unaffected region
    c = rng.uniform(12.0, 14.5, size=(1800, 1))
    d = rng.uniform(8.0, 10.0, size=(1800, 1))
    z3 = rng.normal(0.4, 0.04, size=(1800, 1))
    static2 = np.column_stack([c, d, z3])

    points = np.vstack([static, dynamic, static2]).astype(np.float64)
    np.savetxt(INPUT, points, fmt="%.6f")

    objects = [
        {
            "center": car_center.tolist(),
            "size": (car_size * 1.2).tolist(),
            "yaw": 0.0,
            "label": "vehicle_demo",
        }
    ]
    OBJECTS.write_text(json.dumps({"objects": objects}, ensure_ascii=False, indent=2), encoding="utf-8")

    boxes = core.parse_boxes_payload(objects, skip_invalid=False)
    filtered, keep = core.remove_points_in_boxes(points, boxes, [0.0, 0.0, 0.0])
    np.savetxt(OUTPUT, filtered, fmt="%.6f")

    removed = points[~keep]

    generate_plots(points, removed, filtered, car_center, car_size)
    generate_plots_3d(points, removed, filtered, car_center, car_size)
    scene = generate_scene_data(
        points=points,
        removed=removed,
        kept=filtered,
        center=car_center,
        size=car_size,
    )
    generate_standalone_3d_html(scene)

    return {
        "input_points": int(points.shape[0]),
        "output_points": int(filtered.shape[0]),
        "removed_points": int(points.shape[0] - filtered.shape[0]),
    }


def generate_scene_data(
    points: np.ndarray,
    removed: np.ndarray,
    kept: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
) -> dict[str, object]:
    box_half = (size * 1.2) / 2.0
    box_center = center.astype(float).tolist()
    payload = {
        "meta": {
            "input_points": int(points.shape[0]),
            "output_points": int(kept.shape[0]),
            "removed_points": int(removed.shape[0]),
        },
        "box": {
            "center": box_center,
            "size": (size * 1.2).tolist(),
            "yaw": 0.0,
        },
        "points": {
            "input": points.tolist(),
            "removed": removed.tolist(),
            "kept": kept.tolist(),
        },
        "limits": {
            "xmin": float(points[:, 0].min()),
            "xmax": float(points[:, 0].max()),
            "ymin": float(points[:, 1].min()),
            "ymax": float(points[:, 1].max()),
            "zmin": float(points[:, 2].min()),
            "zmax": float(points[:, 2].max()),
            "box_half": box_half.tolist(),
            "box_center": box_center,
        },
    }
    SCENE_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def generate_standalone_3d_html(data: dict[str, object]) -> None:
    plot_data = json.dumps(data, ensure_ascii=False)
    html = """<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <title>Dynamic Object Removal 3D Studio</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.165.0/examples/js/controls/OrbitControls.js"></script>
    <style>
      :root {
        --bg-top: #050711;
        --bg-bottom: #10162d;
        --panel: rgba(16, 24, 46, 0.82);
        --line: #243457;
        --text: #e5ecff;
        --muted: #8ea2d3;
        --accent: #6ee7ff;
        --acc-soft: #7c3aed;
      }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Segoe UI", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
        color: var(--text);
        background: radial-gradient(1200px 700px at 12% 2%, #1e2f5f 0%, transparent 56%),
          linear-gradient(140deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
      }
      .layout {
        min-height: 100vh;
        display: grid;
        grid-template-columns: 340px 1fr;
        gap: 18px;
        padding: 16px;
        box-sizing: border-box;
      }
      .side {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 12px 24px rgba(2, 9, 28, 0.5);
        backdrop-filter: blur(6px);
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: fit-content;
        border: 1px solid rgba(110, 231, 255, 0.45);
        border-radius: 999px;
        color: var(--accent);
        padding: 4px 12px;
        font-size: 12px;
        letter-spacing: 0.02em;
      }
      h1 {
        margin: 0;
        font-size: 20px;
      }
      .note {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.55;
        margin: 0;
      }
      .metric {
        background: rgba(7, 15, 32, 0.75);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px;
      }
      .metric h3 {
        margin: 0 0 4px;
        font-size: 13px;
        color: #bdd6ff;
      }
      .metric p {
        margin: 0;
        font-size: 16px;
        font-weight: 700;
      }
      .control-group {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }
      label {
        display: inline-flex;
        gap: 10px;
        align-items: center;
        font-size: 14px;
      }
      .line {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--line), transparent);
        margin: 4px 0;
      }
      .slider {
        width: 100%;
      }
      .actions {
        display: grid;
        gap: 8px;
      }
      .btn {
        border: 1px solid rgba(110, 231, 255, 0.5);
        background: linear-gradient(130deg, rgba(18, 28, 56, 0.95), rgba(36, 58, 97, 0.82));
        color: #e8f0ff;
        border-radius: 10px;
        padding: 8px 10px;
        cursor: pointer;
        font-weight: 600;
        transition: transform 0.16s ease, box-shadow 0.16s ease;
      }
      .btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(110, 231, 255, 0.18);
      }
      .btn:active {
        transform: translateY(0);
      }
      #stage {
        position: relative;
        border-radius: 16px;
        min-height: 82vh;
        border: 1px solid #243457;
        overflow: hidden;
        background: #05070f;
      }
      #viewer {
        width: 100%;
        height: 100%;
        min-height: 600px;
      }
      #meta {
        position: absolute;
        right: 14px;
        bottom: 12px;
        padding: 8px 10px;
        border-radius: 9px;
        background: rgba(5, 9, 24, 0.58);
        border: 1px solid rgba(138, 159, 199, 0.35);
        color: #a8b6db;
        font-size: 12px;
        backdrop-filter: blur(3px);
      }
      .warn {
        color: #ffc27a;
        font-size: 12px;
      }
      @media (max-width: 1080px) {
        .layout {
          grid-template-columns: 1fr;
        }
        #stage {
          min-height: 68vh;
        }
      }
    </style>
  </head>
  <body>
    <div class="layout">
      <aside class="side">
        <span class="badge">Dynamic Object Removal Viewer</span>
        <h1>3D Studio（Web）</h1>
        <p class="note" id="status"></p>
        <div class="metric">
          <h3>点群サマリ</h3>
          <p id="summary"></p>
        </div>
        <div class="line"></div>
        <div class="control-group">
          <strong>表示レイヤー</strong>
          <label><input type="checkbox" id="show-input" checked>入力</label>
          <label><input type="checkbox" id="show-kept" checked>除去後（静的）</label>
          <label><input type="checkbox" id="show-removed" checked>除去点（動的）</label>
          <label><input type="checkbox" id="show-bbox" checked>バウンディングボックス</label>
        </div>
        <div class="line"></div>
        <div class="control-group">
          <div class="row">
            <span>点サイズ</span>
            <strong id="point-size-value">2.0</strong>
          </div>
          <input id="point-size" class="slider" type="range" min="0.8" max="7" step="0.2" value="2">
          <div class="row">
            <span>透過</span>
            <strong id="alpha-value">0.80</strong>
          </div>
          <input id="alpha" class="slider" type="range" min="0.2" max="1" step="0.05" value="0.8">
        </div>
        <div class="line"></div>
        <div class="control-group">
          <div class="row">
            <span>自動回転</span>
            <label><input type="checkbox" id="auto-rotate"></label>
          </div>
          <div class="actions">
            <button class="btn" id="fit-view">視点リセット</button>
            <button class="btn" id="screenshot">PNG保存</button>
          </div>
        </div>
      </aside>
      <main id="stage" class="side">
        <div id="viewer"></div>
        <div id="meta"></div>
      </main>
    </div>
    <script>
      const DEMO_DATA = """ + plot_data + """;

      if (!window.WebGLRenderingContext) {
        document.getElementById("status").textContent =
          "WebGL が利用できない環境です。ブラウザの設定を確認してください。";
      }

      const meta = DEMO_DATA.meta;
      const limits = DEMO_DATA.limits;
      document.getElementById("status").textContent =
        "center: " + DEMO_DATA.box.center.map((v) => v.toFixed(2)).join(", ");
      document.getElementById("summary").textContent =
        "input " + meta.input_points + " 点 / kept " + meta.output_points + " 点 / removed " + meta.removed_points + " 点";

      const stage = document.getElementById("stage");
      const viewer = document.getElementById("viewer");
      const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
      const camera = new THREE.PerspectiveCamera(54, 16 / 9, 0.05, 1500);
      const scene = new THREE.Scene();
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      const center = new THREE.Vector3(
        (limits.xmin + limits.xmax) * 0.5,
        (limits.ymin + limits.ymax) * 0.5,
        (limits.zmin + limits.zmax) * 0.5
      );
      const extent = Math.max(
        limits.xmax - limits.xmin,
        limits.ymax - limits.ymin,
        limits.zmax - limits.zmin,
        1
      );

      const state = {
        showInput: true,
        showKept: true,
        showRemoved: true,
        showBbox: true,
        pointSize: 2,
        alpha: 0.8,
        autoRotate: false,
      };

      scene.background = new THREE.Color(0x05070f);
      scene.fog = new THREE.FogExp2(0x070d21, 0.004);

      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      viewer.appendChild(renderer.domElement);

      controls.enableDamping = true;
      controls.dampingFactor = 0.1;
      camera.position.set(center.x + extent * 0.9, center.y + extent * 0.8, center.z + extent * 0.6);
      controls.target.copy(center);

      const layerInput = buildPointCloud("input", DEMO_DATA.points.input, 0x94a3b8);
      const layerKept = buildPointCloud("kept", DEMO_DATA.points.kept, 0x3b82f6);
      const layerRemoved = buildPointCloud("removed", DEMO_DATA.points.removed, 0xef4444);
      const bbox = buildBbox(DEMO_DATA.box);
      const grid = new THREE.GridHelper(Math.max(30, extent), 30, 0x3b82f6, 0x253a66);
      grid.material.opacity = 0.2;
      grid.material.transparent = true;

      scene.add(grid);
      scene.add(layerInput.mesh);
      scene.add(layerKept.mesh);
      scene.add(layerRemoved.mesh);
      scene.add(bbox);
      fitCamera();

      controls.addEventListener("change", () => {
        updateHud();
      });

      const layers = {
        input: layerInput,
        kept: layerKept,
        removed: layerRemoved,
        bbox,
      };

      function buildPointCloud(key, points, color) {
        const positions = new Float32Array(points.length * 3);
        for (let i = 0; i < points.length; i++) {
          positions[i * 3] = points[i][0];
          positions[i * 3 + 1] = points[i][1];
          positions[i * 3 + 2] = points[i][2];
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const material = new THREE.PointsMaterial({
          size: state.pointSize,
          color,
          opacity: state.alpha,
          transparent: true,
          depthWrite: false,
          vertexColors: false,
        });
        const mesh = new THREE.Points(geometry, material);
        return {mesh};
      }

      function buildBbox(box) {
        const geometry = new THREE.BoxGeometry(box.size[0], box.size[1], box.size[2]);
        const wire = new THREE.EdgesGeometry(geometry);
        const material = new THREE.LineBasicMaterial({color: 0x2dd4bf, transparent: true, opacity: 0.9});
        const line = new THREE.LineSegments(wire, material);
        line.position.set(box.center[0], box.center[1], box.center[2]);
        line.rotation.z = box.yaw || 0.0;
        return line;
      }

      const ctrlInput = document.getElementById("show-input");
      const ctrlKept = document.getElementById("show-kept");
      const ctrlRemoved = document.getElementById("show-removed");
      const ctrlBbox = document.getElementById("show-bbox");
      const ctrlPointSize = document.getElementById("point-size");
      const ctrlAlpha = document.getElementById("alpha");
      const ctrlRotate = document.getElementById("auto-rotate");
      const pointSizeValue = document.getElementById("point-size-value");
      const alphaValue = document.getElementById("alpha-value");
      const btnFit = document.getElementById("fit-view");
      const btnShot = document.getElementById("screenshot");

      ctrlInput.addEventListener("change", applyUI);
      ctrlKept.addEventListener("change", applyUI);
      ctrlRemoved.addEventListener("change", applyUI);
      ctrlBbox.addEventListener("change", applyUI);
      ctrlPointSize.addEventListener("input", applyUI);
      ctrlAlpha.addEventListener("input", applyUI);
      ctrlRotate.addEventListener("change", () => {
        state.autoRotate = ctrlRotate.checked;
        controls.autoRotate = state.autoRotate;
      });
      btnFit.addEventListener("click", () => {
        controls.reset();
        fitCamera();
      });
      btnShot.addEventListener("click", () => {
        const a = document.createElement("a");
        a.href = renderer.domElement.toDataURL("image/png");
        a.download = "dynamic_object_removal_view.png";
        a.click();
      });

      function applyUI() {
        state.showInput = ctrlInput.checked;
        state.showKept = ctrlKept.checked;
        state.showRemoved = ctrlRemoved.checked;
        state.showBbox = ctrlBbox.checked;
        state.pointSize = parseFloat(ctrlPointSize.value);
        state.alpha = parseFloat(ctrlAlpha.value);

        layerInput.mesh.visible = state.showInput;
        layerKept.mesh.visible = state.showKept;
        layerRemoved.mesh.visible = state.showRemoved;
        bbox.visible = state.showBbox;

        [layerInput.mesh, layerKept.mesh, layerRemoved.mesh].forEach((mesh) => {
          mesh.material.size = state.pointSize;
          mesh.material.opacity = state.alpha;
        });

        pointSizeValue.textContent = state.pointSize.toFixed(1);
        alphaValue.textContent = state.alpha.toFixed(2);
      }

      function fitCamera() {
        camera.near = extent / 30;
        camera.far = Math.max(extent * 40, 300);
        controls.target.copy(center);
        controls.enablePan = true;
        controls.autoRotate = state.autoRotate;
        controls.update();
        camera.updateProjectionMatrix();
      }

      function resize() {
        const rect = stage.getBoundingClientRect();
        camera.aspect = rect.width / rect.height;
        camera.updateProjectionMatrix();
        renderer.setSize(rect.width, rect.height, false);
      }

      function updateHud() {
        const x = camera.position.x.toFixed(1);
        const y = camera.position.y.toFixed(1);
        const z = camera.position.z.toFixed(1);
        document.getElementById("meta").textContent =
          "cam=" + x + ", " + y + ", " + z;
      }

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }

      function init() {
        resize();
        applyUI();
        updateHud();
        animate();
      }

      window.addEventListener("resize", resize);
      init();
    </script>
  </body>
</html>
"""
    OUT_STANDALONE_3D.write_text(html, encoding="utf-8")


def generate_plots(points: np.ndarray, removed: np.ndarray, kept: np.ndarray, center: np.ndarray, size: np.ndarray) -> None:
    # panel 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], s=1.5, alpha=0.35, c="tab:gray")
    ax.scatter(removed[:, 0], removed[:, 1], s=2.0, alpha=0.8, c="red", label="removed")
    ax.add_patch(
        Rectangle(
            (center[0] - size[0] * 1.2 / 2, center[1] - size[1] * 1.2 / 2),
            size[0] * 1.2,
            size[1] * 1.2,
            linewidth=1.5,
            edgecolor="tab:red",
            facecolor="none",
        )
    )
    ax.set_title("Before filtering (top view)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    ax.scatter(kept[:, 0], kept[:, 1], s=1.5, alpha=0.45, c="tab:blue")
    ax.set_title("After filtering (top view)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_BEFORE_AFTER)
    plt.close(fig)

    # compact summary
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6), dpi=180)
    ax2.scatter(points[:, 0], points[:, 1], s=1.5, alpha=0.25, c="tab:gray", label="input")
    ax2.scatter(removed[:, 0], removed[:, 1], s=2.5, alpha=0.8, c="tab:red", label="removed")
    ax2.scatter(kept[:, 0], kept[:, 1], s=1.0, alpha=0.35, c="tab:blue", label="kept")
    ax2.add_patch(
        Rectangle(
            (center[0] - size[0] * 1.2 / 2, center[1] - size[1] * 1.2 / 2),
            size[0] * 1.2,
            size[1] * 1.2,
            linewidth=1.5,
            edgecolor="tab:red",
            facecolor="none",
        )
    )
    ax2.set_title("Dynamic object removal result (top view)")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right", fontsize=8)

    fig2.tight_layout()
    fig2.savefig(OUT_COMPARISON)
    plt.close(fig2)


def generate_plots_3d(
    points: np.ndarray,
    removed: np.ndarray,
    kept: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(14, 6), dpi=180)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1.0,
        alpha=0.28,
        c="tab:gray",
    )
    if removed.size > 0:
        ax1.scatter(
            removed[:, 0],
            removed[:, 1],
            removed[:, 2],
            s=2.0,
            alpha=0.8,
            c="tab:red",
            label="removed",
        )

    # wireframe box (not exact oriented box, yaw only)
    yaw = 0.0
    xw = np.array([-size[0] * 1.2 / 2, size[0] * 1.2 / 2])
    yw = np.array([-size[1] * 1.2 / 2, size[1] * 1.2 / 2])
    zw = np.array([0.0, size[2] * 1.2 / 2])
    corners = []
    for x in xw:
        for y in yw:
            for z in zw:
                corners.append([x, y, z])
    corners = np.array(corners)
    rot = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    corners = corners @ rot.T + center
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a_idx, b_idx in edges:
        ax1.plot(
            [corners[a_idx, 0], corners[b_idx, 0]],
            [corners[a_idx, 1], corners[b_idx, 1]],
            [corners[a_idx, 2], corners[b_idx, 2]],
            c="tab:red",
            linewidth=1.0,
        )

    ax1.set_title("Before (3D view)")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")
    ax1.view_init(elev=25, azim=45)
    ax1.set_box_aspect((1.0, 1.0, 0.35))
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        kept[:, 0],
        kept[:, 1],
        kept[:, 2],
        s=1.0,
        alpha=0.4,
        c="tab:blue",
    )
    ax2.set_title("After (3D view)")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_zlabel("z [m]")
    ax2.view_init(elev=25, azim=45)
    ax2.set_box_aspect((1.0, 1.0, 0.35))

    fig.tight_layout()
    fig.savefig(OUT_BEFORE_AFTER_3D)
    plt.close(fig)


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    counts = generate_data()
    print(f"input_points={counts['input_points']}")
    print(f"output_points={counts['output_points']}")
    print(f"removed_points={counts['removed_points']}")
    print(f"input_cloud={INPUT}")
    print(f"output_cloud={OUTPUT}")
    print(f"before_after={OUT_BEFORE_AFTER}")
    print(f"before_after_3d={OUT_BEFORE_AFTER_3D}")
    print(f"comparison={OUT_COMPARISON}")
    print(f"scene_data={SCENE_JSON}")
    print(f"standalone_3d={OUT_STANDALONE_3D}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
