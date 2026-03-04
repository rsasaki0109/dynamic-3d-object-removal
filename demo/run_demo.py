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
    template = (ROOT / "index_3d_standalone.html").read_text(encoding="utf-8")
    marker = "      const DEMO_DATA = "
    anchor = ";\n\n      if (!window.WebGLRenderingContext)"
    start = template.index(marker)
    end = template.index(anchor, start)
    html = template[:start] + marker + plot_data + template[end:]
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
