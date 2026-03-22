"""Shared fixtures for dynamic_object_removal tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
DEMO_PCD = DEMO_DIR / "actual_scan_20240820_cloud.pcd"
DEMO_OBJECTS_JSON = DEMO_DIR / "actual_scan_20240820_objects.json"


@pytest.fixture
def demo_pcd_path() -> Path:
    assert DEMO_PCD.exists(), f"Demo PCD not found: {DEMO_PCD}"
    return DEMO_PCD


@pytest.fixture
def demo_objects_path() -> Path:
    assert DEMO_OBJECTS_JSON.exists(), f"Demo objects not found: {DEMO_OBJECTS_JSON}"
    return DEMO_OBJECTS_JSON


@pytest.fixture
def sample_points() -> np.ndarray:
    """A small 10-point cloud in a 1x1x1 cube centered at origin."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.5, 0.5, size=(10, 3))


@pytest.fixture
def sample_box():
    """A DetectionBox centered at origin with size 1x1x1."""
    from dynamic_object_removal import DetectionBox
    return DetectionBox(
        center=np.array([0.0, 0.0, 0.0]),
        size=np.array([1.0, 1.0, 1.0]),
        yaw=0.0,
    )
