from __future__ import annotations

import numpy as np

from scripts.run_height_candidate_ablation import height_persistence_candidate


def test_height_persistence_selects_transient_above_ground_only():
    ground = np.array([[0.1, 0.1, 0.0]])
    transient = np.array([[0.1, 0.1, 1.0]])
    scans = [np.vstack([ground, transient]), ground.copy(), ground.copy()]
    map_points = np.concatenate(scans)

    candidate, evidence = height_persistence_candidate(
        map_points,
        scans,
        xy_cell=1.0,
        z_bin=0.25,
        min_cell_height=0.5,
        ground_margin=0.2,
        min_visits=3,
        max_persistence=0.5,
    )

    assert candidate[1]
    assert not candidate[[0, 2, 3]].any()
    assert evidence["hits"][1] == 1
    assert evidence["visits"][1] == 3


def test_height_persistence_keeps_persistent_structure():
    wall = np.array([[0.1, 0.1, 0.0], [0.1, 0.1, 1.0]])
    scans = [wall.copy(), wall.copy(), wall.copy()]
    candidate, _ = height_persistence_candidate(
        np.concatenate(scans), scans,
        xy_cell=1.0, z_bin=0.25, min_cell_height=0.5,
        ground_margin=0.2, min_visits=3, max_persistence=0.5,
    )
    assert not candidate.any()
