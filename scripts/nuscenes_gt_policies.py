"""Explicit nuScenes moving-track GT policies, separate from production filtering."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

POLICIES = ("xyz", "xy", "xy_or_moving_attribute")
MOVING_ATTRIBUTES = frozenset(("vehicle.moving", "pedestrian.moving"))


def definition(policy="xyz", threshold=2., box_margin=.25):
    if policy not in POLICIES:
        raise ValueError("unknown nuScenes GT policy")
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(box_margin) or box_margin < 0:
        raise ValueError("GT threshold and box margin must be finite and nonnegative")
    return {"kind": "moving_track_boxes", "motion_policy": policy,
            "displacement_axes": "xyz" if policy == "xyz" else "xy",
            "displacement_threshold_m": float(threshold), "comparison": "strictly_greater",
            "anchor": "first selected observation of each track",
            "moving_attribute_names": sorted(MOVING_ATTRIBUTES) if policy == "xy_or_moving_attribute" else [],
            "attribute_scope": "any selected observation; OR displacement" if policy == "xy_or_moving_attribute" else "unused",
            "single_observation": "may qualify via explicit moving attribute" if policy == "xy_or_moving_attribute" else "excluded",
            "point_label_scope": "per-frame boxes of qualifying tracks across the entire selected window",
            "box_margin_m": float(box_margin), "pose_contract": "nuScenes ego localization is 2D; Z translation is fixed to zero",
            "limitations": "A diagnostic GT definition, not independent physical-motion truth or corrected 3D registration."}


def select_tracks(annotations, selected, *, policy="xyz", threshold=2., attribute_names=None):
    definition(policy, threshold)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected samples must be a nonempty ordered unique list")
    if policy == "xy_or_moving_attribute" and attribute_names is None:
        raise ValueError("attribute-aware policy requires the official attribute mapping")
    rows = defaultdict(list)
    for sample in selected:
        seen = set()
        for annotation in annotations.get(sample, []):
            token = annotation["instance_token"]
            if token in seen:
                raise ValueError("duplicate instance annotation in one selected sample")
            seen.add(token)
            center = np.asarray(annotation["translation"], dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError("invalid annotation center")
            rows[token].append(annotation)
    decisions = {}
    for token, observations in sorted(rows.items()):
        centers = np.asarray([a["translation"] for a in observations], dtype=float)
        relative = centers - centers[0]
        xyz = float(np.linalg.norm(relative, axis=1).max())
        xy = float(np.linalg.norm(relative[:, :2], axis=1).max())
        names = []
        if attribute_names is not None:
            for a in observations:
                for attribute in a.get("attribute_tokens", []):
                    if attribute not in attribute_names:
                        raise ValueError("annotation attribute token missing from official mapping")
                    names.append(attribute_names[attribute])
        moving_attributes = sorted(set(names) & MOVING_ATTRIBUTES)
        displacement = xyz if policy == "xyz" else xy
        enough_motion = len(observations) > 1 and displacement > threshold
        explicit_motion = policy == "xy_or_moving_attribute" and bool(moving_attributes)
        decisions[token] = {"observations": len(observations), "max_xyz_displacement_m": xyz,
                            "max_xy_displacement_m": xy, "explicit_moving_attributes": moving_attributes,
                            "qualifies_by_displacement": enough_motion, "qualifies_by_attribute": explicit_motion,
                            "moving": bool(enough_motion or explicit_motion)}
    return {"moving": {token for token, decision in decisions.items() if decision["moving"]}, "tracks": decisions}
