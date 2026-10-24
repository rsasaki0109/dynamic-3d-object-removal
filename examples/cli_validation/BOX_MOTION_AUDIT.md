# nuScenes motion-label and vertical-coordinate audit

This audits why barrier-semantic common misses were labelled moving by the frozen benchmark. It preserves the original GT, parameters, API outputs and benchmark results. It is not a filter improvement or a revised accuracy score.

[Full audit and annotation trajectories](nuscenes_box_motion_audit.json) · [Prior point/track/semantic attribution](NEARER_RETURN_ATTRIBUTION.md)

## Key finding: the vertical coordinate contract matters

The official [nuScenes ego-pose schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#ego_pose) states:

> The localization is 2-dimensional in the x-y plane.
>
> Note that z is always 0.

The retrieved schema has git blob SHA `e69415ca97e49f50fd659aac5cafcd39bffa3cf0`. All **36 selected raw ego-pose records actually have translation Z = 0**. The small variation of the derived LiDAR-origin Z reflects calibrated sensor geometry/orientation; it is not an independently measured altitude trajectory.

The frozen moving-track rule uses **3D** annotation-center displacement >2 m. Its global-Z differences therefore need a coordinate-contract audit before being interpreted as physical motion. The observed issue is not confined to boxes: barrier-semantic point-height populations also change in the same XY footprints. This does not establish an annotation-only mistake or prove that an object is stationary.

## Audit population and bookkeeping

- Focus: the 13,998 **distinct** common-miss points with official lidarseg class `movable_object.barrier`, previously found in scene-0006. Scenes 0004/0005 are retained with zero focus points.
- Every focus point is tested against every moving annotation box at its own acquisition frame. No first-owner tie-break is used to discard overlapping memberships.
- There are **10,200** points in one moving box and **3,798** in multiple moving boxes. The 17,796 point/track memberships must not be summed as distinct points.
- The contributing set contains **38 tracks**: 37 annotated barriers and one adult-pedestrian track with two shared target points. The pedestrian has `pedestrian.moving` on all ten selected observations.
- Each contributing track reproduces the original selected-window motion rule exactly. Full-scene trajectories are retained separately as post-result context; they do not redefine the frozen GT.

## Measured geometry

| Quantity across 38 contributing tracks | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Maximum 3D displacement from first selected center (m) | 2.030 | 2.546 | 3.224 |
| Maximum XY displacement from first selected center (m) | 0.031 | 0.253 | 1.582 |
| Maximum absolute Z displacement from first selected center (m) | 1.590 | 2.492 | 3.216 |
| Derived sensor-origin Z span over those observations (m) | 0.004 | 0.021 | 0.045 |
| Barrier-point median-Z span in the XY footprint (m) | 0.000 | 2.642 | 3.875 |

All 38 tracks exceed 2 m under the original 3D rule, while **none exceeds 2 m horizontally**. All 37 barrier tracks have exactly constant annotated width/length/height over their selected observations. Dimension changes do not explain these selected barrier tracks. The projected box-face spans, center steps, annotation point counts and official attributes are retained in JSON.

The point-height context uses **all barrier-semantic points inside each annotation’s XY footprint**, with the existing 0.25 m XY margin and **no box-Z gate or common-miss/GT restriction**. It therefore avoids selecting points merely because their Z matches the changing box. However, it is not a per-object correspondence: sampling, partial visibility and neighboring barriers can change the observed point population. Frames with no points are explicitly undefined. A zero span from one observed frame is not evidence of stationary geometry.

The largest membership tracks illustrate the difference between horizontal and vertical changes:

| Track prefix | Category | Target memberships | Max 3D (m) | Max XY (m) | Max absolute Z (m) | Sensor Z span (m) | Point-median Z span (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| `8a454bc5` | `movable_object.barrier` | 899 | 2.851 | 0.209 | 2.843 | 0.045 | 3.170 |
| `19e50e1e` | `movable_object.barrier` | 875 | 2.732 | 0.194 | 2.726 | 0.031 | 2.755 |
| `ff4d5fb2` | `movable_object.barrier` | 862 | 2.349 | 0.048 | 2.349 | 0.045 | 1.900 |
| `1e4987b6` | `movable_object.barrier` | 776 | 2.959 | 0.128 | 2.956 | 0.021 | 3.227 |
| `9496b67a` | `movable_object.barrier` | 768 | 3.224 | 0.243 | 3.216 | 0.031 | 3.875 |
| `8d9cd163` | `movable_object.barrier` | 688 | 2.454 | 0.175 | 2.451 | 0.021 | 1.840 |
| `1df401e0` | `movable_object.barrier` | 686 | 2.434 | 0.174 | 2.428 | 0.045 | 2.575 |
| `df7e8f12` | `movable_object.barrier` | 669 | 2.933 | 0.100 | 2.932 | 0.045 | 2.819 |
| `20956819` | `movable_object.barrier` | 658 | 2.482 | 0.270 | 2.480 | 0.021 | 2.681 |
| `94b9bc45` | `movable_object.barrier` | 656 | 3.065 | 0.126 | 3.064 | 0.021 | 3.427 |

## Diagnostic horizontal-only counterfactual

The subsequent [explicit GT-policy implementation and comparison](GT_POLICY_COMPARISON.md) exports legacy XYZ, XY and XY-or-moving-attribute labels while sharing identical prediction masks. It preserves the original GT and separates label-definition effects from filter performance.

With the same first-observation anchor and strict 2 m threshold, but XY distance only, no contributing track would qualify. Thus **none of these 13,998 focus points is covered by a contributing horizontal-rule moving box**. This is a diagnostic of the original motion proxy, not a new GT label set or improved recall result. It also excludes the officially moving pedestrian track, illustrating why a universal XY displacement cutoff alone is not a complete physical-motion label.

The audit supports two separate follow-ups: compare explicit nuScenes motion-label policies while retaining the legacy result, and validate/correct vertical frame consistency before treating this dataset as a fully registered 3D map-cleaning accuracy test. Neither a physical stationary/moving conclusion nor a new performance headline follows from this audit alone. Reliable 3D ego-motion or independent point/object motion evidence would be needed to settle those questions.

## Generated evidence

Each `scene-XXXX/target_memberships.npz` retains focus map indices, every moving-track membership pair, per-point multiplicity, global XYZ and acquisition frames. Track-index tokens are in the JSON. Selected and full-scene observation records retain centers, width/length/height, yaw, center-displacement components, projected bounds, attribute names and annotation LiDAR point counts. Selected records additionally retain sensor-origin Z and annotation-minus-sensor Z.

For projection records, U/V are the first annotation’s local box X/Y axes (length/width conventions), not an assumption about which dimension is physically longest. Projected U/V positions are relative to the first center; Z bounds are in the recorded global frame.

The PNG/SVG audit figure selects the three largest target-membership tracks, with lexicographic token tie-breaks. It shows all their focus points in XY, first/last boxes, center trajectories, 3D/XY/Z/sensor and point-height changes, and box dimensions. This is diagnostic case selection; no point sampling or accuracy claim is made. Large arrays/plots remain in ignored `output/`, and compact JSON fingerprints and all-track measurements are retained here.

## Reproduce

Use the existing frozen manifests and prior point/semantic traces. The original trainval metadata archive also contains the three extra tables needed for category/attribute interpretation. Extract them separately, without replacing the lidarseg category mapping:

```bash
mkdir -p data/nuscenes_annotation_audit
tar -xzf work/trainval_meta.tgz -C data/nuscenes_annotation_audit \
  v1.0-trainval/instance.json v1.0-trainval/category.json v1.0-trainval/attribute.json
```

The source is the same official `public/v1.0/v1.0-trainval_meta.tgz` used in [the frozen validation](FROZEN_NUSCENES.md#reproduce). The audit records the extra-table hashes in addition to verifying the seven original metadata-table hashes. Run from the repository root with fresh output/report paths:

```bash
mkdir -p work/mpl-cache work/font-cache
MPLCONFIGDIR=work/mpl-cache XDG_CACHE_HOME=work/font-cache \
python scripts/audit_nuscenes_box_motion.py \
  --frozen-results examples/cli_validation/nuscenes_frozen_suite_results.json \
  --attribution-report examples/cli_validation/nuscenes_nearer_return_attribution.json \
  --semantics-report examples/cli_validation/nuscenes_nearer_return_semantics.json \
  --point-evidence-root output/nuscenes_common_miss_evidence \
  --semantic-root output/nuscenes_nearer_return_semantics \
  --manifest-root output/nuscenes_frozen_suite/manifests \
  --data-root data/nuscenes_trainval --lidarseg-root data/nuscenes_lidarseg \
  --annotation-metadata-root data/nuscenes_annotation_audit \
  --output output/box-motion-audit-replay \
  --report-json output/box-motion-audit-replay.json --render
```

NumPy is sufficient without rendering; Matplotlib is optional for PNG/SVG. Saved point/semantic NPZ hashes, full input-map/GT hashes, original moving-track sets, metadata and source-code hashes are verified. Every focus point must be covered by the retained moving-box memberships. All **326 tests passed**, including eight new tests for extent change versus rigid translation, chronological anchors/strict cutoffs, overlap bookkeeping, missing/duplicate annotations, vertical common shifts, Z-independent point context and raw-pose contract reporting.

No production filtering behavior, original GT file, original benchmark result or public demo is changed. Source: nuScenes Dataset and nuScenes-lidarseg, https://www.nuscenes.org.
