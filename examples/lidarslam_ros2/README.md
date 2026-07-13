# lidar_slam_ros2 integration (experimental)

This example places pose-aware dynamic-object removal between RKO-LIO and the
`graph_based_slam` map backend:

```text
rosbag -> RKO-LIO (deskew + odom) -> DOR stamp pairing -> raw/cleaned map backends
```

This ordering is intentional. Detector-free temporal filtering needs ego pose, and
the cloud must already be deskewed. Filtering the raw cloud before the component that
provides those inputs would create a circular dependency. RKO-LIO frontend odometry
therefore remains identical in baseline and filtered branches; this example evaluates
**online static mapping**, not an odometry improvement and not offline map cleaning.

## Prerequisites

- ROS 2 Jazzy (the adjacent `lidar_slam_ros2` checkout currently targets Jazzy)
- a built `rsasaki0109/lidar_slam_ros2` workspace with `rko_lio` and
  `graph_based_slam`
- this project installed so `dynamic-object-removal-realtime` is on `PATH`
- a deskewable LiDAR + IMU rosbag, initially NTU VIRAL `tnp_01`

The launch can build both maps from one frontend run. DOR republishes the unfiltered
and filtered clouds with the same exact-stamp odometry. This guarantees identical
*published* stamp sets, but not that two loaded SLAM processes accept every message:
verify their pose graphs after the run. Under backpressure, record the three DOR output
topics once and replay each backend sequentially.

## Run

Source ROS and the SLAM workspace, then install this checkout:

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/lidarslam_ws/install/setup.bash
python3 -m pip install -e /path/to/dynamic-3d-object-removal
```

Invoke this file by absolute path (supported by the ROS 2 launch CLI). Set
`run_baseline_backend:=true` to build the same-stamp baseline and filtered maps in one
run; `frontend_mode:=online` uses rate-controlled `ros2 bag play`:

```bash
ros2 launch /path/to/dynamic-3d-object-removal/examples/lidarslam_ros2/dor_lidarslam.launch.py \
  bag_path:=/data/ntu_viral/tnp_01 \
  rko_param_file:=/path/to/lidar_slam_ros2/lidarslam/param/rko_lio_ntu_viral.yaml \
  main_param_dir:=/path/to/lidar_slam_ros2/lidarslam/param/lidarslam.yaml \
  frontend_mode:=online bag_play_rate:=1.0 \
  run_baseline_backend:=true \
  baseline_save_dir:=/tmp/tnp01_baseline \
  save_dir:=/tmp/tnp01_dor
```

The launch defaults match the current NTU helper (`/os1_cloud_node1/points`,
`/imu/imu`, cloud frame `sensor1/os_sensor`) and publishes `base_link <- lidar`
from the translation in `rko_lio_ntu_viral.yaml`. If the RKO extrinsic changes,
override `lidar_to_base_{x,y,z}` as well; a mismatched TF invalidates the comparison.

The OS1-16 is sparse, so `range` is the first candidate. The `1.0 x 2.0 degree`
starting resolution is not a claimed optimum; sweep it around the sensor's effective
beam spacing and record every tested configuration. AV2 online results favor `range`
over `temporal` for static preservation, but they do not establish NTU performance.
The rolling window starts at 3 because a 64k-point ROS replay measured callback p95
near 30 ms at this setting; it still requires an NTU accuracy/map-quality check.

### External pose/GT sequence

`frontend_mode:=external` skips RKO-LIO and plays an already deskewed cloud plus
exact-stamp odometry. The AV2 benchmark exporter preserves integer nanoseconds and the
ROS bag converter rebases every pose to the first frame without changing relative
transforms. Point labels stay in the manifest and are never published to DOR:

```bash
python3 scripts/run_av2_benchmark.py --frames 12 --stride 3 \
  --online-manifest /tmp/av2_manifest.json --online-only

source /opt/ros/jazzy/setup.bash
python3 scripts/prepare_online_manifest_rosbag.py \
  /tmp/av2_manifest.json /tmp/av2_rosbag

ros2 launch /path/to/dor_lidarslam.launch.py \
  bag_path:=/tmp/av2_rosbag frontend_mode:=external \
  deskewed_topic:=/av2/points frontend_odometry_topic:=/av2/odometry \
  filter_lidar_frame:=lidar lidar_to_base_x:=0 lidar_to_base_y:=0 lidar_to_base_z:=0 \
  main_param_dir:=/path/to/lidarslam_mid360_rko_graph.yaml
```

For the strict comparison, record DOR's three output topics once. Start the recorder
before the launch; the measured run used `bag_play_rate:=0.1`,
`sensor_rate_hz:=3.3333333333`, both graph backends disabled, and the default range
window/resolution. It recorded 11 baseline clouds, 11 cleaned clouds, and 11 odometry
messages with the same exact stamps. The filter saw no TF failure or fail-open frame;
callback p95 was 124.4 ms, below the AV2 selection's 300 ms frame period.

```bash
# Terminal 1
ros2 bag record -o /tmp/av2_dor_outputs \
  /dor/odometry /dor/baseline_points /dor/cleaned_points

# Terminal 2
ros2 launch /path/to/dor_lidarslam.launch.py \
  bag_path:=/tmp/av2_rosbag frontend_mode:=external bag_play_rate:=0.1 \
  deskewed_topic:=/av2/points frontend_odometry_topic:=/av2/odometry \
  filter_lidar_frame:=lidar lidar_to_base_x:=0 lidar_to_base_y:=0 lidar_to_base_z:=0 \
  sensor_rate_hz:=3.3333333333 run_filtered_backend:=false run_baseline_backend:=false \
  dor_summary_json:=/tmp/av2_dor_summary.json \
  main_param_dir:=/path/to/lidarslam_mid360_rko_graph.yaml
```

Live DDS playback was not accepted as the proof path: simultaneous backends yielded
different graph sizes, and an early sequential attempt yielded baseline 4 versus
cleaned 7 vertices because the heavy callback path did not consume identical message
sets. Instead, run the adjacent project's existing `graph_slam_offline_runner`. It
reads the fixed bag directly, pairs by exact stamp without DDS scheduling, and uses
the same submap creation, loop search, and pose-graph code as the live component:

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/lidarslam_ws/install/setup.bash

for branch in raw cleaned; do
  if [ "$branch" = raw ]; then topic=/dor/baseline_points; else topic=/dor/cleaned_points; fi
  ros2 run graph_based_slam graph_slam_offline_runner --ros-args \
    --params-file /path/to/lidarslam_mid360_rko_graph.yaml \
    -p bag_path:=/tmp/av2_dor_outputs \
    -p output_dir:=/tmp/av2_${branch}_map \
    -p offline_odom_topic:=/dor/odometry -p offline_cloud_topic:=$topic \
    -p submap_distance_threshold:=0.1 -p ndt_num_threads:=1 \
    -p refine:=true -p refine_save_maps:=true
done
```

Both runs report 11 pairs, 11 submaps, zero unpaired messages, and zero loop edges.
Their `trajectory_raw.tum`, `trajectory_optimized.tum`, and `loop_edges.csv` files are
byte-identical. Evaluation deliberately uses `map_optimized.pcd`, before the optional
cloud-driven refinement, then transforms manifest GT with that same trajectory:

```bash
python3 -m pip install scipy matplotlib

python3 scripts/compare_downstream_gt_maps.py \
  --manifest /tmp/av2_manifest.json \
  --baseline-map /tmp/av2_raw_map/map_optimized.pcd \
  --cleaned-map /tmp/av2_cleaned_map/map_optimized.pcd \
  --baseline-trajectory /tmp/av2_raw_map/trajectory_optimized.tum \
  --cleaned-trajectory /tmp/av2_cleaned_map/trajectory_optimized.tum \
  --baseline-raw-trajectory /tmp/av2_raw_map/trajectory_raw.tum \
  --cleaned-raw-trajectory /tmp/av2_cleaned_map/trajectory_raw.tum \
  --baseline-loop-edges /tmp/av2_raw_map/loop_edges.csv \
  --cleaned-loop-edges /tmp/av2_cleaned_map/loop_edges.csv \
  --dor-summary /tmp/av2_dor_summary.json \
  --output-json examples/lidarslam_ros2/av2_downstream_gt_map_proof.json \
  --output-png demo/av2_downstream_gt_map_proof.png
```

![AV2 downstream map GT proof](../../demo/av2_downstream_gt_map_proof.png)

The raw map has 1,132,807 points: 78,270 moving GT and 1,054,537 static GT.
Realtime `range` removes 50,839 points, including 11,058 moving GT and 39,781 static
GT: moving-GT contamination falls 14.1%, static-GT preservation is 96.2%, and removed
point precision is 21.8%. Every map point matches the reconstructed GT source within
0.01 m (observed maximum 7.7 µm). These modest numbers are reported rather than hidden:
the result proves same-pose downstream integration and some ghost reduction, while the
offline `fusion` proof remains the accuracy headline. Machine-readable counts and
SHA-256 contracts are in
[`av2_downstream_gt_map_proof.json`](av2_downstream_gt_map_proof.json).

### TIERS Indoor02 engineering result

The local public TIERS Indoor02 bag provided a real end-to-end integration check when
NTU VIRAL was unavailable locally. Its stored `/velodyne_points` and IMU topics were
copied into a 240 MiB derived bag so RKO's offline reader sees the future IMU before
each scan; message headers and sensor payloads are unchanged. The final source LiDAR
frame is omitted because the source bag has no IMU sample beyond that scan end:

```bash
source /opt/ros/jazzy/setup.bash
python3 scripts/prepare_offline_lio_bag.py /data/tiers/indoor02 /tmp/tiers_lio \
  --lidar-topic /velodyne_points --imu-topic /os_cloud_nodee/imu \
  --lidar-storage-delay-ms 50 --drop-trailing-lidar 1
```

The same-frontend comparison used rate-controlled online playback at 5 Hz. DOR
processed and paired 377 frames, with zero TF failures/fail-open frames. Both graph
backends used the exact same stamps and produced byte-identical pose graphs. At a
0.2 m nearest-neighbor radius, 99.74% of the dense-structure proxy was preserved and
100% of filtered map points were supported by the baseline. The 70 baseline-only
candidates were locally sparse (median 6 neighbors within 0.5 m versus 29 for
supported points). See
[`tiers_indoor02_stepa_results.json`](tiers_indoor02_stepa_results.json) and reproduce
the spatial proxy analysis with `scripts/compare_stepa_maps.py`.

These are not dynamic-object accuracy numbers: TIERS has no point-wise dynamic GT,
and the identity LiDAR/IMU extrinsics used here are an engineering approximation.
The result supports “sparse baseline-only contamination decreased while dense
structure stayed matched,” not “all ghosts were identified.” The separate 10 Hz
window-3 profile remains the realtime latency evidence; this 5 Hz run is the map
quality/integration proof.

## Acceptance checks

Run baseline and filtered mapping from the same frontend branch, graph parameters,
frame stamps, and map rendering settings. Record:

- DOR input/output points, TF fail-open count, and callback p50/p95/max
- graph/map input frames and final point count
- ghost-region point reduction and matched static-region retention
- a two-panel map screenshot from the same camera and point-size settings
- both pose-graph hashes and vertex counts; reject a pose-fixed claim if they differ

The result passes only if callback p95 is below the sensor period with zero dropped
frames, TF fail-open is understood and negligible, ghost contamination decreases,
and matched static structures remain intact. Point-count reduction alone is not
evidence of successful dynamic-object removal.
