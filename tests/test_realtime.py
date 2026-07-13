from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import dynamic_object_removal as core
import realtime


def _stamp(value: float) -> SimpleNamespace:
    sec = int(value)
    return SimpleNamespace(sec=sec, nanosec=int(round((value - sec) * 1e9)))


def _transform_message(
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    stamp: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(stamp)),
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=translation[0], y=translation[1], z=translation[2]),
            rotation=SimpleNamespace(
                x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
            ),
        ),
    )


def _cloud_message(frame_id: str = "lidar", stamp: float = 10.0) -> SimpleNamespace:
    return SimpleNamespace(header=SimpleNamespace(frame_id=frame_id, stamp=_stamp(stamp)))


def _wall() -> np.ndarray:
    ys = np.linspace(-2.0, 2.0, 41)
    zs = np.linspace(-1.0, 1.0, 21)
    return np.array([[10.0, y, z] for y in ys for z in zs], dtype=np.float64)


def _filter_node(algorithm: str) -> realtime.DynamicObjectRemovalNode:
    node = realtime.DynamicObjectRemovalNode.__new__(realtime.DynamicObjectRemovalNode)
    node._algorithm = algorithm
    node._fixed_frame = "odom"
    node._temporal_filter = None
    node._range_filter = None
    if algorithm == "temporal":
        node._temporal_filter = core.TemporalConsistencyFilter(
            voxel_size=0.1, window_size=3, min_hits=3
        )
    else:
        node._range_filter = core.RangeImageGhostFilter(
            window_size=3, h_res_deg=0.4, v_res_deg=1.0, range_margin=0.5
        )
    return node


class TestRigidTransform:
    def test_identity_and_translation(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
        tf = realtime._rigid_transform_from_message(
            _transform_message(translation=(10.0, -2.0, 0.5))
        )
        np.testing.assert_allclose(
            realtime._transform_points(points, tf),
            [[11.0, 0.0, 3.5], [9.0, -2.0, 2.5]],
            atol=1e-12,
        )

    def test_yaw_rotation(self):
        half = np.sqrt(0.5)
        tf = realtime._rigid_transform_from_message(
            _transform_message(quaternion=(0.0, 0.0, half, half))
        )
        np.testing.assert_allclose(
            realtime._transform_points(np.array([[1.0, 0.0, 0.0]]), tf),
            [[0.0, 1.0, 0.0]],
            atol=1e-12,
        )

    def test_invalid_zero_quaternion(self):
        with pytest.raises(ValueError, match="zero norm"):
            realtime._rigid_transform_from_message(
                _transform_message(quaternion=(0.0, 0.0, 0.0, 0.0))
            )

    def test_odometry_composes_lidar_to_base(self):
        odometry = SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=10.0, y=0.0, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )
        )
        transform = realtime._rigid_transform_from_odometry(
            odometry,
            realtime._RigidTransform(
                rotation=np.eye(3), translation=np.array([1.0, 2.0, 3.0])
            ),
        )
        np.testing.assert_allclose(transform.translation, [11.0, 2.0, 3.0])


class TestPoseAwareFiltering:
    @pytest.mark.parametrize("algorithm", ["temporal", "range"])
    def test_moving_sensor_preserves_static_wall(self, algorithm):
        world = _wall()
        node = _filter_node(algorithm)

        kept_counts = []
        for x in (0.0, 1.0, 2.0, 3.0):
            origin = np.array([x, 0.0, 0.0])
            local_points = world - origin
            node._fixed_transform = lambda _msg, o=origin: realtime._RigidTransform(
                rotation=np.eye(3), translation=o
            )
            filtered, aligned = node._filter_detector_free(local_points, _cloud_message())
            kept_counts.append(len(filtered))
            assert aligned

        if algorithm == "temporal":
            assert kept_counts == [0, 0, len(world), len(world)]
        else:
            assert kept_counts == [len(world)] * 4

    @pytest.mark.parametrize("algorithm", ["temporal", "range"])
    def test_moving_sensor_removes_new_transient_but_keeps_wall(self, algorithm):
        world = _wall()
        node = _filter_node(algorithm)

        for x in (0.0, 1.0, 2.0):
            origin = np.array([x, 0.0, 0.0])
            node._fixed_transform = lambda _msg, o=origin: realtime._RigidTransform(
                rotation=np.eye(3), translation=o
            )
            node._filter_detector_free(world - origin, _cloud_message())

        origin = np.array([3.0, 0.0, 0.0])
        transient_world = np.array([[5.0, 0.0, 0.0]])
        incoming_world = np.vstack([world, transient_world])
        node._fixed_transform = lambda _msg: realtime._RigidTransform(
            rotation=np.eye(3), translation=origin
        )
        filtered, _ = node._filter_detector_free(incoming_world - origin, _cloud_message())

        assert len(filtered) == len(world)


class _FakeTime:
    @classmethod
    def from_msg(cls, stamp):
        return stamp


class _FakeDuration:
    def __init__(self, *, seconds: float):
        self.seconds = seconds


class _FakeBuffer:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls = []

    def lookup_transform(self, target, source, stamp, *, timeout):
        self.calls.append((target, source, stamp, timeout.seconds))
        if self.error is not None:
            raise self.error
        return self.result


def _tf_node(buffer: _FakeBuffer) -> realtime.DynamicObjectRemovalNode:
    node = realtime.DynamicObjectRemovalNode.__new__(realtime.DynamicObjectRemovalNode)
    node._fixed_frame = "odom"
    node._tf_timeout = 0.05
    node._tf_stale_time = 0.25
    node._tf_buffer = buffer
    node._tf_duration_class = _FakeDuration
    node._tf_time_class = _FakeTime
    return node


class TestTimestampedTfLookup:
    def test_identity_when_cloud_is_already_in_fixed_frame(self):
        buffer = _FakeBuffer(error=AssertionError("lookup should not run"))
        tf = _tf_node(buffer)._fixed_transform(_cloud_message(frame_id="odom"))
        np.testing.assert_allclose(tf.rotation, np.eye(3))
        np.testing.assert_allclose(tf.translation, np.zeros(3))
        assert not buffer.calls

    def test_lookup_uses_cloud_timestamp(self):
        buffer = _FakeBuffer(
            result=_transform_message(translation=(2.0, 0.0, 0.0), stamp=10.0)
        )
        tf = _tf_node(buffer)._fixed_transform(_cloud_message(stamp=10.0))
        np.testing.assert_allclose(tf.translation, [2.0, 0.0, 0.0])
        assert buffer.calls[0][0:2] == ("odom", "lidar")
        assert realtime._stamp_to_sec(buffer.calls[0][2]) == pytest.approx(10.0)
        assert buffer.calls[0][3] == pytest.approx(0.05)

    @pytest.mark.parametrize(
        "message, match",
        [
            (_cloud_message(frame_id=""), "frame_id"),
            (SimpleNamespace(header=SimpleNamespace(frame_id="lidar", stamp=None)), "stamp"),
        ],
    )
    def test_missing_cloud_transform_metadata(self, message, match):
        node = _tf_node(_FakeBuffer())
        with pytest.raises(realtime._TransformUnavailable, match=match):
            node._fixed_transform(message)

    def test_missing_transform(self):
        node = _tf_node(_FakeBuffer(error=RuntimeError("not connected")))
        with pytest.raises(realtime._TransformUnavailable, match="no transform"):
            node._fixed_transform(_cloud_message())

    def test_stale_transform(self):
        node = _tf_node(_FakeBuffer(result=_transform_message(stamp=9.0)))
        with pytest.raises(realtime._TransformStale, match="exceeds"):
            node._fixed_transform(_cloud_message(stamp=10.0))

    def test_static_zero_stamp_is_not_stale(self):
        assert not realtime._transform_is_stale(10.0, 0.0, 0.25)


class _Logger:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)

    def info(self, _message):
        pass

    def error(self, _message):
        pass


class _NodeFacade:
    def __init__(self):
        self.logger = _Logger()

    def get_logger(self):
        return self.logger


def test_tf_failure_publishes_original_cloud_and_updates_stats(monkeypatch):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    node = realtime.DynamicObjectRemovalNode.__new__(realtime.DynamicObjectRemovalNode)
    node._PointCloud2 = object
    node._pointfield = object
    node._algorithm = "temporal"
    node._stats = realtime._RealtimeStats()
    node._stats_period = 100
    node._expected_rate_hz = 0.0
    node._last_cloud_stamp = None
    node._quiet = True
    node._node = _NodeFacade()
    node._filter_detector_free = lambda _points, _msg: (_ for _ in ()).throw(
        realtime._TransformUnavailable("missing")
    )
    published = []
    node._publish = lambda _msg, output: published.append(output.copy())
    monkeypatch.setattr(realtime, "_point_cloud2_to_xyz", lambda *_args: points.copy())

    node._on_pointcloud(_cloud_message())

    np.testing.assert_array_equal(published[0], points)
    summary = node._stats.summary()
    assert summary["tf_lookup_failures"] == 1
    assert summary["tf_stale_frames"] == 0
    assert summary["fail_open_frames"] == 1
    assert summary["removed_points"] == 0
    assert set(summary["filter_latency"]) == {"mean_ms", "p50_ms", "p95_ms", "max_ms"}
    assert node._node.logger.warnings and "fail-open" in node._node.logger.warnings[0]


def test_realtime_parser_exposes_tf_options():
    args = realtime._build_parser().parse_args(
        ["--algorithm", "range", "--fixed-frame", "odom", "--tf-timeout", "0.1"]
    )
    assert args.fixed_frame == "odom"
    assert args.tf_timeout == pytest.approx(0.1)
    assert args.tf_stale_time == pytest.approx(0.25)


def test_realtime_parser_exposes_paired_odometry_options():
    args = realtime._build_parser().parse_args(
        [
            "--odometry-topic",
            "/rko_lio/odometry",
            "--output-odometry-topic",
            "/dor/odometry",
        ]
    )
    assert args.odometry_topic == "/rko_lio/odometry"
    assert args.output_odometry_topic == "/dor/odometry"


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _pairing_node() -> realtime.DynamicObjectRemovalNode:
    node = realtime.DynamicObjectRemovalNode.__new__(realtime.DynamicObjectRemovalNode)
    node._pub_pc = _Publisher()
    node._pub_odometry = _Publisher()
    node._pub_baseline = _Publisher()
    node._pair_lock = realtime.Lock()
    node._pair_condition = realtime.Condition(node._pair_lock)
    node._pair_cache_size = 4
    node._odometry_by_stamp = realtime.OrderedDict()
    node._cloud_by_stamp = realtime.OrderedDict()
    node._stats = realtime._RealtimeStats()
    node._node = _NodeFacade()
    node._PointCloud2 = object
    node._pointfield = object
    return node


@pytest.mark.parametrize("odometry_first", [True, False])
def test_cleaned_cloud_and_odometry_are_republished_as_exact_stamp_pair(
    monkeypatch, odometry_first
):
    node = _pairing_node()
    cloud = _cloud_message(stamp=12.25)
    odometry = SimpleNamespace(header=cloud.header)
    monkeypatch.setattr(realtime, "_xyz_to_point_cloud2", lambda *_args: cloud)

    if odometry_first:
        node._on_odometry(odometry)
        node._publish(cloud, np.zeros((1, 3)))
    else:
        node._publish(cloud, np.zeros((1, 3)))
        node._on_odometry(odometry)

    assert node._pub_pc.messages == [cloud]
    assert node._pub_odometry.messages == [odometry]
    assert node._pub_baseline.messages == [cloud]
    assert node._stats.paired_odometry_published == 1
    assert not node._cloud_by_stamp
    assert not node._odometry_by_stamp


def test_timestamp_gaps_are_counted_as_inferred_drops(monkeypatch):
    node = realtime.DynamicObjectRemovalNode.__new__(realtime.DynamicObjectRemovalNode)
    node._PointCloud2 = object
    node._pointfield = object
    node._algorithm = "range"
    node._expected_rate_hz = 10.0
    node._last_cloud_stamp = None
    node._stats = realtime._RealtimeStats()
    node._stats_period = 100
    node._quiet = True
    node._node = _NodeFacade()
    node._filter_detector_free = lambda points, _msg: (points, True)
    node._publish = lambda *_args: None
    monkeypatch.setattr(
        realtime, "_point_cloud2_to_xyz", lambda *_args: np.array([[1.0, 0.0, 0.0]])
    )

    node._on_pointcloud(_cloud_message(stamp=10.0))
    node._on_pointcloud(_cloud_message(stamp=10.3))

    assert node._stats.inferred_dropped_frames == 2


def test_main_initializes_and_shuts_down_rclpy_in_order(monkeypatch):
    events = []

    class FakeRclpy:
        active = False

        @classmethod
        def init(cls, *, args):
            assert args is None
            cls.active = True
            events.append("init")

        @classmethod
        def ok(cls):
            return cls.active

        @classmethod
        def shutdown(cls):
            cls.active = False
            events.append("shutdown")

    class FakeRemovalNode:
        def __init__(self, **_kwargs):
            assert FakeRclpy.active
            events.append("construct")

        def spin(self):
            events.append("spin")

        def destroy(self):
            events.append("destroy")

    monkeypatch.setattr(realtime, "_ros_imports", lambda: {"rclpy": FakeRclpy})
    monkeypatch.setattr(realtime, "DynamicObjectRemovalNode", FakeRemovalNode)

    assert realtime.main(["--algorithm", "range"]) == 0
    assert events == ["init", "construct", "spin", "destroy", "shutdown"]


def test_main_treats_keyboard_interrupt_as_clean_shutdown(monkeypatch):
    events = []

    class FakeRclpy:
        active = False

        @classmethod
        def init(cls, *, args):
            cls.active = True

        @classmethod
        def ok(cls):
            return cls.active

        @classmethod
        def shutdown(cls):
            cls.active = False
            events.append("shutdown")

    class FakeRemovalNode:
        def __init__(self, **_kwargs):
            pass

        def spin(self):
            raise KeyboardInterrupt

        def destroy(self):
            events.append("destroy")

    monkeypatch.setattr(realtime, "_ros_imports", lambda: {"rclpy": FakeRclpy})
    monkeypatch.setattr(realtime, "DynamicObjectRemovalNode", FakeRemovalNode)

    assert realtime.main(["--algorithm", "range"]) == 0
    assert events == ["destroy", "shutdown"]
