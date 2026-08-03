#!/usr/bin/env python3
"""Spoof LocalMap + control_state + inject_path; measure state→cmd latency via ROS.

Prereq: run hound_nav with planner bypass, e.g.

  ros2 run hound_nav nav_node --ros-args \\
    -p skip_planner:=true \\
    -p event_driven:=true

Then:

  ros2 run hound_nav nav_ipc_latency_probe --ros-args -p samples:=50
"""

from __future__ import annotations

import math
import statistics
import time
from typing import List, Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import MapMetaData, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Header
from hound_mapping.msg import LocalMap


def _f32_image(arr: np.ndarray, stamp, frame_id: str = "odom") -> Image:
    img = Image()
    img.header.stamp = stamp
    img.header.frame_id = frame_id
    if arr.ndim == 2:
        img.height, img.width = arr.shape
        img.encoding = "32FC1"
        img.step = img.width * 4
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img.height, img.width = arr.shape[0], arr.shape[1]
        img.encoding = "32FC3"
        img.step = img.width * 12
    else:
        raise ValueError(f"bad arr shape {arr.shape}")
    img.is_bigendian = 0
    img.data = arr.astype(np.float32).tobytes()
    return img


class NavIpcLatencyProbe(Node):
    def __init__(self) -> None:
        super().__init__("nav_ipc_latency_probe")
        self.declare_parameter("map_topic", "/hound_mapping/local_map")
        self.declare_parameter("state_topic", "/hound_fcu_control/control_state")
        self.declare_parameter("inject_path_topic", "/hound_nav/inject_path")
        self.declare_parameter("cmd_topic", "/hound_nav/cmd_ackermann")
        self.declare_parameter("latency_topic", "/hound_nav/state_to_cmd_ms")
        self.declare_parameter("map_size_m", 100.0)
        self.declare_parameter("map_res", 0.25)
        self.declare_parameter("path_length_m", 40.0)
        self.declare_parameter("path_step_m", 0.5)
        self.declare_parameter("speed_mps", 3.0)
        self.declare_parameter("samples", 50)
        self.declare_parameter("state_hz", 20.0)
        self.declare_parameter("warmup", 5)
        self.declare_parameter("settle_s", 2.0)

        self._latencies_nav: List[float] = []
        self._latencies_rtt: List[float] = []
        self._pending_pub_sec: Optional[float] = None
        self._matched = False

        qos = qos_profile_sensor_data
        self._map_pub = self.create_publisher(
            LocalMap, str(self.get_parameter("map_topic").value), 1
        )
        self._state_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("state_topic").value), qos
        )
        self._inject_pub = self.create_publisher(
            Path, str(self.get_parameter("inject_path_topic").value), 1
        )
        self.create_subscription(
            AckermannDriveStamped,
            str(self.get_parameter("cmd_topic").value),
            self._on_cmd,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("latency_topic").value),
            self._on_nav_latency,
            10,
        )

    def _on_nav_latency(self, msg: Float64MultiArray) -> None:
        # Require publisher stamp in data[2] — never accept unmatched/stale samples.
        if self._pending_pub_sec is None or len(msg.data) < 3:
            return
        if float(msg.data[2]) <= 0.0:
            return
        if abs(float(msg.data[2]) - self._pending_pub_sec) > 1e-4:
            return
        lat_ms = float(msg.data[0])
        now = self.get_clock().now().nanoseconds * 1e-9
        self._latencies_nav.append(lat_ms)
        self._latencies_rtt.append((now - self._pending_pub_sec) * 1e3)
        self._pending_pub_sec = None
        self._matched = True

    def _on_cmd(self, msg: AckermannDriveStamped) -> None:
        del msg  # correlation happens via latency topic + stamp
        return

    def publish_fake_map(self) -> None:
        size_m = float(self.get_parameter("map_size_m").value)
        res = float(self.get_parameter("map_res").value)
        n = int(round(size_m / res))
        elev = np.zeros((n, n), dtype=np.float32)
        cost = np.zeros((n, n), dtype=np.float32)  # 0 free .. 1 lethal
        # Soft obstacle blob ahead so cost isn't trivially zero.
        cost[n // 2 - 10 : n // 2 + 10, n // 2 + 40 : n // 2 + 55] = 1.0

        now = self.get_clock().now()
        stamp = now.to_msg()
        msg = LocalMap()
        msg.header = Header(stamp=stamp, frame_id="odom")
        msg.info = MapMetaData()
        msg.info.map_load_time = stamp
        msg.info.resolution = res
        msg.info.width = n
        msg.info.height = n
        msg.info.origin.position.x = -0.5 * size_m
        msg.info.origin.position.y = -0.5 * size_m
        msg.info.origin.orientation.w = 1.0
        msg.elevation = _f32_image(elev, stamp)
        msg.costmap = _f32_image(cost, stamp)
        normals = np.zeros((n, n, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        msg.normals = _f32_image(normals, stamp)
        # empty observed (optional)
        obs = Image()
        obs.header = msg.header
        obs.height, obs.width = n, n
        obs.encoding = "mono8"
        obs.step = n
        obs.data = bytes(n * n)
        msg.observed = obs
        self._map_pub.publish(msg)
        self.get_logger().info(f"Published fake LocalMap {n}x{n} @ {res}m")

    def publish_inject_path(self) -> None:
        length = float(self.get_parameter("path_length_m").value)
        step = float(self.get_parameter("path_step_m").value)
        speed = float(self.get_parameter("speed_mps").value)
        xs = np.arange(0.0, length + 1e-9, step, dtype=np.float64)
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "odom"
        for x in xs:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = 0.0
            ps.pose.orientation.w = 1.0  # yaw filled from diffs in nav
            path.poses.append(ps)
        self._inject_pub.publish(path)
        self.get_logger().info(
            f"Injected path: {len(path.poses)} poses, {length}m @ {speed} m/s hint"
        )

    def publish_state(self) -> float:
        """Publish 17-D state + stamp in data[17]. Returns publish ROS time (s)."""
        now = self.get_clock().now()
        stamp_sec = now.nanoseconds * 1e-9
        state = np.zeros(18, dtype=np.float64)
        # At origin, facing +x, moving forward.
        state[6] = float(self.get_parameter("speed_mps").value)
        state[11] = 9.8
        state[16] = state[6]  # wheelspeed
        state[17] = stamp_sec
        msg = Float64MultiArray()
        msg.data = state.tolist()
        self._pending_pub_sec = stamp_sec
        self._matched = False
        self._state_pub.publish(msg)
        return stamp_sec

    def run(self) -> None:
        samples = int(self.get_parameter("samples").value)
        warmup = int(self.get_parameter("warmup").value)
        state_hz = float(self.get_parameter("state_hz").value)
        settle = float(self.get_parameter("settle_s").value)
        period = 1.0 / max(state_hz, 1.0)

        time.sleep(0.5)
        self.publish_fake_map()
        time.sleep(0.2)
        self.publish_inject_path()
        self.get_logger().info(f"Settling {settle}s for nav to become ready…")
        # Seed a few states so the async BEV worker has a ready map before timing.
        t_end = time.perf_counter() + settle
        while time.perf_counter() < t_end and rclpy.ok():
            self.publish_state()
            self._pending_pub_sec = None
            self._matched = False
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)

        self.get_logger().info(
            f"Measuring {samples} samples (+{warmup} warmup) at {state_hz} Hz"
        )
        measured = 0
        attempt = 0
        while measured < samples and rclpy.ok():
            attempt += 1
            self.publish_state()
            t_deadline = time.perf_counter() + max(0.75, 4.0 * period)
            while (
                not self._matched
                and time.perf_counter() < t_deadline
                and rclpy.ok()
            ):
                rclpy.spin_once(self, timeout_sec=0.01)
            if attempt > warmup and self._matched:
                measured += 1
            elif not self._matched:
                self.get_logger().warn("Timeout waiting for matched latency sample")
            time.sleep(period)

        if not self._latencies_rtt:
            self.get_logger().error("No latency samples collected")
            return

        def _stats(xs: List[float]) -> str:
            xs = sorted(xs)
            p50 = xs[len(xs) // 2]
            p95 = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
            return (
                f"n={len(xs)} mean={statistics.fmean(xs):.2f} "
                f"p50={p50:.2f} p95={p95:.2f} min={xs[0]:.2f} max={xs[-1]:.2f} ms"
            )

        self.get_logger().info(
            "RTT (probe state pub → probe latency recv): " + _stats(self._latencies_rtt)
        )
        if self._latencies_nav:
            self.get_logger().info(
                "Nav-internal (state recv → cmd pub): " + _stats(self._latencies_nav)
            )
        else:
            self.get_logger().warn("No /hound_nav/state_to_cmd_ms samples")


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = NavIpcLatencyProbe()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
