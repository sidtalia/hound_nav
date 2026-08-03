"""ROS2 node: fill PDefBuffer from topics; run MPPI (+ optional IGHA*) on main thread."""

from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from hound_mapping.msg import LocalMap
from hound_nav.control_loop import run_control_loop
from hound_nav.pdef_buffer import LocalMapSnapshot, PDefBuffer


def _img_f32(msg) -> Optional[np.ndarray]:
    if msg.width == 0 or msg.height == 0 or not msg.data:
        return None
    h, w = int(msg.height), int(msg.width)
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
    if msg.encoding == "32FC3":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w, 3).copy()
    return None


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _path_to_nx4(msg: Path, default_speed: float = 3.0) -> np.ndarray:
    n = len(msg.poses)
    out = np.zeros((n, 4), dtype=np.float64)
    for i, ps in enumerate(msg.poses):
        out[i, 0] = ps.pose.position.x
        out[i, 1] = ps.pose.position.y
        q = ps.pose.orientation
        # Non-identity quat → use it; else fill from diffs below.
        if abs(q.w - 1.0) > 1e-6 or abs(q.x) + abs(q.y) + abs(q.z) > 1e-6:
            out[i, 2] = _yaw_from_quat(q.x, q.y, q.z, q.w)
        else:
            out[i, 2] = np.nan
        out[i, 3] = default_speed
    # Fill missing yaw from consecutive points.
    for i in range(n - 1):
        if np.isnan(out[i, 2]):
            out[i, 2] = math.atan2(out[i + 1, 1] - out[i, 1], out[i + 1, 0] - out[i, 0])
    if n >= 2 and np.isnan(out[-1, 2]):
        out[-1, 2] = out[-2, 2]
    elif n == 1 and np.isnan(out[0, 2]):
        out[0, 2] = 0.0
    return out


class NavRosBridge(Node):
    def __init__(self, buffer: PDefBuffer) -> None:
        super().__init__("hound_nav")
        self._buffer = buffer

        self.declare_parameter("config_path", "")
        self.declare_parameter("local_map_topic", "/hound_mapping/local_map")
        self.declare_parameter(
            "state_topic", "/hound_fcu_control/control_state"
        )
        self.declare_parameter("path_topic", "/mission/path")
        # Injected local plan (Nx poses → x,y,yaw,vel). When present, IGHA* is skipped.
        self.declare_parameter("inject_path_topic", "/hound_nav/inject_path")
        self.declare_parameter("skip_planner", False)
        self.declare_parameter("cmd_topic", "/hound_nav/cmd_ackermann")
        self.declare_parameter("plan_topic", "/hound_nav/local_plan")
        self.declare_parameter("latency_topic", "/hound_nav/state_to_cmd_ms")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("event_driven", True)
        self.declare_parameter("async_bev", True)
        self.declare_parameter("bev_hz", 20.0)
        self.declare_parameter("max_steer_rad", 0.6)
        self.declare_parameter("max_speed_mps", 20.0)

        self._config_path = str(self.get_parameter("config_path").value)
        self._max_steer = float(self.get_parameter("max_steer_rad").value)
        self._max_speed = float(self.get_parameter("max_speed_mps").value)
        self._rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._event_driven = bool(self.get_parameter("event_driven").value)
        self._async_bev = bool(self.get_parameter("async_bev").value)
        self._bev_hz = float(self.get_parameter("bev_hz").value)
        self._skip_planner = bool(self.get_parameter("skip_planner").value)

        qos = qos_profile_sensor_data
        self.create_subscription(
            LocalMap,
            str(self.get_parameter("local_map_topic").value),
            self._on_local_map,
            1,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("state_topic").value),
            self._on_control_state,
            qos,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("path_topic").value),
            self._on_path,
            1,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("inject_path_topic").value),
            self._on_inject_path,
            1,
        )
        self._cmd_pub = self.create_publisher(
            AckermannDriveStamped, str(self.get_parameter("cmd_topic").value), 1
        )
        self._plan_pub = self.create_publisher(
            Path, str(self.get_parameter("plan_topic").value), 1
        )
        self._latency_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("latency_topic").value), 10
        )
        self._stop = threading.Event()
        self.get_logger().info(
            f"hound_nav bridge ready; config={self._config_path or '(unset)'} "
            f"state={self.get_parameter('state_topic').value} "
            f"skip_planner={self._skip_planner} "
            f"event_driven={self._event_driven} "
            f"async_bev={self._async_bev}@{self._bev_hz}Hz "
            f"inject={self.get_parameter('inject_path_topic').value}"
        )

    def _on_local_map(self, msg: LocalMap) -> None:
        elev = _img_f32(msg.elevation)
        cost = _img_f32(msg.costmap)
        if elev is None or cost is None:
            return
        if elev.shape != cost.shape:
            return
        normals = _img_f32(msg.normals)
        if normals is not None and (
            normals.ndim != 3 or normals.shape[:2] != elev.shape
        ):
            normals = None
        self._buffer.set_local_map(
            LocalMapSnapshot(
                elevation=elev,
                cost=cost,
                origin_x=float(msg.info.origin.position.x),
                origin_y=float(msg.info.origin.position.y),
                resolution=float(msg.info.resolution),
                normals=normals,
                stamp_sec=float(msg.header.stamp.sec)
                + 1e-9 * float(msg.header.stamp.nanosec),
            )
        )

    def _on_control_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 17:
            return
        # Latency stamp = receive time (isolates nav compute from transport).
        # Optional data[17] = publisher ROS time, forwarded only for probe correlation.
        recv_sec = self.get_clock().now().nanoseconds * 1e-9
        pub_sec = 0.0
        if len(msg.data) >= 18 and float(msg.data[17]) > 0.0:
            pub_sec = float(msg.data[17])
        self._buffer.set_state_vector(
            np.asarray(msg.data[:17], dtype=np.float64),
            stamp_sec=recv_sec,
            pub_stamp_sec=pub_sec,
        )

    def _on_path(self, msg: Path) -> None:
        if not msg.poses:
            return
        wp = np.array(
            [[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses],
            dtype=np.float64,
        )
        self._buffer.set_waypoints(wp)

    def _on_inject_path(self, msg: Path) -> None:
        if len(msg.poses) < 2:
            return
        self._buffer.set_injected_path(_path_to_nx4(msg))
        self.get_logger().info(
            f"Injected local plan with {len(msg.poses)} poses (planner bypass)"
        )

    def send_ctrl(
        self,
        action: np.ndarray,
        dynamics_cfg: dict,
        state_stamp_sec: float = 0.0,
        state_pub_stamp_sec: float = 0.0,
    ) -> None:
        thr_to_ws = float(dynamics_cfg.get("throttle_to_wheelspeed", self._max_speed))
        now = self.get_clock().now()
        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = float(
            np.clip(action[0], -self._max_steer, self._max_steer)
        )
        msg.drive.speed = float(np.clip(action[1] * thr_to_ws, -thr_to_ws, thr_to_ws))
        self._cmd_pub.publish(msg)

        if state_stamp_sec > 0.0:
            lat = Float64MultiArray()
            # [latency_ms from recv, recv_stamp_sec, pub_stamp_sec (0 if none)]
            lat.data = [
                (now.nanoseconds * 1e-9 - state_stamp_sec) * 1e3,
                float(state_stamp_sec),
                float(state_pub_stamp_sec),
            ]
            self._latency_pub.publish(lat)

    def publish_plan(self, path: np.ndarray) -> None:
        out = Path()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "odom"
        for row in path:
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(row[0])
            ps.pose.position.y = float(row[1])
            ps.pose.position.z = float(row[2]) if row.shape[0] > 2 else 0.0
            out.poses.append(ps)
        self._plan_pub.publish(out)

    def request_stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set() or not rclpy.ok()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    buffer = PDefBuffer()
    node = NavRosBridge(buffer)

    config_path = node._config_path
    if not config_path:
        from ament_index_python.packages import get_package_share_directory
        import os

        config_path = os.path.join(
            get_package_share_directory("hound_nav"), "config", "nav_example.yaml"
        )
        node.get_logger().info(f"Using default config: {config_path}")

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        run_control_loop(
            buffer,
            config_path,
            send_ctrl=node.send_ctrl,
            should_stop=node.should_stop,
            publish_path=node.publish_plan,
            rate_hz=node._rate_hz,
            skip_planner=node._skip_planner,
            event_driven=node._event_driven,
            async_bev=node._async_bev,
            bev_hz=node._bev_hz,
        )
    except KeyboardInterrupt:
        pass
    finally:
        node.request_stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
