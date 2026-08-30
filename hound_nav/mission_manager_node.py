"""GPS mission manager: FCU lat/lon waypoints → one PoseStamped on goal_topic.

GPS mode
--------
Vehicle pose comes from NavSatFix + IMU (heading). A new GPS waypoint set
(nav_msgs/Path, frame wgs84, x=lat y=lon z=alt) is accepted only when
MAVLink fix_type >= min_fix_type (3 = 3D) and horizontal accuracy < max_h_acc_m.

Local ENU is east/north about a GPS origin frozen at accept (unless
update_wp_local_from_gps). Goal pose heading is always
atan2(target_y - robot_y, target_x - robot_x) in that frame.

If control_state is available, the GPS ENU delta is applied at the EKF
robot xy so the planner's odom frame matches. Current target goes to
goal_topic; the full local-frame list is latched on path_viz_topic for RViz.

RViz mode is not implemented.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import Float64MultiArray, UInt8

_EARTH_RADIUS_M = 6378137.0


def wgs84_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> np.ndarray:
    dlat = math.radians(lat_deg - lat0_deg)
    dlon = math.radians(lon_deg - lon0_deg)
    cos_lat0 = math.cos(math.radians(lat0_deg))
    north = dlat * _EARTH_RADIUS_M
    east = dlon * _EARTH_RADIUS_M * cos_lat0
    up = alt_m - alt0_m
    return np.array([east, north, up], dtype=np.float64)


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class MissionManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_manager")

        self.declare_parameter("mode", "gps")
        self.declare_parameter("gps_topic", "/hound_fcu_control/gps/fix")
        self.declare_parameter("gps_fix_type_topic", "/hound_fcu_control/gps/fix_type")
        self.declare_parameter("imu_topic", "/hound_fcu_control/imu")
        self.declare_parameter(
            "gps_waypoints_topic", "/hound_fcu_control/mission/gps"
        )
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("path_viz_topic", "/hound_nav/mission/waypoints")
        self.declare_parameter("state_topic", "/hound_fcu_control/control_state")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("min_fix_type", 3)
        self.declare_parameter("max_h_acc_m", 2.5)
        self.declare_parameter("update_wp_local_from_gps", False)
        self.declare_parameter("wp_radius", 1.0)
        # Periodic goal republish; 0 disables the timer (still publishes on WP reach / mission accept).
        self.declare_parameter("publish_hz", 0.25)

        self._mode = str(self.get_parameter("mode").value).strip().lower()
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._min_fix = int(self.get_parameter("min_fix_type").value)
        self._max_h_acc = float(self.get_parameter("max_h_acc_m").value)
        self._update_local = bool(self.get_parameter("update_wp_local_from_gps").value)
        self._wp_radius = float(self.get_parameter("wp_radius").value)

        goal_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        gps_wp_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._goal_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("goal_topic").value), goal_qos
        )
        self._path_viz_pub = self.create_publisher(
            Path, str(self.get_parameter("path_viz_topic").value), goal_qos
        )

        self._fix: Optional[NavSatFix] = None
        self._fix_type: int = 0
        self._have_fix_type = False
        self._imu_yaw: Optional[float] = None
        self._ekf_xy: Optional[np.ndarray] = None

        self._pending_gps: Optional[np.ndarray] = None  # Nx3 lat,lon,alt
        self._gps_wps: Optional[np.ndarray] = None
        self._wp_local: Optional[np.ndarray] = None
        self._origin: Optional[np.ndarray] = None  # lat, lon, alt
        self._curr_wp_index = 0
        self._final_reached = False
        self._logged_rviz = False
        self._last_reject_log = 0.0

        if self._mode != "gps":
            self.get_logger().warn(
                f"mode={self._mode!r} is not implemented (RViz needs a saved map). "
                "Idle until mode:=gps."
            )
            return

        self.create_subscription(
            NavSatFix,
            str(self.get_parameter("gps_topic").value),
            self._on_gps,
            10,
        )
        self.create_subscription(
            UInt8,
            str(self.get_parameter("gps_fix_type_topic").value),
            self._on_fix_type,
            10,
        )
        self.create_subscription(
            Imu, str(self.get_parameter("imu_topic").value), self._on_imu, 10
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("gps_waypoints_topic").value),
            self._on_gps_waypoints,
            gps_wp_qos,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("state_topic").value),
            self._on_state,
            10,
        )
        hz = float(self.get_parameter("publish_hz").value)
        if hz > 0.0:
            self.create_timer(1.0 / hz, self._on_timer)

        self.get_logger().info(
            f"GPS mission manager: wps={self.get_parameter('gps_waypoints_topic').value} "
            f"→ {self.get_parameter('goal_topic').value} "
            f"viz={self.get_parameter('path_viz_topic').value} "
            f"min_fix={self._min_fix} max_h_acc={self._max_h_acc:.2f}m "
            f"update_local={self._update_local} publish_hz={hz:g} "
            "(also publishes on waypoint reach)"
        )

    def _on_fix_type(self, msg: UInt8) -> None:
        self._fix_type = int(msg.data)
        self._have_fix_type = True
        self._try_activate()

    def _on_gps(self, msg: NavSatFix) -> None:
        self._fix = msg
        if not self._have_fix_type:
            # FCU maps MAVLink >=3 → STATUS_FIX; treat as 3D if topic is late.
            if msg.status.status >= NavSatStatus.STATUS_FIX:
                self._fix_type = max(self._fix_type, self._min_fix)
        if self._update_local and self._gps_wps is not None and self._gps_ok():
            self._origin = np.array(
                [msg.latitude, msg.longitude, msg.altitude], dtype=np.float64
            )
            self._rebuild_local()
            self._publish_waypoints()
        self._try_activate()
        if self._maybe_advance():
            self._publish_goal()

    def _on_imu(self, msg: Imu) -> None:
        self._imu_yaw = yaw_from_quat(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )

    def _on_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            self._ekf_xy = np.array([msg.data[0], msg.data[1]], dtype=np.float64)

    def _on_gps_waypoints(self, msg: Path) -> None:
        if not msg.poses:
            return
        rows: List[List[float]] = []
        for ps in msg.poses:
            rows.append(
                [ps.pose.position.x, ps.pose.position.y, ps.pose.position.z]
            )
        self._pending_gps = np.array(rows, dtype=np.float64)
        self.get_logger().info(
            f"GPS mission received n={len(rows)} frame={msg.header.frame_id!r}"
        )
        self._try_activate()

    def _h_acc_m(self) -> float:
        if self._fix is None:
            return float("inf")
        cov0 = float(self._fix.position_covariance[0])
        if cov0 > 0.0 and math.isfinite(cov0):
            return math.sqrt(cov0)
        return float("inf")

    def _gps_ok(self) -> bool:
        if self._fix is None:
            return False
        if self._fix_type < self._min_fix:
            return False
        return self._h_acc_m() < self._max_h_acc

    def _try_activate(self) -> None:
        if self._pending_gps is None:
            return
        if not self._gps_ok():
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_reject_log >= 2.0:
                self._last_reject_log = now
                self.get_logger().warn(
                    "holding GPS mission: need fix_type>="
                    f"{self._min_fix} (have {self._fix_type}) and "
                    f"h_acc<{self._max_h_acc:.2f}m (have {self._h_acc_m():.2f}m)"
                )
            return
        assert self._fix is not None
        self._gps_wps = np.copy(self._pending_gps)
        self._pending_gps = None
        self._origin = np.array(
            [self._fix.latitude, self._fix.longitude, self._fix.altitude],
            dtype=np.float64,
        )
        self._curr_wp_index = 0
        self._final_reached = False
        self._rebuild_local()
        self.get_logger().info(
            f"GPS mission active n={int(self._gps_wps.shape[0])} "
            f"origin=({self._origin[0]:.7f},{self._origin[1]:.7f}) "
            f"idx=0 local=({self._wp_local[0, 0]:.2f},{self._wp_local[0, 1]:.2f})"
        )
        self._publish_goal()
        self._publish_waypoints()

    def _rebuild_local(self) -> None:
        if self._gps_wps is None or self._origin is None:
            self._wp_local = None
            return
        n = int(self._gps_wps.shape[0])
        out = np.zeros((n, 3), dtype=np.float64)
        lat0, lon0, alt0 = (float(self._origin[0]), float(self._origin[1]), float(self._origin[2]))
        for i in range(n):
            out[i] = wgs84_to_enu(
                float(self._gps_wps[i, 0]),
                float(self._gps_wps[i, 1]),
                float(self._gps_wps[i, 2]),
                lat0,
                lon0,
                alt0,
            )
        self._wp_local = out

    def _robot_local(self) -> Optional[np.ndarray]:
        if self._fix is None or self._origin is None:
            return None
        return wgs84_to_enu(
            self._fix.latitude,
            self._fix.longitude,
            self._fix.altitude,
            float(self._origin[0]),
            float(self._origin[1]),
            float(self._origin[2]),
        )

    def _maybe_advance(self) -> bool:
        """Return True if a waypoint was reached (index advanced or final WP)."""
        if self._wp_local is None or self._curr_wp_index >= len(self._wp_local):
            return False
        robot = self._robot_local()
        if robot is None:
            return False
        wp = self._wp_local[self._curr_wp_index]
        d = float(np.linalg.norm(wp[:2] - robot[:2]))
        last = int(self._wp_local.shape[0]) - 1
        if d >= self._wp_radius:
            return False
        if self._curr_wp_index < last:
            self._curr_wp_index += 1
            nxt = self._wp_local[self._curr_wp_index]
            self.get_logger().info(
                f"waypoint reached → {self._curr_wp_index}/{last} "
                f"local=({nxt[0]:.2f},{nxt[1]:.2f})"
            )
            return True
        if not self._final_reached:
            self._final_reached = True
            self.get_logger().info(f"final waypoint {last} reached")
            return True
        return False

    def _world_xy(self, wp_xy: np.ndarray) -> Optional[np.ndarray]:
        """Waypoint in goal/odom frame (GPS ENU, plus EKF offset when available)."""
        robot = self._robot_local()
        if robot is None:
            return None
        xy = np.asarray(wp_xy, dtype=np.float64)[:2]
        if self._ekf_xy is not None:
            return self._ekf_xy + (xy - robot[:2])
        return xy.copy()

    def _publish_waypoints(self) -> None:
        if self._wp_local is None or int(self._wp_local.shape[0]) < 1:
            return
        pts = []
        for i in range(int(self._wp_local.shape[0])):
            xy = self._world_xy(self._wp_local[i])
            if xy is None:
                return
            pts.append(xy)
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = now
        path.header.frame_id = self._frame_id
        n = len(pts)
        for i, xy in enumerate(pts):
            if i + 1 < n:
                yaw = math.atan2(
                    float(pts[i + 1][1] - xy[1]), float(pts[i + 1][0] - xy[0])
                )
            elif i > 0:
                yaw = math.atan2(
                    float(xy[1] - pts[i - 1][1]), float(xy[0] - pts[i - 1][0])
                )
            else:
                yaw = 0.0
            qx, qy, qz, qw = yaw_to_quat(yaw)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(xy[0])
            ps.pose.position.y = float(xy[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)
        self._path_viz_pub.publish(path)

    def _publish_goal(self) -> None:
        if self._wp_local is None or int(self._wp_local.shape[0]) < 1:
            return
        idx = min(self._curr_wp_index, int(self._wp_local.shape[0]) - 1)
        robot = self._robot_local()
        if robot is None:
            return
        robot_xy = self._world_xy(robot)
        target_xy = self._world_xy(self._wp_local[idx])
        if target_xy is None or robot_xy is None:
            return
        dx = float(target_xy[0] - robot_xy[0])
        dy = float(target_xy[1] - robot_xy[1])
        if dx * dx + dy * dy < 1.0e-8 and self._imu_yaw is not None:
            yaw = float(self._imu_yaw)
        else:
            yaw = math.atan2(dy, dx)
        qx, qy, qz, qw = yaw_to_quat(yaw)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = float(target_xy[0])
        msg.pose.position.y = float(target_xy[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._goal_pub.publish(msg)

    def _on_timer(self) -> None:
        if self._mode != "gps":
            if not self._logged_rviz:
                self._logged_rviz = True
                self.get_logger().warn("RViz mission mode not implemented")
            return
        self._publish_goal()
        self._publish_waypoints()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
