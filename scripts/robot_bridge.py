#!/usr/bin/env python3
"""Feed hound_nav from the robot's navigation contract, and lease the base.

Subscribes to /map, /odom and TF (map → odom composed with odom → base_link).
Publishes the LocalMap crop and 17-vector control_state the stack already
consumes. Forwards the controller Twist to /base/<owner>/cmd_vel and keeps
the /base/claim lease alive. The same Viser page as the floor sim shows that
crop and takes a click-drag as /goal_pose in the map frame. Does not start
SLAM and does not publish /cmd_vel.

Run inside opendubs-gates-sim with rmw_fastrtps_cpp and
FASTDDS_BUILTIN_TRANSPORTS=UDPv4. The container's bashrc uses Zenoh and
will not see the host graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from hound_mapping.msg import LocalMap
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float64MultiArray, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from gates_floor_sim import _segs, ground_from_screen, yaw_quat

STATE_DIM = 17


def _yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _f32_image(arr: np.ndarray, stamp) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = "map"
    msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    return msg


def crop_grid(grid: np.ndarray, ox: float, oy: float, res: float,
              x: float, y: float, size_m: float, out_res: float):
    """Robot-centered square. Row 0 is min Y, same as the occupancy grid and LocalMap.

    Free cells (occupancy 0) become 255. Occupied, unknown, and off-map become 0.
    """
    n = int(round(size_m / out_res))
    origin_x = x - 0.5 * n * out_res
    origin_y = y - 0.5 * n * out_res
    h, w = grid.shape
    cols = origin_x + (np.arange(n) + 0.5) * out_res
    rows = origin_y + (np.arange(n) + 0.5) * out_res
    ic = np.floor((cols - ox) / res).astype(np.int32)
    ir = np.floor((rows - oy) / res).astype(np.int32)
    cc, rr = np.meshgrid(ic, ir, indexing="xy")
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    raw = np.full((n, n), -1, dtype=np.int16)
    raw[ok] = grid[rr[ok], cc[ok]]
    out = np.zeros((n, n), dtype=np.float32)
    out[raw == 0] = 255.0
    return out, origin_x, origin_y


class RobotBridge(Node):
    def __init__(self, args) -> None:
        super().__init__("hound_robot_bridge")
        self._owner = args.owner
        self._map_size = float(args.map_size)
        self._map_res = float(args.map_res)
        self._grid = None
        self._grid_meta = None
        self._twist = (0.0, 0.0, 0.0)
        self._odom_t = 0.0
        self._logged_pose = False
        self._logged_tf = False
        self._lock = threading.Lock()
        self._goal_vis = None
        self._goal_pending = None
        self._goal_committed = None
        self._init_pending = None
        self._cancel_pending = False
        self._drag_on = False
        self._pose_mode = "goal"
        self._cam_set = False
        self._plan_xy = None
        self._mppi = None
        self._mppi_c = None
        self._server = None
        self._map_h = None
        self._robot_h = None
        self._goal_h = None
        self._drawn = {}
        self._vis_pose = None
        self._tf = Buffer()
        TransformListener(self._tf, self, spin_thread=False)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, "/map", self._on_map, latched)
        self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        self.create_subscription(String, "/base/owner", self._on_owner, latched)
        self.create_subscription(
            Twist, args.cmd_in, self._on_cmd, qos_profile_sensor_data
        )
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._init_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self._cancel_pub = self.create_publisher(Empty, "/hound_nav/cancel", 10)
        self.create_subscription(Path, "/hound_nav/local_plan", self._on_plan, 1)
        self.create_subscription(MarkerArray, "/hound_nav/mppi_topk", self._on_mppi, 1)
        self._start_viser(int(args.viser_port))
        self._state_pub = self.create_publisher(
            Float64MultiArray, args.state_topic, qos_profile_sensor_data
        )
        self._map_pub = self.create_publisher(
            LocalMap, args.map_topic, qos_profile_sensor_data
        )
        self._cmd_pub = self.create_publisher(Twist, f"/base/{self._owner}/cmd_vel", 10)
        self._claim_pub = self.create_publisher(String, "/base/claim", 10)
        self._release_pub = self.create_publisher(String, "/base/release", 10)
        # shortcut: repeated between /odom samples so the manager's 0.02 s clock
        # stays in real time. Upgrade when that clock follows message stamps.
        self.create_timer(0.02, self._publish_state)
        self.create_timer(0.2, self._publish_map)
        self.create_timer(0.1, self._draw_viser)
        self.create_timer(1.0, self._renew)
        self.get_logger().info(
            f"bridge owner={self._owner} crop={self._map_size:.0f}m "
            f"@{self._map_res:.2f}m cmd {args.cmd_in} -> /base/{self._owner}/cmd_vel"
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        w = int(msg.info.width)
        h = int(msg.info.height)
        raw = np.asarray(msg.data, dtype=np.int16)
        if raw.size != w * h or w < 1 or h < 1:
            self.get_logger().warn(f"/map ignored: {w}x{h} data={raw.size}")
            return
        self._grid = raw.reshape(h, w)
        self._grid_meta = msg.info
        self.get_logger().info(
            f"/map {w}x{h} res={msg.info.resolution:.3f} "
            f"origin=({msg.info.origin.position.x:.2f},{msg.info.origin.position.y:.2f})"
        )

    def _on_odom(self, msg: Odometry) -> None:
        t = msg.twist.twist
        self._twist = (float(t.linear.x), float(t.linear.y), float(t.angular.z))
        self._odom_t = self.get_clock().now().nanoseconds * 1e-9

    def _on_owner(self, msg: String) -> None:
        self.get_logger().info(f"/base/owner {msg.data}", throttle_duration_sec=5.0)

    def _pose(self):
        try:
            tf = self._tf.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return None
        p = tf.transform.translation
        yaw = _yaw(tf.transform.rotation)
        return float(p.x), float(p.y), yaw

    def _publish_state(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._odom_t <= 0.0 or (now - self._odom_t) > 0.5:
            return
        pose = self._pose()
        if pose is None:
            if not self._logged_tf:
                self._logged_tf = True
                self.get_logger().warn(
                    "odom is arriving but TF map -> base_link is not. "
                    "map -> odom is not on /tf, so the base pose in the map is unknown"
                )
            return
        x, y, yaw = pose
        vx, vy, wz = self._twist
        s = [0.0] * STATE_DIM
        s[0], s[1], s[5] = x, y, yaw
        s[6], s[7], s[14] = vx, vy, wz
        msg = Float64MultiArray()
        msg.data = s
        self._state_pub.publish(msg)
        if not self._logged_pose:
            self._logged_pose = True
            self.get_logger().info(
                f"map pose ({x:.2f},{y:.2f},{math.degrees(yaw):.0f} deg) "
                f"body twist ({vx:.2f},{vy:.2f},{wz:.2f})"
            )
        self._vis_pose = (x, y, yaw)
        self._publish_goal()

    def _publish_map(self) -> None:
        if self._grid is None or self._grid_meta is None:
            return
        pose = self._pose()
        if pose is None:
            return
        info = self._grid_meta
        cost, ox, oy = crop_grid(
            self._grid,
            float(info.origin.position.x),
            float(info.origin.position.y),
            float(info.resolution),
            pose[0],
            pose[1],
            self._map_size,
            self._map_res,
        )
        stamp = self.get_clock().now().to_msg()
        elev = np.zeros_like(cost)
        lm = LocalMap()
        lm.header.stamp = stamp
        lm.header.frame_id = "map"
        lm.info = MapMetaData()
        lm.info.resolution = float(self._map_res)
        lm.info.width = int(cost.shape[1])
        lm.info.height = int(cost.shape[0])
        lm.info.origin.position.x = float(ox)
        lm.info.origin.position.y = float(oy)
        lm.info.origin.orientation.w = 1.0
        lm.elevation = _f32_image(elev, stamp)
        lm.costmap = _f32_image(cost, stamp)
        self._map_pub.publish(lm)
        self._show_map(cost, ox, oy)

    def _start_viser(self, port: int) -> None:
        import viser

        server = viser.ViserServer(port=port)
        server.scene.set_up_direction("+z")
        server.initial_camera.position = (0.0, -12.0, 24.0)
        server.initial_camera.look_at = (0.0, 0.0, 0.0)
        self._robot_h = server.scene.add_box(
            "/robot",
            dimensions=(0.7, 0.75, 0.1),
            color=(30, 120, 220),
            side="double",
            position=(0.0, 0.0, 0.05),
        )
        self._goal_h = server.scene.add_box(
            "/goal",
            dimensions=(0.35, 0.35, 0.04),
            color=(220, 40, 40),
            side="double",
            position=(0.0, 0.0, 0.02),
            visible=False,
        )
        self._set_goal = server.gui.add_checkbox("set goal", initial_value=True)
        self._set_init = server.gui.add_checkbox(
            "set initial pose", initial_value=False
        )
        self._cancel_btn = server.gui.add_button("cancel goal")
        self._cancel_btn.on_click(lambda _ev: self._request_cancel())
        server.gui.add_markdown(
            "check **set goal** or **set initial pose**, then drag on the map. "
            "the press is the position and the drag direction is the heading. "
            "with both off, the drag orbits the camera.\n\n"
            "the blue box is where localization places the robot. "
            "green = plan · orange = best MPPI rollouts. "
            "**cancel goal** drops the path and the controller stops."
        )
        self._server = server
        self.get_logger().info(f"viser http://0.0.0.0:{port}")

    def _on_plan(self, msg: Path) -> None:
        if not msg.poses:
            with self._lock:
                self._plan_xy = None
            return
        xy = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            np.float32,
        )
        with self._lock:
            self._plan_xy = xy

    def _on_mppi(self, msg: MarkerArray) -> None:
        chunks, cols = [], []
        for m in msg.markers:
            if len(m.points) < 2:
                continue
            xy = np.array([[p.x, p.y] for p in m.points], np.float32)
            seg = _segs(xy, 0.08)
            if seg is None:
                continue
            c = np.array([m.color.r, m.color.g, m.color.b], np.float32) * 255.0
            chunks.append(seg)
            cols.append(np.broadcast_to(c, (seg.shape[0], 2, 3)).copy())
        with self._lock:
            if chunks:
                self._mppi = np.concatenate(chunks)
                self._mppi_c = np.concatenate(cols)
            else:
                self._mppi = self._mppi_c = None

    async def _on_goal_drag(self, ev) -> None:
        hit = ground_from_screen(ev.client, ev.end_screen_pos)
        if hit is None:
            return
        x0 = float(ev.start_position[0])
        y0 = float(ev.start_position[1])
        dx = float(hit[0]) - x0
        dy = float(hit[1]) - y0
        if dx * dx + dy * dy < 0.25 * 0.25:
            if ev.phase == "end":
                with self._lock:
                    self._goal_vis = self._goal_committed
            return
        pose = (x0, y0, math.atan2(dy, dx))
        init = bool(self._set_init.value)
        with self._lock:
            self._goal_vis = pose
            if ev.phase == "end" and init:
                self._init_pending = pose
            elif ev.phase == "end":
                self._goal_committed = pose
                self._goal_pending = pose

    def _request_cancel(self) -> None:
        with self._lock:
            self._cancel_pending = True
            self._goal_vis = None
            self._goal_committed = None
            self._goal_pending = None
            self._plan_xy = None
            self._mppi = None
            self._mppi_c = None

    def _sync_drag(self) -> None:
        if self._map_h is None:
            return
        goal_on = bool(self._set_goal.value)
        init_on = bool(self._set_init.value)
        if goal_on and init_on:
            # The box that was already on is the old mode. Keep the one just checked.
            if self._pose_mode == "init":
                self._set_init.value = False
                init_on = False
            else:
                self._set_goal.value = False
                goal_on = False
        self._pose_mode = "init" if init_on else "goal" if goal_on else "look"
        want = goal_on or init_on
        if want and not self._drag_on:
            self._map_h.on_drag(self._on_goal_drag)
            self._drag_on = True
        elif not want and self._drag_on:
            self._map_h.remove_drag_callback(self._on_goal_drag)
            self._drag_on = False

    def _publish_goal(self) -> None:
        with self._lock:
            goal = self._goal_pending
            init = self._init_pending
            cancel = self._cancel_pending
            self._goal_pending = None
            self._init_pending = None
            self._cancel_pending = False
        if cancel:
            self._cancel_pub.publish(Empty())
            self.get_logger().info("cancel goal")
        if init is not None:
            self._publish_initial_pose(init)
        if goal is None:
            return
        g = PoseStamped()
        g.header.stamp = self.get_clock().now().to_msg()
        g.header.frame_id = "map"
        g.pose.position.x = float(goal[0])
        g.pose.position.y = float(goal[1])
        qw, qx, qy, qz = yaw_quat(float(goal[2]))
        g.pose.orientation.w = qw
        g.pose.orientation.x = qx
        g.pose.orientation.y = qy
        g.pose.orientation.z = qz
        self._goal_pub.publish(g)
        self.get_logger().info(
            f"goal ({goal[0]:.2f},{goal[1]:.2f},{math.degrees(goal[2]):.0f} deg) frame=map"
        )

    def _publish_initial_pose(self, pose) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        qw, qx, qy, qz = yaw_quat(float(pose[2]))
        msg.pose.pose.orientation.w = qw
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.25
        cov[35] = 0.07
        msg.pose.covariance = cov
        self._init_pub.publish(msg)
        self.get_logger().info(
            f"initial pose ({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f} deg) frame=map"
        )

    def _show_map(self, cost: np.ndarray, ox: float, oy: float) -> None:
        if self._server is None:
            return
        rgb = np.repeat(np.clip(cost, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
        side_x = float(cost.shape[1]) * float(self._map_res)
        side_y = float(cost.shape[0]) * float(self._map_res)
        center = (ox + 0.5 * side_x, oy + 0.5 * side_y, 0.0)
        if self._map_h is None:
            self._map_h = self._server.scene.add_image(
                "/map",
                rgb,
                side_x,
                side_y,
                format="png",
                cast_shadow=False,
                receive_shadow=False,
                position=center,
            )
        self._sync_drag()
        self._map_h.image = rgb
        self._map_h.position = center

    def _draw(self, name, segs, color, width) -> None:
        old = self._drawn.pop(name, None)
        if old is not None:
            old.remove()
        if segs is None:
            return
        self._drawn[name] = self._server.scene.add_line_segments(
            name, segs, color, line_width=width
        )

    def _draw_viser(self) -> None:
        if self._server is None:
            return
        self._sync_drag()
        if self._vis_pose is None:
            return
        x, y, yaw = self._vis_pose
        if not self._cam_set:
            self._cam_set = True
            self._server.initial_camera.position = (x, y - 12.0, 24.0)
            self._server.initial_camera.look_at = (x, y, 0.0)
        self._robot_h.position = (x, y, 0.05)
        self._robot_h.wxyz = yaw_quat(yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        self._draw(
            "/heading",
            _segs(np.array([[x, y], [x + 0.7 * c, y + 0.7 * s]], np.float32), 0.16),
            (255, 210, 40),
            6,
        )
        with self._lock:
            shown = None if self._goal_vis is None else tuple(self._goal_vis)
            plan = None if self._plan_xy is None else self._plan_xy.copy()
            mppi, mppi_c = self._mppi, self._mppi_c
        if shown is None:
            self._goal_h.visible = False
            self._draw("/goal_heading", None, (220, 40, 40), 6)
        else:
            gx, gy, gyaw = shown
            self._goal_h.position = (gx, gy, 0.02)
            self._goal_h.visible = True
            gc, gs = math.cos(gyaw), math.sin(gyaw)
            self._draw(
                "/goal_heading",
                _segs(
                    np.array([[gx, gy], [gx + 0.7 * gc, gy + 0.7 * gs]], np.float32),
                    0.18,
                ),
                (220, 40, 40),
                6,
            )
        self._draw("/plan", None if plan is None else _segs(plan, 0.12), (40, 210, 70), 4)
        self._draw("/mppi", mppi, (255, 140, 20) if mppi_c is None else mppi_c, 2)

    def _on_cmd(self, msg: Twist) -> None:
        out = Twist()
        out.linear.x = float(msg.linear.x)
        out.linear.y = float(msg.linear.y)
        out.angular.z = float(msg.angular.z)
        self._cmd_pub.publish(out)

    def _renew(self) -> None:
        # No lease until the map pose exists, so a missing transform does not hold the base.
        if not self._logged_pose:
            return
        self._claim_pub.publish(
            String(data=json.dumps({"owner": self._owner, "ttl_s": 5}))
        )

    def release(self) -> None:
        self._release_pub.publish(String(data=json.dumps({"owner": self._owner})))


def _nav(path: str, key: str, default):
    doc = yaml.safe_load(open(path)) or {}
    nav = doc.get("nav") or {}
    sim = doc.get("sim") or {}
    if key in nav:
        return nav[key]
    if key in sim:
        return sim[key]
    return default


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=os.path.join(here, "..", "config", "gates_r6_f2.yaml"),
    )
    p.add_argument("--owner", default="nav")
    args = p.parse_args()
    cfg = os.path.abspath(args.config)
    args.map_size = float(_nav(cfg, "map_size", 30.0))
    args.map_res = float(_nav(cfg, "map_res", 0.05))
    args.state_topic = str(_nav(cfg, "state_topic", "/hound_fcu_control/control_state"))
    args.map_topic = str(_nav(cfg, "local_map_topic", "/hound_mapping/local_map"))
    args.cmd_in = str(_nav(cfg, "cmd_topic", "/hound_nav/cmd_ackermann"))
    args.viser_port = int(_nav(cfg, "viser_port", 8081))
    rclpy.init()
    node = RobotBridge(args)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
