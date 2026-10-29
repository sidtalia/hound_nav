#!/usr/bin/env python3
"""Feed hound_nav from the robot's navigation contract, and lease the base.

Subscribes to /costmap, /map, /odom and TF (map → odom composed with odom → base_link).
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


def apply_lethal_mask(cost, ox, oy, res, mask, mask_ox, mask_oy, mask_res):
    """AND a 1-free / 0-lethal mask onto an IGHA crop. Both grids are map-frame.

    Out-of-mask cells stay as they are (the author treats those as free).
    A lethal mask cell forces the crop cell to 0. A free mask cell leaves it.
    """
    n = cost.shape[0]
    cols = ox + (np.arange(n) + 0.5) * res
    rows = oy + (np.arange(n) + 0.5) * res
    ic = np.floor((cols - mask_ox) / mask_res).astype(np.int32)
    ir = np.floor((rows - mask_oy) / mask_res).astype(np.int32)
    cc, rr = np.meshgrid(ic, ir, indexing="xy")
    h, w = mask.shape
    inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    blocked = np.zeros(cost.shape, dtype=bool)
    blocked[inside] = mask[rr[inside], cc[inside]] <= 0.5
    out = np.array(cost, dtype=np.float32, copy=True)
    out[blocked] = 0.0
    return out


def load_lethal_mask(path: str):
    """Return (mask, meta) or None. Mask is float32, 1 = free, 0 = lethal."""
    path = str(path or "").strip()
    if not path or not os.path.isfile(path):
        return None
    data = yaml.safe_load(open(path, encoding="utf-8")) or {}
    # Aligned floor-plan view yaml: same file the Viser picture uses.
    if "image" in data and "width_m" in data:
        return floor_plan_as_lethal(path, data)
    name = str(data.get("lethal_f32") or "")
    w = int(data.get("width") or 0)
    h = int(data.get("height") or 0)
    res = float(data.get("resolution") or 0.0)
    if not name or w < 1 or h < 1 or res <= 0.0:
        return None
    bin_path = name if os.path.isfile(name) else os.path.join(os.path.dirname(path), name)
    if not os.path.isfile(bin_path):
        return None
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size != w * h:
        return None
    meta = {
        "frame_id": str(data.get("frame_id") or "map"),
        "origin_x": float(data.get("origin_x") or 0.0),
        "origin_y": float(data.get("origin_y") or 0.0),
        "resolution": res,
    }
    return arr.reshape(h, w), meta


def floor_plan_as_lethal(path: str, data: dict | None = None, res: float = 0.05):
    """Rasterize the aligned plan into a map-frame mask. Ink = lethal, paper = free.

    Same pose as the Viser floor picture: centre, yaw, width_m, height_m. Image row 0
    is min Y in the page frame (plan_align flipud). Outside the page is free.
    """
    path = str(path or "").strip()
    if not path or not os.path.isfile(path):
        return None
    if data is None:
        data = yaml.safe_load(open(path, encoding="utf-8")) or {}
    name = str(data.get("image") or "")
    img_path = name if os.path.isfile(name) else os.path.join(os.path.dirname(path), name)
    if not name or not os.path.isfile(img_path):
        return None
    img = np.load(img_path)
    if img.ndim != 3 or img.shape[0] < 2 or img.shape[1] < 2:
        return None
    # plan_align: white paper → alpha 0; ink → alpha 230.
    if img.shape[2] >= 4:
        ink = img[:, :, 3] > 0
    else:
        ink = img[:, :, :3].min(axis=2) < 245
    h_img, w_img = ink.shape
    width_m = float(data["width_m"])
    height_m = float(data["height_m"])
    cx, cy = (float(v) for v in data["center"])
    phi = float(data["yaw"])
    c, s = math.cos(phi), math.sin(phi)
    # Page corners in the map → axis-aligned bounds for the raster.
    corners = np.array(
        [
            [-0.5 * width_m, -0.5 * height_m],
            [0.5 * width_m, -0.5 * height_m],
            [0.5 * width_m, 0.5 * height_m],
            [-0.5 * width_m, 0.5 * height_m],
        ],
        dtype=np.float64,
    )
    world = corners @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
    ox = float(np.floor(world[:, 0].min() / res) * res)
    oy = float(np.floor(world[:, 1].min() / res) * res)
    x1 = float(np.ceil(world[:, 0].max() / res) * res)
    y1 = float(np.ceil(world[:, 1].max() / res) * res)
    nw = max(1, int(round((x1 - ox) / res)))
    nh = max(1, int(round((y1 - oy) / res)))
    xs = ox + (np.arange(nw) + 0.5) * res
    ys = oy + (np.arange(nh) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    dx, dy = xx - cx, yy - cy
    # Inverse of R_z(phi): page-local x right, y toward page +Y (row increases).
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    col = np.floor((local_x + 0.5 * width_m) / width_m * w_img).astype(np.int32)
    row = np.floor((local_y + 0.5 * height_m) / height_m * h_img).astype(np.int32)
    inside = (col >= 0) & (col < w_img) & (row >= 0) & (row < h_img)
    mask = np.ones((nh, nw), dtype=np.float32)
    hit = np.zeros((nh, nw), dtype=bool)
    hit[inside] = ink[row[inside], col[inside]]
    mask[hit] = 0.0
    meta = {
        "frame_id": "map",
        "origin_x": ox,
        "origin_y": oy,
        "resolution": float(res),
    }
    return mask, meta


def mission_slug(name: str) -> str:
    """Safe file stem for a mission name."""
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name).strip())
    return s.strip("_") or "mission"


def save_mission_yaml(path: str, waypoints, frame_id: str = "map") -> None:
    """Write map-frame waypoints. Same shape as mission_pose_builder."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "frame_id": str(frame_id),
        "waypoints": [
            {
                "x": float(wp[0]),
                "y": float(wp[1]),
                "z": 0.0,
                "yaw": float(wp[2]),
            }
            for wp in waypoints
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def load_mission_yaml(path: str):
    """Return (frame_id, [(x, y, yaw), ...]) or None."""
    if not path or not os.path.isfile(path):
        return None
    data = yaml.safe_load(open(path, encoding="utf-8")) or {}
    frame = str(data.get("frame_id") or "map")
    out = []
    for wp in data.get("waypoints") or []:
        if isinstance(wp, dict):
            out.append(
                (float(wp.get("x", 0.0)), float(wp.get("y", 0.0)), float(wp.get("yaw", 0.0)))
            )
        elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
            yaw = float(wp[3]) if len(wp) > 3 else 0.0
            out.append((float(wp[0]), float(wp[1]), yaw))
    return frame, out


def crop_grid(grid: np.ndarray, ox: float, oy: float, res: float,
              x: float, y: float, size_m: float, out_res: float, only_100: bool):
    """Robot-centered square. Row 0 is min Y, same as the occupancy grid and LocalMap.

    The planner blocks a cell at or below 250, and this crop stores that as 0.
    Free is 255. Off the grid stays 0. Occupancy / Nav2 cost 0 is free; every
    other value is blocked — for /costmap that includes inflated cells (1..99)
    and lethal (100). ``only_100`` is unused; kept so callers do not change.
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
    # Nav2 costmap: 0 free, 1..99 inflated, 100 lethal. Same rule for /map.
    out[ok & (raw == 0)] = 255.0
    return out, origin_x, origin_y


def cost_picture(grid: np.ndarray, ox: float, oy: float, res: float,
                 x: float, y: float, size_m: float, out_res: float):
    """The costmap drawn over a floor plan. Row 0 is min Y, same as crop_grid.

    Free and off-map cells are clear. Lethal (occupancy 100) is red, inflation is orange.
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
    rgba = np.zeros((n, n, 4), np.uint8)
    lethal = ok & (raw == 100)
    infl = ok & (raw > 0) & (raw < 100)
    rgba[lethal] = (170, 25, 25, 230)
    if infl.any():
        rgba[infl, 0] = 210
        rgba[infl, 1] = 100
        rgba[infl, 2] = 30
        rgba[infl, 3] = np.clip(raw[infl].astype(np.int32) * 2, 50, 190).astype(np.uint8)
    return rgba, origin_x, origin_y


class RobotBridge(Node):
    def __init__(self, args) -> None:
        super().__init__("hound_robot_bridge")
        self._owner = args.owner
        self._map_size = float(args.map_size)
        self._map_res = float(args.map_res)
        self._grids = {}
        self._logged_grids = set()
        self._display = None
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
        self._marker_save_pending = False
        self._marker = None  # x, y, z, qx, qy, qz, qw in the map
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
        self._missions_dir = str(getattr(args, "missions_dir", "") or "")
        self._mission_wp_radius = float(getattr(args, "mission_wp_radius", 0.6))
        self._mission_wps = []  # (x, y, yaw) in map
        self._mission_idx = 0
        self._mission_running = False
        self._mission_name = ""
        self._mission_status = None
        self._tf = Buffer()
        TransformListener(self._tf, self, spin_thread=False)
        # The aligned floor plan is the lethal mask (ink blocks). A separate
        # lethal_f32 file still loads if that path is set and is not a view yaml.
        lethal_path = str(getattr(args, "lethal_map", "") or "")
        floor_path = str(getattr(args, "floor_plan", "") or "")
        self._lethal = load_lethal_mask(lethal_path) if lethal_path else None
        if self._lethal is None and floor_path:
            self._lethal = floor_plan_as_lethal(floor_path, res=float(self._map_res))
            lethal_path = floor_path
        if self._lethal is None:
            self.get_logger().info(f"lethal overlay off ({lethal_path or 'empty path'})")
        else:
            mask, meta = self._lethal
            self.get_logger().info(
                f"lethal overlay {mask.shape[1]}x{mask.shape[0]} "
                f"frame={meta['frame_id']} res={meta['resolution']:.3f} "
                f"origin=({meta['origin_x']:.2f},{meta['origin_y']:.2f}) "
                f"from {lethal_path}"
            )

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        for topic in ("/costmap", "/map"):
            self.create_subscription(
                OccupancyGrid,
                topic,
                lambda msg, topic=topic: self._on_grid(topic, msg),
                latched,
            )
        self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        self.create_subscription(String, "/base/owner", self._on_owner, latched)
        self.create_subscription(
            Twist, args.cmd_in, self._on_cmd, qos_profile_sensor_data
        )
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._mission_path_pub = self.create_publisher(
            Path, "/hound_nav/mission/waypoints", latched
        )
        self._init_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self._save_marker_pub = self.create_publisher(
            Empty, "/aruco_registration/save_marker", 10
        )
        self.create_subscription(
            PoseStamped,
            "/aruco_registration/marker_in_map",
            self._on_marker,
            latched,
        )
        self._cancel_pub = self.create_publisher(Empty, "/hound_nav/cancel", 10)
        self.create_subscription(Path, "/hound_nav/local_plan", self._on_plan, 1)
        self.create_subscription(MarkerArray, "/hound_nav/mppi_topk", self._on_mppi, 1)
        self._floor_plan = str(getattr(args, "floor_plan", "") or "")
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

    def _on_grid(self, topic: str, msg: OccupancyGrid) -> None:
        w = int(msg.info.width)
        h = int(msg.info.height)
        raw = np.asarray(msg.data, dtype=np.int16)
        if raw.size != w * h or w < 1 or h < 1:
            self.get_logger().warn(f"{topic} ignored: {w}x{h} data={raw.size}")
            return
        self._grids[topic] = (raw.reshape(h, w), msg.info)
        if topic not in self._logged_grids:
            self._logged_grids.add(topic)
            self.get_logger().info(
                f"{topic} {w}x{h} res={msg.info.resolution:.3f} "
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
        self._mission_tick()
        self._publish_goal()

    def _crop(self, topic: str, pose, only_100: bool):
        item = self._grids.get(topic)
        if item is None:
            return None
        grid, info = item
        return crop_grid(
            grid,
            float(info.origin.position.x),
            float(info.origin.position.y),
            float(info.resolution),
            pose[0],
            pose[1],
            self._map_size,
            self._map_res,
            only_100,
        )

    def _publish_map(self) -> None:
        pose = self._pose()
        if pose is None:
            return
        # Same crop LocalMap publishes: /costmap (not free → blocked), then AND the plan.
        planner = None
        cost = self._crop("/costmap", pose, True)
        if cost is not None:
            grid, ox, oy = cost
            grid = self._and_lethal(grid, ox, oy)
            stamp = self.get_clock().now().to_msg()
            self._send_local((grid, ox, oy), stamp)
            planner = (grid, ox, oy)
        # Dropdown only changes the picture. The planner always gets `planner`.
        src = "/costmap" if self._display is None else str(self._display.value)
        if src == "planner":
            if planner is not None:
                grid, ox, oy = planner
                gray = np.repeat(
                    np.clip(grid, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2
                )
                rgba = np.dstack([gray, np.full(gray.shape[:2], 255, np.uint8)])
                self._show_map(rgba, ox, oy)
        elif src == "/costmap":
            item = self._grids.get("/costmap")
            if item is not None:
                grid, info = item
                self._show_map(*cost_picture(
                    grid, float(info.origin.position.x), float(info.origin.position.y),
                    float(info.resolution), pose[0], pose[1], self._map_size, self._map_res,
                ))
        else:
            view = self._crop("/map", pose, False)
            if view is not None:
                grid, ox, oy = view
                gray = np.repeat(np.clip(grid, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
                rgba = np.dstack([gray, np.full(gray.shape[:2], 255, np.uint8)])
                self._show_map(rgba, ox, oy)

    def _and_lethal(self, cost, ox, oy):
        if self._lethal is None:
            return cost
        mask, meta = self._lethal
        if meta["frame_id"] != "map":
            return cost
        return apply_lethal_mask(
            cost, ox, oy, self._map_res,
            mask, meta["origin_x"], meta["origin_y"], meta["resolution"],
        )

    def _send_local(self, cost_origin, stamp) -> None:
        cost, ox, oy = cost_origin
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
        self._marker_h = server.scene.add_box(
            "/aruco",
            dimensions=(0.18, 0.18, 0.01),
            color=(170, 40, 220),
            side="double",
            position=(0.0, 0.0, 0.0),
            visible=False,
        )
        self._save_marker_btn = server.gui.add_button("save marker pose")
        self._save_marker_btn.on_click(lambda _ev: self._request_save_marker())
        self._display = server.gui.add_dropdown(
            "display",
            options=("/costmap", "/map", "planner"),
            initial_value="/costmap",
        )
        self._edit_mission = server.gui.add_checkbox(
            "edit mission", initial_value=False
        )
        self._mission_name_gui = server.gui.add_text(
            "mission name", initial_value="idle"
        )
        self._mission_loop = server.gui.add_checkbox("loop mission", initial_value=True)
        self._mission_status = server.gui.add_markdown(self._mission_status_text())
        save_m = server.gui.add_button("save mission")
        save_m.on_click(lambda _ev: self._mission_save())
        load_m = server.gui.add_button("load mission")
        load_m.on_click(lambda _ev: self._mission_load())
        undo_m = server.gui.add_button("undo waypoint")
        undo_m.on_click(lambda _ev: self._mission_undo())
        clear_m = server.gui.add_button("clear mission")
        clear_m.on_click(lambda _ev: self._mission_clear())
        run_m = server.gui.add_button("run mission")
        run_m.on_click(lambda _ev: self._mission_run())
        stop_m = server.gui.add_button("stop mission")
        stop_m.on_click(lambda _ev: self._mission_stop())
        server.gui.add_markdown(
            "check **set goal**, **set initial pose**, or **edit mission**, then drag. "
            "the press is the position and the drag direction is the heading. "
            "**edit mission** appends waypoints (cyan). Name them, **save** / **load**, "
            "**run** (loops if checked). Reach radius "
            f"{self._mission_wp_radius:.2f} m.\n\n"
            "**display** changes the picture only. The planner always gets "
            "the **planner** view (costmap + floor-plan ink).\n\n"
            "blue = robot · purple = ArUco · green = plan · orange = MPPI · "
            "cyan = mission. **cancel goal** stops the controller path."
        )
        self._server = server
        self._add_floor_plan(str(getattr(self, "_floor_plan", "") or ""))
        if self._missions_dir:
            os.makedirs(self._missions_dir, exist_ok=True)
            self.get_logger().info(f"missions dir {self._missions_dir}")
        self.get_logger().info(f"viser http://0.0.0.0:{port}")

    def _add_floor_plan(self, path: str) -> None:
        """The aligned page, under the costmap. Same image convention as plan_align.py."""
        if not path or not os.path.isfile(path):
            self.get_logger().info(f"floor plan off ({path or 'empty path'})")
            return
        meta = yaml.safe_load(open(path))
        image = np.load(os.path.join(os.path.dirname(path), meta["image"]))
        phi = float(meta["yaw"])
        cx, cy = meta["center"]
        self._server.scene.add_image(
            "/floor",
            image,
            float(meta["width_m"]),
            float(meta["height_m"]),
            format="png",
            cast_shadow=False,
            receive_shadow=False,
            position=(float(cx), float(cy), 0.0),
            wxyz=(math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2)),
        )
        self.get_logger().info(
            f"floor plan {meta['width_m']:.0f}x{meta['height_m']:.0f} m "
            f"centre=({float(cx):.1f},{float(cy):.1f}) yaw={math.degrees(phi):.1f} deg"
        )

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
        edit = bool(self._edit_mission.value)
        with self._lock:
            self._goal_vis = pose
            if ev.phase != "end":
                return
            if init:
                self._init_pending = pose
            elif edit:
                self._mission_wps.append(pose)
                self._mission_running = False
            else:
                self._goal_committed = pose
                self._goal_pending = pose
        if edit and ev.phase == "end":
            self._mission_publish_path()
            self._mission_refresh_status()
            self.get_logger().info(
                f"mission WP {len(self._mission_wps)}: "
                f"({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f} deg)"
            )

    def _mission_status_text(self) -> str:
        names = []
        if self._missions_dir and os.path.isdir(self._missions_dir):
            names = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(self._missions_dir)
                if f.endswith(".yaml")
            )
        n = len(self._mission_wps)
        run = "running" if self._mission_running else "stopped"
        listed = ", ".join(names) if names else "(none saved)"
        return (
            f"**mission** {n} WP · {run} · idx {self._mission_idx} · "
            f"saved: {listed}"
        )

    def _mission_refresh_status(self) -> None:
        if self._mission_status is not None:
            self._mission_status.content = self._mission_status_text()

    def _mission_path(self, name: str) -> str:
        return os.path.join(self._missions_dir, f"{mission_slug(name)}.yaml")

    def _mission_save(self) -> None:
        name = str(self._mission_name_gui.value).strip() or "idle"
        if not self._missions_dir:
            self.get_logger().error("missions_dir not set")
            return
        with self._lock:
            wps = list(self._mission_wps)
        if not wps:
            self.get_logger().warn("empty mission — not saving")
            return
        path = self._mission_path(name)
        save_mission_yaml(path, wps)
        self._mission_name = mission_slug(name)
        self._mission_refresh_status()
        self.get_logger().info(f"saved mission {self._mission_name} ({len(wps)} WP) → {path}")

    def _mission_load(self) -> None:
        name = str(self._mission_name_gui.value).strip()
        if not name or not self._missions_dir:
            return
        path = self._mission_path(name)
        loaded = load_mission_yaml(path)
        if loaded is None or not loaded[1]:
            self.get_logger().error(f"mission not found: {path}")
            return
        _frame, wps = loaded
        with self._lock:
            self._mission_wps = list(wps)
            self._mission_idx = 0
            self._mission_running = False
            self._mission_name = mission_slug(name)
        self._mission_publish_path()
        self._mission_refresh_status()
        self.get_logger().info(f"loaded mission {self._mission_name} ({len(wps)} WP)")

    def _mission_undo(self) -> None:
        with self._lock:
            if self._mission_wps:
                self._mission_wps.pop()
            self._mission_running = False
            self._mission_idx = min(self._mission_idx, max(0, len(self._mission_wps) - 1))
        self._mission_publish_path()
        self._mission_refresh_status()

    def _mission_clear(self) -> None:
        with self._lock:
            self._mission_wps.clear()
            self._mission_idx = 0
            self._mission_running = False
        self._mission_publish_path()
        self._mission_refresh_status()

    def _mission_run(self) -> None:
        with self._lock:
            if not self._mission_wps:
                self.get_logger().warn("no waypoints to run")
                return
            self._mission_idx = 0
            self._mission_running = True
            pose = self._mission_wps[0]
            self._goal_pending = pose
            self._goal_committed = pose
            self._goal_vis = pose
        self._edit_mission.value = False
        self._set_goal.value = False
        self._mission_refresh_status()
        self.get_logger().info(
            f"mission run ({len(self._mission_wps)} WP, loop={bool(self._mission_loop.value)})"
        )

    def _mission_stop(self) -> None:
        with self._lock:
            self._mission_running = False
            self._cancel_pending = True
        self._mission_refresh_status()
        self.get_logger().info("mission stopped")

    def _mission_publish_path(self) -> None:
        with self._lock:
            wps = list(self._mission_wps)
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        for x, y, yaw in wps:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            qw, qx, qy, qz = yaw_quat(float(yaw))
            ps.pose.orientation.w = qw
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            path.poses.append(ps)
        self._mission_path_pub.publish(path)

    def _mission_tick(self) -> None:
        """Advance when close to the current WP; republish the active goal."""
        with self._lock:
            if not self._mission_running or not self._mission_wps:
                return
            if self._vis_pose is None:
                return
            x, y, _yaw = self._vis_pose
            idx = self._mission_idx
            wps = list(self._mission_wps)
            loop = bool(self._mission_loop.value)
        wx, wy, wyaw = wps[idx]
        if math.hypot(x - wx, y - wy) > self._mission_wp_radius:
            return
        nxt = idx + 1
        if nxt >= len(wps):
            if not loop:
                with self._lock:
                    self._mission_running = False
                self._mission_refresh_status()
                self.get_logger().info("mission finished")
                return
            nxt = 0
        pose = wps[nxt]
        with self._lock:
            self._mission_idx = nxt
            self._goal_pending = pose
            self._goal_committed = pose
            self._goal_vis = pose
        self._mission_refresh_status()
        self.get_logger().info(f"mission advance → {nxt}/{len(wps) - 1}")

    def _on_marker(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        with self._lock:
            self._marker = (
                float(p.x), float(p.y), float(p.z),
                float(q.x), float(q.y), float(q.z), float(q.w),
            )

    def _request_save_marker(self) -> None:
        with self._lock:
            self._marker_save_pending = True

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
        edit_on = bool(self._edit_mission.value)
        on = [("goal", goal_on), ("init", init_on), ("edit", edit_on)]
        active = [k for k, v in on if v]
        if len(active) > 1:
            # Keep the mode that was not already selected; drop the old one.
            keep = active[-1] if self._pose_mode not in active else (
                active[0] if active[-1] == self._pose_mode else active[-1]
            )
            self._set_goal.value = keep == "goal"
            self._set_init.value = keep == "init"
            self._edit_mission.value = keep == "edit"
            goal_on, init_on, edit_on = keep == "goal", keep == "init", keep == "edit"
        self._pose_mode = (
            "init" if init_on else "edit" if edit_on else "goal" if goal_on else "look"
        )
        want = goal_on or init_on or edit_on
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
            save_marker = self._marker_save_pending
            self._goal_pending = None
            self._init_pending = None
            self._cancel_pending = False
            self._marker_save_pending = False
        if cancel:
            self._cancel_pub.publish(Empty())
            self.get_logger().info("cancel goal")
        if save_marker:
            self._save_marker_pub.publish(Empty())
            self.get_logger().info("save marker pose")
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

    def _show_map(self, rgba: np.ndarray, ox: float, oy: float) -> None:
        if self._server is None:
            return
        side_x = float(rgba.shape[1]) * float(self._map_res)
        side_y = float(rgba.shape[0]) * float(self._map_res)
        center = (ox + 0.5 * side_x, oy + 0.5 * side_y, 0.02)
        if self._map_h is None:
            self._map_h = self._server.scene.add_image(
                "/map",
                rgba,
                side_x,
                side_y,
                format="png",
                cast_shadow=False,
                receive_shadow=False,
                position=center,
            )
        self._sync_drag()
        self._map_h.image = rgba
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
        with self._lock:
            mission = list(self._mission_wps)
            marker = None if self._marker is None else tuple(self._marker)
        if len(mission) >= 2:
            xy = np.array([[p[0], p[1]] for p in mission], np.float32)
            self._draw("/mission", _segs(xy, 0.14), (40, 200, 220), 5)
        elif len(mission) == 1:
            x0, y0, _ = mission[0]
            self._draw(
                "/mission",
                _segs(np.array([[x0, y0], [x0 + 0.01, y0]], np.float32), 0.14),
                (40, 200, 220),
                5,
            )
        else:
            self._draw("/mission", None, (40, 200, 220), 5)
        if marker is None or self._marker_h is None:
            if self._marker_h is not None:
                self._marker_h.visible = False
            self._draw("/marker_facing", None, (170, 40, 220), 6)
            self._draw("/marker_face", None, (240, 240, 255), 6)
        else:
            mx, my, mz, qx, qy, qz, qw = marker
            self._marker_h.position = (mx, my, mz)
            self._marker_h.wxyz = (qw, qx, qy, qz)
            self._marker_h.visible = True
            # +Z of the tag frame points out of the face. +X lies in the face.
            zx = 2.0 * (qx * qz + qy * qw)
            zy = 2.0 * (qy * qz - qx * qw)
            xx = 1.0 - 2.0 * (qy * qy + qz * qz)
            xy = 2.0 * (qx * qy + qz * qw)
            half = 0.09  # half of the 0.18 m tag, same as the plate
            self._draw(
                "/marker_facing",
                _segs(
                    np.array([[mx, my], [mx + 0.5 * zx, my + 0.5 * zy]], np.float32),
                    0.16,
                ),
                (170, 40, 220),
                6,
            )
            self._draw(
                "/marker_face",
                _segs(
                    np.array(
                        [[mx - half * xx, my - half * xy], [mx + half * xx, my + half * xy]],
                        np.float32,
                    ),
                    0.12,
                ),
                (240, 240, 255),
                8,
            )

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
    args.lethal_map = str(_nav(cfg, "lethal_map_path", "") or "")
    args.floor_plan = str(_nav(cfg, "floor_plan", "") or "")
    args.missions_dir = str(_nav(cfg, "missions_dir", "") or "")
    args.mission_wp_radius = float(_nav(cfg, "mission_wp_radius", 0.6))
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
