#!/usr/bin/env python3
"""Manager: ROS in, world-frame traj buffer, Dora fan-out (pdef / map / track)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import math

import numpy as np
import rclpy
from dora import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node as RosNode
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import ColorRGBA, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from hound_mapping.msg import LocalMap

from hound_nav.deps_path import setup_dependency_paths
from hound_nav.pdef_buffer import LocalMapSnapshot, PDefBuffer
from hound_nav.pdef_codec import pack_map, pack_pdef, pack_track, unpack_plan
from hound_nav.planner_viz import PlannerVisFrame, PlannerVisWorker
from hound_nav.traj_buffer import TrajBuffer
from hound_nav.utils import update_goal


def _img_f32(msg) -> Optional[np.ndarray]:
    if msg.width == 0 or msg.height == 0 or not msg.data:
        return None
    h, w = int(msg.height), int(msg.width)
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
    if msg.encoding == "32FC3":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w, 3).copy()
    return None


class ManagerRos(RosNode):
    def __init__(self, buffer: PDefBuffer, launch_cfg: Dict[str, Any]) -> None:
        super().__init__("hound_nav_manager")
        self._buffer = buffer
        self._state_dims = int(launch_cfg.get("control_state_dims", buffer.state_dims))
        local_map_topic = str(
            launch_cfg.get("local_map_topic", "/hound_mapping/local_map")
        )
        state_topic = str(
            launch_cfg.get("state_topic", "/hound_fcu_control/control_state")
        )
        path_topic = str(launch_cfg.get("path_topic", "/mission/path"))
        goal_topic = str(launch_cfg.get("goal_topic", "/goal_pose"))
        plan_topic = str(launch_cfg.get("plan_topic", "/hound_nav/local_plan"))
        plan_markers_topic = str(
            launch_cfg.get("plan_markers_topic", "/hound_nav/local_plan_arrows")
        )
        qos = qos_profile_sensor_data
        self.create_subscription(
            LocalMap, local_map_topic, self._on_local_map, qos_profile_sensor_data
        )
        self.create_subscription(
            Float64MultiArray, state_topic, self._on_control_state, qos
        )
        self.create_subscription(Path, path_topic, self._on_path, 1)
        # RViz "2D Goal Pose" (PoseStamped, reliable). Each click = one-WP mission.
        self.create_subscription(PoseStamped, goal_topic, self._on_goal_pose, 10)
        self._plan_pub = self.create_publisher(Path, plan_topic, 1)
        self._plan_marker_pub = self.create_publisher(
            MarkerArray, plan_markers_topic, 1
        )
        self._plan_frame = "odom"
        self.get_logger().info(
            f"manager ROS: map={local_map_topic} state={state_topic} "
            f"dims={self._state_dims} wps={path_topic} goal={goal_topic} "
            f"plan={plan_topic} arrows={plan_markers_topic}"
        )

    def _on_local_map(self, msg: LocalMap) -> None:
        elev = _img_f32(msg.elevation)
        cost = _img_f32(msg.costmap)
        if elev is None or cost is None or elev.shape != cost.shape:
            return
        self._buffer.set_local_map(
            LocalMapSnapshot(
                elevation=elev,
                cost=cost,
                origin_x=float(msg.info.origin.position.x),
                origin_y=float(msg.info.origin.position.y),
                resolution=float(msg.info.resolution),
                stamp_sec=float(msg.header.stamp.sec)
                + 1e-9 * float(msg.header.stamp.nanosec),
            )
        )

    def _on_control_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < self._state_dims:
            return
        self._buffer.set_state_vector(
            np.asarray(msg.data[: self._state_dims], dtype=np.float64),
            stamp_sec=self.get_clock().now().nanoseconds * 1e-9,
        )

    def _on_path(self, msg: Path) -> None:
        if not msg.poses:
            return
        wp = np.array(
            [[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses],
            dtype=np.float64,
        )
        self._buffer.set_waypoints(wp)
        frame = str(msg.header.frame_id or "")
        print(
            f"[hound_manager] mission path: n={len(wp)} "
            f"frame={frame!r} first=({wp[0, 0]:.2f},{wp[0, 1]:.2f}) "
            f"last=({wp[-1, 0]:.2f},{wp[-1, 1]:.2f})",
            flush=True,
        )

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        """RViz 2D Goal Pose: replace mission with this one waypoint."""
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        self._buffer.set_waypoints(np.array([[x, y]], dtype=np.float64))
        frame = str(msg.header.frame_id or "")
        print(
            f"[hound_manager] rviz goal: ({x:.2f},{y:.2f}) frame={frame!r}",
            flush=True,
        )

    def publish_plan(self, path: np.ndarray) -> None:
        """Path: x,y,yaw, stamp=t0+|g|, z=g*time_direction. Arrows: green fwd, blue back."""
        arr = np.asarray(path, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1:
            return
        now = self.get_clock().now()
        frame = self._plan_frame
        out = Path()
        out.header.stamp = now.to_msg()
        out.header.frame_id = frame
        markers = MarkerArray()
        wipe = Marker()
        wipe.header.stamp = out.header.stamp
        wipe.header.frame_id = frame
        wipe.ns = "local_plan"
        wipe.action = Marker.DELETEALL
        markers.markers.append(wipe)
        n = int(arr.shape[0])
        for i, row in enumerate(arr):
            x = float(row[0])
            y = float(row[1])
            yaw = float(row[2]) if row.size > 2 else 0.0
            signed_t = float(row[-1]) if row.size > 4 else float(i)
            t_abs = abs(signed_t)
            fwd = signed_t >= 0.0
            travel_yaw = yaw if fwd else yaw + math.pi
            stamp = (now + Duration(seconds=float(t_abs))).to_msg()
            q = _yaw_to_quat(yaw)
            q_travel = _yaw_to_quat(travel_yaw)
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = frame
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = signed_t
            ps.pose.orientation = q
            out.poses.append(ps)
            if i + 1 < n:
                dx = float(arr[i + 1, 0]) - x
                dy = float(arr[i + 1, 1]) - y
                step = math.hypot(dx, dy)
            else:
                step = 0.25
            shaft = min(0.55, max(0.18, step * 0.85))
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame
            m.ns = "local_plan"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.12
            m.pose.orientation = q_travel
            m.scale.x = shaft
            m.scale.y = 0.07
            m.scale.z = 0.07
            if fwd:
                m.color = ColorRGBA(r=0.05, g=0.85, b=0.15, a=1.0)
            else:
                m.color = ColorRGBA(r=0.15, g=0.40, b=1.0, a=1.0)
            markers.markers.append(m)
        self._plan_pub.publish(out)
        self._plan_marker_pub.publish(markers)


def _yaw_to_quat(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


def main() -> None:
    setup_dependency_paths()

    from hound_nav.nav_config import as_float, load_hound_nav_config, stack_config

    launch_cfg = load_hound_nav_config()
    Config = stack_config(launch_cfg)
    timesteps = int(Config["MPPI_config"]["TIMESTEPS"])
    lookahead = as_float(Config["lookahead"])
    wp_radius = as_float(Config["wp_radius"])
    planner_cfg = Config["Planner_config"]
    hysteresis = int(planner_cfg["experiment_info_default"]["hysteresis"])
    default_expansion_limit = int(
        planner_cfg["experiment_info_default"]["max_expansions"]
    )
    expansion_limit = default_expansion_limit
    unstuck = int(planner_cfg["unstuck_expansions"])

    buffer = PDefBuffer(state_dims=int(launch_cfg.get("control_state_dims", 17)))
    traj = TrajBuffer()
    rclpy.init()
    ros = ManagerRos(buffer, launch_cfg)
    executor = SingleThreadedExecutor()
    executor.add_node(ros)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    dora = Node()
    last_map_gen_sent = -1
    query_outstanding = False
    query_sent_t = 0.0
    query_timeout_s = 5.0
    last_pdef_t = 0.0
    planner_hz = float(launch_cfg.get("planner_hz", 5.0))
    pdef_period = 1.0 / max(planner_hz, 0.1)
    track_ref_metric = str(launch_cfg.get("track_ref_metric", "screw")).lower()
    screw_length_m = float(launch_cfg.get("screw_length_m", 1.0))
    goal = None
    current_wp_index = 0
    first_path = False
    goal_reached = False
    last_track_stamp = 0.0
    last_wp_gen = -1
    last_vis_path: Optional[np.ndarray] = None
    last_vis_ok = False
    last_vis_exp = 0
    # Snapshot of last pdef sent → OpenCV viz (background thread).
    last_query: Optional[Dict[str, Any]] = None
    planner_cv_viz = bool(launch_cfg.get("planner_cv_viz", True))
    vis = PlannerVisWorker(
        enabled=planner_cv_viz,
        window_name=str(launch_cfg.get("planner_cv_viz_window", "hound_planner_vis")),
        map_size=int(launch_cfg.get("planner_cv_viz_size", 480)),
    )

    print(
        f"[hound_manager] up T={timesteps} exp={expansion_limit} "
        f"track_ref={track_ref_metric} L={screw_length_m:.3f} "
        f"cv_viz={planner_cv_viz} "
        "(plan start = live robot, LocalMap native grid)",
        flush=True,
    )

    def _push_vis(
        *,
        path: Optional[np.ndarray],
        ok: bool,
        expansions: int,
        note: str = "",
        query: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not planner_cv_viz:
            return
        q = query if query is not None else last_query
        if q is None:
            return
        live = buffer.get_state_copy()
        if live is not None and live.size >= 2:
            yaw = float(live[5]) if live.size > 5 else 0.0
            state_xy_yaw = np.array(
                [float(live[0]), float(live[1]), yaw], dtype=np.float64
            )
        else:
            start = np.asarray(q["start"], dtype=np.float64).reshape(-1)
            yaw = float(start[5]) if start.size > 5 else 0.0
            state_xy_yaw = np.array(
                [float(start[0]), float(start[1]), yaw], dtype=np.float64
            )
        vis.update(
            PlannerVisFrame(
                costmap=np.asarray(q["costmap"], dtype=np.float32),
                height=np.asarray(q["height"], dtype=np.float32),
                path=None if path is None else np.asarray(path, dtype=np.float64),
                goal_xy=np.asarray(q["goal"], dtype=np.float64).reshape(-1)[:2],
                state_xy_yaw=state_xy_yaw,
                map_center=np.asarray(q["map_center"], dtype=np.float64).reshape(-1)[
                    :2
                ],
                map_res=float(q["map_res"]),
                wp_radius=float(wp_radius),
                ok=bool(ok),
                expansions=int(expansions),
                note=str(note),
            )
        )

    def _maybe_send_map(dora_node: Node) -> None:
        nonlocal last_map_gen_sent
        gen = buffer.map_generation()
        if gen == last_map_gen_sent or not buffer.ready():
            return
        pdef = buffer.snapshot_pdef()
        if pdef is None:
            return
        arr, meta = pack_map(
            pdef.costmap,
            pdef.height_bev,
            pdef.normal_bev,
            map_center=pdef.map_center,
            map_res=pdef.map_res,
            map_gen=gen,
        )
        dora_node.send_output("map", arr, meta)
        last_map_gen_sent = gen

    def _maybe_send_pdef(dora_node: Node) -> None:
        nonlocal query_outstanding, query_sent_t, goal, current_wp_index
        nonlocal expansion_limit, goal_reached, last_pdef_t, last_wp_gen
        nonlocal last_query
        now = time.perf_counter()
        if (now - last_pdef_t) < pdef_period:
            return
        if query_outstanding:
            if time.perf_counter() - query_sent_t > query_timeout_s:
                query_outstanding = False
            else:
                return
        if not buffer.ready():
            return
        wp_gen = buffer.waypoint_generation()
        if wp_gen != last_wp_gen:
            last_wp_gen = wp_gen
            goal = None
            current_wp_index = 0
            goal_reached = False
            print(
                f"[hound_manager] new mission (gen={wp_gen}); reset goal index",
                flush=True,
            )
        pdef = buffer.snapshot_pdef()
        if pdef is None or pdef.target_wp is None or len(pdef.target_wp) < 1:
            return
        state = pdef.state
        pos = np.copy(state[:2])
        goal, success, current_wp_index = update_goal(
            goal,
            pos,
            pdef.target_wp,
            current_wp_index,
            lookahead,
            wp_radius=wp_radius,
        )
        goal_reached = bool(success)
        if goal is None or goal_reached:
            if goal_reached and (now - last_pdef_t) >= pdef_period:
                # Rate-limit so a too-large wp_radius does not spam.
                print(
                    f"[hound_manager] skip plan: within wp_radius of goal "
                    f"(idx={current_wp_index})",
                    flush=True,
                )
                last_pdef_t = now
            return
        start = np.copy(state)
        query_t = time.perf_counter()
        goal_arr = np.asarray(goal, dtype=np.float64)
        last_query = {
            "costmap": np.copy(pdef.costmap),
            "height": np.copy(pdef.height_bev),
            "map_center": np.copy(pdef.map_center),
            "map_res": float(pdef.map_res),
            "start": np.copy(start),
            "goal": np.copy(goal_arr),
        }
        _push_vis(path=None, ok=False, expansions=0, note="query")
        arr, meta = pack_pdef(
            pdef.costmap,
            pdef.height_bev,
            map_center=pdef.map_center,
            map_res=pdef.map_res,
            start=start,
            goal=goal_arr,
            hysteresis=hysteresis,
            expansion_limit=expansion_limit,
            query_t=query_t,
        )
        dora_node.send_output("pdef", arr, meta)
        query_outstanding = True
        query_sent_t = query_t
        last_pdef_t = now
        print(
            f"[hound_manager] pdef sent exp={expansion_limit} "
            f"goal=({float(goal[0]):.2f},{float(goal[1]):.2f})",
            flush=True,
        )

    def _maybe_send_track(dora_node: Node) -> None:
        nonlocal last_track_stamp
        state = buffer.get_state_copy()
        if state is None:
            return
        stamp = buffer.get_state_stamp_sec()
        if stamp <= last_track_stamp:
            return
        last_track_stamp = stamp
        world = traj.get()
        if world is None:
            horizon = np.zeros((timesteps, 4), dtype=np.float64)
            hold = True
        else:
            yaw = float(state[5]) if state.shape[0] > 5 else 0.0
            horizon, stop = traj.horizon(
                state[:2],
                timesteps,
                yaw=yaw,
                metric=track_ref_metric,
                screw_length_m=screw_length_m,
            )
            if horizon is None:
                horizon = np.zeros((timesteps, 4), dtype=np.float64)
                hold = True
            else:
                hold = bool(stop or goal_reached)
        arr, meta = pack_track(state, horizon, goal_reached=hold)
        dora_node.send_output("track", arr, meta)
        if last_query is not None:
            _push_vis(
                path=last_vis_path,
                ok=last_vis_ok,
                expansions=last_vis_exp,
            )

    try:
        for event in dora:
            if event is None or event.get("type") == "STOP":
                break
            if event.get("type") != "INPUT":
                continue
            eid = event.get("id")
            if eid == "plan":
                path, ok, expansions, query_t = unpack_plan(
                    event.get("value"), event.get("metadata") or {}
                )
                query_outstanding = False
                last_vis_ok = bool(ok)
                last_vis_exp = int(expansions)
                last_vis_path = path if ok else None
                _push_vis(
                    path=last_vis_path,
                    ok=last_vis_ok,
                    expansions=last_vis_exp,
                    note="plan",
                )
                if ok and path is not None:
                    traj.replace(path)
                    first_path = True
                    expansion_limit = default_expansion_limit
                    ros.publish_plan(path)
                elif not first_path:
                    expansion_limit = min(expansion_limit * 2, unstuck)
                continue
            if eid == "tick":
                _maybe_send_map(dora)
                _maybe_send_pdef(dora)
                _maybe_send_track(dora)
    finally:
        vis.stop()
        ros.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
