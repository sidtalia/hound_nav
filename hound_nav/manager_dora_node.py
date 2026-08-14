#!/usr/bin/env python3
"""Manager: ROS in, world-frame traj buffer, Dora fan-out (pdef / map / track)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import rclpy
from dora import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node as RosNode
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from hound_mapping.msg import LocalMap

from hound_nav.deps_path import setup_dependency_paths
from hound_nav.pdef_buffer import LocalMapSnapshot, PDefBuffer
from hound_nav.pdef_codec import pack_map, pack_pdef, pack_track, unpack_plan
from hound_nav.traj_buffer import TrajBuffer
from hound_nav.utils import path_pose_to_start_state, update_goal


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
        plan_topic = str(launch_cfg.get("plan_topic", "/hound_nav/local_plan"))
        qos = qos_profile_sensor_data
        self.create_subscription(LocalMap, local_map_topic, self._on_local_map, 1)
        self.create_subscription(
            Float64MultiArray, state_topic, self._on_control_state, qos
        )
        self.create_subscription(Path, path_topic, self._on_path, 1)
        self._plan_pub = self.create_publisher(Path, plan_topic, 1)
        self.get_logger().info(
            f"manager ROS: map={local_map_topic} state={state_topic} "
            f"dims={self._state_dims} wps={path_topic} viz={plan_topic}"
        )

    def _on_local_map(self, msg: LocalMap) -> None:
        elev = _img_f32(msg.elevation)
        cost = _img_f32(msg.costmap)
        if elev is None or cost is None or elev.shape != cost.shape:
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


def main() -> None:
    setup_dependency_paths()

    from hound_nav.nav_config import as_float, load_hound_nav_config, stack_config

    launch_cfg = load_hound_nav_config()
    Config = stack_config(launch_cfg)
    timesteps = int(Config["MPPI_config"]["TIMESTEPS"])
    lookahead = as_float(Config["lookahead"])
    wp_radius = as_float(Config["wp_radius"])
    planner_cfg = Config["Planner_config"]
    hysteresis = float(planner_cfg["experiment_info_default"]["hysteresis"])
    default_expansion_limit = int(
        planner_cfg["experiment_info_default"]["max_expansions"]
    )
    bidirectional = bool(
        planner_cfg["experiment_info_default"].get("bidirectional", False)
    )
    expansion_limit = (
        default_expansion_limit // 4 if bidirectional else default_expansion_limit
    )
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
    planning_margin_s = float(launch_cfg.get("planning_margin_s", 0.05))
    plan_start_max_ref_dist_m = float(
        launch_cfg.get("plan_start_max_ref_dist_m", 2.0)
    )
    # Planner traj is timed at Dynamics_config.dt (override via SSoT plan_traj_dt_s).
    plan_traj_dt_s = float(
        launch_cfg.get(
            "plan_traj_dt_s",
            float(Config.get("Dynamics_config", {}).get("dt", 0.05)),
        )
    )
    planner_dt = 0.0  # EMA of measured query→solution RTT (perf_counter)
    planner_dt_ema_alpha = 0.9
    goal = None
    current_wp_index = 0
    first_path = False
    goal_reached = False
    last_track_stamp = 0.0

    print(
        f"[hound_manager] up T={timesteps} exp={expansion_limit} "
        f"track_ref={track_ref_metric} L={screw_length_m:.3f} "
        f"plan_margin={planning_margin_s:.2f}s "
        f"traj_dt={plan_traj_dt_s:.3f}s "
        f"max_ref={plan_start_max_ref_dist_m:.2f}m "
        "(LocalMap native grid, no warp)",
        flush=True,
    )

    def _plan_start_horizon_s() -> float:
        return float(planner_dt) + float(planning_margin_s)

    def _planner_start_state(robot_state: np.ndarray) -> np.ndarray:
        """Robot state, or traj pose (planner_dt + margin) after closest ref."""
        yaw = float(robot_state[5]) if robot_state.shape[0] > 5 else 0.0
        if traj.empty() or traj.robot_too_far_from_ref(
            robot_state[:2],
            yaw,
            max_dist_m=plan_start_max_ref_dist_m,
            metric=track_ref_metric,
            screw_length_m=screw_length_m,
        ):
            return np.copy(robot_state)
        ahead = traj.pose_after(
            robot_state[:2],
            yaw,
            dt_ahead_s=_plan_start_horizon_s(),
            dt_s=plan_traj_dt_s,
            metric=track_ref_metric,
            screw_length_m=screw_length_m,
        )
        if ahead is None:
            return np.copy(robot_state)
        return path_pose_to_start_state(robot_state, ahead)

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
        nonlocal expansion_limit, goal_reached, last_pdef_t
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
            return
        start = _planner_start_state(state)
        query_t = time.perf_counter()
        arr, meta = pack_pdef(
            pdef.costmap,
            pdef.height_bev,
            map_center=pdef.map_center,
            map_res=pdef.map_res,
            start=start,
            goal=np.asarray(goal, dtype=np.float64),
            hysteresis=hysteresis,
            expansion_limit=expansion_limit,
            query_t=query_t,
        )
        dora_node.send_output("pdef", arr, meta)
        query_outstanding = True
        query_sent_t = query_t
        last_pdef_t = now

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
                # RTT EMA after unpack (same clock as packed query_t).
                if query_t > 0.0:
                    measured_dt = max(0.0, time.perf_counter() - float(query_t))
                    a = planner_dt_ema_alpha
                    planner_dt = a * planner_dt + (1.0 - a) * measured_dt
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
        ros.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
