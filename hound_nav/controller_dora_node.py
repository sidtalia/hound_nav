#!/usr/bin/env python3
"""Dora controller: MPPI track only. Inputs map + track → AckermannDriveStamped.

MPPI samples physical units: steer [rad], wheelspeed [m/s] — published as-is.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HOUND_NAV_ROOT = Path(__file__).resolve().parent.parent
if _HOUND_NAV_ROOT.is_dir():
    sys.path.insert(0, str(_HOUND_NAV_ROOT))

import time
import traceback
from typing import Any, Dict

from hound_nav.cuda_cache import ensure_torch_extensions_dir
from hound_nav.deps_path import load_class, setup_dependency_paths

_jit_dir = ensure_torch_extensions_dir()
print(f"[hound_controller] TORCH_EXTENSIONS_DIR={_jit_dir}", flush=True)

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from dora import Node
from rclpy.node import Node as RosNode
from rclpy.qos import qos_profile_sensor_data
from hound_nav.pdef_codec import unpack_map, unpack_track


def main() -> None:
    setup_dependency_paths()
    from hound_nav.nav_config import load_hound_nav_config, stack_config

    launch_cfg = load_hound_nav_config()
    Config = stack_config(launch_cfg)
    Dynamics_config = dict(Config["Dynamics_config"])
    Cost_config = Config["Cost_config"]
    Sampling_config = Config["Sampling_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    cooldown_s = 1.0
    state_dims = int(launch_cfg.get("control_state_dims", 17))
    Dynamics_config["NX"] = state_dims
    Dynamics_config["state_dims"] = state_dims
    # Physical control units (identity plant scale).
    Dynamics_config["throttle_to_wheelspeed"] = 1.0
    Dynamics_config["steering_max"] = 1.0
    cmd_topic = str(launch_cfg.get("cmd_topic", "/hound_nav/cmd_ackermann"))
    nu = int(Sampling_config["control_dim"])

    if "steer_lim" in Sampling_config:
        steer_lim = float(Sampling_config["steer_lim"])
        max_spd = float(Sampling_config["max_spd"])
        min_spd = float(Sampling_config.get("min_spd", 0.0))
    else:
        steer_lim = 1.0
        max_spd = float(Sampling_config.get("max_thr", 1.0))
        min_spd = float(Sampling_config.get("min_thr", 0.0))

    # Register with Dora *before* CUDA imports/JIT. Otherwise the whole graph
    # waits (no manager ticks → goal sits in the buffer with no pdef/plan).
    rclpy.init()
    ros = RosNode("hound_nav_controller")
    cmd_type = Twist if nu == 3 else AckermannDriveStamped
    cmd_pub = ros.create_publisher(cmd_type, cmd_topic, qos_profile_sensor_data)
    viz_k = int(MPPI_config.get("viz_k", 0))
    viz_pub = ros.create_publisher(MarkerArray, "/hound_nav/mppi_topk", 1)
    viz_next = 0.0
    dora = Node()
    print(
        "[hound_controller] dora registered; compiling CUDA MPPI "
        "(first Orin JIT can take minutes; manager/planner already running)",
        flush=True,
    )
    ros.get_logger().info(
        "dora up; JIT CUDA MPPI next (graph is live, tracking waits)"
    )
    nxt = getattr(dora, "next", None)
    if callable(nxt):
        try:
            nxt(timeout=0.1)
        except TypeError:
            pass
        except Exception:
            pass

    from mppi.MPPI import MPPI

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    t_jit = time.perf_counter()
    costs = load_class(Cost_config["file"], Cost_config["class"])(
        Cost_config, Map_config, device=device
    )
    sampling = load_class(Sampling_config["file"], Sampling_config["class"])(
        Sampling_config, MPPI_config, device=device
    )
    dynamics = load_class(Dynamics_config["file"], Dynamics_config["class"])(
        Dynamics_config, Map_config, MPPI_config, device=device
    )
    controller = MPPI(dynamics, costs, sampling, MPPI_config, device)
    jit_s = time.perf_counter() - t_jit
    print(f"[hound_controller] MPPI ready in {jit_s:.1f}s", flush=True)
    ros.get_logger().info(
        f"controller up cmd={cmd_topic} "
        f"steer_lim={steer_lim:.3f} rad spd=[{min_spd:.2f},{max_spd:.2f}] m/s "
        f"jit={jit_s:.1f}s"
    )

    last_map_gen = -1
    map_center = np.zeros(2, dtype=np.float64)
    action = np.zeros(nu, dtype=np.float64)
    cooldown_until = 0.0
    map_ready = False
    dt_sum = 0.0
    dt_n = 0
    dt_log_t = 0.0
    log_period_s = 1.0
    cv_log_t = 0.0

    def _publish(act: np.ndarray) -> None:
        if nu == 3:
            msg = Twist()
            msg.linear.x = float(act[0])
            msg.linear.y = float(act[1])
            msg.angular.z = float(act[2])
        else:
            msg = AckermannDriveStamped()
            msg.header.stamp = ros.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.drive.steering_angle = float(np.clip(act[0], -steer_lim, steer_lim))
            # Allow reverse on the wire (cooldown); sampling still respects min_spd.
            msg.drive.speed = float(np.clip(act[1], -max_spd, max_spd))
        cmd_pub.publish(msg)

    def _publish_topk() -> None:
        nonlocal viz_next
        if viz_k < 1:
            return
        now_v = time.perf_counter()
        if now_v < viz_next:
            return
        viz_next = now_v + 0.1
        st = controller.Dynamics.states.detach()
        cost = getattr(controller, "last_cost", None)
        if cost is None:
            return
        if st.dim() == 4:
            st = st[0]
        cost = cost.detach().reshape(-1)
        k = min(viz_k, int(cost.shape[0]), int(st.shape[0]))
        if k < 1 or st.shape[-1] < 2:
            return
        idx = torch.topk(cost, k, largest=False).indices
        xy = st[idx, :, :2].float().cpu().numpy()
        xy[..., 0] += float(map_center[0])
        xy[..., 1] += float(map_center[1])
        msg = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        msg.markers.append(wipe)
        for rank, traj in enumerate(xy):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = ros.get_clock().now().to_msg()
            m.ns = "mppi"
            m.id = rank
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.03
            m.pose.orientation.w = 1.0
            fade = 1.0 - rank / float(k)
            m.color = ColorRGBA(r=1.0, g=0.45 * fade, b=0.05, a=0.35 + 0.65 * fade)
            for p in traj:
                m.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.08))
            msg.markers.append(m)
        viz_pub.publish(msg)

    for event in dora:
        if event is None or event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT":
            continue
        eid = event.get("id")
        if eid == "map":
            try:
                m = unpack_map(event.get("value"), event.get("metadata") or {})
            except Exception:
                traceback.print_exc()
                continue
            gen = int(m["map_gen"])
            if gen == last_map_gen and map_ready:
                continue
            height_t = torch.from_numpy(np.ascontiguousarray(m["height"])).to(
                device=device, dtype=dtype
            )
            normal_t = torch.from_numpy(np.ascontiguousarray(m["normals"])).to(
                device=device, dtype=dtype
            )
            cost_t = torch.from_numpy(np.ascontiguousarray(m["costmap"])).to(
                device=device, dtype=dtype
            )
            # Grid size/res come from SSoT Map_config at MPPI construct.
            controller.Dynamics.set_BEV(height_t, normal_t)
            controller.Costs.set_BEV(height_t, normal_t, cost_t)
            mc = np.asarray(m["map_center"], dtype=np.float64).reshape(-1)
            if mc.size >= 2:
                map_center[0] = float(mc[0])
                map_center[1] = float(mc[1])
            last_map_gen = gen
            map_ready = True
            continue

        if eid != "track" or not map_ready:
            continue
        try:
            tr = unpack_track(event.get("value"), event.get("metadata") or {})
        except Exception:
            traceback.print_exc()
            continue

        if tr["goal_reached"]:
            action[:] = 0.0
            _publish(action)
            continue

        state = tr["state"]
        if state.size < state_dims:
            continue
        reference = tr["reference"]
        if reference.shape[0] < 1:
            action[:] = 0.0
            _publish(action)
            continue

        try:
            t0 = time.perf_counter()
            # LocalMap is odom-aligned; origin is SW. MPPI BEV lookup is
            # (xy + map_size/2)/res, so xy=0 is the grid center (origin + half
            # extent), not the robot. Track horizon is robot-relative: restore
            # world then shift into that same map-center frame.
            cx = float(map_center[0])
            cy = float(map_center[1])
            ref = np.ascontiguousarray(reference, dtype=np.float64)
            if ref.ndim == 1:
                ref = ref.reshape(1, -1)
            ref = np.copy(ref)
            ref[:, 0] += float(state[0]) - cx
            ref[:, 1] += float(state[1]) - cy
            ref_t = torch.from_numpy(ref).to(device=device, dtype=dtype)
            controller.Costs.set_path(ref_t)
            state_to_ctrl = np.copy(state)
            state_to_ctrl[0] = float(state[0]) - cx
            state_to_ctrl[1] = float(state[1]) - cy
            state_to_ctrl[2] = 0.0  # height BEV is already relative to center cell
            # nu==2: trailing slots are the last steer/speed command.
            # nu==3: sampler reads measured vx, vy, yaw rate (indices 6, 7, 14).
            if nu == 2 and state_to_ctrl.size >= 2:
                state_to_ctrl[-2:] = action
            action = np.array(
                controller.forward(
                    torch.from_numpy(state_to_ctrl).to(device=device, dtype=dtype),
                    num_iters=int(MPPI_config["n_iter"]),
                )
                .cpu()
                .numpy(),
                dtype=np.float64,
            )[0]
            if nu == 2:
                action[0] = float(np.clip(action[0], -steer_lim, steer_lim))
                action[1] = float(np.clip(action[1], min_spd, max_spd))
            _publish_topk()

            now = time.perf_counter()
            dt = now - t0
            dt_sum += dt
            dt_n += 1
            if dt_log_t <= 0.0:
                dt_log_t = now
            elif now - dt_log_t >= log_period_s:
                mean_dt = dt_sum / max(dt_n, 1)
                print(
                    f"[hound_controller] dt={mean_dt*1e3:.1f}ms "
                    f"(n={dt_n} last={dt*1e3:.1f}ms) "
                    f"cmd={np.array2string(action, precision=3)}",
                    flush=True,
                )
                dt_log_t = now
                dt_sum = 0.0
                dt_n = 0
            if controller.Costs.constraint_violation:
                action = np.zeros(2)
                if state.size > 6 and state[6] > 0.5:
                    controller.reset()
                cooldown_until = now + cooldown_s
                if now - cv_log_t >= 1.0:
                    cv_log_t = now
                    print(
                        f"\033[31m[hound_controller] constraint_violation "
                        f"(all samples lethal) halt {dt*1e3:.1f}ms\033[0m",
                        flush=True,
                    )
            if now < cooldown_until:
                action = np.zeros(2)
                action[1] = 0.0  # m/s light reverse during cooldown
            _publish(action)
        except Exception:
            traceback.print_exc()

    ros.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
