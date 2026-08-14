#!/usr/bin/env python3
"""Dora controller: MPPI track only. Inputs map + track → AckermannDriveStamped.

MPPI samples physical units: steer [rad], wheelspeed [m/s] — published as-is.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from dora import Node
from rclpy.node import Node as RosNode
from rclpy.qos import qos_profile_sensor_data

from hound_nav.deps_path import setup_dependency_paths
from hound_nav.pdef_codec import unpack_map, unpack_track


def main() -> None:
    setup_dependency_paths()
    from mppi.MPPI import MPPI
    from mppi.Dynamics.SimpleCarDynamicsTCUDA import SimpleCarDynamics
    from mppi.Sampling.Delta_Sampling import Delta_Sampling
    from hound_nav.trackingCostCUDA import SimpleCarCost
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

    if "steer_lim" in Sampling_config:
        steer_lim = float(Sampling_config["steer_lim"])
        max_spd = float(Sampling_config["max_spd"])
        min_spd = float(Sampling_config.get("min_spd", 0.0))
    else:
        steer_lim = 1.0
        max_spd = float(Sampling_config.get("max_thr", 1.0))
        min_spd = float(Sampling_config.get("min_thr", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("mppi SimpleCarDynamicsTCUDA requires CUDA")
    dtype = torch.float32

    costs = SimpleCarCost(Cost_config, Map_config, device=device)
    sampling = Delta_Sampling(Sampling_config, MPPI_config, device=device)
    dynamics = SimpleCarDynamics(
        Dynamics_config, Map_config, MPPI_config, device=device
    )
    controller = MPPI(dynamics, costs, sampling, MPPI_config, device)

    rclpy.init()
    ros = RosNode("hound_nav_controller")
    cmd_pub = ros.create_publisher(
        AckermannDriveStamped, cmd_topic, qos_profile_sensor_data
    )
    ros.get_logger().info(
        f"controller up cmd={cmd_topic} "
        f"steer_lim={steer_lim:.3f} rad spd=[{min_spd:.2f},{max_spd:.2f}] m/s"
    )

    last_map_gen = -1
    action = np.zeros(2, dtype=np.float64)
    cooldown_until = 0.0
    map_ready = False

    def _publish(act: np.ndarray) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = ros.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = float(np.clip(act[0], -steer_lim, steer_lim))
        # Allow reverse on the wire (cooldown); sampling still respects min_spd.
        msg.drive.speed = float(np.clip(act[1], -max_spd, max_spd))
        cmd_pub.publish(msg)

    dora = Node()
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
            h, w = int(height_t.shape[0]), int(height_t.shape[1])
            res = float(m["map_res"])
            size = h * res
            dynamics = controller.Dynamics
            dynamics.BEVmap_size_px = np.int32(h)
            dynamics.BEVmap_res = np.float32(res)
            dynamics.BEVmap_size = np.float32(size)
            costs = controller.Costs
            costs.BEVmap_size_px = h
            costs.BEVmap_res = res
            costs.BEVmap_size = size
            dynamics.set_BEV(height_t, normal_t)
            costs.set_BEV(height_t, normal_t, cost_t)
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
            ref_t = torch.from_numpy(np.ascontiguousarray(reference)).to(
                device=device, dtype=dtype
            )
            controller.Costs.set_path(ref_t)
            state_to_ctrl = np.copy(state)
            state_to_ctrl[:3] = 0.0
            if state_to_ctrl.size >= 2:
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
            action[0] = float(np.clip(action[0], -steer_lim, steer_lim))
            action[1] = float(np.clip(action[1], min_spd, max_spd))

            now = time.perf_counter()
            if controller.Costs.constraint_violation:
                action = np.zeros(2)
                if state.size > 6 and state[6] > 0.5:
                    controller.reset()
                cooldown_until = now + cooldown_s
            if now < cooldown_until:
                action = np.zeros(2)
                action[1] = -0.2  # m/s light reverse during cooldown
            _publish(action)
        except Exception:
            traceback.print_exc()

    ros.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
