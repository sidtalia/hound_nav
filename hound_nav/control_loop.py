"""MPPI (main thread) + IGHAStarMP planner process — port of IGHAStar BeamNG example.py."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import yaml

from hound_nav.deps_path import setup_dependency_paths
from hound_nav.pdef_buffer import PDefBuffer


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_control_loop(
    buffer: PDefBuffer,
    config_path: str,
    *,
    send_ctrl: Callable[..., None],
    should_stop: Callable[[], bool],
    device: Optional[torch.device] = None,
    publish_path: Optional[Callable[[np.ndarray], None]] = None,
    rate_hz: float = 20.0,
    skip_planner: bool = False,
    event_driven: bool = True,
    async_bev: bool = True,
    bev_hz: float = 20.0,
) -> None:
    """
    Mirror of IGHAStar/examples/BeamNG/example.py inner loop:
    - planner.update / set_query (planner runs in its own process)
    - MPPI.forward on this thread
    Map/state come from PDefBuffer instead of BeamNG.

    If skip_planner is True (or an injected path is present), IGHA* is not
    started and /hound_nav/inject_path supplies the local plan.

    event_driven (default): wake on each new control_state and run one MPPI
    cycle immediately (no rate sleep). rate_hz is only used when event_driven
    is False.

    async_bev (default): background thread crops LocalMap to nav size and H2D
    at bev_hz (and on every LocalMap). MPPI uses the latest ready GPU BEV + state.
    """
    setup_dependency_paths()

    # Deferred imports so PYTHONPATH is ready (do not edit those packages).
    from BeamNGRL.control.UW_mppi.MPPI import MPPI
    from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsTCUDA import (
        SimpleCarDynamics,
    )
    from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
    from hound_nav.bev_worker import start_bev_worker
    from hound_nav.trackingCostCUDA import SimpleCarCost
    from utils import update_goal, steering_limiter

    Config = _load_config(config_path)
    Dynamics_config = Config["Dynamics_config"]
    Cost_config = Config["Cost_config"]
    Sampling_config = Config["Sampling_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    map_res = float(Map_config["map_res"])
    map_size = float(Map_config["map_size"])
    # Constraint cooldown wall time (was cycle-counted via burn_time in rate mode).
    cooldown_s = 1.0
    lookahead = float(Config["lookahead"][0])
    wp_radius = float(Config["wp_radius"][0])
    hysteresis = float(Config["hysteresis"][0])

    dtype = torch.float32
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("UW_mppi SimpleCarDynamicsTCUDA requires CUDA")

    costs = SimpleCarCost(Cost_config, Map_config, device=device)
    sampling = Delta_Sampling(Sampling_config, MPPI_config, device=device)
    dynamics = SimpleCarDynamics(
        Dynamics_config, Map_config, MPPI_config, device=device
    )
    controller = MPPI(dynamics, costs, sampling, MPPI_config, device)

    planner = None
    default_expansion_limit = int(
        Config["Planner_config"]["experiment_info_default"]["max_expansions"]
    )
    bidirectional = bool(
        Config["Planner_config"]["experiment_info_default"].get("bidirectional", False)
    )
    expansion_limit = (
        default_expansion_limit // 4 if bidirectional else default_expansion_limit
    )
    if not skip_planner:
        from IGHAStarMP import IGHAStarMP

        if bidirectional:
            Config["Planner_config"]["experiment_info_default"][
                "max_expansions"
            ] = expansion_limit
        planner = IGHAStarMP(Config["Planner_config"], bidirectional=bidirectional)
        time.sleep(1.0)
        planner.reset()

    if async_bev:
        start_bev_worker(
            buffer,
            map_size=map_size,
            map_res=map_res,
            device=device,
            dtype=dtype,
            should_stop=should_stop,
            bev_hz=bev_hz,
        )

    action = np.zeros(2, dtype=np.float64)
    goal = None
    current_wp_index = 0
    first_path = False
    cooldown_until = 0.0
    dt = 1.0 / max(rate_hz, 1.0)
    last_published_inject_gen = -1

    print(
        f"[hound_nav] control loop start (cuda cost+dynamics, map {map_size}m @ "
        f"{map_res}m, skip_planner={skip_planner}, event_driven={event_driven}, "
        f"async_bev={async_bev}@{bev_hz}Hz, expansions={expansion_limit}, "
        f"bi={bidirectional})"
    )

    try:
        while not should_stop():
            t0 = time.perf_counter()

            if event_driven:
                if not buffer.ready():
                    buffer.wait_for_state(timeout=0.05)
                    continue
                # Block until a fresh state (coalesce: latest is already in buffer).
                if not buffer.wait_for_state(timeout=0.1):
                    continue
                while buffer.state_pending():
                    buffer.wait_for_state(timeout=0.0)
            else:
                if not buffer.ready():
                    time.sleep(0.05)
                    continue

            injected = buffer.get_injected_path()
            use_inject = injected is not None and len(injected) >= 2

            if async_bev:
                ready = buffer.get_ready_bev()
                if ready is None:
                    # State may arrive before the first background BEV finishes.
                    deadline = time.perf_counter() + 0.15
                    while (
                        ready is None
                        and time.perf_counter() < deadline
                        and not should_stop()
                    ):
                        time.sleep(0.005)
                        ready = buffer.get_ready_bev()
                if ready is None:
                    if not event_driven:
                        time.sleep(0.05)
                    continue
                state = buffer.get_state_copy()
                if state is None:
                    continue
                pos = np.copy(state[:2])
                BEV_height = ready.height_t
                BEV_normal = ready.normal_t
                costmap_t = ready.costmap_t
                costmap_np = ready.costmap_np
                height_np = ready.height_np
                map_center = ready.map_center
                target_wp = ready.target_wp

                if use_inject:
                    goal = np.asarray(injected[-1, :2], dtype=np.float64)
                    success = False
                    path = injected
                else:
                    if skip_planner:
                        if not event_driven:
                            time.sleep(0.05)
                        continue
                    success = False
                    if target_wp is not None and len(target_wp) > 0:
                        goal, success, current_wp_index = update_goal(
                            goal,
                            pos,
                            target_wp,
                            current_wp_index,
                            lookahead,
                            wp_radius=wp_radius,
                        )
                    elif goal is None:
                        if not event_driven:
                            time.sleep(0.05)
                        continue

                    assert planner is not None
                    _, path, expansions = planner.update()
                    planner.set_query(
                        map_center,
                        state,
                        np.asarray(goal, dtype=np.float64),
                        costmap_np,
                        height_np,
                        hysteresis,
                        expansion_limit,
                    )
                    if path is None:
                        if not first_path:
                            expansion_limit = min(
                                expansion_limit * 2,
                                int(Config["Planner_config"]["unstuck_expansions"]),
                            )
                        if not event_driven:
                            elapsed = time.perf_counter() - t0
                            time.sleep(max(0.0, dt - elapsed))
                        continue
                    first_path = True
                    expansion_limit = default_expansion_limit
            elif use_inject:
                goal = np.asarray(injected[-1, :2], dtype=np.float64)
                pdef = buffer.snapshot_pdef(
                    map_size, map_res, goal_xy=goal
                )
                if pdef is None:
                    if not event_driven:
                        time.sleep(0.05)
                    continue
                state = pdef.state
                pos = np.copy(state[:2])
                success = False
                path = injected
                BEV_height = torch.from_numpy(pdef.height_bev).to(
                    dtype=dtype, device=device
                )
                BEV_normal = torch.from_numpy(pdef.normal_bev).to(
                    dtype=dtype, device=device
                )
                costmap_t = torch.from_numpy(pdef.costmap).to(
                    dtype=dtype, device=device
                )
            else:
                if skip_planner:
                    if not event_driven:
                        time.sleep(0.05)
                    continue

                pdef = buffer.snapshot_pdef(
                    map_size, map_res, goal_xy=None
                )
                if pdef is None:
                    if not event_driven:
                        time.sleep(0.05)
                    continue

                state = pdef.state
                pos = np.copy(state[:2])
                success = False

                if pdef.target_wp is not None and len(pdef.target_wp) > 0:
                    goal, success, current_wp_index = update_goal(
                        goal,
                        pos,
                        pdef.target_wp,
                        current_wp_index,
                        lookahead,
                        wp_radius=wp_radius,
                    )
                elif goal is None:
                    if not event_driven:
                        time.sleep(0.05)
                    continue

                pdef = buffer.snapshot_pdef(
                    map_size, map_res, goal_xy=goal
                )
                if pdef is None or pdef.goal is None:
                    continue

                BEV_height = torch.from_numpy(pdef.height_bev).to(
                    dtype=dtype, device=device
                )
                BEV_normal = torch.from_numpy(pdef.normal_bev).to(
                    dtype=dtype, device=device
                )
                costmap_t = torch.from_numpy(pdef.costmap).to(
                    dtype=dtype, device=device
                )

                assert planner is not None
                _, path, expansions = planner.update()
                planner.set_query(
                    pdef.map_center,
                    state,
                    np.asarray(pdef.goal, dtype=np.float64),
                    pdef.costmap,
                    pdef.height_bev,
                    hysteresis,
                    expansion_limit,
                )

                if path is None:
                    if not first_path:
                        expansion_limit = min(
                            expansion_limit * 2,
                            int(Config["Planner_config"]["unstuck_expansions"]),
                        )
                    if not event_driven:
                        elapsed = time.perf_counter() - t0
                        time.sleep(max(0.0, dt - elapsed))
                    continue

                first_path = True
                expansion_limit = default_expansion_limit

            if publish_path is not None:
                inj_gen = buffer.inject_generation()
                if use_inject:
                    if inj_gen != last_published_inject_gen:
                        publish_path(path)
                        last_published_inject_gen = inj_gen
                else:
                    publish_path(path)

            controller_path = np.copy(path)
            controller_path[:, :2] -= np.copy(pos)
            reference_index = int(
                np.argmin(np.linalg.norm(controller_path[:, :2], axis=1))
            )
            T = int(MPPI_config["TIMESTEPS"])
            if reference_index < len(controller_path) - T:
                reference_path = controller_path[
                    reference_index : reference_index + T, :4
                ]
            else:
                reference_path = np.zeros((T, 4), dtype=np.float64)
                available = controller_path[reference_index:, :4]
                reference_path[: len(available)] = available
                if len(available) > 0:
                    reference_path[len(available) :, :3] = reference_path[
                        len(available) - 1, :3
                    ]
                if reference_index >= len(controller_path) - 10:
                    action *= 0
                    buffer.set_last_action(action)
                    send_ctrl(
                        action,
                        Dynamics_config,
                        state_stamp_sec=buffer.get_state_stamp_sec(),
                        state_pub_stamp_sec=buffer.get_state_pub_stamp_sec(),
                    )
                    if not event_driven:
                        elapsed = time.perf_counter() - t0
                        time.sleep(max(0.0, dt - elapsed))
                    continue

            reference_path_t = torch.from_numpy(reference_path).to(
                device=device, dtype=dtype
            )

            # Stamps for the state this cycle consumes (set at recv in callback).
            cycle_state_stamp = buffer.get_state_stamp_sec()
            cycle_pub_stamp = buffer.get_state_pub_stamp_sec()

            controller.Dynamics.set_BEV(BEV_height, BEV_normal)
            controller.Costs.set_BEV(BEV_height, BEV_normal, costmap_t)
            controller.Costs.set_path(reference_path_t)

            state_to_ctrl = np.copy(state)
            state_to_ctrl[:3] = 0.0
            state_to_ctrl[15:17] = action
            action = np.array(
                controller.forward(
                    torch.from_numpy(state_to_ctrl).to(device=device, dtype=dtype),
                    num_iters=int(MPPI_config["n_iter"]),
                )
                .cpu()
                .numpy(),
                dtype=np.float64,
            )[0]
            action[1] = np.clip(
                action[1], Sampling_config["min_thr"], Sampling_config["max_thr"]
            )
            action[0] = steering_limiter(action[0], state, Config["RPS_config"])

            now = time.perf_counter()
            if controller.Costs.constraint_violation:
                action = np.zeros(2)
                if state[6] > 0.5:
                    controller.reset()
                cooldown_until = now + cooldown_s

            if now < cooldown_until:
                action = np.zeros(2)
                action[1] = -0.2

            if success:
                action *= 0.0

            buffer.set_last_action(action)
            send_ctrl(
                action,
                Dynamics_config,
                state_stamp_sec=cycle_state_stamp,
                state_pub_stamp_sec=cycle_pub_stamp,
            )

            if not event_driven:
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, dt - elapsed))

    except Exception:
        traceback.print_exc()
    finally:
        if planner is not None:
            try:
                planner.shutdown()
            except Exception:  # noqa: BLE001
                pass
        print("[hound_nav] control loop stopped")
