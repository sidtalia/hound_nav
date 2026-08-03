#!/usr/bin/env python3
"""Warm-started full MPPI timing (dynamics TCUDA + cost + sampling)."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml

from hound_nav.deps_path import setup_dependency_paths


def main() -> None:
    setup_dependency_paths()

    from BeamNGRL.control.UW_mppi.MPPI import MPPI
    from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsTCUDA import (
        SimpleCarDynamics,
    )
    from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
    from TrackingCost import SimpleCarCost as OldCost
    from hound_nav.trackingCostCUDA import SimpleCarCost as CudaCost

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "nav_example.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        Config = yaml.safe_load(f)

    Dynamics_config = Config["Dynamics_config"]
    Cost_config = Config["Cost_config"]
    Sampling_config = Config["Sampling_config"]
    MPPI_config = dict(Config["MPPI_config"])
    Map_config = Config["Map_config"]

    device = torch.device("cuda")
    dtype = torch.float32
    K = int(MPPI_config["ROLLOUTS"])
    T = int(MPPI_config["TIMESTEPS"])
    M = int(MPPI_config["BINS"])
    H = int(Map_config["map_size"] / Map_config["map_res"])

    # Synthetic BEV + path + state (body-centric like control_loop)
    height = torch.zeros(H, H, device=device, dtype=dtype)
    normal = torch.zeros(H, H, 3, device=device, dtype=dtype)
    normal[..., 2] = 1.0
    costmap = torch.full((H, H), 255.0, device=device, dtype=dtype)
    costmap[180:220, 220:260] = 0.0
    path = torch.zeros(T, 4, device=device, dtype=dtype)
    path[:, 0] = torch.linspace(0, 8, T, device=device)
    path[:, 3] = 3.0
    state = torch.zeros(17, device=device, dtype=dtype)
    state[6] = 2.0
    state[11] = 9.8
    state[15:17] = torch.tensor([0.0, 0.3], device=device, dtype=dtype)

    warmup = 20
    iters = 50
    n_iters_list = [1, 4, 10]

    def make_controller(cost_cls):
        costs = cost_cls(Cost_config, Map_config, device=device)
        sampling = Delta_Sampling(Sampling_config, MPPI_config, device=device)
        dynamics = SimpleCarDynamics(
            Dynamics_config, Map_config, MPPI_config, device=device
        )
        ctrl = MPPI(dynamics, costs, sampling, MPPI_config, device)
        ctrl.Dynamics.set_BEV(height, normal)
        ctrl.Costs.set_BEV(height, normal, costmap)
        ctrl.Costs.set_path(path)
        return ctrl

    def bench(ctrl, n_iter: int) -> float:
        ctrl.reset()
        for _ in range(warmup):
            _ = ctrl.forward(state, num_iters=n_iter)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = ctrl.forward(state, num_iters=n_iter)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    print(
        f"MPPI full pipeline  K={K} T={T} M={M}  "
        f"map={Map_config['map_size']}@{Map_config['map_res']}  "
        f"warmup={warmup} iters={iters}"
    )
    print(
        f"{'cost':<10} {'n_iter':>6} {'ms/call':>10} {'ms/opt':>10} {'Hz':>8}"
    )
    print("-" * 48)

    for label, cost_cls in (("cuda", CudaCost), ("old", OldCost)):
        ctrl = make_controller(cost_cls)
        # one extra warm construct/JIT settle
        _ = ctrl.forward(state, num_iters=1)
        torch.cuda.synchronize()
        for n_iter in n_iters_list:
            ms = bench(ctrl, n_iter)
            print(
                f"{label:<10} {n_iter:6d} {ms:10.3f} {ms / n_iter:10.3f} "
                f"{1000.0 / ms:8.1f}"
            )


if __name__ == "__main__":
    main()
