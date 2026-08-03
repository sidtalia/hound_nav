#!/usr/bin/env python3
"""Warm-started timing: IGHAStar TrackingCost.py vs hound trackingCostCUDA."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "IGHAStar" / "examples" / "BeamNG"))

from TrackingCost import SimpleCarCost as OldCost  # noqa: E402
from hound_nav.trackingCostCUDA import SimpleCarCost as NewCost  # noqa: E402


def main() -> None:
    cfg = dict(
        pos_w=1.0,
        speed_w=0.1,
        heading_w=0.1,
        scaling_factor=1.0,
        critical_RI=0.8,
        critical_vert_acc=4.0,
        roll_ditch_w=50.0,
        lethal_w=50.0,
        car_bb_length=2.6,
        car_bb_width=1.6,
    )
    mp = dict(map_size=100, map_res=0.25)
    device = torch.device("cuda")
    dtype = torch.float32

    shapes = [
        (1, 1024, 30, 17),
        (1, 512, 30, 17),
        (1, 256, 30, 17),
    ]
    warmup = 30
    iters = 100

    def make_inputs(M: int, K: int, T: int, NX: int):
        state = torch.zeros(M, K, T, NX, device=device, dtype=dtype)
        state[..., 0] = torch.linspace(-8, 8, K, device=device).view(1, K, 1)
        state[..., 5] = 0.2
        state[..., 6] = 2.0
        state[..., 7] = 0.1
        state[..., 11] = 9.8
        controls = torch.zeros(M, K, T, 2, device=device, dtype=dtype)
        path = torch.zeros(T, 4, device=device, dtype=dtype)
        path[:, 0] = torch.linspace(0, 5, T, device=device)
        path[:, 3] = 2.0
        H = int(mp["map_size"] / mp["map_res"])
        costmap = torch.full((H, H), 255.0, device=device, dtype=dtype)
        costmap[180:220, 220:260] = 0.0
        height = torch.zeros(H, H, device=device, dtype=dtype)
        normal = torch.zeros(H, H, 3, device=device, dtype=dtype)
        normal[..., 2] = 1.0
        return state, controls, path, height, normal, costmap

    def bench(cost_mod, state, controls, path, height, normal, costmap):
        cost_mod.set_BEV(height, normal, costmap)
        cost_mod.set_path(path)
        for _ in range(warmup):
            out = cost_mod.forward(state, controls)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = cost_mod.forward(state, controls)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) / iters * 1e3
        return dt_ms, float(out.mean())

    header = (
        f"{'M,K,T':<14} {'old_ms':>10} {'cuda_ms':>10} "
        f"{'speedup':>8} {'old_mean':>10} {'cuda_mean':>10}"
    )
    print(header)
    print("-" * len(header))
    for M, K, T, NX in shapes:
        state, controls, path, height, normal, costmap = make_inputs(M, K, T, NX)
        old = OldCost(cfg, mp, dtype=dtype, device=device)
        new = NewCost(cfg, mp, dtype=dtype, device=device)
        old_ms, old_mean = bench(old, state, controls, path, height, normal, costmap)
        new_ms, new_mean = bench(new, state, controls, path, height, normal, costmap)
        print(
            f"{M},{K},{T:<8} {old_ms:10.3f} {new_ms:10.3f} "
            f"{old_ms / new_ms:7.2f}x {old_mean:10.2f} {new_mean:10.2f}"
        )
    print()
    print(
        f"warmup={warmup} iters={iters} map={mp['map_size']}m@{mp['map_res']}m "
        f"bb={cfg['car_bb_length']}x{cfg['car_bb_width']}m"
    )
    print(
        "Note: CUDA cost rasters full footprint (IGHA* check_crop); "
        "old cost samples 4 corners only — timings are not apples-to-apples FLOPs."
    )


if __name__ == "__main__":
    main()
