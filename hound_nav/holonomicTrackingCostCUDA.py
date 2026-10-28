"""Tracking cost for the holonomic plant: x, y, yaw, vx, vy, yaw rate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from hound_nav.cuda_cache import load_extension


def _load_kernel():
    cuda_dir = Path(__file__).resolve().parent / "cuda"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    return load_extension(
        "hound_holonomic_tracking_cost",
        [str(cuda_dir / "holonomic_tracking_cost.cpp"), str(cuda_dir / "holonomic_tracking_cost.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _load_kernel()
    return _KERNEL


class HolonomicTrackingCost(torch.nn.Module):
    def __init__(
        self,
        Cost_config: Dict[str, Any],
        Map_config: Dict[str, Any],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        if device.type != "cuda":
            raise RuntimeError("HolonomicTrackingCost requires CUDA")
        self.dtype = dtype
        self.d = device
        self.critical_RI = float(Cost_config["critical_RI"])
        self.lethal_w = float(Cost_config["lethal_w"])
        self.critical_vert_acc = float(Cost_config["critical_vert_acc"])
        self.pos_w = float(Cost_config["pos_w"])
        self.roll_ditch_w = float(Cost_config["roll_ditch_w"])
        self.speed_w = float(Cost_config["speed_w"])
        self.vy_w = float(Cost_config.get("vy_w", self.speed_w))
        self.omega_w = float(Cost_config.get("omega_w", self.speed_w))
        self.heading_w = float(Cost_config["heading_w"])
        self.scaling_factor = float(Cost_config["scaling_factor"])
        self.car_w2 = float(Cost_config["car_bb_width"]) * 0.5
        self.car_l2 = float(Cost_config["car_bb_length"]) * 0.5
        self.BEVmap_size = float(Map_config["map_size"])
        self.BEVmap_res = float(Map_config["map_res"])
        self.BEVmap_size_px = int(self.BEVmap_size / self.BEVmap_res)
        self.GRAVITY = 9.81
        self.BEVmap_cost = torch.zeros(
            (self.BEVmap_size_px, self.BEVmap_size_px), device=self.d, dtype=self.dtype
        )
        self.path = torch.zeros((1, 6), device=self.d, dtype=self.dtype)
        self.scaling: Optional[torch.Tensor] = None
        self.constraint_violation = False
        self.block_dim = 32
        _kernel()

    def set_BEV(self, BEVmap_height, BEVmap_normal, BEVmap_cost) -> None:
        del BEVmap_height, BEVmap_normal
        self.BEVmap_cost = BEVmap_cost.to(dtype=self.dtype)

    def set_path(self, path) -> None:
        if not torch.is_tensor(path):
            path = torch.tensor(path, dtype=self.dtype, device=self.d)
        else:
            path = path.to(device=self.d, dtype=self.dtype)
        self.path = path.contiguous()
        t = int(self.path.shape[0])
        self.scaling = torch.linspace(
            0.1, self.scaling_factor, t, device=self.d, dtype=self.dtype
        ).contiguous()

    def forward(self, state: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        del controls
        if state.dim() != 4:
            raise ValueError(f"state must be M,K,T,NX got {tuple(state.shape)}")
        M, K, T, NX = (int(x) for x in state.shape)
        if self.scaling is None or int(self.scaling.numel()) != T:
            self.scaling = torch.linspace(
                0.1, self.scaling_factor, T, device=self.d, dtype=self.dtype
            ).contiguous()
        if int(self.path.shape[0]) != T:
            c = int(self.path.shape[1])
            path = torch.zeros((T, c), device=self.d, dtype=self.dtype)
            n = min(T, int(self.path.shape[0]))
            path[:n] = self.path[:n]
            if n > 0 and n < T:
                path[n:] = path[n - 1]
            self.path = path.contiguous()
        out = torch.empty(K, device=self.d, dtype=self.dtype)
        out_cons = torch.empty(K, device=self.d, dtype=self.dtype)
        _kernel().tracking_cost(
            state.contiguous(),
            self.path,
            self.BEVmap_cost.contiguous(),
            self.scaling,
            out,
            out_cons,
            M,
            K,
            T,
            NX,
            int(self.path.shape[1]),
            self.BEVmap_size_px,
            float(self.BEVmap_size),
            float(self.BEVmap_res),
            float(self.car_l2),
            float(self.car_w2),
            float(self.pos_w),
            float(self.heading_w),
            float(self.speed_w),
            float(self.vy_w),
            float(self.omega_w),
            float(self.roll_ditch_w),
            float(self.lethal_w),
            float(self.critical_RI),
            float(self.critical_vert_acc),
            float(self.GRAVITY),
            int(self.block_dim),
            int((K + self.block_dim - 1) // self.block_dim),
        )
        self.constraint_violation = bool(torch.all(out_cons > self.lethal_w * 0.9).item())
        return out
