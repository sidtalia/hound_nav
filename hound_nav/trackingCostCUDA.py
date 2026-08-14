"""CUDA SimpleCarCost — Jetson-fused port of IGHAStar TrackingCost.SimpleCarCost.

Drop-in for MPPI: same set_BEV / set_path / forward / constraint_violation API.
Does not modify IGHAStar; lives in hound_nav.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.cpp_extension import load


def _load_kernel():
    cuda_dir = Path(__file__).resolve().parent / "cuda"
    cpp = str(cuda_dir / "tracking_cost.cpp")
    cu = str(cuda_dir / "tracking_cost.cu")
    # Jetson: prefer sm_87 (Orin) when TORCH_CUDA_ARCH_LIST unset.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    kwargs = {}
    ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if ext_dir:
        kwargs["build_directory"] = ext_dir
    return load(
        name="hound_tracking_cost",
        sources=[cpp, cu],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        **kwargs,
    )


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _load_kernel()
    return _KERNEL


class SimpleCarCost(torch.nn.Module):
    """Same interface as IGHAStar examples/BeamNG/TrackingCost.SimpleCarCost."""

    def __init__(
        self,
        Cost_config: Dict[str, Any],
        Map_config: Dict[str, Any],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        if device.type != "cuda":
            raise RuntimeError("trackingCostCUDA requires CUDA")
        self.dtype = dtype
        self.d = device

        self.critical_RI = float(Cost_config["critical_RI"])
        self.lethal_w = float(Cost_config["lethal_w"])
        self.critical_vert_acc = float(Cost_config["critical_vert_acc"])
        self.pos_w = float(Cost_config["pos_w"])
        self.roll_ditch_w = float(Cost_config["roll_ditch_w"])
        self.speed_w = float(Cost_config["speed_w"])
        self.heading_w = float(Cost_config["heading_w"])
        self.scaling_factor = float(Cost_config["scaling_factor"])
        self.car_w2 = float(Cost_config["car_bb_width"]) * 0.5
        self.car_l2 = float(Cost_config["car_bb_length"]) * 0.5

        self.BEVmap_size = float(Map_config["map_size"])
        self.BEVmap_res = float(Map_config["map_res"])
        self.BEVmap_size_px = int(self.BEVmap_size / self.BEVmap_res)
        self.GRAVITY = 9.8

        self.BEVmap_height = torch.zeros(
            (self.BEVmap_size_px, self.BEVmap_size_px), device=self.d, dtype=self.dtype
        )
        self.BEVmap_normal = torch.zeros(
            (self.BEVmap_size_px, self.BEVmap_size_px, 3),
            device=self.d,
            dtype=self.dtype,
        )
        self.BEVmap_cost = torch.zeros(
            (self.BEVmap_size_px, self.BEVmap_size_px), device=self.d, dtype=self.dtype
        )
        self.path = torch.zeros((1, 4), device=self.d, dtype=self.dtype)
        self.scaling: Optional[torch.Tensor] = None
        self.speed_target = torch.tensor(0.0, device=self.d, dtype=self.dtype)
        self.constraint_violation = False

        # Jetson-friendly launch (match SimpleCarDynamicsTCUDA).
        self.block_dim = 32
        _kernel()  # JIT compile once at construct

    def set_BEV(
        self,
        BEVmap_height: torch.Tensor,
        BEVmap_normal: torch.Tensor,
        BEVmap_cost: torch.Tensor,
    ) -> None:
        self.BEVmap_height = BEVmap_height
        self.BEVmap_normal = BEVmap_normal
        self.BEVmap_cost = (255.0 - BEVmap_cost.to(dtype=self.dtype)) / 255.0

    def set_goal(self, goal_state: torch.Tensor) -> None:
        self.goal_state = goal_state[:2]

    def set_path(self, path: torch.Tensor) -> None:
        if not torch.is_tensor(path):
            path = torch.tensor(path, dtype=self.dtype, device=self.d)
        else:
            path = path.to(device=self.d, dtype=self.dtype)
        self.path = path.contiguous()
        T = int(self.path.shape[0])
        self.scaling = torch.linspace(
            0.1, self.scaling_factor, T, device=self.d, dtype=self.dtype
        ).contiguous()

    def set_speed_limit(self, speed_lim: float) -> None:
        self.speed_target = torch.tensor(speed_lim, dtype=self.dtype, device=self.d)

    def forward(self, state: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        del controls  # unused (same as TrackingCost.py)
        # Expect [M, K, T, NX]
        if state.dim() != 4:
            raise ValueError(f"state must be M,K,T,NX got {tuple(state.shape)}")
        M, K, T, NX = (int(x) for x in state.shape)
        state_c = state.contiguous()
        if self.scaling is None or int(self.scaling.numel()) != T:
            self.scaling = torch.linspace(
                0.1, self.scaling_factor, T, device=self.d, dtype=self.dtype
            ).contiguous()
        if int(self.path.shape[0]) != T:
            # Pad / trim path to T (MPPI always uses TIMESTEPS).
            path = torch.zeros((T, 4), device=self.d, dtype=self.dtype)
            n = min(T, int(self.path.shape[0]))
            path[:n] = self.path[:n]
            if n > 0 and n < T:
                path[n:] = path[n - 1]
            self.path = path.contiguous()

        bev = self.BEVmap_cost.contiguous()
        out = torch.empty(K, device=self.d, dtype=self.dtype)
        out_cons = torch.empty(K, device=self.d, dtype=self.dtype)
        grid_dim = int((K + self.block_dim - 1) // self.block_dim)

        _kernel().tracking_cost(
            state_c,
            self.path,
            bev,
            self.scaling,
            out,
            out_cons,
            M,
            K,
            T,
            NX,
            self.BEVmap_size_px,
            float(self.BEVmap_size),
            float(self.BEVmap_res),
            float(self.car_l2),
            float(self.car_w2),
            float(self.pos_w),
            float(self.heading_w),
            float(self.speed_w),
            float(self.roll_ditch_w),
            float(self.lethal_w),
            float(self.critical_RI),
            float(self.critical_vert_acc),
            float(self.GRAVITY),
            int(self.block_dim),
            int(grid_dim),
        )

        thresh = self.lethal_w * 0.9
        self.constraint_violation = bool(torch.all(out_cons > thresh).item())
        return out
