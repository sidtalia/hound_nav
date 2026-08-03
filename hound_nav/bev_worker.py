"""Background BEV prep: crop mapper LocalMap to nav size and H2D for MPPI."""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from hound_nav.pdef_buffer import PDefBuffer


@dataclass
class ReadyBEV:
    """Latest finished BEV (CPU + CUDA). Pose may lag state by ~1/bev_hz."""

    height_np: np.ndarray
    normal_np: np.ndarray
    costmap_np: np.ndarray
    height_t: torch.Tensor
    normal_t: torch.Tensor
    costmap_t: torch.Tensor
    map_center: np.ndarray
    target_wp: Optional[np.ndarray]
    map_gen: int
    build_xy_yaw: np.ndarray  # (3,) pose at build time


def start_bev_worker(
    buffer: PDefBuffer,
    *,
    map_size: float,
    map_res: float,
    device: torch.device,
    dtype: torch.dtype,
    should_stop: Callable[[], bool],
    bev_hz: float = 20.0,
) -> threading.Thread:
    """Daemon thread: rebuild BEV at up to bev_hz (immediate on new LocalMap)."""

    period = 1.0 / max(bev_hz, 1.0)

    def _run() -> None:
        last_build = 0.0
        last_map_gen = -1
        try:
            while not should_stop():
                map_gen = buffer.map_generation()
                map_changed = map_gen != last_map_gen
                buffer.wait_for_bev_kick(timeout=period)
                if should_stop():
                    break
                now = time.perf_counter()
                map_gen = buffer.map_generation()
                map_changed = map_gen != last_map_gen
                if not map_changed and (now - last_build) < period:
                    continue
                if not buffer.ready():
                    continue

                injected = buffer.get_injected_path()
                goal = None
                if injected is not None and len(injected) >= 2:
                    goal = np.asarray(injected[-1, :2], dtype=np.float64)

                pdef = buffer.snapshot_pdef(
                    map_size,
                    map_res,
                    goal_xy=goal,
                )
                if pdef is None:
                    continue

                height_t = torch.from_numpy(np.ascontiguousarray(pdef.height_bev)).to(
                    device=device, dtype=dtype, non_blocking=True
                )
                normal_t = torch.from_numpy(np.ascontiguousarray(pdef.normal_bev)).to(
                    device=device, dtype=dtype, non_blocking=True
                )
                costmap_t = torch.from_numpy(np.ascontiguousarray(pdef.costmap)).to(
                    device=device, dtype=dtype, non_blocking=True
                )
                if device.type == "cuda":
                    torch.cuda.current_stream(device).synchronize()

                ready = ReadyBEV(
                    height_np=pdef.height_bev,
                    normal_np=pdef.normal_bev,
                    costmap_np=pdef.costmap,
                    height_t=height_t,
                    normal_t=normal_t,
                    costmap_t=costmap_t,
                    map_center=np.copy(pdef.map_center),
                    target_wp=None
                    if pdef.target_wp is None
                    else np.copy(pdef.target_wp),
                    map_gen=map_gen,
                    build_xy_yaw=np.array(
                        [pdef.state[0], pdef.state[1], pdef.state[5]], dtype=np.float64
                    ),
                )
                buffer.set_ready_bev(ready)
                last_build = time.perf_counter()
                last_map_gen = map_gen
        except Exception:
            traceback.print_exc()
            print("[hound_nav] BEV worker stopped with error")

    t = threading.Thread(target=_run, name="hound_nav_bev", daemon=True)
    t.start()
    return t
