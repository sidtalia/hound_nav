"""Index an IGHA* path by its stamped time, not by row count.

``path[:, -1]`` is cumulative cost ``g`` in seconds. It is non-decreasing.
Holonomic search writes one ``g`` onto every intermediate of an edge
(the edge cost is ``timesteps * dt``). This module does not invent a time
per row.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def path_index(path: np.ndarray, t_now: float) -> int:
    # First row whose stamped sim time is at or after t_now.
    i = int(np.searchsorted(path[:, -1], t_now, side="left"))
    return min(i, len(path) - 1)


def ref_window(path: np.ndarray, t_now: float, horizon: int, stride: int) -> np.ndarray:
    # Path rows are physics substeps. One tracker step spans `stride` of them.
    i = path_index(path, t_now)
    idx = np.clip(i + np.arange(horizon) * stride, 0, len(path) - 1)
    # State columns only. The last column is g.
    return path[idx, :-1]


def need_plan(path, t_now: float, stride: int) -> bool:
    # Hold the current path until the clock reaches the last usable edge.
    # Prevents thrashing when the planner returns faster than one tracker step.
    if path is None:
        return True
    return path_index(path, t_now) + stride >= len(path)


def open_loop_command(path, controls, t_now: float) -> Optional[np.ndarray]:
    """Action at the first row strictly after ``t_now``.

    ``None`` when there is no tape or the clock is past the last stamp.
    The caller must not repeat the previous command in that case.
    """
    if path is None or len(path) == 0 or len(controls) == 0:
        return None
    if float(t_now) > float(path[-1, -1]):
        return None
    i = int(np.searchsorted(path[:, -1], t_now, side="right"))
    i = min(i, len(controls) - 1)
    return np.asarray(controls[i], dtype=np.float64).copy()


def clamp_twist(
    cmd,
    max_vel: float,
    vy_max: float,
    max_omega: float,
    min_vel: float = 0.0,
) -> np.ndarray:
    """Same per-axis caps the holonomic planner uses. A zero cap is left alone."""
    out = np.asarray(cmd, dtype=np.float64).reshape(-1).copy()
    if max_vel > 0.0:
        out[0] = float(np.clip(out[0], float(min_vel), float(max_vel)))
    if vy_max > 0.0:
        out[1] = float(np.clip(out[1], -float(vy_max), float(vy_max)))
    if max_omega > 0.0:
        out[2] = float(np.clip(out[2], -float(max_omega), float(max_omega)))
    return out
