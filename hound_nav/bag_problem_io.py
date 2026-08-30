"""Snapshot / start–goal problem IO for bag mapping playback."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


def find_ssot_yaml() -> Path:
    candidates = [
        Path("/home/hound/colcon_ws/src/hound_core/config/SSoT.yaml"),
        Path("/root/colcon_ws/src/hound_core/config/SSoT.yaml"),
        Path(__file__).resolve().parents[2] / "hound_core" / "config" / "SSoT.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("Could not find hound_core/config/SSoT.yaml")


def load_planner_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return a deep copy of Planner_config (SSoT nav: or a standalone YAML)."""
    path = Path(config_path) if config_path else find_ssot_yaml()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "Planner_config" in raw:
        cfg = deepcopy(raw["Planner_config"])
    elif "nav" in raw and isinstance(raw["nav"], dict) and "Planner_config" in raw["nav"]:
        cfg = deepcopy(raw["nav"]["Planner_config"])
    elif "experiment_info_default" in raw:
        cfg = deepcopy(raw)
    else:
        raise ValueError(f"{path} has no Planner_config / experiment_info_default")
    return cfg


def dump_planner_config(cfg: Dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def map_center(origin_xy: np.ndarray, shape_hw: Sequence[int], res: float) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    return np.array([ox + 0.5 * w * res, oy + 0.5 * h * res], dtype=np.float64)


def map_xy_bounds(
    origin_xy: np.ndarray, shape_hw: Sequence[int], res: float
) -> Tuple[float, float, float, float]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    half = 0.5 * float(res)
    return ox + half, oy + half, ox + w * res - half, oy + h * res - half


def project_xy_to_map(
    xy: np.ndarray, origin_xy: np.ndarray, shape_hw: Sequence[int], res: float
) -> Tuple[np.ndarray, bool]:
    xmin, ymin, xmax, ymax = map_xy_bounds(origin_xy, shape_hw, res)
    x, y = float(xy[0]), float(xy[1])
    px = float(np.clip(x, xmin, xmax))
    py = float(np.clip(y, ymin, ymax))
    moved = abs(px - x) > 1e-6 or abs(py - y) > 1e-6
    out = np.asarray(xy, dtype=np.float64).copy()
    out[0], out[1] = px, py
    return out, moved


def relative_height(elev: np.ndarray, z_fallback: float) -> np.ndarray:
    height = np.ascontiguousarray(elev, dtype=np.float32)
    h, w = height.shape[:2]
    z0 = float(height[h // 2, w // 2])
    if not np.isfinite(z0):
        z0 = float(z_fallback)
    return np.nan_to_num(height - z0, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def costmap_igha(cost: np.ndarray) -> np.ndarray:
    cost_s = np.ascontiguousarray(cost, dtype=np.float32)
    if float(np.nanmax(cost_s)) <= 1.0 + 1e-3:
        return np.where(cost_s < 0.5, 255.0, 0.0).astype(np.float32)
    return np.clip(cost_s, 0.0, 255.0).astype(np.float32)


def plant_to_planner_state(state: np.ndarray) -> np.ndarray:
    """control_state [x,y,z,r,p,yaw,vx,vy,...] → planner [x,y,yaw,speed]."""
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    out = np.zeros(4, dtype=np.float32)
    out[0] = float(s[0])
    out[1] = float(s[1])
    out[2] = float(s[5]) if s.size > 5 else 0.0
    vx = float(s[6]) if s.size > 6 else 0.0
    vy = float(s[7]) if s.size > 7 else 0.0
    speed = float(np.hypot(vx, vy))
    out[3] = speed
    if speed > 1.0 and vx > 0.5 and s.size > 5:
        theta = float(s[5])
        dx = vx * np.cos(theta) - vy * np.sin(theta)
        dy = vx * np.sin(theta) + vy * np.cos(theta)
        out[2] = float(np.arctan2(dy, dx))
    return out


def world_to_map_xy(
    xy: np.ndarray, map_center_xy: np.ndarray, map_res: float, shape_hw: Sequence[int]
) -> np.ndarray:
    offset = float(map_res) * np.array(shape_hw[:2], dtype=np.float64) * 0.5
    return np.asarray(xy[:2], dtype=np.float64) + offset - np.asarray(map_center_xy[:2])


def interpolate_state(
    times: np.ndarray, states: np.ndarray, t: float
) -> Optional[np.ndarray]:
    if times.size < 1:
        return None
    if t <= float(times[0]):
        return states[0].copy()
    if t >= float(times[-1]):
        return states[-1].copy()
    i = int(np.searchsorted(times, t, side="left"))
    t0, t1 = float(times[i - 1]), float(times[i])
    if abs(t1 - t0) < 1e-9:
        return states[i].copy()
    a = (t - t0) / (t1 - t0)
    return ((1.0 - a) * states[i - 1] + a * states[i]).astype(np.float64)


def first_state_at_xy_dist(
    times: np.ndarray,
    states: np.ndarray,
    t0: float,
    start_xy: np.ndarray,
    min_xy_dist_m: float,
) -> Optional[Tuple[float, np.ndarray]]:
    """First logged state after t0 whose XY is at least min_xy_dist_m from start."""
    i0 = int(np.searchsorted(times, t0, side="right"))
    sx, sy = float(start_xy[0]), float(start_xy[1])
    min_d2 = float(min_xy_dist_m) ** 2
    for i in range(i0, len(times)):
        dx = float(states[i, 0]) - sx
        dy = float(states[i, 1]) - sy
        if dx * dx + dy * dy >= min_d2:
            return float(times[i]), states[i].copy()
    return None


def snapshot_paths(root: Path) -> List[Path]:
    d = root / "snapshots"
    if not d.is_dir():
        return []
    return sorted(d.glob("*.npz"))


def save_snapshot(root: Path, idx: int, payload: Dict[str, np.ndarray]) -> Path:
    d = root / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{idx:06d}.npz"
    np.savez_compressed(path, **payload)
    return path


def save_state_history(root: Path, times: np.ndarray, states: np.ndarray) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "state_history.npz"
    np.savez_compressed(path, t=times, state=states)
    return path


def load_state_history(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = root / "state_history.npz"
    data = np.load(path)
    return np.asarray(data["t"]), np.asarray(data["state"])


def generate_problem_pairs(
    root: Path,
    *,
    horizon_s: float = 5.0,
    min_xy_dist_m: float = 0.5,
    until_min_dist: bool = False,
    problems_subdir: str = "problems",
) -> int:
    """Build start/goal pairs.

    Default: goal = state at t+horizon (skip if XY dist < min).
    until_min_dist: goal = first later state with XY dist >= min (horizon unused).
    """
    snaps = snapshot_paths(root)
    if not snaps:
        return 0
    times, states = load_state_history(root)
    out_dir = root / problems_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    index: List[Dict[str, Any]] = []
    n = 0
    for path in snaps:
        snap = np.load(path)
        t0 = float(snap["t"])
        start_state = np.asarray(snap["state"], dtype=np.float64)
        if until_min_dist:
            hit = first_state_at_xy_dist(
                times, states, t0, start_state[:2], min_xy_dist_m
            )
            if hit is None:
                continue
            t1, goal_raw = hit
        else:
            t1 = t0 + float(horizon_s)
            if t1 > float(times[-1]) + 1e-3:
                continue
            goal_raw = interpolate_state(times, states, t1)
            if goal_raw is None:
                continue
        elev = np.asarray(snap["elevation"])
        cost = np.asarray(snap["cost"])
        origin = np.asarray(snap["origin"], dtype=np.float64)
        res = float(snap["resolution"])
        goal_state, projected = project_xy_to_map(goal_raw, origin, elev.shape, res)
        dist = float(np.linalg.norm(goal_state[:2] - start_state[:2]))
        if dist < float(min_xy_dist_m):
            continue
        center = map_center(origin, elev.shape, res)
        dest = out_dir / f"{n:06d}.npz"
        np.savez_compressed(
            dest,
            t_start=np.float64(t0),
            t_goal=np.float64(t1),
            start_state=start_state,
            goal_state=goal_state,
            goal_state_raw=goal_raw,
            elevation=elev,
            cost=cost,
            origin=origin,
            resolution=np.float64(res),
            map_center=center,
            projected=np.bool_(projected),
        )
        index.append(
            {
                "id": n,
                "file": dest.name,
                "t_start": t0,
                "t_goal": t1,
                "horizon_s": t1 - t0,
                "xy_dist_m": dist,
                "projected": bool(projected),
            }
        )
        n += 1
    index_name = (
        "problems_index.json"
        if problems_subdir == "problems"
        else f"{problems_subdir}_index.json"
    )
    (root / index_name).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return n


def list_problems(root: Path, subdir: str = "problems") -> List[Path]:
    d = root / subdir
    if not d.is_dir():
        return []
    return sorted(d.glob("*.npz"))
