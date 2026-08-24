#!/usr/bin/env python3
"""Dora planner: IGHA* search only. Input pdef (start, goal, map) → plan."""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
import torch
from dora import Node

from hound_nav.deps_path import setup_dependency_paths
from hound_nav.pdef_codec import pack_plan, unpack_pdef


def _search(
    planner: Any,
    *,
    map_center: np.ndarray,
    start_state: np.ndarray,
    goal_xy: np.ndarray,
    costmap: np.ndarray,
    heightmap: np.ndarray,
    map_res: float,
    hysteresis: int,
    expansion_limit: int,
    stop: bool,
    cruise_speed_mps: float,
    min_goal_dist_m: float = 0.3,
) -> tuple[bool, Optional[np.ndarray], int]:
    goal_dist = float(np.linalg.norm(start_state[:2] - goal_xy[:2]))
    # Too close to bother searching (already at goal). Was hard-coded 5.0 m,
    # which rejected normal RViz goals a few meters away.
    if goal_dist < min_goal_dist_m:
        print(
            f"[hound_planner] skip search: goal_dist={goal_dist:.2f}m "
            f"< min={min_goal_dist_m:.2f}m",
            flush=True,
        )
        return False, None, 0

    bitmap = torch.ones(
        (costmap.shape[0], costmap.shape[1], 2), dtype=torch.float32
    )
    bitmap[..., 0] = torch.from_numpy(np.ascontiguousarray(costmap))
    bitmap[..., 1] = torch.from_numpy(np.ascontiguousarray(heightmap))
    offset = map_res * np.array(bitmap.shape[:2], dtype=np.float64) * 0.5

    start = torch.zeros(4, dtype=torch.float32)
    goal = torch.zeros(4, dtype=torch.float32)
    start[:2] = torch.from_numpy(start_state[:2] + offset - map_center)
    goal[:2] = torch.from_numpy(goal_xy[:2] + offset - map_center)
    start[2] = float(start_state[5])
    start[3] = float(np.linalg.norm(start_state[6:8]))
    if start[3] > 1.0 and start_state[6] > 0.5:
        v = start_state[6:8]
        theta = float(start_state[5])
        dx = v[0] * np.cos(theta) - v[1] * np.sin(theta)
        dy = v[0] * np.sin(theta) + v[1] * np.cos(theta)
        start[2] = float(np.arctan2(dy, dx))
    dx = float(goal[0] - start[0])
    dy = float(goal[1] - start[1])
    goal[2] = float(np.arctan2(dy, dx))
    goal[3] = 0.0 if stop else float(cruise_speed_mps)

    # IGHA* pybind expects int hysteresis_threshold (not float).
    success = planner.search(
        start, goal, bitmap, int(expansion_limit), int(hysteresis), True
    )
    prof = planner.get_profiler_info()
    expansions = int(prof[7]) if prof is not None and len(prof) > 7 else 0
    if not success:
        return False, None, expansions
    path = planner.get_best_path().numpy()
    path = np.flip(path, axis=0)
    path[..., :2] -= offset
    path[..., :2] += map_center
    return True, path, expansions


def main() -> None:
    setup_dependency_paths()
    from ighastar.scripts.common_utils import create_planner
    from hound_nav.nav_config import load_hound_nav_config, stack_config

    launch_cfg = load_hound_nav_config()
    Config = stack_config(launch_cfg)
    cruise_speed_mps = float(launch_cfg.get("cruise_speed_mps", 10.0))

    planner_cfg = Config["Planner_config"]
    bidirectional = bool(
        planner_cfg["experiment_info_default"].get("bidirectional", False)
    )

    planner = None
    dora = Node()
    for event in dora:
        if event is None or event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT" or event.get("id") != "pdef":
            continue
        try:
            pdef = unpack_pdef(event.get("value"), event.get("metadata") or {})
        except Exception:
            traceback.print_exc()
            arr, meta = pack_plan(None, success=False, expansions=0, query_t=0.0)
            dora.send_output("plan", arr, meta)
            continue

        query_t = float(pdef.get("query_t", 0.0))

        if planner is None:
            planner_cfg["experiment_info_default"]["node_info"]["map_res"] = float(
                pdef["map_res"]
            )
            planner = create_planner(planner_cfg, bidirectional=bidirectional)
            print(
                f"[hound_planner] IGHA* ready bi={bidirectional} "
                f"map_res={pdef['map_res']} cruise={cruise_speed_mps:.1f} m/s",
                flush=True,
            )

        t0 = time.perf_counter()
        try:
            ok, path, expansions = _search(
                planner,
                map_center=pdef["map_center"],
                start_state=pdef["start"],
                goal_xy=pdef["goal"],
                costmap=pdef["costmap"],
                heightmap=pdef["height"],
                map_res=pdef["map_res"],
                hysteresis=pdef["hysteresis"],
                expansion_limit=pdef["expansion_limit"],
                stop=False,
                cruise_speed_mps=cruise_speed_mps,
            )
        except Exception:
            traceback.print_exc()
            ok, path, expansions = False, None, 0
        dt = time.perf_counter() - t0
        arr, meta = pack_plan(
            path, success=ok, expansions=expansions, query_t=query_t
        )
        dora.send_output("plan", arr, meta)
        n = 0 if path is None else len(path)
        gxy = pdef["goal"]
        sxy = pdef["start"]
        gdist = float(np.linalg.norm(sxy[:2] - gxy[:2]))
        print(
            f"[hound_planner] ok={ok} n={n} exp={expansions} dt={dt:.3f}s "
            f"goal_dist={gdist:.2f}m",
            flush=True,
        )


if __name__ == "__main__":
    main()
