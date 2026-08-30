#!/usr/bin/env python3
"""Run workspace IGHA* / BiIGHA* on logged bag problems; plot cost vs expansions.

Uses Planner_config copied from SSoT (or --config). Does not use IGHAStar_private.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hound_nav.bag_problem_io import (
    costmap_igha,
    dump_planner_config,
    list_problems,
    load_planner_config,
    plant_to_planner_state,
    relative_height,
    world_to_map_xy,
)
from hound_nav.deps_path import setup_dependency_paths


def _bitmap(cost: np.ndarray, height: np.ndarray) -> torch.Tensor:
    bitmap = torch.ones((cost.shape[0], cost.shape[1], 2), dtype=torch.float32)
    bitmap[..., 0] = torch.from_numpy(np.ascontiguousarray(cost))
    bitmap[..., 1] = torch.from_numpy(np.ascontiguousarray(height))
    return bitmap


def _start_goal_tensors(
    prob, *, free_costmap: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    elev = np.asarray(prob["elevation"])
    if free_costmap:
        cost = np.full(elev.shape[:2], 255.0, dtype=np.float32)
    else:
        cost = costmap_igha(np.asarray(prob["cost"]))
    start_w = np.asarray(prob["start_state"], dtype=np.float64)
    goal_w = np.asarray(prob["goal_state"], dtype=np.float64)
    res = float(prob["resolution"])
    center = np.asarray(prob["map_center"], dtype=np.float64)
    height = relative_height(elev, float(start_w[2]) if start_w.size > 2 else 0.0)
    bitmap = _bitmap(cost, height)

    ps = plant_to_planner_state(start_w)
    pg = plant_to_planner_state(goal_w)
    sm = world_to_map_xy(ps, center, res, elev.shape)
    gm = world_to_map_xy(pg, center, res, elev.shape)
    start = torch.zeros(4, dtype=torch.float32)
    goal = torch.zeros(4, dtype=torch.float32)
    start[0] = float(sm[0])
    start[1] = float(sm[1])
    start[2] = float(ps[2])
    start[3] = float(ps[3])
    goal[0] = float(gm[0])
    goal[1] = float(gm[1])
    goal[2] = float(pg[2])
    goal[3] = float(pg[3])
    return start, goal, bitmap


def _profiler_result(prof, solve_time_s: float) -> Dict[str, Any]:
    cost = list(prof[9]) if prof is not None and len(prof) > 9 else []
    n_exp = int(prof[7]) if prof is not None and len(prof) > 7 else len(cost)
    avg_succ_us = float(prof[0]) if prof is not None and len(prof) > 0 else float("nan")
    if n_exp < 1:
        n_exp = max(len(cost), 1)
    exp_list = list(range(1, len(cost) + 1))
    dummy = [0.0] * len(cost)
    return {
        "expansions": exp_list,
        "Seen_size": dummy,
        "Q_v_size": dummy,
        "best_cost": cost,
        "solve_time_s": float(solve_time_s),
        "n_expansions": int(n_exp),
        "avg_successor_us": avg_succ_us,
        "time_per_expansion_us": 1e6 * float(solve_time_s) / float(n_exp),
    }


def _save_pkl(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def run_playback(
    problems_dir: Path,
    *,
    config_path: Optional[str],
    results_dir: Path,
    exp: int,
    run_igha: bool,
    run_bi: bool,
    cruise_speed: Optional[float],
    max_problems: Optional[int],
    free_costmap: bool = False,
    problems_subdir: str = "problems",
) -> None:
    setup_dependency_paths()
    from ighastar.scripts.common_utils import create_planner

    planner_cfg = load_planner_config(config_path)
    dump_planner_config(planner_cfg, results_dir / "planner_config.yaml")
    info = planner_cfg.setdefault("experiment_info_default", {})
    hyst = int(info.get("hysteresis", 100))
    problems = list_problems(problems_dir, subdir=problems_subdir)
    if max_problems is not None:
        problems = problems[: int(max_problems)]
    if not problems:
        raise SystemExit(f"no problems in {problems_dir / problems_subdir}")

    first = np.load(problems[0])
    info["node_info"]["map_res"] = float(first["resolution"])

    uni = create_planner(deepcopy(planner_cfg), bidirectional=False) if run_igha else None
    bi = create_planner(deepcopy(planner_cfg), bidirectional=True) if run_bi else None

    n = len(problems)
    for i, path in enumerate(problems):
        prob = np.load(path)
        start, goal, bitmap = _start_goal_tensors(prob, free_costmap=free_costmap)
        if cruise_speed is not None:
            goal[3] = float(cruise_speed)
        print(
            f"[{i + 1}/{n}] {path.name} start=({float(start[0]):.2f},{float(start[1]):.2f}) "
            f"goal=({float(goal[0]):.2f},{float(goal[1]):.2f}) v={float(start[3]):.2f}",
            flush=True,
        )
        if uni is not None:
            t0 = time.perf_counter()
            uni.search(start, goal, bitmap, int(exp), hyst, True)
            dt = time.perf_counter() - t0
            _save_pkl(
                _profiler_result(uni.get_profiler_info(), dt),
                results_dir / f"IGHAStar_test_{i}_0_bag.pkl",
            )
        if bi is not None:
            t0 = time.perf_counter()
            bi.search(start, goal, bitmap, int(exp), hyst, True)
            dt = time.perf_counter() - t0
            _save_pkl(
                _profiler_result(bi.get_profiler_info(), dt),
                results_dir / f"BiIGHAStar_LCR_1_{i}_0_bag.pkl",
            )
        gc.collect()
    print(f"results → {results_dir}", flush=True)


def plot_cost_playback(
    results_dir: Path,
    algorithm_names: List[str],
    exp: int,
    out_path: Optional[Path] = None,
) -> None:
    import matplotlib.pyplot as plt

    aggregated: Dict[str, List[np.ndarray]] = defaultdict(list)
    timing: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for file in sorted(results_dir.glob("*.pkl")):
        name = file.name
        with file.open("rb") as f:
            results = pickle.load(f)
        if "IGHAStar_test" in name and "BiIGHAStar" not in name:
            algo = "IGHAStar_test"
        elif "BiIGHAStar_LCR_1" in name:
            algo = "BiIGHA*_LCR_1"
        else:
            continue
        aggregated[algo].append(np.asarray(results["best_cost"], dtype=np.float64))
        n_exp = float(results.get("n_expansions") or len(results["best_cost"]) or 1)
        solve_s = float(results.get("solve_time_s", float("nan")))
        timing[algo].append(
            {
                "solve_time_s": solve_s,
                "n_expansions": n_exp,
                "time_per_expansion_us": float(
                    results.get(
                        "time_per_expansion_us",
                        1e6 * solve_s / max(n_exp, 1.0),
                    )
                ),
                "avg_successor_us": float(results.get("avg_successor_us", float("nan"))),
            }
        )

    algo_to_label = {"IGHAStar_test": "IGHA*", "BiIGHA*_LCR_1": "Bi-IGHA*"}
    color_map = {"IGHA*": "#f4a582", "Bi-IGHA*": "#ca0020"}
    all_expansions = np.arange(int(exp))

    def first_sol(costs: np.ndarray) -> float:
        idx = np.where(costs < 1e4)[0]
        return float(idx[0]) if len(idx) else float("inf")

    lists: Dict[str, List[np.ndarray]] = {}
    valid = {}
    for algo in algorithm_names:
        padded = []
        ok = set()
        for i, raw in enumerate(aggregated.get(algo, [])):
            row = np.full(len(all_expansions), np.nan, dtype=float)
            n = min(len(raw), len(all_expansions))
            vals = raw[:n].copy()
            vals[vals >= 1e4] = np.nan
            row[:n] = vals
            if n:
                last = row[n - 1]
                if np.isfinite(last):
                    row[n:] = last
            padded.append(row)
            if first_sol(row) < float(exp):
                ok.add(i)
        lists[algo] = padded
        valid[algo] = ok
        print(f"{algo}: {len(ok)}/{len(padded)} solved within {exp} expansions")

    nonempty = [valid[a] for a in algorithm_names if lists[a]]
    if not nonempty:
        raise SystemExit(f"no result pkls in {results_dir}")
    common = (
        sorted(set.intersection(*nonempty)) if len(nonempty) > 1 else sorted(nonempty[0])
    )
    print(f"common problems: {len(common)}")

    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(4, 3))
    for algo in algorithm_names:
        if not lists[algo]:
            continue
        data = np.array([lists[algo][i] for i in common if i < len(lists[algo])])
        if data.size == 0:
            continue
        with np.errstate(all="ignore"):
            mean = np.nanmean(data, axis=0)
            n_valid = np.sum(~np.isnan(data), axis=0)
            std = np.nanstd(data, axis=0)
        ci = 1.96 * std / np.sqrt(np.maximum(n_valid, 1))
        label = algo_to_label.get(algo, algo)
        color = color_map.get(label, "black")
        ax.plot(all_expansions, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(all_expansions, mean - ci, mean + ci, color=color, alpha=0.2)
    ax.set_xlabel("expansions")
    ax.set_ylabel("best cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    dest = out_path or (results_dir / "cost_vs_expansions.pdf")
    fig.savefig(dest)
    fig.savefig(dest.with_suffix(".png"))
    print(f"wrote {dest} and {dest.with_suffix('.png')}")

    fig_s, ax_s = plt.subplots(figsize=(4, 3))
    for algo in algorithm_names:
        if not lists[algo]:
            continue
        data = np.array(lists[algo])
        if data.size == 0:
            continue
        success = np.mean(~np.isnan(data), axis=0)
        label = algo_to_label.get(algo, algo)
        color = color_map.get(label, "black")
        ax_s.plot(all_expansions, success, color=color, label=label, linewidth=1.5)
    ax_s.set_xlabel("expansions")
    ax_s.set_ylabel("success rate")
    ax_s.set_ylim(0.0, 1.05)
    ax_s.legend()
    ax_s.grid(True, alpha=0.3)
    fig_s.tight_layout()
    succ_path = results_dir / "success_vs_expansions.pdf"
    fig_s.savefig(succ_path)
    fig_s.savefig(succ_path.with_suffix(".png"))
    print(f"wrote {succ_path} and {succ_path.with_suffix('.png')}")

    metrics: Dict[str, Any] = {}
    print("\nTiming (mean ± std over problems):")
    for algo in algorithm_names:
        rows = timing.get(algo) or []
        if not rows:
            continue
        tpe = np.array([r["time_per_expansion_us"] for r in rows], dtype=np.float64)
        succ = np.array([r["avg_successor_us"] for r in rows], dtype=np.float64)
        nexp = np.array([r["n_expansions"] for r in rows], dtype=np.float64)
        wall = np.array([r["solve_time_s"] for r in rows], dtype=np.float64)
        label = algo_to_label.get(algo, algo)
        entry = {
            "n": int(len(rows)),
            "solve_time_s_mean": float(np.nanmean(wall)),
            "n_expansions_mean": float(np.nanmean(nexp)),
            "solve_time_per_expansion_us_mean": float(np.nanmean(tpe)),
            "solve_time_per_expansion_us_std": float(np.nanstd(tpe)),
            "avg_successor_us_mean": float(np.nanmean(succ)),
            "avg_successor_us_std": float(np.nanstd(succ)),
        }
        metrics[label] = entry
        print(
            f"  {label}: solve_time/exp = {entry['solve_time_per_expansion_us_mean']:.1f} "
            f"± {entry['solve_time_per_expansion_us_std']:.1f} µs   "
            f"avg_succ = {entry['avg_successor_us_mean']:.1f} "
            f"± {entry['avg_successor_us_std']:.1f} µs   "
            f"(n={entry['n']}, mean expansions={entry['n_expansions_mean']:.0f})"
        )
    (results_dir / "timing_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {results_dir / 'timing_metrics.json'}")


def viz_one_problem(
    problems_dir: Path,
    *,
    config_path: Optional[str],
    problem_id: int,
    exp: int,
    free_costmap: bool,
    bidirectional: bool,
    cruise_speed: Optional[float],
    port: int,
) -> None:
    import math

    deps = setup_dependency_paths()
    from ighastar.scripts.common_utils import create_planner

    standalone = deps["IGHAStar"] / "examples" / "standalone"
    if str(standalone) not in sys.path:
        sys.path.insert(0, str(standalone))
    from viser_replay import replay_trajectory

    planner_cfg = load_planner_config(config_path)
    info = planner_cfg.setdefault("experiment_info_default", {})
    hyst = int(info.get("hysteresis", 100))
    problems = list_problems(problems_dir)
    if not problems:
        raise SystemExit(f"no problems in {problems_dir / 'problems'}")
    if problem_id < 0 or problem_id >= len(problems):
        raise SystemExit(f"--problem {problem_id} out of range 0..{len(problems) - 1}")

    path_npz = problems[problem_id]
    prob = np.load(path_npz)
    info["node_info"]["map_res"] = float(np.load(problems[0])["resolution"])

    start, goal, bitmap = _start_goal_tensors(prob, free_costmap=free_costmap)
    if cruise_speed is not None:
        goal[3] = float(cruise_speed)

    start_w = np.asarray(prob["start_state"], dtype=np.float64)
    goal_w = np.asarray(prob["goal_state"], dtype=np.float64)
    print(
        f"problem {problem_id} {path_npz.name}\n"
        f"  logged start xyz/yaw/speed: "
        f"({start_w[0]:.2f},{start_w[1]:.2f},{start_w[2]:.2f})  "
        f"yaw={math.degrees(float(start_w[5])):.1f} deg  "
        f"|v|={float(np.hypot(start_w[6], start_w[7])):.2f}\n"
        f"  logged goal  xyz/yaw/speed: "
        f"({goal_w[0]:.2f},{goal_w[1]:.2f},{goal_w[2]:.2f})  "
        f"yaw={math.degrees(float(goal_w[5])):.1f} deg  "
        f"|v|={float(np.hypot(goal_w[6], goal_w[7])):.2f}\n"
        f"  planner start (map xy, yaw, v): "
        f"({float(start[0]):.2f},{float(start[1]):.2f}, "
        f"{math.degrees(float(start[2])):.1f} deg, {float(start[3]):.2f})\n"
        f"  planner goal  (map xy, yaw, v): "
        f"({float(goal[0]):.2f},{float(goal[1]):.2f}, "
        f"{math.degrees(float(goal[2])):.1f} deg, {float(goal[3]):.2f})",
        flush=True,
    )

    planner = create_planner(deepcopy(planner_cfg), bidirectional=bidirectional)
    ok = planner.search(start, goal, bitmap, int(exp), hyst, True)
    path = planner.get_best_path()
    path_np = path.numpy() if hasattr(path, "numpy") else np.asarray(path)
    print(f"search ok={ok} path_len={len(path_np)}", flush=True)
    if path_np.size == 0 or path_np.ndim != 2:
        raise SystemExit(
            "no path to visualize — try another --problem or --free-costmap"
        )

    replay_trajectory(
        path_np,
        bitmap,
        info["node_info"],
        stride=2,
        port=int(port),
        fps=20.0,
        block=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="IGHA*/BiIGHA* playback on bag problems")
    p.add_argument("problems_dir", type=Path)
    p.add_argument("--config", default="", help="YAML with Planner_config (default: SSoT)")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--exp", type=int, default=5000)
    p.add_argument("--no-igha", action="store_true")
    p.add_argument("--no-bi", action="store_true")
    p.add_argument("--cruise-speed", type=float, default=None)
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument(
        "--free-costmap",
        action="store_true",
        help="Set every cost cell to 255 (free); keep elevation",
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["IGHAStar_test", "BiIGHA*_LCR_1"],
    )
    p.add_argument(
        "--viser",
        action="store_true",
        help="Plan one problem and replay it in Viser (IGHAStar standalone viz)",
    )
    p.add_argument("--problem", type=int, default=10, help="Problem index for --viser")
    p.add_argument("--viser-port", type=int, default=8081)
    p.add_argument(
        "--uni",
        action="store_true",
        help="With --viser, use IGHA* instead of BiIGHA*",
    )
    p.add_argument(
        "--problems-subdir",
        default="problems",
        help="Problem folder under problems_dir (e.g. problems_8m)",
    )
    args = p.parse_args()
    problems_dir = args.problems_dir.expanduser().resolve()
    if args.viser:
        viz_one_problem(
            problems_dir,
            config_path=args.config or None,
            problem_id=int(args.problem),
            exp=args.exp,
            free_costmap=args.free_costmap,
            bidirectional=not args.uni,
            cruise_speed=args.cruise_speed,
            port=int(args.viser_port),
        )
        return
    if args.results_dir is not None:
        results_dir = args.results_dir.expanduser().resolve()
    elif args.problems_subdir != "problems":
        tag = "free" if args.free_costmap else "logged"
        results_dir = problems_dir / f"results_{tag}_{args.problems_subdir}"
    else:
        results_dir = problems_dir / (
            "results_free_cost" if args.free_costmap else "results"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg = args.config or None
    if not args.plot_only:
        run_playback(
            problems_dir,
            config_path=cfg,
            results_dir=results_dir,
            exp=args.exp,
            run_igha=not args.no_igha,
            run_bi=not args.no_bi,
            cruise_speed=args.cruise_speed,
            max_problems=args.max_problems,
            free_costmap=args.free_costmap,
            problems_subdir=args.problems_subdir,
        )
    plot_cost_playback(results_dir, args.algorithms, args.exp)


if __name__ == "__main__":
    main()
