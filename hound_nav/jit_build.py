"""Force JIT build of nav CUDA kernels into ``hound_nav/cache/``.

Builds:
  - mppi analytical bicycle dynamics
  - hound_nav tracking cost
  - IGHA* kinodynamic planner environment

Usage::

  ros2 run hound_nav jit_build_cuda --clean
  python3 scripts/jit_build_cuda.py --clean
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe hound_nav/cache/torch_extensions before building",
    )
    parser.add_argument(
        "--skip-planner",
        action="store_true",
        help="Skip IGHA* kinodynamic extension",
    )
    parser.add_argument(
        "--skip-dynamics",
        action="store_true",
        help="Skip mppi bicycle dynamics",
    )
    parser.add_argument(
        "--skip-cost",
        action="store_true",
        help="Skip tracking cost CUDA",
    )
    args = parser.parse_args(argv)

    from hound_nav.cuda_cache import (
        ensure_torch_extensions_dir,
        repo_root,
        torch_extensions_dir,
    )
    from hound_nav.deps_path import setup_dependency_paths

    cache = ensure_torch_extensions_dir(clean=bool(args.clean))
    print(f"[jit_build] TORCH_EXTENSIONS_DIR={cache}", flush=True)
    if args.clean:
        print("[jit_build] cache wiped", flush=True)

    deps = setup_dependency_paths()
    print(f"[jit_build] deps={ {k: str(v) for k, v in deps.items()} }", flush=True)

    if not args.skip_dynamics:
        t0 = time.perf_counter()
        print("[jit_build] building mppi_analytical_bicycle …", flush=True)
        from mppi.Dynamics.SimpleCarDynamicsTCUDA import _load_bicycle_kernel

        _load_bicycle_kernel()
        print(
            f"[jit_build] dynamics OK  ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    if not args.skip_cost:
        t0 = time.perf_counter()
        print("[jit_build] building hound_tracking_cost …", flush=True)
        from hound_nav.trackingCostCUDA import _load_kernel

        _load_kernel()
        print(
            f"[jit_build] tracking cost OK  ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    if not args.skip_planner:
        t0 = time.perf_counter()
        print("[jit_build] building ighastar kinodynamic …", flush=True)
        import yaml
        from ighastar.scripts.common_utils import create_planner

        ssot = repo_root().parent / "hound_core" / "config" / "SSoT.yaml"
        if not ssot.is_file():
            raise FileNotFoundError(f"SSoT not found: {ssot}")
        with ssot.open(encoding="utf-8") as f:
            nav = yaml.safe_load(f)["nav"]
        planner_cfg = dict(nav["Planner_config"])
        exp = dict(planner_cfg["experiment_info_default"])
        node = dict(exp["node_info"])
        node["node_type"] = "kinodynamic"
        exp["node_info"] = node
        planner_cfg["experiment_info_default"] = exp
        bi = bool(exp.get("bidirectional", False))
        create_planner(planner_cfg, bidirectional=bi)
        print(
            f"[jit_build] planner OK  ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    print(f"[jit_build] done → {torch_extensions_dir()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
