"""Locate IGHAStar / mppi without modifying those repos."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _candidates() -> list[Path]:
    env_ws = os.environ.get("HOUND_DEPS_ROOT") or os.environ.get("COLCON_PREFIX_PATH")
    roots: list[Path] = []
    here = Path(__file__).resolve()
    roots.append(here.parents[2])  # .../src
    roots.append(Path("/root/colcon_ws/src"))
    roots.append(Path("/home/hound/colcon_ws/src"))
    if env_ws:
        for part in str(env_ws).split(os.pathsep):
            p = Path(part)
            if p.name == "install":
                roots.append(p.parent / "src")
            roots.append(p)
    return roots


def find_repo(name: str) -> Path:
    for root in _candidates():
        cand = root / name
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find {name}/ under workspace src. "
        "Clone it next to hound_nav or set HOUND_DEPS_ROOT."
    )


def load_class(spec: str, class_name: str):
    """Load ``class_name`` from a .py path.

    ``spec`` is absolute, or ``<repo>/<path-inside-repo>`` under workspace src
    (``mppi/mppi/Dynamics/SimpleCarDynamicsTCUDA.py``).
    """
    path = Path(spec)
    if not path.is_file():
        repo, _, rest = spec.partition("/")
        path = find_repo(repo) / rest
    if not path.is_file():
        raise FileNotFoundError(spec)
    # ponytail: one module name per stem; two files with the same stem collide.
    name = "hound_cfg_" + path.stem
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(spec)
    mod = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = mod
    module_spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def setup_dependency_paths() -> dict[str, Path]:
    """Insert dependency roots on sys.path (idempotent).

    Also pins Torch CUDA JIT builds to ``$ROS_WORKSPACE/cache/torch_extensions``.
    """
    from hound_nav.cuda_cache import ensure_torch_extensions_dir

    ensure_torch_extensions_dir(clean=False)

    igha = find_repo("IGHAStar")
    paths = [
        str(igha),  # import ighastar.*
        str(igha / "examples" / "BeamNG"),  # create_planner helpers
    ]
    # Prefer installed/editable mppi; also allow src/mppi checkout.
    try:
        mppi_root = find_repo("mppi")
        paths.insert(0, str(mppi_root))
    except FileNotFoundError:
        mppi_root = None
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    out: dict[str, Path] = {"IGHAStar": igha}
    if mppi_root is not None:
        out["mppi"] = mppi_root
    return out


if __name__ == "__main__":
    cls = load_class("mppi/mppi/Sampling/Delta_Sampling.py", "Delta_Sampling")
    assert cls.__name__ == "Delta_Sampling"
    try:
        load_class("/no/such/file.py", "X")
    except FileNotFoundError:
        pass
    else:
        raise SystemExit("missing file should fail")
    print("ok")
