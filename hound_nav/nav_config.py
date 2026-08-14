"""Load HOUND_NAV_CONFIG JSON (SSoT ``nav:`` flattened by launch)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

_STACK_KEYS: List[str] = [
    "MPPI_config",
    "Map_config",
    "Dynamics_config",
    "Sampling_config",
    "Cost_config",
    "Planner_config",
    "lookahead",
    "wp_radius",
]


def load_hound_nav_config() -> Dict[str, Any]:
    """Return launch JSON from ``HOUND_NAV_CONFIG`` (required)."""
    path = str(os.environ.get("HOUND_NAV_CONFIG", "") or "")
    if not path:
        raise RuntimeError("HOUND_NAV_CONFIG is required")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, (list, tuple)):
        return float(x[0]) if x else float(default)
    return float(x)


def stack_config(launch_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """MPPI / planner stack dict (must be embedded in HOUND_NAV_CONFIG from SSoT)."""
    missing = [k for k in _STACK_KEYS if k not in launch_cfg]
    if missing:
        raise RuntimeError(
            "HOUND_NAV_CONFIG missing SSoT nav stack keys: "
            + ", ".join(missing)
            + " (set under nav: in SSoT.yaml; launch embeds them)"
        )
    return {k: launch_cfg[k] for k in _STACK_KEYS}
