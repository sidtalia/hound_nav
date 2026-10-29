#!/usr/bin/env python3
"""Nav bring-up. Same graph as hound_core, but this file owns HOUND_NAV_YAML.

SSoT stays the default. Set the env var to a yaml with a ``nav:`` block::

  HOUND_NAV_YAML=/root/colcon_ws/src/hound_nav/config/gates_r6.yaml \\
    ros2 launch /root/colcon_ws/src/hound_nav/launch/hound_nav.launch.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import LogInfo

_core_launch = Path(__file__).resolve().parents[2] / "hound_core" / "launch"
sys.path.insert(0, str(_core_launch))
from hound_launch_common import build_nav_dora_actions, find_ssot  # noqa: E402

# PyYAML has no negation. `-*max_vel` in a flow list is the string "-*max_vel".
_ACTION_LISTS = ("ax_list", "ay_list", "alpha_list")
_CAP_KEYS = ("max_vel", "vy_max", "max_omega")


def _resolve_action_lists(nav: dict) -> None:
    info = (
        (nav.get("Planner_config") or {})
        .get("experiment_info_default", {})
        .get("node_info")
    )
    if not isinstance(info, dict):
        return
    sampling = nav.get("Sampling_config") or {}
    caps = {}
    for key in _CAP_KEYS:
        raw = info.get(key, sampling.get(key))
        if isinstance(raw, (int, float)):
            caps[key] = float(raw)
    for name in _ACTION_LISTS:
        seq = info.get(name)
        if not isinstance(seq, list):
            continue
        resolved = []
        for item in seq:
            if isinstance(item, str) and item.startswith("-*"):
                key = item[2:]
                if key not in caps:
                    raise RuntimeError(f"{name} entry {item} has no numeric cap {key}")
                resolved.append(-caps[key])
            else:
                resolved.append(float(item))
        info[name] = resolved


def generate_launch_description():
    override = os.environ.get("HOUND_NAV_YAML", "").strip()
    if override:
        print(f"[hound_nav] Using {override} (SSoT untouched)")
        with open(override, "r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle) or {}
        nav = dict(doc["nav"] if isinstance(doc, dict) and "nav" in doc else doc)
        if not nav:
            raise RuntimeError(f"no nav: block in {override}")
    else:
        ssot_file = find_ssot()
        print(f"[hound_nav] Using SSoT file: {ssot_file}")
        with open(ssot_file, "r", encoding="utf-8") as handle:
            ssot = yaml.safe_load(handle) or {}
        nav = dict(ssot.get("nav") or {})
        if not nav:
            raise RuntimeError(f"SSoT has no nav: block ({ssot_file})")

    _resolve_action_lists(nav)

    map_topic = str(nav.get("local_map_topic", "/hound_mapping/local_map"))
    state_topic = str(nav.get("state_topic", "/hound_fcu_control/control_state"))
    goal_topic = str(nav.get("goal_topic", "/goal_pose"))
    cmd_topic = str(nav.get("cmd_topic", "/hound_nav/cmd_ackermann"))

    if nav.get("enabled", False):
        print(
            "[hound_nav] WARN: nav.enabled is true in SSoT — do not also run "
            "hound_core.launch.py with nav on, or you get two Dora graphs"
        )

    acts = build_nav_dora_actions(nav)
    return LaunchDescription(
        [
            LogInfo(
                msg=(
                    f"[hound_nav] standalone (ignores nav.enabled): "
                    f"map={map_topic} state={state_topic} "
                    f"goal={goal_topic} cmd={cmd_topic}"
                )
            ),
            *acts,
        ]
    )
