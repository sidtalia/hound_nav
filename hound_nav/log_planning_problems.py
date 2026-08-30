#!/usr/bin/env python3
"""Log LocalMap + control_state at 1 Hz; on Ctrl+C emit start–goal pairs.

  ros2 run hound_nav log_planning_problems --ros-args \\
    -p use_sim_time:=true \\
    -p local_map_topic:=/debug/hound_mapping/local_map

  python3 -m hound_nav.log_planning_problems --pairs-only /path/to/logdir
"""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

from hound_mapping.msg import LocalMap

from hound_nav.bag_problem_io import generate_problem_pairs, save_snapshot, save_state_history


def _img_f32(msg) -> Optional[np.ndarray]:
    if msg.width == 0 or msg.height == 0 or not msg.data:
        return None
    h, w = int(msg.height), int(msg.width)
    if msg.encoding != "32FC1":
        return None
    return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()


def _default_log_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for root in (Path("/home/hound/colcon_ws"), Path("/root/colcon_ws")):
        if root.is_dir():
            return root / "planning_problems" / stamp
    return Path.cwd() / "planning_problems" / stamp


class ProblemLogger(Node):
    def __init__(self) -> None:
        super().__init__("hound_planning_problem_logger")
        self.declare_parameter("local_map_topic", "/debug/hound_mapping/local_map")
        self.declare_parameter("state_topic", "/hound_fcu_control/control_state")
        self.declare_parameter("control_state_dims", 17)
        self.declare_parameter("log_hz", 1.0)
        self.declare_parameter("goal_horizon_s", 5.0)
        self.declare_parameter("min_xy_dist_m", 0.5)
        self.declare_parameter("problems_dir", str(_default_log_dir()))

        self._map_topic = str(self.get_parameter("local_map_topic").value)
        self._state_topic = str(self.get_parameter("state_topic").value)
        self._dims = int(self.get_parameter("control_state_dims").value)
        self._horizon = float(self.get_parameter("goal_horizon_s").value)
        self._min_dist = float(self.get_parameter("min_xy_dist_m").value)
        self._root = Path(str(self.get_parameter("problems_dir").value)).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

        self._latest_map: Optional[dict] = None
        self._latest_state: Optional[np.ndarray] = None
        self._state_t: list[float] = []
        self._state_x: list[np.ndarray] = []
        self._snap_i = 0
        self._last_log_t = -1e9
        self._done = False

        qos = qos_profile_sensor_data
        self.create_subscription(LocalMap, self._map_topic, self._on_map, qos)
        self.create_subscription(
            Float64MultiArray, self._state_topic, self._on_state, qos
        )
        hz = float(self.get_parameter("log_hz").value)
        period = 1.0 / max(hz, 0.1)
        self.create_timer(period, self._on_timer)
        self.get_logger().info(
            f"logging → {self._root} map={self._map_topic} state={self._state_topic} "
            f"{hz:.2f} Hz horizon={self._horizon:.1f}s"
        )

    def _stamp_sec(self, stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_map(self, msg: LocalMap) -> None:
        elev = _img_f32(msg.elevation)
        cost = _img_f32(msg.costmap)
        if elev is None or cost is None or elev.shape != cost.shape:
            return
        t = self._stamp_sec(msg.header.stamp)
        if t <= 0.0:
            t = self._now_sec()
        self._latest_map = {
            "t": t,
            "elevation": elev,
            "cost": cost,
            "origin": np.array(
                [float(msg.info.origin.position.x), float(msg.info.origin.position.y)],
                dtype=np.float64,
            ),
            "resolution": float(msg.info.resolution),
        }

    def _on_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < self._dims:
            return
        st = np.asarray(msg.data[: self._dims], dtype=np.float64)
        t = self._now_sec()
        self._latest_state = st
        self._state_t.append(t)
        self._state_x.append(st.copy())

    def _on_timer(self) -> None:
        if self._latest_map is None or self._latest_state is None:
            return
        t = float(self._latest_map["t"])
        if t - self._last_log_t < 0.5:
            return
        self._last_log_t = t
        save_snapshot(
            self._root,
            self._snap_i,
            {
                "t": np.float64(t),
                "state": self._latest_state,
                "elevation": self._latest_map["elevation"],
                "cost": self._latest_map["cost"],
                "origin": self._latest_map["origin"],
                "resolution": np.float64(self._latest_map["resolution"]),
            },
        )
        self._snap_i += 1
        if self._snap_i % 10 == 0:
            if self._state_t:
                save_state_history(
                    self._root,
                    np.asarray(self._state_t, dtype=np.float64),
                    np.stack(self._state_x, axis=0),
                )
            self.get_logger().info(f"snapshots={self._snap_i} t={t:.2f}")

    def finalize(self) -> None:
        if self._done:
            return
        self._done = True
        if self._state_t:
            save_state_history(
                self._root,
                np.asarray(self._state_t, dtype=np.float64),
                np.stack(self._state_x, axis=0),
            )
        n = generate_problem_pairs(
            self._root, horizon_s=self._horizon, min_xy_dist_m=self._min_dist
        )
        self.get_logger().info(
            f"wrote {self._snap_i} snapshots, {n} problems under {self._root}"
        )


def _pairs_only(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-only", type=Path, required=True)
    p.add_argument("--horizon", type=float, default=5.0)
    p.add_argument("--min-dist", type=float, default=0.5)
    p.add_argument(
        "--until-min-dist",
        action="store_true",
        help="Goal = first later pose with XY dist >= --min-dist (not t+horizon)",
    )
    p.add_argument("--problems-subdir", default="problems")
    args = p.parse_args(argv)
    n = generate_problem_pairs(
        args.pairs_only,
        horizon_s=args.horizon,
        min_xy_dist_m=args.min_dist,
        until_min_dist=args.until_min_dist,
        problems_subdir=args.problems_subdir,
    )
    print(f"wrote {n} problems under {args.pairs_only}")
    return 0


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--pairs-only" in argv:
        raise SystemExit(_pairs_only(argv))

    rclpy.init(args=None)
    node = ProblemLogger()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def _stop(*_a) -> None:
        node.finalize()
        executor.shutdown()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
