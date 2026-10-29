#!/usr/bin/env python3
"""Log the MPPI twist and /odom while CH5 is AUTO and the deadman is held.

Started by run_nav.sh. One CSV per hold, under logs/. The command columns are
the twist on /hound_nav/cmd_ackermann (vx, vy, wz), which is the MPPI action
divided by the speed knob. Ctrl-C closes the open file.
"""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

# /api/stamplc drive.mode is MECANUM/SKID, not CH5. AUTO and the deadman are here.
_URL = "http://127.0.0.1:8080/api/snapshot"
_CMD_TOPIC = "/hound_nav/cmd_ackermann"
_ODOM_TOPIC = "/odom"
_HORIZON_TOPIC = "/hound_nav/mppi_topk_twist"
_RATE_HZ = 20.0
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_gate_lock = threading.Lock()
_gate_open = False


def gate(rc) -> bool:
    """True only in AUTO with the deadman held. rc is snapshot telemetry.rc."""
    return isinstance(rc, dict) and rc.get("mode") == "AUTO" and bool(rc.get("deadman"))


def _selfcheck() -> None:
    assert gate({"mode": "AUTO", "deadman": True})
    assert not gate({"mode": "AUTO", "deadman": False})
    assert not gate({"mode": "MANUAL", "deadman": True})
    assert not gate({"mode": "OFF", "deadman": True})
    assert not gate({})
    assert not gate(None)
    print("ok")


def _fetch_gate() -> bool:
    try:
        with urllib.request.urlopen(_URL, timeout=1.0) as resp:
            rc = (json.load(resp).get("telemetry") or {}).get("rc") or {}
    except (OSError, ValueError):
        return False
    return gate(rc)


def _gate_loop() -> None:
    """Snapshot takes ~0.15 s. Poll it off the log timer so the rows stay at 20 Hz."""
    global _gate_open
    while True:
        opened = _fetch_gate()
        with _gate_lock:
            _gate_open = opened
        time.sleep(0.25)


def main() -> None:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float64MultiArray

    class Log(Node):
        def __init__(self) -> None:
            super().__init__("nav_drive_log")
            self.create_subscription(Twist, _CMD_TOPIC, self._on_cmd, qos_profile_sensor_data)
            self.create_subscription(Odometry, _ODOM_TOPIC, self._on_odom, qos_profile_sensor_data)
            self.create_subscription(Float64MultiArray, _HORIZON_TOPIC, self._on_horizon, 10)
            self._cmd = None
            self._odom = None
            self._horizon = None
            self._open = False
            self._fh = None
            self._path = None
            self._t0 = 0.0
            self._rows = 0
            self._shown = 0.0
            self._noted = False
            self.create_timer(1.0 / _RATE_HZ, self._tick)
            self.get_logger().info(
                "waiting for AUTO and the deadman; then logging "
                f"{_CMD_TOPIC} and {_ODOM_TOPIC} to {_LOG_DIR}"
            )

        def _on_cmd(self, msg: Twist) -> None:
            self._cmd = (msg.linear.x, msg.linear.y, msg.angular.z)

        def _on_odom(self, msg: Odometry) -> None:
            t = msg.twist.twist
            self._odom = (t.linear.x, t.linear.y, t.angular.z)

        def _on_horizon(self, msg: Float64MultiArray) -> None:
            d = msg.data
            if len(d) < 5:
                return
            T = int(d[1])
            if T < 1 or len(d) < 2 + 3 * T:
                return
            self._horizon = (float(d[0]), T, [float(x) for x in d[2 : 2 + 3 * T]])

        def _armed_now(self) -> bool:
            with _gate_lock:
                return _gate_open

        def _begin(self) -> None:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._path = _LOG_DIR / f"nav_drive_{stamp}.csv"
            self._fh = open(self._path, "w", encoding="utf-8")
            self._fh.write(
                "t,cmd_vx,cmd_vy,cmd_wz,odom_vx,odom_vy,odom_wz,"
                "hdt,steps,vx0,vy0,wz0,...\n"
            )
            latest = _LOG_DIR / "latest.csv"
            tmp = _LOG_DIR / ".latest.csv.tmp"
            tmp.unlink(missing_ok=True)
            tmp.symlink_to(self._path.name)
            tmp.replace(latest)
            self._t0 = time.monotonic()
            self._rows = 0
            self._shown = 0.0
            self._open = True
            self.get_logger().info(f"logging {self._path}")

        def _end(self) -> None:
            if self._fh is None:
                return
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._open = False
            self.get_logger().info(f"hold ended, {self._rows} rows in {self._path}")

        def _tick(self) -> None:
            if not self._armed_now():
                self._noted = False
                if self._open:
                    self._end()
                return
            if self._cmd is None or self._odom is None:
                if not self._noted:
                    self._noted = True
                    self.get_logger().info("AUTO and deadman held, waiting for cmd and odom")
                return
            if not self._open:
                self._begin()
            t = time.monotonic() - self._t0
            cx, cy, cw = self._cmd
            ox, oy, ow = self._odom
            if self._horizon is None:
                extra = ",nan,0"
            else:
                hdt, T, vals = self._horizon
                extra = f",{hdt:.3f},{T}," + ",".join(f"{v:.4f}" for v in vals)
            self._fh.write(
                f"{t:.3f},{cx:.4f},{cy:.4f},{cw:.4f},{ox:.4f},{oy:.4f},{ow:.4f}{extra}\n"
            )
            self._rows += 1
            self._fh.flush()
            if t - self._shown >= 1.0:
                self._shown = t
                self.get_logger().info(
                    f"cmd {cx:+.3f} {cy:+.3f} {cw:+.3f}  odom {ox:+.3f} {oy:+.3f} {ow:+.3f}"
                )

        def close(self) -> None:
            self._end()

    from rclpy.executors import ExternalShutdownException

    threading.Thread(target=_gate_loop, daemon=True).start()
    rclpy.init()
    node = Log()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selfcheck()
    else:
        main()
