#!/usr/bin/env python3
"""Hold the base and sweep one axis at a time with a square wave.

After the lease, publish zero for kArmHoldMs (the firmware's ESC arm hold) so the
motors are armed before the waveform. Order: strafe, then forward, then yaw.
Each lasts one period (pi seconds). Records the twist we send and the /odom
twist, then writes a plot with the fundamental gain and phase of each axis.
Ctrl-C stops early, still writes whatever was recorded, and releases the lease.

Stop the nav stack first. It holds the lease as "nav", and this claim is refused
while that lease is live.
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hound_nav.wire_scale import knob, to_wire

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String

OWNER = "test"
RATE_HZ = 20.0
# firmware/rc_serial_probe/src/main.cpp kArmHoldMs. Zeros on the wire for this
# long, deadman held, are what arm the ESCs. The square starts after it.
ARM_HOLD_S = 2.0
# One cycle of the square. +amp for the first half, -amp for the second.
# Amplitudes are the symmetric caps in config/gates_r6_f2.yaml (max_vel, vy_max, max_omega).
PERIOD_S = math.pi
PHASES = ("y", "x", "yaw")
AMP = {"x": 0.38, "y": 0.25, "yaw": 0.52}
# Column of cmd_* and odom_* for each phase. cmd starts at column 1, odom at 4.
AXIS = {"x": 0, "y": 1, "yaw": 2}
LABEL = {"x": "vx", "y": "vy", "yaw": "wz"}
CSV_PATH = "/tmp/drive_sine.csv"
PLOT_PATH = str(Path(__file__).resolve().parents[1] / "plots" / "drive_sine.png")


def command(phase: str, t: float) -> tuple[float, float, float]:
    s = AMP[phase] if (t % PERIOD_S) < PERIOD_S / 2.0 else -AMP[phase]
    if phase == "x":
        return s, 0.0, 0.0
    if phase == "y":
        return 0.0, s, 0.0
    return 0.0, 0.0, s


def bode(cmd: np.ndarray, meas: np.ndarray, dt: float, period: float) -> tuple[float, float] | None:
    """Fundamental gain and phase (degrees) of meas against cmd. None when either is flat.

    Phase is negative when odom lags the square. Gain is the ratio of the two
    fundamentals, so 1 means /odom matched the commanded amplitude.
    """
    w = 2.0 * math.pi / period
    t = np.arange(len(cmd)) * dt
    sn, cs = np.sin(w * t), np.cos(w * t)

    def phasor(x: np.ndarray) -> tuple[float, float]:
        # x ≈ S sin(wt) + C cos(wt); magnitude hypot(S, C), phase atan2(C, S).
        return 2.0 * float(np.mean(x * sn)), 2.0 * float(np.mean(x * cs))

    sc, cc = phasor(cmd)
    sm, cm = phasor(meas)
    rc, rm = math.hypot(sc, cc), math.hypot(sm, cm)
    if rc < 1e-6 or rm < 1e-6:
        return None
    phase = math.degrees(math.atan2(cm, sm) - math.atan2(cc, sc))
    phase = (phase + 180.0) % 360.0 - 180.0
    return rm / rc, phase


class Sine(Node):
    def __init__(self) -> None:
        super().__init__("drive_sine")
        self._claim = self.create_publisher(String, "/base/claim", 10)
        self._release = self.create_publisher(String, "/base/release", 10)
        self._cmd = self.create_publisher(Twist, f"/base/{OWNER}/cmd_vel", 10)
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, "/base/owner", self._on_owner, latched)
        self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        self._odom = (0.0, 0.0, 0.0)
        self._odom_t = 0.0
        self._rows: list[tuple] = []
        self._held_at = 0.0
        self._t0 = 0.0
        self._ticks = 0
        self._done = False
        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.get_logger().info(
            f"owner={OWNER} waiting for the lease, then {ARM_HOLD_S:.0f}s of zero, "
            f"then a square on y, then x, then yaw, {PERIOD_S:.1f}s each, "
            f"wire = v / knob ({knob():.2f})"
        )

    def _on_owner(self, msg: String) -> None:
        self.get_logger().info(msg.data, throttle_duration_sec=2.0)
        try:
            owner = json.loads(msg.data).get("owner")
        except json.JSONDecodeError:
            return
        if owner == OWNER and self._held_at == 0.0:
            self._held_at = time.monotonic()
            self.get_logger().info(f"lease held, sending zero for {ARM_HOLD_S:.0f}s")

    def _on_odom(self, msg: Odometry) -> None:
        t = msg.twist.twist
        self._odom = (float(t.linear.x), float(t.linear.y), float(t.angular.z))
        self._odom_t = time.monotonic()

    def _tick(self) -> None:
        if self._ticks % int(RATE_HZ) == 0:
            self._claim.publish(String(data=json.dumps({"owner": OWNER, "ttl_s": 5})))
        self._ticks += 1
        if self._held_at == 0.0:
            return
        if time.monotonic() - self._held_at < ARM_HOLD_S:
            self._cmd.publish(Twist())
            return
        if self._t0 == 0.0:
            self._t0 = time.monotonic()
            self.get_logger().info("square start")
        elapsed = time.monotonic() - self._t0
        phase_i = min(int(elapsed / PERIOD_S), len(PHASES) - 1)
        phase = PHASES[phase_i]
        local = elapsed - phase_i * PERIOD_S
        if elapsed >= len(PHASES) * PERIOD_S:
            self._done = True
            return
        vx, vy, wz = command(phase, local)
        wx, wy, ww = to_wire(vx, vy, wz)
        twist = Twist()
        twist.linear.x = wx
        twist.linear.y = wy
        twist.angular.z = ww
        self._cmd.publish(twist)
        ox, oy, ow = self._odom
        age = time.monotonic() - self._odom_t if self._odom_t else -1.0
        self._rows.append((elapsed, phase, vx, vy, wz, ox, oy, ow, age))

    def halt(self) -> None:
        self._cmd.publish(Twist())
        self._release.publish(String(data=json.dumps({"owner": OWNER})))


def save(rows: list[tuple]) -> None:
    if not rows:
        print("no samples")
        return
    arr = np.array([(r[0], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows], dtype=float)
    np.savetxt(
        CSV_PATH,
        arr,
        delimiter=",",
        header="t,cmd_vx,cmd_vy,cmd_wz,odom_vx,odom_vy,odom_wz",
        comments="",
    )
    dt = 1.0 / RATE_HZ
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    n = int(PERIOD_S * RATE_HZ)
    for i, (ax, phase) in enumerate(zip(axes, PHASES)):
        col = AXIS[phase]
        ax.plot(arr[:, 0], arr[:, 1 + col], label="sent", color="C0")
        ax.plot(arr[:, 0], arr[:, 4 + col], label="odom", color="C1")
        ax.set_ylabel(LABEL[phase])
        ax.grid(True, alpha=0.3)
        sl = slice(i * n, min((i + 1) * n, len(arr)))
        cmd, meas = arr[sl, 1 + col], arr[sl, 4 + col]
        if len(cmd) < 4:
            ax.set_title(f"{phase}: flat")
            continue
        point = bode(cmd, meas, dt, PERIOD_S)
        if point is None:
            title = f"{phase}: flat"
        else:
            gain, phase_deg = point
            title = f"{phase}: gain {gain:.2f}, phase {phase_deg:+.0f}°"
            print(f"{phase}: gain {gain:.3f}, phase {phase_deg:+.1f} deg")
        ax.set_title(title)
    axes[0].legend(loc="upper right")
    axes[2].set_xlabel("t (s)")
    fig.tight_layout()
    Path(PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"wrote {CSV_PATH}")
    print(f"wrote {PLOT_PATH}")


def main() -> None:
    rclpy.init()
    node = Sine()
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    node.halt()
    rclpy.spin_once(node, timeout_sec=0.3)
    rows = node._rows
    node.destroy_node()
    rclpy.shutdown()
    save(rows)


if __name__ == "__main__":
    main()
