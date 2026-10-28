#!/usr/bin/env python3
"""Hold the base as owner "test" and publish a forward speed. Ctrl-C releases it.

The selector only forwards /base/<owner>/cmd_vel while that owner holds the lease.
A command older than 0.5 s is dropped, so this publishes at 10 Hz and renews the
claim every 1 s.
"""

import argparse
import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

OWNER = "test"


class Poke(Node):
    def __init__(self, speed: float) -> None:
        super().__init__("drive_poke")
        self._speed = speed
        self._claim = self.create_publisher(String, "/base/claim", 10)
        self._release = self.create_publisher(String, "/base/release", 10)
        self._cmd = self.create_publisher(Twist, f"/base/{OWNER}/cmd_vel", 10)
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, "/base/owner", self._on_owner, latched)
        self._ticks = 0
        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"owner={OWNER} vx={speed:.3f} m/s on /base/{OWNER}/cmd_vel — Ctrl-C releases"
        )

    def _on_owner(self, msg: String) -> None:
        self.get_logger().info(msg.data, throttle_duration_sec=2.0)

    def _tick(self) -> None:
        if self._ticks % 10 == 0:
            self._claim.publish(String(data=json.dumps({"owner": OWNER, "ttl_s": 5})))
        twist = Twist()
        twist.linear.x = self._speed
        self._cmd.publish(twist)
        self._ticks += 1

    def halt(self) -> None:
        self._cmd.publish(Twist())
        self._release.publish(String(data=json.dumps({"owner": OWNER})))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold the base lease and drive forward.")
    parser.add_argument("vx", nargs="?", type=float, default=1.0, help="forward speed, m/s")
    args = parser.parse_args()
    rclpy.init()
    node = Poke(args.vx)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.halt()
    rclpy.spin_once(node, timeout_sec=0.3)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
