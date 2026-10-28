#!/usr/bin/env python3
"""Gates floor 1 debug plant for hound_nav.

Occupancy (mushr gates.yaml) becomes a flat LocalMap: free = 255, everything
else (occupied and unknown) = 0. MuJoCo is a planar body. It publishes the
17-vector control_state (pose + body twist) and subscribes to the nav twist
(linear.x, linear.y, angular.z). No cameras, no arms.

Viser (default :8081): click sets the goal, shift-click sets the robot xy.
The yaw slider is the robot heading used on that shift-click.

  python3 scripts/gates_floor_sim.py --config ../config/gates_r6.yaml
  python3 scripts/gates_floor_sim.py --check

SSoT nav must be the R6 plant or the command type will not match:
  Dynamics  mppi/mppi/Dynamics/HolonomicDynamicsTCUDA.py  HolonomicDynamics
  Sampling  mppi/mppi/Sampling/HolonomicDeltaSampling.py  HolonomicDeltaSampling
            control_dim: 3
  Cost      hound_nav/hound_nav/holonomicTrackingCostCUDA.py  HolonomicTrackingCost
  Planner   node_type: holonomic_6D
Map_config map_size / map_res must match --map-size / --map-res (30 m, 0.05 m).
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time

import numpy as np
import yaml

# ponytail: qvel is overwritten from the twist every step, so the body has no
# inertia. Integrate accelerations when the plant should lag the command.

XML = """
<mujoco>
  <option timestep="{dt}" gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <joint name="x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="box" size="0.35 0.375 0.05" mass="20"/>
    </body>
  </worldbody>
</mujoco>
"""

STATE_DIM = 17


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def body_to_world(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return vx * c - vy * s, vx * s + vy * c


def world_to_body(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return vx * c + vy * s, -vx * s + vy * c


def load_cost(yaml_path: str) -> tuple[np.ndarray, float, float, float]:
    """pgm + map yaml → uint8 free-ness, origin xy, resolution.

    Image row 0 is the top of the file. The yaml origin is the lower-left
    pixel, same as nav2 map_server. Free pixels stay 255; occupied and
    unknown are 0.
    """
    meta = yaml.safe_load(open(yaml_path))
    img_path = meta["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(yaml_path), img_path)
    import cv2

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {img_path}")
    pix = img.astype(np.float32)
    if int(meta.get("negate", 0)):
        p = pix / 255.0
    else:
        p = (255.0 - pix) / 255.0
    free = p < float(meta.get("free_thresh", 0.196))
    cost = np.where(free, np.uint8(255), np.uint8(0))
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    return cost, ox, oy, float(meta["resolution"])


def pixel_center(col: float, row: float, ox: float, oy: float, res: float, height: int):
    x = ox + (col + 0.5) * res
    y = oy + ((height - 1 - row) + 0.5) * res
    return x, y


def crop_cost(
    cost: np.ndarray,
    ox: float,
    oy: float,
    res: float,
    x: float,
    y: float,
    size_m: float,
    out_res: float,
) -> tuple[np.ndarray, float, float]:
    """Robot-centered square. Row 0 is min Y (LocalMap). Outside the floor is lethal."""
    n = int(round(size_m / out_res))
    origin_x = x - 0.5 * n * out_res
    origin_y = y - 0.5 * n * out_res
    h, w = cost.shape
    cols = origin_x + (np.arange(n) + 0.5) * out_res
    rows = origin_y + (np.arange(n) + 0.5) * out_res
    ic = np.floor((cols - ox) / res).astype(np.int32)
    ir_bottom = np.floor((rows - oy) / res).astype(np.int32)
    ir = (h - 1) - ir_bottom
    cc, rr = np.meshgrid(ic, ir, indexing="xy")
    # meshgrid indexing xy: shape (n_rows, n_cols) = (len(ir), len(ic))
    out = np.zeros((n, n), dtype=np.float32)
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    out[ok] = cost[rr[ok], cc[ok]]
    return out, origin_x, origin_y


def free_spawn(cost: np.ndarray, ox: float, oy: float, res: float) -> tuple[float, float]:
    ys, xs = np.nonzero(cost == 255)
    if xs.size == 0:
        raise SystemExit("gates map has no free cells")
    k = int(np.argmin((xs - np.median(xs)) ** 2 + (ys - np.median(ys)) ** 2))
    return pixel_center(float(xs[k]), float(ys[k]), ox, oy, res, cost.shape[0])


def wall_points(cost: np.ndarray, ox: float, oy: float, res: float, cap: int = 80000):
    lethal = cost == 0
    step = 4
    while step < 64:
        sub = lethal[::step, ::step]
        if int(sub.sum()) <= cap:
            break
        step *= 2
    rr, cc = np.nonzero(lethal[::step, ::step])
    cols = cc.astype(np.float64) * step
    rows = rr.astype(np.float64) * step
    h = cost.shape[0]
    x = ox + (cols + 0.5) * res
    y = oy + ((h - 1 - rows) + 0.5) * res
    pts = np.stack([x, y, np.zeros_like(x)], axis=1).astype(np.float32)
    return pts, step * res


def _segs(xy, z: float):
    xy = np.asarray(xy, np.float32)
    if xy.ndim != 2 or xy.shape[0] < 2:
        return None
    a = np.empty((xy.shape[0] - 1, 2, 3), np.float32)
    a[:, 0, :2] = xy[:-1]
    a[:, 1, :2] = xy[1:]
    a[:, :, 2] = z
    return a


def ray_z0(origin, direction):
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if abs(d[2]) < 1e-8:
        return None
    t = -o[2] / d[2]
    if t <= 0.0:
        return None
    return o + t * d


def pack_state(x, y, yaw, vx, vy, wz) -> list[float]:
    s = [0.0] * STATE_DIM
    s[0], s[1], s[5] = x, y, yaw
    s[6], s[7], s[14] = vx, vy, wz
    return s


def _check() -> None:
    # 2x2: top row (max Y) occupied-left / free-right; bottom row free-left / occupied-right.
    img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    ox, oy, res = 0.0, 0.0, 1.0
    x, y = pixel_center(0, 1, ox, oy, res, 2)  # bottom-left, free
    crop, cx, cy = crop_cost(img, ox, oy, res, x, y, size_m=2.0, out_res=1.0)
    assert crop.shape == (2, 2), crop.shape
    # Center of the window is the free pixel. LocalMap row 0 is min Y.
    assert crop[0, 0] == 255.0, crop
    assert crop[1, 0] == 0.0, crop  # top-left of the image is lethal
    wx, wy = body_to_world(1.0, 0.0, math.pi / 2)
    assert abs(wx) < 1e-6 and abs(wy - 1.0) < 1e-6
    import mujoco

    model = mujoco.MjModel.from_xml_string(XML.format(dt=0.02))
    data = mujoco.MjData(model)
    assert model.nq == 3
    for _ in range(50):
        yaw = float(data.qpos[2])
        data.qvel[0], data.qvel[1] = body_to_world(1.0, 0.0, yaw)
        data.qvel[2] = 0.0
        mujoco.mj_step(model, data)
    assert abs(float(data.qpos[0]) - 1.0) < 1e-3, data.qpos[0]
    print("gates_floor_sim check ok")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_map = os.path.join(
        here, "..", "..", "mushr", "mushr_sim", "maps", "gates.yaml"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="", help="gates_r6.yaml; its sim: block overrides map/res")
    p.add_argument("--map", default=default_map)
    p.add_argument("--map-size", type=float, default=30.0)
    p.add_argument("--map-res", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--map-hz", type=float, default=10.0)
    p.add_argument("--cmd-timeout", type=float, default=0.5)
    p.add_argument("--viser-port", type=int, default=8081)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.config:
        doc = yaml.safe_load(open(args.config)) or {}
        sim = doc.get("sim") or {}
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        if sim.get("map"):
            m = sim["map"]
            args.map = m if os.path.isabs(m) else os.path.join(cfg_dir, m)
        if "map_res" in sim:
            args.map_res = float(sim["map_res"])
        if "map_size" in sim:
            args.map_size = float(sim["map_size"])
        if "viser_port" in sim:
            args.viser_port = int(sim["viser_port"])
    if args.check:
        _check()
        return

    import mujoco
    import rclpy
    import viser
    from geometry_msgs.msg import PoseStamped, Twist
    from hound_mapping.msg import LocalMap
    from nav_msgs.msg import MapMetaData, Path
    from visualization_msgs.msg import MarkerArray
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64MultiArray

    cost, ox, oy, res = load_cost(args.map)
    x0, y0 = free_spawn(cost, ox, oy, res)
    print(
        f"gates {cost.shape[1]}x{cost.shape[0]} res={res} origin=({ox:.2f},{oy:.2f}) "
        f"spawn=({x0:.2f},{y0:.2f})",
        flush=True,
    )

    model = mujoco.MjModel.from_xml_string(XML.format(dt=args.dt))
    data = mujoco.MjData(model)
    data.qpos[:] = (x0, y0, 0.0)
    mujoco.mj_forward(model, data)

    lock = threading.Lock()
    cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "t": 0.0}
    pending = {"pose": None, "goal": None}  # pose: (x, y, yaw)
    viz = {"plan": None, "mppi": None, "mppi_c": None}

    rclpy.init()
    node = Node("gates_floor_sim")
    map_pub = node.create_publisher(LocalMap, "/hound_mapping/local_map", qos_profile_sensor_data)
    state_pub = node.create_publisher(
        Float64MultiArray, "/hound_fcu_control/control_state", qos_profile_sensor_data
    )
    goal_pub = node.create_publisher(PoseStamped, "/goal_pose", 10)

    def on_twist(msg: Twist) -> None:
        with lock:
            cmd["vx"] = float(msg.linear.x)
            cmd["vy"] = float(msg.linear.y)
            cmd["wz"] = float(msg.angular.z)
            cmd["t"] = time.monotonic()

    node.create_subscription(Twist, "/hound_nav/cmd_ackermann", on_twist, qos_profile_sensor_data)

    def on_plan(msg: Path) -> None:
        if not msg.poses:
            return
        xy = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses], np.float32
        )
        with lock:
            viz["plan"] = xy

    def on_mppi(msg: MarkerArray) -> None:
        chunks, cols = [], []
        for m in msg.markers:
            if len(m.points) < 2:
                continue
            xy = np.array([[p.x, p.y] for p in m.points], np.float32)
            seg = _segs(xy, 0.08)
            if seg is None:
                continue
            c = np.array([m.color.r, m.color.g, m.color.b], np.float32) * 255.0
            chunks.append(seg)
            cols.append(np.broadcast_to(c, (seg.shape[0], 2, 3)).copy())
        with lock:
            if chunks:
                viz["mppi"] = np.concatenate(chunks)
                viz["mppi_c"] = np.concatenate(cols)
            else:
                viz["mppi"] = viz["mppi_c"] = None

    node.create_subscription(Path, "/hound_nav/local_plan", on_plan, 1)
    node.create_subscription(MarkerArray, "/hound_nav/mppi_topk", on_mppi, 1)

    server = viser.ViserServer(port=args.viser_port)
    server.scene.set_up_direction("+z")
    pts, spacing = wall_points(cost, ox, oy, res)
    server.scene.add_point_cloud("/walls", pts, (40, 40, 40), point_size=float(spacing))
    cx = ox + 0.5 * cost.shape[1] * res
    cy = oy + 0.5 * cost.shape[0] * res
    server.initial_camera.position = (cx, cy - 20.0, 40.0)
    server.initial_camera.look_at = (cx, cy, 0.0)
    robot_h = server.scene.add_box(
        "/robot",
        dimensions=(0.7, 0.75, 0.1),
        color=(30, 120, 220),
        side="double",
        position=(x0, y0, 0.05),
    )
    goal_h = server.scene.add_box(
        "/goal",
        dimensions=(0.35, 0.35, 0.04),
        color=(220, 40, 40),
        side="double",
        position=(x0, y0, 0.02),
        visible=False,
    )
    yaw_ui = server.gui.add_slider("robot yaw deg", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
    server.gui.add_markdown(
        "click = goal · shift-click = robot pose\n\n"
        "yellow = heading · green = plan · orange = best MPPI rollouts"
    )
    drawn = {}

    def _draw(name, segs, color, width):
        old = drawn.pop(name, None)
        if old is not None:
            old.remove()
        if segs is None:
            return
        drawn[name] = server.scene.add_line_segments(
            name, segs, color, line_width=width
        )

    def _click(ev, place_robot: bool) -> None:
        hit = ray_z0(ev.ray_origin, ev.ray_direction)
        if hit is None:
            return
        with lock:
            if place_robot:
                pending["pose"] = (float(hit[0]), float(hit[1]), math.radians(float(yaw_ui.value)))
            else:
                pending["goal"] = (float(hit[0]), float(hit[1]))

    @server.scene.on_click()
    def _goal(ev):
        _click(ev, False)

    @server.scene.on_click(modifier="shift")
    def _pose(ev):
        _click(ev, True)

    print(f"viser http://0.0.0.0:{args.viser_port}", flush=True)

    def f32_image(arr: np.ndarray, stamp) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = msg.width * 4
        msg.data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        return msg

    next_map = 0.0
    next_vis = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            with lock:
                if now - cmd["t"] > args.cmd_timeout:
                    vx = vy = wz = 0.0
                else:
                    vx, vy, wz = cmd["vx"], cmd["vy"], cmd["wz"]
                pose = pending["pose"]
                goal = pending["goal"]
                pending["pose"] = None
                pending["goal"] = None
            if pose is not None:
                data.qpos[:] = pose
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            yaw = float(data.qpos[2])
            data.qvel[0], data.qvel[1] = body_to_world(vx, vy, yaw)
            data.qvel[2] = wz
            mujoco.mj_step(model, data)
            x, y, yaw = (float(v) for v in data.qpos)
            yaw = wrap(yaw)
            bvx, bvy = world_to_body(float(data.qvel[0]), float(data.qvel[1]), yaw)
            bwz = float(data.qvel[2])

            stamp = node.get_clock().now().to_msg()
            st = Float64MultiArray()
            st.data = pack_state(x, y, yaw, bvx, bvy, bwz)
            state_pub.publish(st)

            if now >= next_map:
                next_map = now + 1.0 / args.map_hz
                elev_c, origin_x, origin_y = crop_cost(
                    cost, ox, oy, res, x, y, args.map_size, args.map_res
                )
                elev = np.zeros_like(elev_c)
                lm = LocalMap()
                lm.header.stamp = stamp
                lm.header.frame_id = "odom"
                lm.info = MapMetaData()
                lm.info.resolution = float(args.map_res)
                lm.info.width = int(elev_c.shape[1])
                lm.info.height = int(elev_c.shape[0])
                lm.info.origin.position.x = float(origin_x)
                lm.info.origin.position.y = float(origin_y)
                lm.info.origin.orientation.w = 1.0
                lm.elevation = f32_image(elev, stamp)
                lm.costmap = f32_image(elev_c, stamp)
                map_pub.publish(lm)

            if goal is not None:
                g = PoseStamped()
                g.header.stamp = stamp
                g.header.frame_id = "odom"
                g.pose.position.x = goal[0]
                g.pose.position.y = goal[1]
                g.pose.orientation.w = 1.0
                goal_pub.publish(g)
                goal_h.position = (goal[0], goal[1], 0.02)
                goal_h.visible = True
                print(f"goal ({goal[0]:.2f}, {goal[1]:.2f})", flush=True)
            if pose is not None:
                print(f"pose ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg)", flush=True)

            if now >= next_vis:
                next_vis = now + 0.1
                robot_h.position = (x, y, 0.05)
                robot_h.wxyz = yaw_quat(yaw)
                c, s = math.cos(yaw), math.sin(yaw)
                _draw(
                    "/heading",
                    _segs(
                        np.array([[x, y], [x + 0.7 * c, y + 0.7 * s]], np.float32),
                        0.16,
                    ),
                    (255, 210, 40),
                    6,
                )
                with lock:
                    plan = None if viz["plan"] is None else viz["plan"].copy()
                    mppi, mppi_c = viz["mppi"], viz["mppi_c"]
                _draw("/plan", None if plan is None else _segs(plan, 0.12), (40, 210, 70), 4)
                _draw("/mppi", mppi, (255, 140, 20) if mppi_c is None else mppi_c, 2)
            time.sleep(args.dt)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
