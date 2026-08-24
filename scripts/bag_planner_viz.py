#!/usr/bin/env python3
"""Render nav-manager planner viz from a hound rosbag to MP4.

LocalMap drives the background at bag time (real-time clock). A MarkerArray
with more than one ADD arrow replaces the overlay (green = forward, blue =
backward). Earlier bags without arrows are skipped by --list.

  python3 bag_planner_viz.py --list
  python3 bag_planner_viz.py /root/colcon_ws/bags/hound_2026_08_24-00_23_52
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hound_nav.planner_viz import paint_cost_elev_map  # noqa: E402

def _default_bag_root() -> Path:
    for cand in (
        Path("/root/colcon_ws/bags"),
        Path("/home/hound/colcon_ws/bags"),
    ):
        try:
            if cand.is_dir():
                return cand
        except OSError:
            continue
    return Path("/home/hound/colcon_ws/bags")


BAG_ROOT = _default_bag_root()

MAP_TOPIC = "/hound_mapping/local_map"
ARROW_TOPIC = "/hound_nav/local_plan_arrows"
MARKER_ADD = 0
MARKER_DELETEALL = 3


def _topic_counts(bag: Path) -> dict[str, int]:
    meta = bag / "metadata.yaml"
    if not meta.is_file():
        return {}
    info = yaml.safe_load(meta.read_text())["rosbag2_bagfile_information"]
    out: dict[str, int] = {}
    for item in info.get("topics_with_message_count") or []:
        name = item["topic_metadata"]["name"]
        out[name] = int(item.get("message_count") or 0)
    return out


def list_bags(root: Path) -> None:
    print(f"{'bag':<36} {'local_map':>10} {'arrows':>8}  usable")
    for bag in sorted(root.glob("hound_*")):
        if not bag.is_dir():
            continue
        c = _topic_counts(bag)
        n_map = c.get(MAP_TOPIC)
        n_arr = c.get(ARROW_TOPIC)
        ok = bool(n_map and n_arr)
        print(
            f"{bag.name:<36} {n_map if n_map is not None else '-':>10} "
            f"{n_arr if n_arr is not None else '-':>8}  "
            f"{'yes' if ok else 'no'}"
        )


def _img_f32(img) -> np.ndarray | None:
    w, h = int(img.width), int(img.height)
    if w < 1 or h < 1:
        return None
    enc = str(img.encoding)
    if enc != "32FC1":
        return None
    data = np.frombuffer(bytes(img.data), dtype=np.float32)
    if data.size < w * h:
        return None
    return data[: w * h].reshape(h, w).copy()


def _yaw_from_quat(q) -> float:
    siny = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny, cosy)


def _parse_map(msg) -> dict | None:
    elev = _img_f32(msg.elevation)
    cost = _img_f32(msg.costmap)
    if elev is None or cost is None or elev.shape != cost.shape:
        return None
    info = msg.info
    origin = info.origin.position
    res = float(info.resolution)
    h, w = cost.shape
    return {
        "cost": cost,
        "elev": elev,
        "cx": float(origin.x) + 0.5 * w * res,
        "cy": float(origin.y) + 0.5 * h * res,
        "res": res,
        "extent": max(w, h) * res,
    }


def _parse_arrows(msg) -> list[dict] | None:
    arrows: list[dict] = []
    for m in msg.markers:
        action = int(m.action)
        if action == MARKER_DELETEALL:
            continue
        if action != MARKER_ADD:
            continue
        p = m.pose.position
        yaw = _yaw_from_quat(m.pose.orientation)
        col = m.color
        length = float(getattr(m.scale, "x", 0.3) or 0.3)
        arrows.append(
            {
                "x": float(p.x),
                "y": float(p.y),
                "yaw": yaw,
                "length": max(0.12, length),
                "bgr": (
                    int(np.clip(float(col.b) * 255.0, 0, 255)),
                    int(np.clip(float(col.g) * 255.0, 0, 255)),
                    int(np.clip(float(col.r) * 255.0, 0, 255)),
                ),
            }
        )
    if len(arrows) <= 1:
        return None
    return arrows


def _world_to_px(
    x: float, y: float, cx: float, cy: float, res_inv: float, map_size: int
) -> tuple[int, int]:
    px = int(round((x - cx) * res_inv + map_size / 2.0))
    py = int(round((y - cy) * res_inv + map_size / 2.0))
    py = map_size - 1 - py
    return px, py


def _draw_arrows(
    img: np.ndarray, arrows: list[dict], cx: float, cy: float, res_inv: float
) -> None:
    h, w = img.shape[:2]
    for a in arrows:
        x0, y0 = _world_to_px(a["x"], a["y"], cx, cy, res_inv, w)
        x1 = a["x"] + a["length"] * math.cos(a["yaw"])
        y1 = a["y"] + a["length"] * math.sin(a["yaw"])
        x1, y1 = _world_to_px(x1, y1, cx, cy, res_inv, w)
        if not (0 <= x0 < w and 0 <= y0 < h) and not (0 <= x1 < w and 0 <= y1 < h):
            continue
        cv2.arrowedLine(
            img, (x0, y0), (x1, y1), a["bgr"], 2, cv2.LINE_AA, tipLength=0.35
        )


def _render(m: dict, arrows: list[dict] | None, map_size: int) -> np.ndarray:
    img = paint_cost_elev_map(m["cost"], m["elev"], map_size)
    img = cv2.flip(img, 0)
    res_inv = float(map_size) / max(m["extent"], 1e-6)
    if arrows:
        _draw_arrows(img, arrows, m["cx"], m["cy"], res_inv)
    return img


def render_bag(
    bag: Path,
    out: Path,
    *,
    map_size: int,
    fps: float,
    map_topic: str,
    arrow_topic: str,
) -> None:
    from rosbags.highlevel import AnyReader

    events: list[tuple[int, str, object]] = []
    with AnyReader([bag]) as reader:
        conns = [
            c
            for c in reader.connections
            if c.topic in (map_topic, arrow_topic)
        ]
        if not conns:
            raise SystemExit(f"no {map_topic} / {arrow_topic} in {bag}")
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            kind = "map" if conn.topic == map_topic else "plan"
            events.append((int(ts), kind, msg))
    events.sort(key=lambda e: e[0])

    latest_map: dict | None = None
    latest_plan: list[dict] | None = None
    frames: list[tuple[int, np.ndarray]] = []
    n_map = 0
    n_plan = 0
    for ts, kind, msg in events:
        if kind == "map":
            parsed = _parse_map(msg)
            if parsed is None:
                continue
            latest_map = parsed
            n_map += 1
        else:
            arrows = _parse_arrows(msg)
            if arrows is None:
                continue
            latest_plan = arrows
            n_plan += 1
            if latest_map is None:
                continue
        if latest_map is None:
            continue
        frames.append((ts, _render(latest_map, latest_plan, map_size)))

    if not frames:
        raise SystemExit(f"no renderable LocalMap frames in {bag}")

    t0 = frames[0][0]
    t1 = frames[-1][0]
    span_s = max((t1 - t0) * 1e-9, 1.0 / max(fps, 1.0))
    n_out = max(1, int(round(span_s * fps)))
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (map_size, map_size),
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open VideoWriter for {out}")
    idx = 0
    for i in range(n_out):
        t = t0 + int(i * 1e9 / fps)
        while idx + 1 < len(frames) and frames[idx + 1][0] <= t:
            idx += 1
        writer.write(frames[idx][1])
    writer.release()
    print(
        f"wrote {out}  {n_out} frames @ {fps:.1f} Hz  "
        f"({span_s:.1f}s)  maps={n_map} plans={n_plan}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "bag",
        nargs="?",
        help="rosbag2 directory (hound_*). Omit with --list.",
    )
    ap.add_argument(
        "--bags",
        type=Path,
        default=BAG_ROOT,
        help=f"bag root (default {BAG_ROOT})",
    )
    ap.add_argument("--list", action="store_true", help="list hound bags + topic counts")
    ap.add_argument("-o", "--output", type=Path, help="mp4 path (default <bag>.mp4)")
    ap.add_argument("--map-size", type=int, default=480)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--map-topic", default=MAP_TOPIC)
    ap.add_argument("--arrow-topic", default=ARROW_TOPIC)
    args = ap.parse_args()

    if args.list or args.bag is None:
        list_bags(args.bags)
        if args.bag is None:
            return

    bag = Path(args.bag)
    if not bag.is_dir():
        cand = args.bags / bag
        if cand.is_dir():
            bag = cand
    if not bag.is_dir():
        raise SystemExit(f"bag not found: {args.bag}")
    out = args.output or Path(str(bag) + "_planner.mp4")
    render_bag(
        bag,
        out,
        map_size=int(args.map_size),
        fps=float(args.fps),
        map_topic=str(args.map_topic),
        arrow_topic=str(args.arrow_topic),
    )


if __name__ == "__main__":
    main()
