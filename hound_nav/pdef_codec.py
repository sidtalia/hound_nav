"""Arrow pack/unpack for manager ↔ planner / controller (CPU float buffers + JSON meta)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyarrow as pa


def _arrow_f32(x: np.ndarray) -> pa.Array:
    return pa.array(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))


def _arrow_f64(x: np.ndarray) -> pa.Array:
    return pa.array(np.ascontiguousarray(x, dtype=np.float64).reshape(-1))


def _from_arrow(buf: Any, dtype: np.dtype) -> np.ndarray:
    if buf is None:
        return np.zeros((0,), dtype=dtype)
    if hasattr(buf, "to_numpy"):
        return np.asarray(buf.to_numpy(), dtype=dtype)
    if hasattr(buf, "values") and hasattr(buf.values, "to_numpy"):
        return np.asarray(buf.values.to_numpy(), dtype=dtype)
    return np.asarray(buf, dtype=dtype)


def pack_map(
    costmap: np.ndarray,
    height: np.ndarray,
    normals: np.ndarray,
    *,
    map_center: np.ndarray,
    map_res: float,
    map_gen: int,
) -> Tuple[pa.Array, Dict[str, Any]]:
    """Controller map: HxWx5 [cost, height, nx, ny, nz]."""
    cost = np.ascontiguousarray(costmap, dtype=np.float32)
    hgt = np.ascontiguousarray(height, dtype=np.float32)
    nrm = np.ascontiguousarray(normals, dtype=np.float32)
    if nrm.ndim != 3 or nrm.shape[2] < 3 or nrm.shape[:2] != cost.shape:
        raise ValueError("normals must be HxWx3 matching cost/height")
    stacked = np.concatenate(
        [cost[..., None], hgt[..., None], nrm[..., :3]], axis=-1
    )
    h, w = int(stacked.shape[0]), int(stacked.shape[1])
    meta = {
        "H": h,
        "W": w,
        "C": 5,
        "map_res": float(map_res),
        "map_gen": int(map_gen),
        "map_center": np.asarray(map_center, dtype=np.float64).reshape(-1).tolist(),
    }
    return _arrow_f32(stacked), meta


def unpack_map(buf: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    h = int(meta.get("H", 0))
    w = int(meta.get("W", 0))
    c = int(meta.get("C", 5))
    flat = _from_arrow(buf, np.float32)
    if h <= 0 or w <= 0 or c < 5 or flat.size < h * w * c:
        raise ValueError("invalid map buffer")
    stacked = flat[: h * w * c].reshape(h, w, c)
    return {
        "costmap": np.ascontiguousarray(stacked[..., 0]),
        "height": np.ascontiguousarray(stacked[..., 1]),
        "normals": np.ascontiguousarray(stacked[..., 2:5]),
        "map_center": np.asarray(meta.get("map_center", [0.0, 0.0]), dtype=np.float64),
        "map_res": float(meta.get("map_res", 0.25)),
        "map_gen": int(meta.get("map_gen", 0)),
    }


def pack_pdef(
    costmap: np.ndarray,
    height: np.ndarray,
    *,
    map_center: np.ndarray,
    map_res: float,
    start: np.ndarray,
    goal: np.ndarray,
    hysteresis: int,
    expansion_limit: int,
    query_t: float = 0.0,
) -> Tuple[pa.Array, Dict[str, Any]]:
    """Planner query: HxWx2 [cost, height] + start/goal in metadata."""
    cost = np.ascontiguousarray(costmap, dtype=np.float32)
    hgt = np.ascontiguousarray(height, dtype=np.float32)
    if cost.shape != hgt.shape:
        raise ValueError(f"cost {cost.shape} != height {hgt.shape}")
    bitmap = np.stack([cost, hgt], axis=-1)
    h, w = int(bitmap.shape[0]), int(bitmap.shape[1])
    meta = {
        "H": h,
        "W": w,
        "map_res": float(map_res),
        "map_center": np.asarray(map_center, dtype=np.float64).reshape(-1).tolist(),
        "start": np.asarray(start, dtype=np.float64).reshape(-1).tolist(),
        "goal": np.asarray(goal, dtype=np.float64).reshape(-1)[:2].tolist(),
        "hysteresis": int(hysteresis),
        "expansion_limit": int(expansion_limit),
        "state_dims": int(np.asarray(start).reshape(-1).size),
        "query_t": float(query_t),
    }
    return _arrow_f32(bitmap), meta


def unpack_pdef(buf: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    h = int(meta.get("H", 0))
    w = int(meta.get("W", 0))
    flat = _from_arrow(buf, np.float32)
    if h <= 0 or w <= 0 or flat.size < h * w * 2:
        raise ValueError("invalid pdef buffer")
    bitmap = flat[: h * w * 2].reshape(h, w, 2)
    start = np.asarray(meta.get("start", []), dtype=np.float64).reshape(-1)
    goal = np.asarray(meta.get("goal", []), dtype=np.float64).reshape(-1)
    state_dims = int(meta.get("state_dims", start.size if start.size > 0 else 17))
    if start.size < state_dims or goal.size < 2:
        raise ValueError(
            f"pdef start must be {state_dims} and goal 2 (got {start.size}, {goal.size})"
        )
    return {
        "costmap": np.ascontiguousarray(bitmap[..., 0]),
        "height": np.ascontiguousarray(bitmap[..., 1]),
        "map_center": np.asarray(meta.get("map_center", [0.0, 0.0]), dtype=np.float64),
        "map_res": float(meta.get("map_res", 0.25)),
        "start": start[:state_dims].copy(),
        "goal": goal[:2].copy(),
        "hysteresis": int(meta.get("hysteresis", 100)),
        "expansion_limit": int(meta.get("expansion_limit", 5000)),
        "state_dims": state_dims,
        "query_t": float(meta.get("query_t", 0.0)),
    }


def pack_plan(
    path: Optional[np.ndarray],
    *,
    success: bool,
    expansions: int,
    query_t: float = 0.0,
) -> Tuple[pa.Array, Dict[str, Any]]:
    if path is None or len(path) == 0:
        arr = np.zeros((0, 4), dtype=np.float64)
        success = False
    else:
        arr = np.ascontiguousarray(path, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 4:
            pad = np.zeros((arr.shape[0], 4), dtype=np.float64)
            pad[:, : arr.shape[1]] = arr
            arr = pad
        arr = arr[:, :4]
    meta = {
        "n": int(arr.shape[0]),
        "cols": 4,
        "success": bool(success),
        "expansions": int(expansions),
        "query_t": float(query_t),
    }
    return _arrow_f64(arr), meta


def unpack_plan(
    buf: Any, meta: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], bool, int, float]:
    n = int(meta.get("n", 0))
    cols = int(meta.get("cols", 4))
    success = bool(meta.get("success", False))
    expansions = int(meta.get("expansions", 0))
    query_t = float(meta.get("query_t", 0.0))
    if n <= 0 or not success:
        return None, False, expansions, query_t
    flat = _from_arrow(buf, np.float64)
    need = n * cols
    if flat.size < need:
        return None, False, expansions, query_t
    return flat[:need].reshape(n, cols), True, expansions, query_t


def pack_track(
    state: np.ndarray,
    reference: np.ndarray,
    *,
    goal_reached: bool,
) -> Tuple[pa.Array, Dict[str, Any]]:
    st = np.ascontiguousarray(state, dtype=np.float64).reshape(-1)
    state_dims = int(st.size)
    ref = np.ascontiguousarray(reference, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    t = int(ref.shape[0])
    cols = int(ref.shape[1])
    packed = np.concatenate([st, ref.reshape(-1)])
    meta = {
        "T": t,
        "cols": cols,
        "state_dims": state_dims,
        "goal_reached": bool(goal_reached),
    }
    return _arrow_f64(packed), meta


def unpack_track(buf: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    t = int(meta.get("T", 0))
    cols = int(meta.get("cols", 4))
    state_dims = int(meta.get("state_dims", 17))
    flat = _from_arrow(buf, np.float64)
    if flat.size < state_dims + t * cols:
        raise ValueError("invalid track buffer")
    state = flat[:state_dims].copy()
    ref = flat[state_dims : state_dims + t * cols].reshape(t, cols)
    return {
        "state": state,
        "reference": ref,
        "goal_reached": bool(meta.get("goal_reached", False)),
    }
