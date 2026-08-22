# hound_nav

IGHA* + **mppi** ROS adapter for HOUND, split into three Dora nodes.
Consumes `hound_mapping/LocalMap` (elevation + cost free-ness) — no yaw
warp. Unobserved cells use mapper `unobserved_luminance` (default 128).
MPPI core lives in the `mppi` package
(BeamNGRL keeps research copies of dynamics/costs as examples).

## Graph

```
ROS LocalMap / control_state / mission Path and RViz /goal_pose
  → manager (CPU crop, world-frame TrajBuffer, update_goal)
      → pdef  ~5 Hz   (start + goal + HxWx2 map)  → planner (IGHA* search)
      ← plan          (world Nx4)
      → map   ~10 Hz  (HxWx5 + map_gen)           → controller
      → track ctrl Hz (state[17] + Tx4 body crop) → controller
                                                    → AckermannDriveStamped
  → nav_msgs/Path viz
```

Mapper BEV is robot-centered in **odom axes**. Manager uses that grid as-is
(relative-Z + cost `[0,255]` IGHA free-ness). No crop/resample.

Planner gets the map **with** the query (no separate planner map stream).
Controller gates `set_BEV` on `map_gen`. Trajectory stitch/invalidate lives in
the manager buffer (`replace` / `append` / `truncate_after`).

## Layout

```
hound_nav/
  dora/nav_dataflow.yml
  hound_nav/
    manager_dora_node.py    ROS + traj server + Dora fan-out
    planner_dora_node.py    create_planner + search
    controller_dora_node.py MPPI track → AckermannDriveStamped
    pdef_codec.py           Arrow pack/unpack
    traj_buffer.py          world-frame path buffer
    pdef_buffer.py          LocalMap + state + waypoints → PDef
    trackingCostCUDA.py     MPPI tracking cost (loads cuda/)
    cuda/                   tracking_cost.cu/.cpp
  config/                   nav_example.yaml
  scripts/                  bench_mppi.py, bench_tracking_cost.py
```

Mission params: `hound_core` SSoT (`planner_hz: 5.0`). Example nav YAML under
`config/`. Slope threshold lives in mapping (`lethal_slope_deg`), not here.

## Build

```bash
cd /root/colcon_ws && colcon build --packages-select hound_nav --symlink-install
source install/setup.bash
```

Needs `IGHAStar` on `PYTHONPATH` (see `deps_path.py`) and the `mppi` package.

## Run

Enable `nav:` in SSoT and launch the stack (starts `dora run` like YOLO):

```bash
ros2 launch hound_core hound_core.launch.py
```

## Notes

- LocalMap is elev+cost only; dynamics API normals are FD from elev if needed.
- `Cost_config.lethal_w` weights the mapper costmap; it does not re-threshold slope.
- Do not use `IGHAStarMP`; planner is in-process IGHA* in its Dora node.
- First-slice traj buffer replaces the whole plan on each successful search.
- Planner start is the live robot (`control_state`). OpenCV box is that pose, not a simulated plant.
