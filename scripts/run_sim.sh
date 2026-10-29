#!/usr/bin/env bash
# Gates floor sim plus the holonomic planner and MPPI, inside opendubs-gates-sim.
# Ctrl-C stops the launch and the sim.
#
# Zenoh, localhost only (sim_env.sh). This does not join the robot's FastDDS
# graph and does not take the base lease. Stop run_nav.sh first: both use
# Viser on 8081.
#
#   /root/colcon_ws/src/hound_nav/scripts/run_sim.sh

source /root/colcon_ws/sim_env.sh
export TORCH_CUDA_ARCH_LIST=11.0
export HOUND_NAV_YAML=/root/colcon_ws/src/hound_nav/config/gates_r6_f2.yaml

root=/root/colcon_ws/src/hound_nav
python3 -u "${root}/scripts/gates_floor_sim.py" --config "${HOUND_NAV_YAML}" &
sim=$!
trap 'kill -TERM "$sim" 2>/dev/null || true' INT TERM EXIT
ros2 launch "${root}/launch/hound_nav.launch.py"
