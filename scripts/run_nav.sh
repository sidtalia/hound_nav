#!/usr/bin/env bash
# Live Gates nav, inside opendubs-gates-sim. Ctrl-C stops the launch and the bridge.
# Do not source bashrc_common: it forces zenoh and localhost-only discovery.

source /opt/ros/jazzy/setup.bash
source /root/colcon_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/opendubs/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST=11.0
export HOUND_NAV_YAML=/root/colcon_ws/src/hound_nav/config/gates_r6_f2.yaml

root=/root/colcon_ws/src/hound_nav
python3 -u "${root}/scripts/robot_bridge.py" --config "${HOUND_NAV_YAML}" &
bridge=$!
trap 'kill -TERM "$bridge" 2>/dev/null || true' INT TERM EXIT
ros2 launch "${root}/launch/hound_nav.launch.py"
