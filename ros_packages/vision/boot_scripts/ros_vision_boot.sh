#!/bin/bash
source /opt/ros/humble/setup.bash
source /home/pib/ros_working_dir/install/setup.bash
source /home/pib/ros_working_dir/ros_config.sh
ros2 run pib_vision vision_node 2>&1 | tee -a ~/ros_working_dir/src/vision/vision_node.log