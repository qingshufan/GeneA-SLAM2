#!/bin/bash

# Source ROS environment (modify for your ROS version)
source /opt/ros/melodic/setup.bash

# Define parameters (update according to your setup)
GeneA_SLAM2_PATH=./
ROSBAG_PATH=./rgbd_dataset_geneaslam_laptop2.bag
PLAY_SPEED=1.5

# Verify paths exist
if [ ! -d "$GeneA_SLAM2_PATH" ]; then
    echo "Error: GeneA_SLAM2 directory not found! Update GeneA_SLAM2_PATH in the script."
    exit 1
fi

if [ ! -f "$ROSBAG_PATH" ]; then
    echo "Error: ROS bag file not found! Update ROSBAG_PATH in the script."
    exit 1
fi

# Start roscore in a new terminal
gnome-terminal --title "roscore" -- bash -c "roscore; exec bash" &

# Wait for roscore initialization
sleep 2

# Start  node in a new terminal
gnome-terminal --title "GeneA_SLAM2 RGBD" -- bash -c "
    export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:${GeneA_SLAM2_PATH}/Examples/ROS;
    rosrun GeneA_SLAM2 RGBD ${GeneA_SLAM2_PATH}/Vocabulary/ORBvoc.txt ${GeneA_SLAM2_PATH}/Examples/RGB-D/GENEA-ROS.yaml;
    exec bash
" &

# Start rosbag play in paused state (press Enter to start)
gnome-terminal --title "rosbag play" -- bash -c "
    rosbag play --pause -r ${PLAY_SPEED} ${ROSBAG_PATH} /camera/rgb/image_color:=/camera/rgb/image_raw /camera/depth/image:=/camera/depth_registered/image_raw;
    exec bash
" &

wait