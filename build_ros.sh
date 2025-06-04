echo "Building ROS nodes"

cd Examples_old/ROS/GeneA-SLAM2
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
