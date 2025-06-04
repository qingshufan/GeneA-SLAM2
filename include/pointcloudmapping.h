/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * source: https://github.com/gaoxiang12/ORBSLAM2_with_pointcloud_map/blob/master/ORB_SLAM2_modified/include/pointcloudmapping.h
 */

 #ifndef POINTCLOUDMAPPING_H
 #define POINTCLOUDMAPPING_H
 
 #include "System.h"
 
 #include <pcl/common/transforms.h>
 #include <pcl/point_types.h>
 #include <pcl/filters/voxel_grid.h>
 #include <pcl/io/pcd_io.h>
 #include <pcl/common/io.h>

 #include <condition_variable>
 
 using namespace ORB_SLAM3;
 
 class PointCloudMapping
 {
 public:
     typedef pcl::PointXYZRGBA PointT;
     typedef pcl::PointCloud<PointT> PointCloud;
     mutex cloudsMutex;
     bool denseMap;
     PointCloud::Ptr globalMap;
 
     PointCloudMapping( double resolution_ );
 
     void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth );
     void shutdown();
     void viewer();
 
 protected:
     PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
 
     shared_ptr<thread>  viewerThread;
 
     bool    shutDownFlag    =false;
     mutex   shutDownMutex;
 
     condition_variable  keyFrameUpdated;
     mutex               keyFrameUpdateMutex;
 
     // data to generate point clouds
     vector<KeyFrame*>       keyframes;
     vector<cv::Mat>         colorImgs;
     vector<cv::Mat>         depthImgs;
     mutex                   keyframeMutex;
     uint16_t                lastKeyframeSize =0;
 
     double resolution = 0.04;
     pcl::VoxelGrid<PointT>  voxel;
 };
 
 #endif // POINTCLOUDMAPPING_H