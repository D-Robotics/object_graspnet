// Copyright (c) 2024，Horizon Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GRASP_COLLISION_DETECTOR_H_
#define GRASP_COLLISION_DETECTOR_H_

#include <string>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "include/grasp_output_parser.h"

class GraspGroupResult;
class GraspGroup;

class GraspCollisionDetector {
 public:
  // GraspCollisionDetector() {}
  GraspCollisionDetector(int topk) {topk_ = topk;}
  ~GraspCollisionDetector() {}

  
  int Process(std::shared_ptr<GraspGroupResult> &input);

  int Init(std::vector<std::vector<float>>& point_clouds);

 private:
  
  std::vector<std::vector<float>> sample_clouds_;
  float voxel_size_ = 0.01;
  float approach_dist_ = 0.05;
  float collision_thresh_ = 0.01;

  float finger_width = 0.023;  // 手指宽度
  float finger_length = 0.075; // 手指长度

  int topk_ = 0;
  int current_masks_ = 0;

  int Detect(std::vector<GraspGroup>& graspgroups,
            std::vector<int>& collision_mask,
            float approach_dist = 0.03,
            float collision_thresh = 0.05, 
            bool return_empty_grasp = false,
            float empty_thresh = 0.01,
            bool return_ious = false);

};

#endif  // GRASP_COLLISION_DETECTOR_H_