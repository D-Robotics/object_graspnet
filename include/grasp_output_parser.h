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

#ifndef GRASP_OUTPUT_PARSER_H_
#define GRASP_OUTPUT_PARSER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dnn_node/dnn_node_data.h"
#include "rclcpp/rclcpp.hpp"

#include "include/grasp_collision_detector.h"

using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::Model;

#define GRASP_MAX_WIDTH 0.067
#define GRASP_MAX_TOLERANCE 0.045
#define THRESH_GOOD 0.7
#define THRESH_BAD 0.1
#define PI 3.14159265358979323846

class GraspGroup {
 public:
  float score;
  float width;
  float height;
  float depth;
  std::vector<float> rotation;
  std::vector<float> translation;
  int object_id;

  // 构造函数初始化成员变量
  GraspGroup()
      : score(0), width(0), height(0), depth(0), object_id(0) {
    rotation.resize(9, 0);
    translation.resize(3, 0);
  }

  void Reset() { 
    rotation.clear();
    translation.clear();
  }

  friend std::ostream& operator<<(std::ostream& os, const GraspGroup& group) {
      os << "[Score]: " << group.score << std::endl;
      os << "[Width]: " << group.width << std::endl;
      os << "[Height]: " << group.height << std::endl;
      os << "[Depth]: " << group.depth << std::endl;
      os << "[Rotation]:" << std::endl;
      for (const auto& rot : group.rotation) {
          os << rot << " ";
      }
      os << std::endl;
      os << "[Translation]: ";
      for (const auto& val : group.translation) {
          os << val << " ";
      }
      os << std::endl;
      os << "[Object ID]: " << group.object_id << std::endl;
      return os;
  }
};


bool compareByScore(const GraspGroup& a, const GraspGroup& b);

class GraspGroupResult {
 public:
  std::vector<GraspGroup> graspgroups;
};

class GraspOutputParser {
 public:
  GraspOutputParser() {}
  ~GraspOutputParser() {}

  int32_t Parse(
      std::shared_ptr<GraspGroupResult> &output,
      std::vector<std::shared_ptr<DNNTensor>> &output_tensors);

  int Decode(
    const float* fp2_xyz,
    const float* objectness_score,
    const float* grasp_top_view_xyz,
    const float* grasp_score_pred,
    const float* grasp_angle_cls_pred,
    const float* grasp_width_pred,
    const float* grasp_tolerance_pred,
    std::shared_ptr<GraspGroupResult> &output
  );
 
 private:
  int fp2_xyz_index_ = 0;
  int objectness_score_index_ = 1;
  int grasp_top_view_xyz_index_ = 5;
  int grasp_score_pred_index_ = 7;
  int grasp_angle_cls_pred_index_ = 8;
  int grasp_width_pred_index_ = 9;
  int grasp_tolerance_pred_index_ = 10;

};

#endif  // GRASP_OUTPUT_PARSER_H_
