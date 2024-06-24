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

#include <Eigen/Dense>

#include "include/grasp_collision_detector.h"

template<typename T>
std::vector<T> computeBoundingBox(const std::vector<std::vector<T>>& points) {
    if (points.empty() || points[0].size() != 3) {
        throw std::invalid_argument("Point cloud must contain points with 3 coordinates.");
    }

    // 初始化最小和最大值
    std::vector<T> min(3, std::numeric_limits<T>::max());

    // 遍历所有点，找出每个坐标的最小值和最大值
    for (const auto& point : points) {
        for (size_t i = 0; i < 3; ++i) {
            if (point[i] < min[i]) {
                min[i] = point[i];
            }
        }
    }
    return min;
}

// 哈希函数，用于 std::unordered_map
struct VectorHash {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t seed = 0;
        for (auto& i : vec) {
            seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

template<typename T>
std::vector<std::vector<T>> voxel_down_sample(const std::vector<std::vector<T>>& point_clouds, float voxel_size_) {
    if (voxel_size_ <= 0.0f) {
        throw std::invalid_argument("voxel_size_ must be greater than 0.");
    }

    if (point_clouds.empty()) {
        return {};
    }

    std::unordered_map<std::vector<int>, std::vector<std::vector<T>>, VectorHash> voxel_dict;

    std::vector<T> min_bound = computeBoundingBox(point_clouds);
    for (size_t i = 0; i < min_bound.size(); i++) {
        min_bound[i] = min_bound[i] - voxel_size_ * 0.5;
    }

    // 计算每个点的体素索引
    for (const auto& point : point_clouds) {
        std::vector<int> voxel_index(point.size());
        for (size_t i = 0; i < point.size(); ++i) {
            voxel_index[i] = static_cast<int>(std::floor((point[i] - min_bound[i]) / voxel_size_));
        }
        voxel_dict[voxel_index].push_back(point);
    }

    // 计算每个体素的平均点
    std::vector<std::vector<T>> downsampled_points;
    for (const auto& voxel : voxel_dict) {
        const auto& voxel_points = voxel.second;
        std::vector<T> voxel_center(voxel_points[0].size(), 0.0);

        for (const auto& voxel_point : voxel_points) {
            for (size_t i = 0; i < voxel_point.size(); ++i) {
                voxel_center[i] += voxel_point[i];
            }
        }

        for (size_t i = 0; i < voxel_center.size(); ++i) {
            voxel_center[i] /= static_cast<T>(voxel_points.size());
        }

        downsampled_points.push_back(voxel_center);
    }

    return downsampled_points;
}

int GraspCollisionDetector::Init(
    std::vector<std::vector<float>>& point_clouds) {
  
  sample_clouds_ = voxel_down_sample(point_clouds, voxel_size_);
  current_masks_ = 0;

  return 0;
}

int GraspCollisionDetector::Detect(std::vector<GraspGroup>& graspgroups,
            std::vector<int>& collision_mask,
            float approach_dist,
            float collision_thresh, 
            bool return_empty_grasp,
            float empty_thresh,
            bool return_ious) {

    int M = graspgroups.size();
    collision_mask.resize(M, 1);

    approach_dist = std::max(approach_dist, finger_width);

    for (int i = 0; i < M; ++i) {
        auto& grasp = graspgroups[i];
        Eigen::Vector3f T(grasp.translation.data());
        Eigen::Matrix3f R;
        R << grasp.rotation[0], grasp.rotation[1], grasp.rotation[2],
             grasp.rotation[3], grasp.rotation[4], grasp.rotation[5],
             grasp.rotation[6], grasp.rotation[7], grasp.rotation[8];
        
        float height = grasp.height;
        float depth = grasp.depth;
        float width = grasp.width;

        Eigen::MatrixXf targets(sample_clouds_.size(), 3);
        for (int j = 0; j < sample_clouds_.size(); ++j) {
            targets.row(j) = Eigen::Vector3f(sample_clouds_[j].data()) - T;
        }
        targets = targets * R;

        // height mask
        Eigen::Array<bool, Eigen::Dynamic, 1> mask1 = ((targets.col(2).array() > -height / 2) && (targets.col(2).array() < height / 2));
        // left finger mask
        Eigen::Array<bool, Eigen::Dynamic, 1> mask2 = ((targets.col(0).array() > depth - finger_length) && (targets.col(0).array() < depth));
        Eigen::Array<bool, Eigen::Dynamic, 1> mask3 = (targets.col(1).array() > -(width / 2 + finger_width));
        Eigen::Array<bool, Eigen::Dynamic, 1> mask4 = (targets.col(1).array() < -width / 2);
        // right finger mask
        Eigen::Array<bool, Eigen::Dynamic, 1> mask5 = (targets.col(1).array() < (width / 2 + finger_width));
        Eigen::Array<bool, Eigen::Dynamic, 1> mask6 = (targets.col(1).array() > width / 2);
        // bottom mask
        Eigen::Array<bool, Eigen::Dynamic, 1> mask7 = ((targets.col(0).array() <= depth - finger_length) && (targets.col(0).array() > depth - finger_length - finger_width));
        // shifting mask
        Eigen::Array<bool, Eigen::Dynamic, 1> mask8 = ((targets.col(0).array() <= depth - finger_length - finger_width) && (targets.col(0).array() > depth - finger_length - finger_width - approach_dist));

        // get collision mask of each point
        Eigen::Array<bool, Eigen::Dynamic, 1> left_mask = (mask1 && mask2 && mask3 && mask4);
        Eigen::Array<bool, Eigen::Dynamic, 1> right_mask = (mask1 && mask2 && mask5 && mask6);
        Eigen::Array<bool, Eigen::Dynamic, 1> bottom_mask = (mask1 && mask3 && mask5 && mask7);
        Eigen::Array<bool, Eigen::Dynamic, 1> shifting_mask = (mask1 && mask3 && mask5 && mask8);
        Eigen::Array<bool, Eigen::Dynamic, 1> global_mask = (left_mask || right_mask || bottom_mask || shifting_mask);

        // calculate equivalent volume of each part
        float left_right_volume = height * finger_length * finger_width / std::pow(voxel_size_, 3);
        float bottom_volume = height * (width + 2 * finger_width) * finger_width / std::pow(voxel_size_, 3);
        float shifting_volume = height * (width + 2 * finger_width) * approach_dist / std::pow(voxel_size_, 3);
        float volume = left_right_volume * 2 + bottom_volume + shifting_volume;

        // get collision IOU of each part
        float global_iou = global_mask.cast<float>().sum() / (volume + 1e-6);

        // get collision mask
        collision_mask[i] = (global_iou > collision_thresh);
        
        if (return_empty_grasp || return_ious) {
            // calculate empty mask if requested
            if (return_empty_grasp) {
                Eigen::Array<bool, Eigen::Dynamic, 1> inner_mask = (mask1 && mask2 && !mask4 && !mask6);
                float inner_volume = height * finger_length * width / std::pow(voxel_size_, 3);
                bool empty = (inner_mask.cast<float>().sum() / inner_volume < empty_thresh);
                collision_mask[i] = empty ? 0 : collision_mask[i]; // 使用0表示非空抓取
            }

            // calculate IOUs if requested
            if (return_ious) {
                // TODO
            }
        }
        current_masks_ += (collision_mask[i] == false);
        // check if get enough masks and return
        if (topk_ > 0 && current_masks_ == topk_) {
            current_masks_ = 0;
            return 0;
        }
    }
    return 0; // 返回0表示成功
}

int GraspCollisionDetector::Process(
    std::shared_ptr<GraspGroupResult> &input) {
  
  auto& graspgroups = input->graspgroups;
  std::vector<int> collision_mask;

  int res = Detect(graspgroups, collision_mask, approach_dist_, collision_thresh_);

  std::vector<GraspGroup> result;
  for (int i = 0; i < collision_mask.size(); i++) {
    if (collision_mask[i] == false) {
        result.push_back(graspgroups[i]);
    }
  }
  swap(input->graspgroups, result);
  return 0;
}