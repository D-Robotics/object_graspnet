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

#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "dnn_node/dnn_node_data.h"

#ifndef DATA_PREPROCESS_H_
#define DATA_PREPROCESS_H_

/**
 * @brief 相机参数
 */
struct CameraInfo
{
    int height;
    int width;
    float fx;
    float fy;
    float cx;
    float cy;
    float scale; // 深度scale
};

std::vector<std::vector<float>> create_point_cloud_from_depth_image(const cv::Mat &depth, const CameraInfo &camera_info, const cv::Mat &workspace_mask);
std::vector<int> gen_unique_random_nums(int range_start, int range_end, int count);
std::vector<std::vector<float>> sample_points(const std::vector<std::vector<float>> &data, const std::vector<int> &indices);

class InputPreProcessor {
 public:
  InputPreProcessor(const CameraInfo& cam_info) : camera_info(cam_info) {}
  ~InputPreProcessor() {}

  /**
   * @brief 数据预处理
   * @param in_depth_data 深度原始数据
   * @param cloud_sampled 采样后的点云数组
   * @param height 深度图高度
   * @param width 深度图宽度
   * @param num_points 采样点云数
   * @return 状态
   */
  int32_t Process(
    const uint16_t *in_depth_data,
    std::vector<std::vector<float>>& cloud_sampled,
    const int height,
    const int width,
    const int num_points=20000);

  /**
   * @brief 图片数据预处理
   * @param in_depth 深度图路径
   * @param cloud_sampled 采样后的点云数组
   * @param num_points 采样点云数
   * @return 状态
   */
  int32_t ProcessImg(
    const std::string& in_depth,
    std::vector<std::vector<float>>& cloud_sampled,
    const int num_points=20000);

  /**
   * @brief 图片数据预处理
   * @param in_data 点云数据
   * @param tensor_properties DNN输入tensor的属性信息
   * @return DNNTensor向量
   */
  static std::shared_ptr<hobot::dnn_node::DNNTensor> GetPointCloudTensor(
        const char *in_data,
        hbDNNTensorProperties tensor_properties);

 private:
   CameraInfo camera_info;
};

#endif // DATA_PREPROCESS_H_