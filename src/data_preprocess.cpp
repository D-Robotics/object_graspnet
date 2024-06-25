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

#include "include/data_preprocess.h"

/**
 * @brief 深度图转点云
 * @param depth 深度图，uint16格式
 * @param CameraInfo 相机参数
 * @param workspace_mask 掩码图，uint8格式
 * @return 点云坐标，格式：
 *         X1 Y1 Z1
 *         X2 Y2 Z2
 *         ......
 */
std::vector<std::vector<float>> create_point_cloud_from_depth_image(const cv::Mat &depth, const CameraInfo &camera_info, const cv::Mat &workspace_mask)
{
    // 根据相机参数将深度图转为点云
    // assert(depth.cols == camera_info.width && depth.rows == camera_info.height);
    std::vector<std::vector<float>> cloud;
    for (int v = 0; v < depth.rows; v++)
    {
        for (int u = 0; u < depth.cols; u++)
        {
            uint16_t depth_val = depth.at<uint16_t>(v, u);
            uint8_t workspace_mask_val = workspace_mask.at<uint8_t>(v, u);
            if (depth_val > 0 && workspace_mask_val > 0)
            {
                float points_z = depth_val / camera_info.scale;
                float points_x = (u - camera_info.cx) * points_z / camera_info.fx;
                float points_y = (v - camera_info.cy) * points_z / camera_info.fy;
                cloud.push_back({points_x, points_y, points_z});
            }
        }
    }
    return cloud;
}

/**
 * @brief 无重复在[range_start, range_end]随机取count个数字
 * @param range_start 起点
 * @param range_end 终点
 * @param count 取count个数
 * @return 选取的随机数
 */
std::vector<int> gen_unique_random_nums(int range_start, int range_end, int count)
{
    // 初始化候选元素
    std::vector<int> candidates;
    for (int i = range_start; i <= range_end; ++i)
    {
        candidates.push_back(i);
    }

    // 随机数生成器
    std::random_device rd;
    std::mt19937 g(rd());

    // 打乱顺序
    shuffle(candidates.begin(), candidates.end(), g);

    // 选择前 count 个元素
    std::vector<int> selected_elements(candidates.begin(), candidates.begin() + count);

    return selected_elements;
}

/**
 * @brief 对数据进行采样
 * @param data 数据
 * @param indices 采样坐标
 * @return 采样结果
 */
std::vector<std::vector<float>> sample_points(const std::vector<std::vector<float>> &data, const std::vector<int> &indices)
{
    std::vector<std::vector<float>> sample_data;
    sample_data.reserve(indices.size()); // 预分配内存以提高性能

    for (auto idx : indices)
    {
        sample_data.push_back(data[idx]);
    }
    return sample_data;
}

int32_t InputPreProcessor::Process(
    const uint16_t *in_depth_data,
    std::vector<std::vector<float>>& cloud_sampled,
    std::vector<std::vector<float>>& cloud_masked,
    const int height,
    const int width,
    const int num_points) {

    cv::Mat depth(width, height, CV_8UC3);
    std::memcpy(depth.data, in_depth_data, height * width * sizeof(uint16_t));
    if (depth.empty())
    {
        return -1;
    }

    cv::Mat workspace_mask = cv::Mat::ones(cv::Size(height, width), CV_8UC1);
    cloud_masked = create_point_cloud_from_depth_image(depth, camera_info, workspace_mask);
    if (cloud_masked.size() >= num_points)
    {
        std::vector<int> indices = gen_unique_random_nums(0, cloud_masked.size() - 1, num_points);
        cloud_sampled = sample_points(cloud_masked, indices);
    }
    else
    {
        std::vector<int> indices = gen_unique_random_nums(0, cloud_masked.size() - 1, num_points - cloud_masked.size());
        std::vector<std::vector<float>> cloud_sampled_tmp = sample_points(cloud_masked, indices);
        cloud_sampled.reserve(cloud_masked.size() + cloud_sampled_tmp.size());
        cloud_sampled.insert(cloud_sampled.end(), cloud_masked.begin(), cloud_masked.end());
        cloud_sampled.insert(cloud_sampled.end(), cloud_sampled_tmp.begin(), cloud_sampled_tmp.end());
    }
    return 0;
}


int32_t InputPreProcessor::ProcessImg(
    const std::string& in_depth,
    std::vector<std::vector<float>>& cloud_sampled,
    std::vector<std::vector<float>>& cloud_masked,
    const int num_points) {

    cv::Mat depth = cv::imread(in_depth, cv::IMREAD_UNCHANGED);
    cv::Mat workspace_mask;

    if (depth.empty())
    {
        throw std::invalid_argument("depth is empty!");
    }
    if (workspace_mask.empty())
    {
        workspace_mask = cv::Mat::ones(cv::Size(1280, 720), CV_8UC1);
    }

    cloud_masked = create_point_cloud_from_depth_image(depth, camera_info, workspace_mask);
    if (cloud_masked.size() >= num_points)
    {
        std::vector<int> indices = gen_unique_random_nums(0, cloud_masked.size() - 1, num_points);
        cloud_sampled = sample_points(cloud_masked, indices);
    }
    else
    {
        std::vector<int> indices = gen_unique_random_nums(0, cloud_masked.size() - 1, num_points - cloud_masked.size());
        std::vector<std::vector<float>> cloud_sampled_tmp = sample_points(cloud_masked, indices);
        cloud_sampled.reserve(cloud_masked.size() + cloud_sampled_tmp.size());
        cloud_sampled.insert(cloud_sampled.end(), cloud_masked.begin(), cloud_masked.end());
        cloud_sampled.insert(cloud_sampled.end(), cloud_sampled_tmp.begin(), cloud_sampled_tmp.end());
    }
    return 0;
}


std::shared_ptr<hobot::dnn_node::DNNTensor> InputPreProcessor::GetPointCloudTensor(
    const char *in_data,
    hbDNNTensorProperties tensor_properties) {

  int src_elem_size = 4;
  int num_points = tensor_properties.alignedShape.dimensionSize[1];

  auto *mem = new hbSysMem;
  hbSysAllocCachedMem(mem, num_points * 3 * src_elem_size);
  //内存初始化
  memset(mem->virAddr, 0, num_points * 3 * src_elem_size);

  const uint8_t *data = reinterpret_cast<const uint8_t *>(in_data);
  auto *hb_mem_addr = reinterpret_cast<uint8_t *>(mem->virAddr);

  memcpy(hb_mem_addr, data, num_points * 3 * src_elem_size);

  hbSysFlushMem(mem, HB_SYS_MEM_CACHE_CLEAN);
  auto input_tensor = new hobot::dnn_node::DNNTensor;

  input_tensor->properties = tensor_properties;
  input_tensor->sysMem[0].virAddr = reinterpret_cast<void *>(mem->virAddr);
  input_tensor->sysMem[0].phyAddr = mem->phyAddr;
  input_tensor->sysMem[0].memSize = num_points * 3 * src_elem_size;
  return std::shared_ptr<hobot::dnn_node::DNNTensor>(
      input_tensor, [mem](hobot::dnn_node::DNNTensor *input_tensor) {
        // Release memory after deletion
        hbSysFreeMem(mem);
        delete mem;
        delete input_tensor;
      });
}
