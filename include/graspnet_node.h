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
#include <string>
#include <vector>

#include "ai_msgs/msg/grasp_group.hpp"
#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node.h"
#ifdef SHARED_MEM_ENABLED
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#endif
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "include/grasp_output_parser.h"
#include "include/data_preprocess.h"

#ifndef GRASPNET_NODE_H_
#define GRASPNET_NODE_H_

using rclcpp::NodeOptions;

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::DnnNodePara;

using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::ModelTaskType;
using hobot::dnn_node::ModelRoiInferTask;
using hobot::dnn_node::NV12PyramidInput;

using ai_msgs::msg::PerceptionTargets;

struct GraspNetOutput : public DnnNodeOutput {
  std::shared_ptr<std_msgs::msg::Header> image_msg_header = nullptr;

  // 原始点云
  std::vector<std::vector<float>> clouds;

  ai_msgs::msg::Perf perf_preprocess;
};

class GrashpNetNode : public DnnNode {
 public:
  GrashpNetNode(const std::string &node_name,
                 const NodeOptions &options = NodeOptions());
  ~GrashpNetNode() override;

 protected:
  int SetNodePara() override;

  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  std::string ai_msg_pub_topic_name_ = "/hobot_object_graspgroup";
  int cache_len_limit_ = 2;
  CameraInfo cam_info_ = {720, 1280, 637.91, 637.91, 639.65, 391.311, 1000.0};
  std::string config_file = "config/cam_info.json";
  // 用于预测的图片来源, 0：本地U16位深度图, 1： 订阅到的image msg
  int feed_type_ = 0;
  std::string image_ = "config/depth.png";
  int is_collision_detect_ = 0;
  int is_sync_mode_ = 0;
  // 使用shared mem通信方式订阅图片
  int is_shared_mem_sub_ = 0;
  std::string model_file_name_ = "config/graspnet_test.bin";
  std::string model_name_ = "graspnet_test";
  int num_points_ = 8000;
  int topk_ = 5;

  int LoadConfig();
  
  int Feedback();

  int Debug();

#ifdef SHARED_MEM_ENABLED
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::ConstSharedPtr
      sharedmem_img_subscription_ = nullptr;
  std::string sharedmem_img_topic_name_ = "/hbmem_img";
  void SharedMemImgProcess(
      const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr msg);
#endif

  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr msg_publisher_ =
      nullptr;

  rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr
      ros_img_subscription_ = nullptr;
  // 目前只支持订阅深度图原图
  std::string ros_img_sub_topic_name_ = "/camera/depth/image_raw";
  void RosImgProcess(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  std::shared_ptr<InputPreProcessor> preprossor_;
  hbDNNTensorProperties tensor_properties_;

  ModelTaskType model_task_type_ = ModelTaskType::ModelInferType;

  std::mutex mtx_img_;  
  int current_cache_ = 0;
};

#endif  // GRASPNET_NODE_H_
