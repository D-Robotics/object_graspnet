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

#include <fstream>
#include <math.h>
#include <memory>
#include <unistd.h>
#include <utility>

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"

#include "include/graspnet_node.h"

builtin_interfaces::msg::Time ConvertToRosTime(
    const struct timespec& time_spec) {
  builtin_interfaces::msg::Time stamp;
  stamp.set__sec(time_spec.tv_sec);
  stamp.set__nanosec(time_spec.tv_nsec);
  return stamp;
}

int CalTimeMsDuration(const builtin_interfaces::msg::Time& start,
                      const builtin_interfaces::msg::Time& end) {
  return (end.sec - start.sec) * 1000 + end.nanosec / 1000 / 1000 -
         start.nanosec / 1000 / 1000;
}

GrashpNetNode::GrashpNetNode(const std::string& node_name,
                               const NodeOptions& options)
    : DnnNode(node_name, options) {
  this->declare_parameter<int>("cache_len_limit", cache_len_limit_);
  this->declare_parameter<int>("feed_type", feed_type_);
  this->declare_parameter<std::string>("image", image_);
  this->declare_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->declare_parameter<int>("is_sync_mode", is_sync_mode_);
  this->declare_parameter<std::string>("model_file_name", model_file_name_);
  this->declare_parameter<int>("num_points", num_points_);
  this->declare_parameter<std::string>("ai_msg_pub_topic_name",
                                       ai_msg_pub_topic_name_);
  this->declare_parameter<std::string>("ros_img_sub_topic_name",
                                       ros_img_sub_topic_name_);

  this->get_parameter<int>("cache_len_limit", cache_len_limit_);
  this->get_parameter<int>("feed_type", feed_type_);
  this->get_parameter<std::string>("image", image_);
  this->get_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->get_parameter<int>("is_sync_mode", is_sync_mode_);
  this->get_parameter<std::string>("model_file_name", model_file_name_);
  this->get_parameter<int>("num_points", num_points_);
  this->get_parameter<std::string>("ai_msg_pub_topic_name",
                                   ai_msg_pub_topic_name_);
  this->get_parameter<std::string>("ros_img_sub_topic_name",
                                   ros_img_sub_topic_name_);

  std::stringstream ss;
  ss << "Parameter:"
     << "\n cache_len_limit: " << cache_len_limit_
     << "\n feed_type(0:local, 1:sub): " << feed_type_
     << "\n image: " << image_
     << "\n is_shared_mem_sub: " << is_shared_mem_sub_
     << "\n is_sync_mode_: " << is_sync_mode_
     << "\n model_file_name_: " << model_file_name_
     << "\n num_points: " << num_points_
     << "\n ai_msg_pub_topic_name: " << ai_msg_pub_topic_name_
     << "\n ros_img_sub_topic_name: " << ros_img_sub_topic_name_;
  RCLCPP_WARN(rclcpp::get_logger("graspnet_node"), "%s", ss.str().c_str());

  // Load DNN model config
  if (Init() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Init failed!");
    return;
  }
  auto model = GetModel();
  if (!model) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Invalid model");
    return;
  }
  model->GetInputTensorProperties(tensor_properties_, 0);

  // Load Cam config
  if (LoadConfig() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Load config failed! use default config");
  }
  preprossor_ = std::make_shared<InputPreProcessor>(cam_info_);


  if (0 == feed_type_) {
    Feedback();
    // Debug();
  } else {
    msg_publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
        ai_msg_pub_topic_name_, 10);

    if (is_shared_mem_sub_) {
#ifdef SHARED_MEM_ENABLED
      RCLCPP_WARN(rclcpp::get_logger("graspnet_node"),
                  "Create hbmem_subscription with topic_name: %s",
                  sharedmem_img_topic_name_.c_str());
      sharedmem_img_subscription_ =
          this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
              sharedmem_img_topic_name_,
              rclcpp::SensorDataQoS(),
              std::bind(&GrashpNetNode::SharedMemImgProcess,
                        this,
                        std::placeholders::_1));
#else
      RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Unsupport shared mem");
#endif
    } else {
      RCLCPP_WARN(rclcpp::get_logger("graspnet_node"),
                  "Create subscription with topic_name: %s",
                  ros_img_sub_topic_name_.c_str());
      ros_img_subscription_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              ros_img_sub_topic_name_,
              10,
              std::bind(
                  &GrashpNetNode::RosImgProcess, this, std::placeholders::_1));
    }
  }
}

GrashpNetNode::~GrashpNetNode() {}

int GrashpNetNode::SetNodePara() {
  RCLCPP_INFO(rclcpp::get_logger("graspnet_node"), "Set node para.");
  if (!dnn_node_para_ptr_) {
    return -1;
  }
  dnn_node_para_ptr_->model_file = model_file_name_;
  dnn_node_para_ptr_->model_name = model_name_;
  dnn_node_para_ptr_->model_task_type = model_task_type_;
  dnn_node_para_ptr_->task_num = 4;
  return 0;
}

int GrashpNetNode::LoadConfig() {
  if (config_file.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"),
                 "Config file [%s] is empty!",
                 config_file.data());
    return -1;
  }
  // Parsing config
  std::ifstream ifs(config_file.c_str());
  if (!ifs) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"),
                 "Read config file [%s] fail!",
                 config_file.data());
    return -1;
  }
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document document;
  document.ParseStream(isw);
  if (document.HasParseError()) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"),
                 "Parsing config file %s failed",
                 config_file.data());
    return -1;
  }

  if (document.HasMember("height")) {
    cam_info_.height = document["height"].GetInt();
  }
  if (document.HasMember("height")) {
    cam_info_.height = document["height"].GetInt();
  }
  if (document.HasMember("fx")) {
    cam_info_.fx = document["fx"].GetFloat();
  }
  if (document.HasMember("fy")) {
    cam_info_.fy = document["fy"].GetFloat();
  }
  if (document.HasMember("cx")) {
    cam_info_.cx = document["cx"].GetFloat();
  }
  if (document.HasMember("cy")) {
    cam_info_.cy = document["cy"].GetFloat();
  }
  if (document.HasMember("scale")) {
    cam_info_.scale = document["scale"].GetFloat();
  }

  return 0;
}

int GrashpNetNode::PostProcess(
    const std::shared_ptr<DnnNodeOutput>& node_output) {
  if (!rclcpp::ok()) {
    return 0;
  }

  std::unique_lock<std::mutex> lg(mtx_img_);
  current_cache_--;
  lg.unlock();

  if (!node_output) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Invalid node output");
    return -1;
  }

  auto grasp_output = std::dynamic_pointer_cast<GraspNetOutput>(node_output);
  if (!grasp_output) {
    return -1;
  }

  // 1. 获取后处理开始时间
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);

  // 2. 模型后处理解析
  auto parser = std::make_shared<GraspOutputParser>();
  auto gg_val = std::make_shared<GraspGroupResult>();
  parser->Parse(gg_val, grasp_output->output_tensors);
  
  if (!gg_val) {
    return -1;
  }

  std::stringstream ss;
  ss << "Output ";
  if (grasp_output->image_msg_header) {
    ss << "from frame_id: " << grasp_output->image_msg_header->frame_id
       << ", stamp: " << grasp_output->image_msg_header->stamp.sec << "_"
       << grasp_output->image_msg_header->stamp.nanosec << ", ";
  }
  ss << "grasp group size: " << gg_val->graspgroups.size()
     << "\nGraspGroup[0] Information:\n" << gg_val->graspgroups[0];
  RCLCPP_INFO(rclcpp::get_logger("graspnet_node"), "%s", ss.str().c_str());

  if (feed_type_ == 0) {
    RCLCPP_INFO(rclcpp::get_logger("graspnet_node"), "feedback finshed!");
    return 0;
  }
  // 3. 发布模型推理话题消息
  if (!msg_publisher_) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Invalid msg_publisher_");
    return -1;
  }
  ai_msgs::msg::PerceptionTargets::UniquePtr pub_data(
      new ai_msgs::msg::PerceptionTargets());
  if (grasp_output->image_msg_header) {
    pub_data->header.set__stamp(grasp_output->image_msg_header->stamp);
    pub_data->header.set__frame_id(grasp_output->image_msg_header->frame_id);
  }

  if (grasp_output->rt_stat) {
    pub_data->set__fps(round(grasp_output->rt_stat->output_fps));
  }

  // render with each task
  auto &graspgroups = gg_val->graspgroups;
  for (auto& graspgroup : graspgroups) {
    ai_msgs::msg::GraspGroup gg;
    gg.set__score(graspgroup.score);
    gg.set__width(graspgroup.width);
    gg.set__height(graspgroup.height);
    gg.set__depth(graspgroup.depth);
    gg.rotation.swap(graspgroup.rotation);
    gg.translation.swap(graspgroup.translation);
    gg.set__object_id(graspgroup.object_id);

    ai_msgs::msg::Target target;
    target.set__type("graspgroup");
    target.graspgroup.emplace_back(std::move(gg));
    pub_data->targets.emplace_back(std::move(target));
  }

  grasp_output->perf_preprocess.set__time_ms_duration(
      CalTimeMsDuration(grasp_output->perf_preprocess.stamp_start,
                        grasp_output->perf_preprocess.stamp_end));
  pub_data->perfs.push_back(grasp_output->perf_preprocess);

  // predict
  if (grasp_output->rt_stat) {
    ai_msgs::msg::Perf perf;
    perf.set__type(model_name_ + "_predict_infer");
    perf.set__stamp_start(
        ConvertToRosTime(grasp_output->rt_stat->infer_timespec_start));
    perf.set__stamp_end(
        ConvertToRosTime(grasp_output->rt_stat->infer_timespec_end));
    perf.set__time_ms_duration(grasp_output->rt_stat->infer_time_ms);
    pub_data->perfs.push_back(perf);

    perf.set__type(model_name_ + "_predict_parse");
    perf.set__stamp_start(
        ConvertToRosTime(grasp_output->rt_stat->parse_timespec_start));
    perf.set__stamp_end(
        ConvertToRosTime(grasp_output->rt_stat->parse_timespec_end));
    perf.set__time_ms_duration(grasp_output->rt_stat->parse_time_ms);
    pub_data->perfs.push_back(perf);
  }

  ai_msgs::msg::Perf perf_postprocess;
  perf_postprocess.set__type(model_name_ + "_postprocess");
  perf_postprocess.set__stamp_start(ConvertToRosTime(time_now));
  clock_gettime(CLOCK_REALTIME, &time_now);
  perf_postprocess.set__stamp_end(ConvertToRosTime(time_now));
  perf_postprocess.set__time_ms_duration(CalTimeMsDuration(
      perf_postprocess.stamp_start, perf_postprocess.stamp_end));
  pub_data->perfs.emplace_back(perf_postprocess);

  // 从发布图像到发布AI结果的延迟
  ai_msgs::msg::Perf perf_pipeline;
  perf_pipeline.set__type(model_name_ + "_pipeline");
  perf_pipeline.set__stamp_start(pub_data->header.stamp);
  perf_pipeline.set__stamp_end(perf_postprocess.stamp_end);
  perf_pipeline.set__time_ms_duration(
      CalTimeMsDuration(perf_pipeline.stamp_start, perf_pipeline.stamp_end));
  pub_data->perfs.push_back(perf_pipeline);

  // 如果当前帧有更新统计信息，输出统计信息
  if (grasp_output->rt_stat->fps_updated) {
    RCLCPP_WARN(rclcpp::get_logger("graspnet_node"),
                "input fps: %.2f, output fps: %.2f, infer time ms: %d, "
                "post process time ms: %d",
                grasp_output->rt_stat->input_fps,
                grasp_output->rt_stat->output_fps,
                grasp_output->rt_stat->infer_time_ms,
                static_cast<int>(perf_postprocess.time_ms_duration));
  }

  msg_publisher_->publish(std::move(pub_data));
  return 0;
}

void GrashpNetNode::RosImgProcess(
    const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
  if (!img_msg || !rclcpp::ok()) {
    return;
  }

  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

  std::stringstream ss;
  ss << "Recved img encoding: " << img_msg->encoding
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step
     << ", frame_id: " << img_msg->header.frame_id
     << ", stamp: " << img_msg->header.stamp.sec << "_"
     << img_msg->header.stamp.nanosec
     << ", data size: " << img_msg->data.size();
  RCLCPP_INFO(rclcpp::get_logger("graspnet_node"), "%s", ss.str().c_str());
  // 1. 将深度图处理成点云, 并保存为模型输入数据类型DNNTensor
  std::shared_ptr<DNNTensor> tensor = nullptr;
  if ("16UC1" == img_msg->encoding) {
    std::vector<std::vector<float>> cloud_sampled;
    preprossor_->Process(
          reinterpret_cast<const uint16_t*>(img_msg->data.data()),
          cloud_sampled, 
          img_msg->height,
          img_msg->width,
          num_points_);

    float* flat_data = new float[num_points_ * 3];
    // Copy data from the 2D vector to the 1D array
    size_t index = 0;
    for (const auto& vec : cloud_sampled) {
        std::memcpy(flat_data + index, vec.data(), vec.size() * sizeof(float));
        index += vec.size();
    }
    const char* data = reinterpret_cast<const char*>(flat_data);

    tensor = preprossor_->GetPointCloudTensor(        
        reinterpret_cast<const char*>(data),
        tensor_properties_);

  } else {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Unsupport img encoding: %s",
    img_msg->encoding.data());
  }
  if (!tensor) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Get Tensor fail");
    return;
  }
  
  std::vector<std::shared_ptr<DNNTensor>> inputs;
  inputs.push_back(tensor);

  // 2. 创建推理输出数据
  auto dnn_output = std::make_shared<GraspNetOutput>();
  // 将图片消息的header填充到输出数据中，用于表示推理输出对应的输入信息
  dnn_output->image_msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->image_msg_header->set__frame_id(img_msg->header.frame_id);
  dnn_output->image_msg_header->set__stamp(img_msg->header.stamp);
  // 将当前的时间戳填充到输出数据中，用于计算perf
  dnn_output->perf_preprocess.stamp_start.sec = time_start.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_start.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  
  if (current_cache_ >= cache_len_limit_) {
    return;
  }
  std::unique_lock<std::mutex> lg(mtx_img_);
  current_cache_++;
  lg.unlock();

  // 3. 开始预测
  if (Run(inputs, dnn_output) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Run Model Error");
    return;
  }
}

#ifdef SHARED_MEM_ENABLED
void GrashpNetNode::SharedMemImgProcess(
    const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr img_msg) {
  if (!img_msg || !rclcpp::ok()) {
    return;
  }

  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

  std::stringstream ss;
  ss << "Recved img encoding: "
     << std::string(reinterpret_cast<const char*>(img_msg->encoding.data()))
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step << ", index: " << img_msg->index
     << ", stamp: " << img_msg->time_stamp.sec << "_"
     << img_msg->time_stamp.nanosec << ", data size: " << img_msg->data_size;
  RCLCPP_INFO(rclcpp::get_logger("graspnet_node"), "%s", ss.str().c_str());

  // 1. 将深度图处理成点云, 并保存为模型输入数据类型DNNTensor
  std::shared_ptr<DNNTensor> tensor = nullptr;
  if ("16UC1" ==
      std::string(reinterpret_cast<const char*>(img_msg->encoding.data()))) {
    std::vector<std::vector<float>> cloud_sampled;
    preprossor_->Process(
          reinterpret_cast<const uint16_t*>(img_msg->data.data()),
          cloud_sampled, 
          img_msg->height,
          img_msg->width,
          num_points_);

    float* flat_data = new float[num_points_ * 3];
    // Copy data from the 2D vector to the 1D array
    size_t index = 0;
    for (const auto& vec : cloud_sampled) {
        std::memcpy(flat_data + index, vec.data(), vec.size() * sizeof(float));
        index += vec.size();
    }
    const char* data = reinterpret_cast<const char*>(flat_data);
    tensor = preprossor_->GetPointCloudTensor(        
        reinterpret_cast<const char*>(data),
        tensor_properties_);
  } else {
    RCLCPP_INFO(rclcpp::get_logger("graspnet_node"),
                "Unsupported img encoding: %s",
                img_msg->encoding);
  }
  if (!tensor) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Get Tensor fail!");
    return;
  }
  std::vector<std::shared_ptr<DNNTensor>> inputs;
  inputs.push_back(tensor);

  // 2. 创建推理输出数据
  auto dnn_output = std::make_shared<GraspNetOutput>();
  // 将图片消息的header填充到输出数据中，用于表示推理输出对应的输入信息
  dnn_output->image_msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->image_msg_header->set__frame_id(std::to_string(img_msg->index));
  dnn_output->image_msg_header->set__stamp(img_msg->time_stamp);
  // 将当前的时间戳填充到输出数据中，用于计算perf
  dnn_output->perf_preprocess.stamp_start.sec = time_start.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_start.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  
  if (current_cache_ >= cache_len_limit_) {
    return;
  }
  std::unique_lock<std::mutex> lg(mtx_img_);
  current_cache_++;
  lg.unlock();

  // 3. 开始预测
  if (Run(inputs, dnn_output) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Run Model Error");
    return;
  }
}
#endif

int GrashpNetNode::Feedback() {

  // 1. preprocess
  std::vector<std::vector<float>> cloud_sampled;
  preprossor_->ProcessImg(image_, cloud_sampled, num_points_);

  float* flat_data = new float[num_points_ * 3];
  // Copy data from the 2D vector to the 1D array
  size_t index = 0;
  for (const auto& vec : cloud_sampled) {
      std::memcpy(flat_data + index, vec.data(), vec.size() * sizeof(float));
      index += vec.size();
  }
  const char* data = reinterpret_cast<const char*>(flat_data);

  // std::ifstream ifs("config/cloud_sample8000.bin", std::ios::in | std::ios::binary);
  // if (!ifs) {
  //   return -1;
  // }
  // ifs.seekg(0, std::ios::end);
  // int len = ifs.tellg();
  // ifs.seekg(0, std::ios::beg);
  // char* data = new char[len];
  // ifs.read(data, len);

  std::shared_ptr<DNNTensor> tensor = nullptr;
  tensor = preprossor_->GetPointCloudTensor(        
      reinterpret_cast<const char*>(data),
      tensor_properties_);

  // 2. 使用pyramid创建DNNInput对象inputs
  // inputs将会作为模型的输入通过RunInferTask接口传入
  std::vector<std::shared_ptr<DNNTensor>> inputs;
  inputs.push_back(tensor);
  auto dnn_output = std::make_shared<GraspNetOutput>();
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;

  // 3. 开始预测
  if (Run(inputs, dnn_output) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("graspnet_node"), "Run Model Error");
    return -1;
  }

  return 0;
}


int readbinary(const std::string &filename, const float* &dataOut) {
  std::string folder = "dump/";
  std::ifstream ifs(folder + filename + ".bin", std::ios::in | std::ios::binary);
  if (!ifs) {
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int len = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  char* data = new char[len];
  ifs.read(data, len);
  dataOut = reinterpret_cast<const float *>(data);
  return len / sizeof(float);
}

int GrashpNetNode::Debug() {

  const float* fp2_xyz;
  const float* objectness_score;
  const float* grasp_top_view_xyz;
  const float* grasp_score_pred;
  const float* grasp_angle_cls_pred;
  const float* grasp_width_pred;
  const float* grasp_tolerance_pred;

  readbinary("fp2_xyz", fp2_xyz);
  readbinary("objectness_score", objectness_score);
  readbinary("grasp_top_view_xyz", grasp_top_view_xyz);
  readbinary("grasp_score_pred", grasp_score_pred);
  readbinary("grasp_angle_cls_pred", grasp_angle_cls_pred);
  readbinary("grasp_width_pred", grasp_width_pred);
  readbinary("grasp_tolerance_pred", grasp_tolerance_pred);

  auto parser = std::make_shared<GraspOutputParser>();
  auto gg_val = std::make_shared<GraspGroupResult>();
  parser->Decode(fp2_xyz, objectness_score, grasp_top_view_xyz, grasp_score_pred,
    grasp_angle_cls_pred, grasp_width_pred, grasp_tolerance_pred, gg_val);

  // 降序排序
  std::sort(gg_val->graspgroups.begin(), gg_val->graspgroups.end(), compareByScore);

  std::stringstream ss;
  ss << gg_val->graspgroups[0];
  RCLCPP_WARN(rclcpp::get_logger("graspnet_node"), "%s", ss.str().c_str());
  return 0;
}