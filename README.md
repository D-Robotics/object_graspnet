# 功能介绍

GraspNet（https://graspnet.net/） 是一个用于机器人抓取研究的大规模数据集和基准测试平台。它由清华大学的研究人员开发，旨在推动机器人抓取任务的研究与应用。

本项目是基于GraspNet的端侧推理节点, 根据置信度大小降序排序, 输出用于下游机械臂抓取的GraspGroup信息。目前无做输出结果数量和置信度阈值限制。

| 彩色图            | 深度图 |
| ------------------ | -------- | 
| ![](./config/bottle.png)  | ![](./config/depth.png) | 

算法输出的数据结构GraspGroup：
```c++
class GraspGroup {
  float score;
  float width;
  float height;
  float depth;
  std::vector<float> rotation;
  std::vector<float> translation;
  int object_id;
}
```

# 开发环境

- 编程语言: C/C++
- 开发平台: X5
- 系统版本：Ubuntu 22.04
- 编译工具链:Linux GCC 11.4.0

# 编译

- X5版本：支持在X5 Ubuntu系统上编译和在PC上使用docker交叉编译两种方式。

## 依赖库

- opencv:3.4.5

ros package：

- dnn node
- cv_bridge
- sensor_msgs
- hbm_img_msgs
- ai_msgs

hbm_img_msgs为自定义的图片消息格式，用于shared mem场景下的图片传输，hbm_img_msgs pkg定义在hobot_msgs中，因此如果使用shared mem进行图片传输，需要依赖此pkg。

## 编译选项

1、SHARED_MEM

- shared mem（共享内存传输）使能开关，默认打开（ON），编译时使用-DSHARED_MEM=OFF命令关闭。
- 如果打开，编译和运行会依赖hbm_img_msgs pkg，并且需要使用tros进行编译。
- 如果关闭，编译和运行不依赖hbm_img_msgs pkg，支持使用原生ros和tros进行编译。
- 对于shared mem通信方式，当前只支持订阅nv12格式图片。

## X5 Ubuntu系统上编译

1、编译环境确认

- 板端已安装X5 Ubuntu系统。
- 当前编译终端已设置TogetherROS环境变量：`source PATH/setup.bash`。其中PATH为TogetherROS的安装路径。
- 已安装ROS2编译工具colcon。安装的ROS不包含编译工具colcon，需要手动安装colcon。colcon安装命令：`pip install -U colcon-common-extensions`
- 已编译dnn node package

2、编译

- 编译命令：`colcon build --packages-select object_graspnet`

## docker交叉编译 X5 版本

1、编译环境确认

- 在docker中编译，并且docker中已经安装好TogetherROS。docker安装、交叉编译说明、TogetherROS编译和部署说明详见机器人开发平台robot_dev_config repo中的README.md。
- 已编译 object_graspnet package
- 已编译 hbm_img_msgs package（编译方法见Dependency部分）

2、编译

- 编译命令：

  ```shell
  # RDK X5
  bash robot_dev_config/build.sh -p X5 -s object_graspnet

- 编译选项中默认打开了shared mem通信方式。

## 注意事项

# 使用介绍

## 参数

| 参数名             | 解释                                  | 是否必须             | 默认值              | 备注                                                                    |
| ------------------ | ------------------------------------- | -------------------- | ------------------- | ----------------------------------------------------------------------- |
| cache_len_limit         | 限制推理数             | 否                   | 2                   |
| feed_type          | 图片来源，0：本地；1：订阅            | 否                   | 0                   |
| image              | 本地深度图片地址                          | 否                   | config/depth.png     |
| is_shared_mem_sub  | 使用shared mem通信方式订阅图片        | 否                   | 0                   | 
| is_sync_mode  | 推理模式，0：同步；1：异步        | 否                   | 0                   | 
| model_file_name        | 模型文件            | 否 | config/graspnet_test.bin                   |
| num_points         | 采样点云数             | 否                   | 8000                   | 
| ai_msg_pub_topic_name         | 智能消息发布话题信息             | 否                   | /hobot_object_graspgroup                   | 
| ros_img_sub_topic_name         | 深度图订阅话题信息            | 否                   | /camera/depth/image_raw                   | 

          

## 运行

## X5 Ubuntu系统上运行

运行方式1，使用可执行文件启动：
```shell
export COLCON_CURRENT_PREFIX=./install
source /opt/ros/humble/setup.bash
source ./install/local_setup.bash
# config中为示例使用的模型，回灌使用的本地图片
cp -r install/object_graspnet/lib/object_graspnet/config/ .

# 运行模式1：使用本地png格式深度图通过同步模式进行回灌预测
ros2 run object_graspnet object_graspnet --ros-args -p feed_type:=0 -p image:=config/depth.png

# 运行模式2：使用订阅到的image msg(topic为/camera/depth/image_raw )通过异步模式进行预测，并设置log级别为warn
ros2 run object_graspnet object_graspnet --ros-args -p feed_type:=1 --ros-args --log-level warn
```

## X5 buildroot系统上运行

```shell
export ROS_LOG_DIR=/userdata/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./install/lib/

# config中为示例使用的模型，回灌使用的本地图片
cp -r install/lib/object_graspnet/config/ .

# 运行模式1：使用本地png格式深度图通过同步模式进行回灌预测
./install/lib/object_graspnet/object_graspnet --ros-args -p  feed_type:=0 -p image:=config/depth.png

# 运行模式2：使用订阅到的image msg(topic为/camera/depth/image_raw )通过异步模式进行预测，并设置log级别为warn
./install/lib/object_graspnet/object_graspnet --ros-args -p feed_type:=1 --log-level warn

```

## 注意事项

- cam_info.json配置文件格式为json格式，存储相机参数信息，具体配置如下：

```json
  {
    "height": 720,
    "width": 1280,
    "fx": 637.91,
    "fy": 637.91,
    "cx": 639.65,
    "cy": 391.311,
    "scale": 1000.0
  }
```
# 结果分析

## X5结果展示

log：

运行命令：`ros2 run object_graspnet object_graspnet --ros-args -p feed_type:=0 -p image:=config/depth.png`

```shell
[WARN] [0000014005.046650719] [graspnet_node]: Parameter:
 cache_len_limit: 2
 feed_type(0:local, 1:sub): 0
 image: config/depth.png
 is_shared_mem_sub: 0
 is_sync_mode_: 0
 model_file_name_: config/graspnet_test.bin
 num_points: 8000
 ai_msg_pub_topic_name: /hobot_object_graspgroup
 ros_img_sub_topic_name: /camera/depth/image_raw
[INFO] [0000014005.047067511] [dnn]: Node init.
[INFO] [0000014005.047105177] [graspnet_node]: Set node para.
[INFO] [0000014005.047182261] [dnn]: Model init.
[BPU_PLAT]BPU Platform Version(1.3.6)!
[HBRT] set log level as 0. version = 3.15.47.0
[DNN] Runtime version = 1.23.5_(3.15.47 HBRT)
[A][DNN][packed_model.cpp:247][Model](1970-01-01,03:53:26.664.510) [HorizonRT] The model builder version = 1.23.6
[W][DNN]bpu_model_info.cpp:491][Version](1970-01-01,03:53:27.330.41) Model: graspnet_test. Inconsistency between the hbrt library version 3.15.47.0 and the model build version 3.15.49.0 detected, in order to ensure correct model results, it is recommended to use compilation tools and the BPU SDK from the same OpenExplorer package.
[INFO] [0000014007.425685179] [dnn]: The model input 0 width is 1 and height is 3
[INFO] [0000014007.425961637] [dnn]:
Model Info:
name: graspnet_test.
[input]
 - (0) Layout: NCHW, Shape: [1, 8000, 3, 1], Type: HB_DNN_TENSOR_TYPE_F32.
[output]
 - (0) Layout: NCHW, Shape: [1, 1024, 3, 1], Type: HB_DNN_TENSOR_TYPE_F32.
 - (1) Layout: NCHW, Shape: [1, 2, 1024, 1], Type: HB_DNN_TENSOR_TYPE_F32.
 - (2) Layout: NCHW, Shape: [1, 1024, 300, 1], Type: HB_DNN_TENSOR_TYPE_F32.
 - (3) Layout: NCHW, Shape: [1, 1024, 1, 1], Type: HB_DNN_TENSOR_TYPE_S64.
 - (4) Layout: NCHW, Shape: [1, 1024, 1, 1], Type: HB_DNN_TENSOR_TYPE_F32.
 - (5) Layout: NCHW, Shape: [1, 1024, 3, 1], Type: HB_DNN_TENSOR_TYPE_F32.
 - (6) Layout: NCHW, Shape: [1, 1024, 3, 3], Type: HB_DNN_TENSOR_TYPE_F32.
 - (7) Layout: NCHW, Shape: [1, 12, 1024, 4], Type: HB_DNN_TENSOR_TYPE_F32.
 - (8) Layout: NCHW, Shape: [1, 12, 1024, 4], Type: HB_DNN_TENSOR_TYPE_F32.
 - (9) Layout: NCHW, Shape: [1, 12, 1024, 4], Type: HB_DNN_TENSOR_TYPE_F32.
 - (10) Layout: NCHW, Shape: [1, 12, 1024, 4], Type: HB_DNN_TENSOR_TYPE_F32.

[INFO] [0000014007.426266220] [dnn]: Task init.
[INFO] [0000014007.428452637] [dnn]: Set task_num [4]
[INFO] [0000014017.167694017] [graspnet_node]: Output grasp group size: 1024
GraspGroup[0] Information:
[Score]: 0.469016
[Width]: 0.067
[Height]: 0.02
[Depth]: 0.01
[Rotation]:
-0.538688 0.205914 -0.816955 -0.40883 0.783967 0.467176 0.736664 0.585658 -0.33813
[Translation]: -0.716535 -0.00787402 0.834646
[Object ID]: -1

[INFO] [0000014017.167984725] [graspnet_node]: feedback finshed!
```
