
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

#include "include/grasp_output_parser.h"

// 模板函数，计算向量的二范数
template<typename T>
T norm(const std::vector<T>& vec) {
    T result = 0;
    for (const auto& val : vec) {
        result += val * val;
    }
    return std::sqrt(result);
}

// 模板函数，对向量进行标准化
template<typename T>
std::vector<T> normalize(const std::vector<T>& vec) {
    T magnitude = norm(vec);
    std::vector<T> result;
    for (const auto& val : vec) {
        result.push_back(val / magnitude);
    }
    return result;
}

// 模板函数，计算向量的叉乘
template<typename T>
std::vector<T> cross_product(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    std::vector<T> result;
    result.push_back(vec1[1] * vec2[2] - vec1[2] * vec2[1]);
    result.push_back(vec1[2] * vec2[0] - vec1[0] * vec2[2]);
    result.push_back(vec1[0] * vec2[1] - vec1[1] * vec2[0]);
    return result;
}

// 定义矩阵乘法函数
template<typename T>
std::vector<T> matrixMultiplication(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
    // 获取矩阵 A 和 B 的行列数
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // 创建结果矩阵
    std::vector<T> result(rowsA * colsB, 0.0);
    
    // 检查矩阵尺寸是否满足乘法要求
    if (colsA != rowsB) {
        std::cerr << "Error: Matrix dimensions do not match for multiplication." << std::endl;
        return result;
    }

    // 执行矩阵乘法
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i * 3 + j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// 模板函数，转换批量的观察向量和角度为旋转矩阵
template<typename T>
std::vector<std::vector<T>> batch_viewpoint_params_to_matrix(const std::vector<std::vector<T>>& batch_towards, const std::vector<T>& batch_angle) {
    // 初始化参数
    size_t batch_size = batch_towards.size();
    std::vector<std::vector<T>> axis_x = batch_towards;
    std::vector<std::vector<T>> axis_y;
    std::vector<bool> mask_y(batch_size, false);

    // 计算axis_y
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<T> temp = {-axis_x[i][1], axis_x[i][0], 0};
        T norm_temp = norm(temp);
        if (norm_temp == 0) {
            temp = {0, 1, 0};
            mask_y[i] = true;
        }
        axis_y.push_back(temp);
    }

    // 标准化向量
    std::transform(axis_x.begin(), axis_x.end(), axis_x.begin(), normalize<T>);
    std::transform(axis_y.begin(), axis_y.end(), axis_y.begin(), normalize<T>);

    // 计算axis_z
    std::vector<std::vector<T>> axis_z;
    for (size_t i = 0; i < batch_size; ++i) {
        axis_z.push_back(cross_product(axis_x[i], axis_y[i]));
    }

    // 计算旋转矩阵
    std::vector<std::vector<std::vector<T>>> R1(batch_size, std::vector<std::vector<T>>(3, std::vector<T>(3)));
    for (size_t i = 0; i < batch_size; ++i) {
      T sin_val = std::sin(batch_angle[i]);
      T cos_val = std::cos(batch_angle[i]);
      R1[i][0] = {1, 0, 0};
      R1[i][1] = {0, cos_val, -sin_val};
      R1[i][2] = {0, sin_val, cos_val};
    }

    std::vector<std::vector<std::vector<T>>> R2(batch_size, std::vector<std::vector<T>>(3, std::vector<T>(3)));
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        R2[i][j] = {axis_x[i][j], axis_y[i][j], axis_z[i][j]};
      }
    }

    std::vector<std::vector<T>> matrix(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      matrix[i] = matrixMultiplication(R2[i], R1[i]);
    }

    return matrix;
}


// 模板函数，根据布尔掩码过滤向量
template <typename T>
std::vector<T> filter_by_mask(const std::vector<T>& data, const std::vector<bool>& mask) {
    std::vector<T> filtered_data;
    for (size_t i = 0; i < data.size(); ++i) {
        if (mask[i]) {
            filtered_data.push_back(data[i]);
        }
    }
    return filtered_data;
}

template <typename T>
std::vector<std::vector<T>> filter_by_mask2(const std::vector<std::vector<T>>& data, const std::vector<bool>& mask) {
    std::vector<std::vector<T>> filtered_data;
    for (size_t i = 0; i < data.size(); ++i) {
        if (mask[i]) {
          filtered_data.push_back(data[i]);
        }
    }
    return filtered_data;
}

std::vector<std::vector<int>> argmax1(std::vector<std::vector<float>>& grasp_score_gather) {
    size_t batch_size = grasp_score_gather.size();
    size_t num_grasps = grasp_score_gather[0].size();

    std::vector<std::vector<int>> grasp_depth_class(batch_size, std::vector<int>(1));

    for (size_t i = 0; i < batch_size; ++i) {
        // 寻找每一行中的最大值的索引
        auto max_index = std::max_element(grasp_score_gather[i].begin(), grasp_score_gather[i].end()) - grasp_score_gather[i].begin();
        grasp_depth_class[i][0] = max_index;
    }

    return grasp_depth_class;
}

template <typename T>
std::vector<std::vector<T>> gather(
          std::vector<std::vector<std::vector<T>>>& grasp_score, 
          std::vector<std::vector<int>> & grasp_angle_class_) {
    size_t num_grasps = grasp_angle_class_.size();
    size_t num_angles = grasp_angle_class_[0].size();

    std::vector<std::vector<T>> result(num_grasps, std::vector<T>(num_angles, 0));

    for (size_t i = 0; i < num_grasps; ++i) {
      for (size_t j = 0; j < num_angles; ++j) {
          int index = grasp_angle_class_[i][j];
          result[i][j] = grasp_score[index][i][j];
      }
    }
    return result;
}

template <typename T>
std::vector<T> gather2(
          std::vector<std::vector<T>>& grasp_score,
          std::vector<std::vector<int>>& grasp_class) {
  size_t num_grasps = grasp_class.size();

  std::vector<T> result(num_grasps, 0);

  for (size_t i = 0; i < num_grasps; ++i) {
      int index = grasp_class[i][0];
      result[i] = grasp_score[i][index];
  }
  return result;
}

int GraspOutputParser::Decode(
    const float* fp2_xyz,
    const float* objectness_score,
    const float* grasp_top_view_xyz,
    const float* grasp_score_pred,
    const float* grasp_angle_cls_pred,
    const float* grasp_width_pred,
    const float* grasp_tolerance_pred,
    std::shared_ptr<GraspGroupResult> &output
) {

    const int num_points = 1024;
    const int num_angles = 12;
    const int num_depths = 4;

    // Load predictions
    std::vector<std::vector<float>> objectness_score_vec(2, std::vector<float>(num_points));
    std::vector<std::vector<std::vector<float>>> grasp_score_vec(num_angles, std::vector<std::vector<float>>(num_points, std::vector<float>(num_depths)));
    std::vector<std::vector<float>> grasp_center_vec(num_points, std::vector<float>(3));
    std::vector<std::vector<float>> approaching_vec(num_points, std::vector<float>(3));
    std::vector<std::vector<std::vector<float>>> grasp_angle_class_score_vec(num_angles, std::vector<std::vector<float>>(num_points, std::vector<float>(num_depths)));
    std::vector<std::vector<std::vector<float>>> grasp_width_vec(num_angles, std::vector<std::vector<float>>(num_points, std::vector<float>(num_depths)));
    std::vector<std::vector<std::vector<float>>> grasp_tolerance_vec(num_angles, std::vector<std::vector<float>>(num_points, std::vector<float>(num_depths)));

    // storage
    // objectness_score
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 1024; j++) {
        objectness_score_vec[i][j] = objectness_score[i * 1024 + j];
      }
    }

    // grasp_score_pred
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 1024; j++) {
        for (int k = 0; k < 4; k++) {
          grasp_score_vec[i][j][k] = grasp_score_pred[(i * 1024 * 4) + (j * 4) + k];
        }
      }
    }


    // grasp_score_pred
    for (int i = 0; i < 1024; i++) {
      for (int j = 0; j < 3; j++) {
        grasp_center_vec[i][j] = fp2_xyz[(i * 3) + j];
      }
    }

    // grasp_top_view_xyz
    for (int i = 0; i < 1024; i++) {
      for (int j = 0; j < 3; j++) {
        approaching_vec[i][j] = -grasp_top_view_xyz[(i * 3) + j];
      }
    }

    // grasp_angle_cls_pred
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 1024; j++) {
        for (int k = 0; k < 4; k++) {
          grasp_angle_class_score_vec[i][j][k] = grasp_angle_cls_pred[(i * 1024 * 4) + (j * 4) + k];
        }
      }
    }


    // grasp_width_pred
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 1024; j++) {
        for (int k = 0; k < 4; k++) {
          grasp_width_vec[i][j][k] = 1.2 * grasp_width_pred[(i * 1024 * 4) + (j * 4) + k];
          grasp_width_vec[i][j][k] = std::clamp(grasp_width_vec[i][j][k], 0.0f, static_cast<float>(GRASP_MAX_WIDTH));
        }
      }
    }

    // grasp_tolerance_pred
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 1024; j++) {
        for (int k = 0; k < 4; k++) {
          grasp_tolerance_vec[i][j][k] = grasp_tolerance_pred[(i * 1024 * 4) + (j * 4) + k];
        }
      }
    }

    std::vector<std::vector<int>> grasp_angle_class(num_points, std::vector<int>(num_depths));
    for (int p = 0; p < num_points; ++p) {
        for (int d = 0; d < num_depths; ++d) {
            float max_score = grasp_angle_class_score_vec[0][p][d];
            for (int a = 1; a < num_angles; ++a) {
                if (grasp_angle_class_score_vec[a][p][d] > max_score) {
                    max_score = grasp_angle_class_score_vec[a][p][d];
                    grasp_angle_class[p][d] = static_cast<int>(a);
                }
            }
        }
    }

    std::vector<std::vector<float>> grasp_angle(num_points, std::vector<float>(num_depths));
    for (int p = 0; p < num_points; ++p) {
      for (int d = 0; d < num_depths; ++d) {
        grasp_angle[p][d] = static_cast<int>(grasp_angle_class[p][d]) * PI / 12;
      }
    }

    auto grasp_score_gather = gather(grasp_score_vec, grasp_angle_class);
    auto grasp_width_gather = gather(grasp_width_vec, grasp_angle_class);
    auto grasp_tolerance_gather = gather(grasp_tolerance_vec, grasp_angle_class);

    std::vector<std::vector<int>> grasp_depth_class = argmax1(grasp_score_gather);
    
    std::vector<float> grasp_depth(grasp_depth_class.size());
    for (int i = 0; i < grasp_depth.size(); i++) {
      grasp_depth[i] = (static_cast<float>(grasp_depth_class[i][0]) + 1) * 0.01;
    }

    auto grasp_score_gather2 = gather2(grasp_score_gather, grasp_depth_class);
    auto grasp_angle_gather2 = gather2(grasp_angle, grasp_depth_class);
    auto grasp_width_gather2 = gather2(grasp_width_gather, grasp_depth_class);
    auto grasp_tolerance_gather2 = gather2(grasp_tolerance_gather, grasp_depth_class);

    std::vector<int> objectness_pred(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        objectness_pred[i] = std::distance(objectness_score_vec.begin(), std::max_element(objectness_score_vec.begin(), objectness_score_vec.end(), 
            [i](const std::vector<float>& a, const std::vector<float>& b) {
                return a[i] < b[i];
            }));
    }
    // 创建objectness_mask并进行比较
    std::vector<bool> objectness_mask(num_points);
    std::transform(objectness_pred.begin(), objectness_pred.end(), objectness_mask.begin(), [](int pred) {
        return pred == 1;
    });


    // 根据objectness_mask过滤向量
    std::vector<float> filtered_grasp_score = filter_by_mask(grasp_score_gather2, objectness_mask);
    std::vector<float> filtered_grasp_width = filter_by_mask(grasp_width_gather2, objectness_mask);
    std::vector<float> filtered_grasp_angle = filter_by_mask(grasp_angle_gather2, objectness_mask);
    std::vector<float> filtered_grasp_tolerance = filter_by_mask(grasp_tolerance_gather2, objectness_mask);
    std::vector<float> filtered_grasp_depth = filter_by_mask(grasp_depth, objectness_mask);

    std::vector<std::vector<float>> filtered_approaching = filter_by_mask2(approaching_vec, objectness_mask);
    std::vector<std::vector<float>> filtered_grasp_center = filter_by_mask2(grasp_center_vec, objectness_mask);

    for (int i = 0; i < filtered_grasp_score.size(); i++) {
      filtered_grasp_score[i] = filtered_grasp_score[i] * filtered_grasp_tolerance[i] / GRASP_MAX_TOLERANCE;
    }
    
    int nums = filtered_grasp_score.size();
  
    auto rotation_matrix = batch_viewpoint_params_to_matrix(filtered_approaching, filtered_grasp_angle);

    for (int i = 0; i < nums; i++) {
      auto graspgroup = GraspGroup();
      graspgroup.score = filtered_grasp_score[i];
      graspgroup.width = filtered_grasp_width[i];
      graspgroup.height = 0.02;
      graspgroup.depth = filtered_grasp_depth[i];
      swap(graspgroup.rotation, rotation_matrix[i]);
      swap(graspgroup.translation, filtered_grasp_center[i]);
      graspgroup.object_id = -1;

      output->graspgroups.push_back(graspgroup);
    }
    
    return 0;
}

// 比较函数，用于按 score 降序排序
bool compareByScore(const GraspGroup& a, const GraspGroup& b) {
    return a.score > b.score;
}

int32_t GraspOutputParser::Parse(
    std::shared_ptr<GraspGroupResult> &output,
    std::vector<std::shared_ptr<DNNTensor>> &output_tensors) {

  for (size_t i = 0; i < output_tensors.size(); i++) {
    hbSysFlushMem(&(output_tensors[i]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  }

  const float *fp2_xyz =
    reinterpret_cast<float *>(output_tensors[fp2_xyz_index_]->sysMem[0].virAddr);
  const float *objectness_score =
    reinterpret_cast<float *>(output_tensors[objectness_score_index_]->sysMem[0].virAddr);
  const float *grasp_top_view_xyz =
    reinterpret_cast<float *>(output_tensors[grasp_top_view_xyz_index_]->sysMem[0].virAddr);
  const float *grasp_score_pred =
    reinterpret_cast<float *>(output_tensors[grasp_score_pred_index_]->sysMem[0].virAddr);
  const float *grasp_angle_cls_pred =
    reinterpret_cast<float *>(output_tensors[grasp_angle_cls_pred_index_]->sysMem[0].virAddr);
  const float *grasp_width_pred =
    reinterpret_cast<float *>(output_tensors[grasp_width_pred_index_]->sysMem[0].virAddr);
  const float *grasp_tolerance_pred =
    reinterpret_cast<float *>(output_tensors[grasp_tolerance_pred_index_]->sysMem[0].virAddr);

  auto parser = std::make_shared<GraspOutputParser>();
  parser->Decode(fp2_xyz, objectness_score, grasp_top_view_xyz, grasp_score_pred,
    grasp_angle_cls_pred, grasp_width_pred, grasp_tolerance_pred, output);

  // 降序排序
  std::sort(output->graspgroups.begin(), output->graspgroups.end(), compareByScore);

  return 0;
}
