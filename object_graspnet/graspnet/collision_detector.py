import os
import sys
import numpy as np
from collections import defaultdict

def voxel_down_sample(points, voxel_size):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than 0.")

    points = np.array(points)
    if points.size == 0:
        return np.array([])

    # 计算体素的最小和最大边界
    min_bound = points.min(axis=0) - voxel_size * 0.5
    max_bound = points.max(axis=0) + voxel_size * 0.5

    if voxel_size * np.iinfo(np.int32).max < (max_bound - min_bound).max():
        raise ValueError("voxel_size is too small.")

    # 计算每个点的体素索引
    ref_coords = (points - min_bound) / voxel_size
    voxel_indices = np.floor(ref_coords).astype(np.int32)

    # 创建一个字典来存储每个体素内的点
    voxel_dict = defaultdict(list)

    for i, voxel_index in enumerate(voxel_indices):
        voxel_index_tuple = tuple(voxel_index)
        voxel_dict[voxel_index_tuple].append(points[i])

    # 计算每个体素的平均点
    # downsampled_points = [np.mean(voxel_dict[voxel_index], axis=0) for voxel_index in voxel_dict]
    downsampled_points = [voxel_index for voxel_index in voxel_dict]

    return np.array(downsampled_points)

class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """

    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.023
        self.finger_length = 0.075
        self.voxel_size = voxel_size
        # TODO voxel_down_sample need to fix. error length of output
        self.scene_points = voxel_down_sample(scene_points, voxel_size)

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False,
               empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious

            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:, np.newaxis]
        depths = grasp_group.depths[:, np.newaxis]
        widths = grasp_group.widths[:, np.newaxis]
        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2))
        # left finger mask
        mask2 = ((targets[:, :, 0] > depths - self.finger_length) & (targets[:, :, 0] < depths))
        mask3 = (targets[:, :, 1] > -(widths / 2 + self.finger_width))
        mask4 = (targets[:, :, 1] < -widths / 2)
        # right finger mask
        mask5 = (targets[:, :, 1] < (widths / 2 + self.finger_width))
        mask6 = (targets[:, :, 1] > widths / 2)
        # bottom mask
        mask7 = ((targets[:, :, 0] <= depths - self.finger_length) \
                 & (targets[:, :, 0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:, :, 0] <= depths - self.finger_length - self.finger_width) \
                 & (targets[:, :, 0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size ** 3)).reshape(-1)
        bottom_volume = (
                    heights * (widths + 2 * self.finger_width) * self.finger_width / (self.voxel_size ** 3)).reshape(-1)
        shifting_volume = (heights * (widths + 2 * self.finger_width) * approach_dist / (self.voxel_size ** 3)).reshape(
            -1)
        volume = left_right_volume * 2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume + 1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask, ]
        if return_empty_grasp:
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size ** 3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1) / inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume + 1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume + 1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume + 1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume + 1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        return ret_value
