import numpy as np

GRASP_MAX_WIDTH = 0.067
GRASP_MAX_TOLERANCE = 0.045

def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):

    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)

    axis_y[mask_y] = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)

    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R1 = R1.reshape([-1,3,3])
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32)

def decode(end_points):
    grasp_preds = []

    objectness_score = end_points['objectness_score'].astype(np.float32)
    grasp_score = end_points['grasp_score_pred'].astype(np.float32)
    grasp_center = end_points['fp2_xyz'].astype(np.float32)
    approaching = -end_points['grasp_top_view_xyz'].astype(np.float32)
    grasp_angle_class_score = end_points['grasp_angle_cls_pred']
    grasp_width = 1.2 * end_points['grasp_width_pred'].astype(np.float32)
    grasp_width = np.clip(grasp_width, a_min=0, a_max=GRASP_MAX_WIDTH)
    grasp_tolerance = end_points['grasp_tolerance_pred']

    grasp_angle_class = np.argmax(grasp_angle_class_score, axis=0)
    grasp_angle = grasp_angle_class.astype(np.float32) / 12 * np.pi

    grasp_angle_class_ = np.expand_dims(grasp_angle_class, axis=0)
    grasp_score = np.take_along_axis(grasp_score, grasp_angle_class_, axis=0).squeeze(axis=0)
    grasp_width = np.take_along_axis(grasp_width, grasp_angle_class_, axis=0).squeeze(axis=0)
    grasp_tolerance = np.take_along_axis(grasp_tolerance, grasp_angle_class_, axis=0).squeeze(axis=0)

    grasp_depth_class = np.argmax(grasp_score, axis=1)
    grasp_depth_class = np.expand_dims(grasp_depth_class, axis=1)

    grasp_depth = (grasp_depth_class.astype(np.float32) + 1) * 0.01

    grasp_score = np.take_along_axis(grasp_score, grasp_depth_class, axis=1)
    grasp_angle = np.take_along_axis(grasp_angle, grasp_depth_class, axis=1)
    grasp_width = np.take_along_axis(grasp_width, grasp_depth_class, axis=1)
    grasp_tolerance = np.take_along_axis(grasp_tolerance, grasp_depth_class, axis=1)

    objectness_pred = np.argmax(objectness_score, axis=0)
    objectness_mask = (objectness_pred == 1)

    grasp_score = grasp_score[objectness_mask]
    grasp_width = grasp_width[objectness_mask]
    grasp_depth = grasp_depth[objectness_mask]
    approaching = approaching[objectness_mask]
    grasp_angle = grasp_angle[objectness_mask]
    grasp_center = grasp_center[objectness_mask]
    grasp_tolerance = grasp_tolerance[objectness_mask]

    grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

    Ns = grasp_angle.shape[0]
    approaching_ = approaching.reshape(Ns, 3)
    grasp_angle_ = grasp_angle.reshape(Ns)
    rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
    rotation_matrix = rotation_matrix.reshape(Ns, 9)
    grasp_height = 0.02 * np.ones_like(grasp_score)
    obj_ids = -1 * np.ones_like(grasp_score)
    grasp_preds.append(np.concatenate([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids], axis=-1))

    return grasp_preds
