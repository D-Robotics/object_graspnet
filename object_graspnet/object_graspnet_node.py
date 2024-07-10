import numpy as np
import onnxruntime
import os
import json
from PIL import Image
import rclpy
import time
import gc

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from object_graspnet.graspnet.data_utils import CameraInfo, create_point_cloud_from_depth_image
from object_graspnet.graspnet.pred_decode import decode
from object_graspnet.graspnet.GraspGroup import GraspGroup
from object_graspnet.graspnet.collision_detector import ModelFreeCollisionDetector

import ai_msgs.msg
import sensor_msgs.msg

def get_float_array(path, dtype=np.float32):
    # 读取二进制文件
    with open(path, "rb") as f:
        data = f.read()

    float_array = np.frombuffer(data, dtype=dtype)
    return float_array

class OrtWrapper:
    def __init__(self, onnxfile: str):
        assert os.path.exists(onnxfile)
        self.onnxfile = onnxfile
        self.sess = onnxruntime.InferenceSession(onnxfile)
        self.inputs = self.sess.get_inputs()        
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        output_tensors = self.sess.run(None, _inputs)
        assert len(output_tensors) == len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor
        return output

class ObjectGraspnetNode(Node):
    def __init__(self):
        super().__init__('object_graspnet_node')
        self.get_logger().warn('Object Graspnet Node has been started.')

        # Declare parameters
        self.declare_parameter('ai_msg_pub_topic_name', '/hobot_object_graspgroup')
        self.declare_parameter('feed_type', False)
        self.declare_parameter('image', 'config/depth.png')
        self.declare_parameter('is_collision_detect', False)
        self.declare_parameter('model_file_name', "config/grasp_horizon_2500.onnx")
        self.declare_parameter('num_points', 2500)
        self.declare_parameter('ros_img_sub_topic_name', '/camera/depth/image_raw')

        # Get parameter values
        self.ai_msg_pub_topic_name = self.get_parameter('ai_msg_pub_topic_name').get_parameter_value().string_value
        self.feed_type = self.get_parameter('feed_type').get_parameter_value().bool_value
        self.image = self.get_parameter('image').get_parameter_value().string_value
        self.is_collision_detect = self.get_parameter('is_collision_detect').get_parameter_value().bool_value
        self.model_file_name = self.get_parameter('model_file_name').get_parameter_value().string_value
        self.num_points = self.get_parameter('num_points').get_parameter_value().integer_value
        self.ros_img_sub_topic_name = self.get_parameter('ros_img_sub_topic_name').get_parameter_value().string_value

        self.voxel_size = 0.01
        self.collision_thresh = 0.01

        with open('config/cam_info.json', 'r') as f:
            camera_params = json.load(f)
        self.camera = CameraInfo(camera_params['width'], 
                                camera_params['height'],
                                camera_params['fx'],
                                camera_params['fy'],
                                camera_params['cx'],
                                camera_params['cy'],
                                camera_params['scale'])

        self.publisher = self.create_publisher(ai_msgs.msg.PerceptionTargets, self.ai_msg_pub_topic_name, 10)
        self.model = OrtWrapper(self.model_file_name)
        if not self.feed_type:
            self.feedback()
        else:
            self.status = True
            self.subscription = self.create_subscription(
                                    sensor_msgs.msg.Image,
                                    self.ros_img_sub_topic_name,
                                    self.listener_callback,
                                    10)

    def preprocess(self, depth):
        
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, self.camera, organized=True)

        mask = (depth > 0)
        cloud_masked = cloud[mask]

        # sample points
        # np.random.seed(0)
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        return cloud_sampled, cloud_masked

    def infer(self, input_data):
        # 1. preprare input data
        inputs = {
            "pointclouds": input_data
        }

        # 2. model inference 
        outputs = self.model.forward(inputs)

        return outputs

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)

        gg = gg[~collision_mask]
        return gg

    def parser(self,
                objectness_score,
                grasp_score_pred,
                fp2_xyz,
                grasp_top_view_xyz,
                grasp_angle_cls_pred,
                grasp_width_pred,
                grasp_tolerance_pred):

        end_points = dict()
        end_points['objectness_score'] = objectness_score
        end_points['grasp_score_pred'] = grasp_score_pred
        end_points['fp2_xyz'] = fp2_xyz
        end_points['grasp_top_view_xyz'] = grasp_top_view_xyz
        end_points['grasp_angle_cls_pred'] = grasp_angle_cls_pred
        end_points['grasp_width_pred'] = grasp_width_pred
        end_points['grasp_tolerance_pred'] = grasp_tolerance_pred

        grasp_preds = decode(end_points)
        gg = GraspGroup(grasp_preds[0])
        
        # sort
        gg.sort_by_score()

        return gg

    def postprocess(self, gg, cloud):

        gg = gg[:50]
        gg = self.collision_detection(gg, cloud)
        return gg

    def feedback(self):
        depth = np.array(Image.open(self.image))
        for i in range(1):
            res = self.run(depth)
            print("feedback result: \n", res[0])
        self.get_logger().info('feedback finshed!')

    def listener_callback(self, msg):
        self.get_logger().info('Receiving image')

        if not self.status:
            return

        if msg.encoding == '16UC1':
            self.status = False
            height = msg.height
            width = msg.width
            step = msg.step

            np_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((height, width))
            # 将OpenCV格式的图像转换为NumPy数组
            self.run(np_depth, msg.header)
            self.status = True

    def run(self, depth, header=None):

        try:
            # 1. preprocess
            start_time = time.time()
            cloud_sampled, cloud = self.preprocess(depth)
            cloud_sampled = cloud_sampled.reshape(1, self.num_points, 3).astype(np.float32)

            # 2. model infer
            record_time_1 = time.time()
            outputs = self.infer(cloud_sampled)
            
            # 3. model parser
            record_time_2 = time.time()
            gg = self.parser(
                outputs["objectness_score"][0],
                outputs["grasp_score_pred"][0],
                outputs["fp2_xyz"][0],
                outputs["grasp_top_view_xyz"][0],
                outputs["grasp_angle_cls_pred"][0],
                outputs["grasp_width_pred"][0],
                outputs["grasp_tolerance_pred"][0])
            num = len(gg)   

            # 4. model postprcess collision detection
            record_time_3 = time.time()
            if self.is_collision_detect:
                gg = self.postprocess(gg, cloud)
            self.get_logger().info('parser num : {0}, postprocess num: {1}.'
                .format(num, len(gg)))

            # 5. pubulish ai msg
            end_time = time.time()
            perceptiontargets = ai_msgs.msg.PerceptionTargets()
            if header is not None:
                perceptiontargets.header = header
            # perf = ai_msgs.msg.Perf()
            for i in range(len(gg)):
                graspgroup = ai_msgs.msg.GraspGroup()
                target = ai_msgs.msg.Target()

                graspgroup.score = float(gg.scores[i])
                graspgroup.width = float(gg.widths[i])
                graspgroup.height = float(gg.heights[i])
                graspgroup.depth = float(gg.depths[i])
                graspgroup.translation = [float(value) for value in gg.translations[i]]
                graspgroup.rotation = [float(value) for value in gg.rotation_matrices[i].reshape(9)]
                graspgroup.object_id = int(gg.object_ids[i])

                target.graspgroup.append(graspgroup)
                perceptiontargets.targets.append(target)

            self.publisher.publish(perceptiontargets)

            self.get_logger().warn('preprocess time: {0:.3f} s, inference time: {1:.3f} s, parser time: {2:.3f} s, collision detect time: {3:.3f} s.'
                .format(record_time_1 - start_time, record_time_2 - record_time_1, record_time_3 - record_time_2, end_time - record_time_3))
            return gg
        finally:
            gc.collect()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectGraspnetNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()