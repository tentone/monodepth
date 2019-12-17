#!/usr/bin/env python2

import time
import numpy as np
import cv2
import rospkg
import rospy
import keras
import tensorflow as tf

from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError

from utils import scale_up, predict
from layers import BilinearUpSampling2D
from loss import depth_loss_function

class MonoDepth():
    def __init__(self):
        # Setup tensorflow session
        self.session = keras.backend.get_session()
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

        # Get parameters
        self.topic_color = rospy.get_param('~topic_color', '/camera/image_raw')
        self.topic_depth = rospy.get_param('~topic_depth', '/camera/image_depth')
        self.topic_pointcloud = rospy.get_param('~topic_depth', '/pointcloud')

        # Read keras model
        self.rospack = rospkg.RosPack()
        self.model_path = self.rospack.get_path("monodepth") + "/models/kitti.h5"

        # Custom object needed for inference and training
        self.start = time.time()
        self.custom_objects = {"BilinearUpSampling2D": BilinearUpSampling2D, "depth_loss_function": depth_loss_function}

        # Load model into GPU / CPU
        self.model = keras.models.load_model(self.model_path, custom_objects=self.custom_objects, compile=False)
        self.model._make_predict_function()

        # Image publisher
        self.image_pub = rospy.Publisher(self.topic_depth, Image, queue_size=1)

        # Image subscriber
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.topic_color, Image, self.image_callback)

    # Create a sensor_msgs.PointCloud2 from an array of points.
    def xyzrgb_array_to_pointcloud2(self, depth, color, stamp=None, frame_id=None, seq=None):
        msg = PointCloud2()
        buf = []

        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if seq:
            msg.header.seq = seq

        height, width, channels = depth.shape

        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            N = len(points)
            xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
            msg.height = 1
            msg.width = N

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)
        ]

        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * N
        msg.is_dense = True;
        msg.data = xyzrgb.tostring()

        return msg

    # Callback to receive and process image published.
    #
    # After processing it publishes back the estimated depth result
    def image_callback(self, data):
        # Convert message to opencv image
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Display image
        # cv2.imshow("Image", image)
        # cv2.waitKey(1)

        # Get image data as a numpy array to be passed for processing.
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (640, 480))
        arr = np.clip(np.asarray(img, dtype=float) / 255, 0, 1)

        # Predict depth image
        with self.session.as_default():
            with self.session.graph.as_default():
                result = predict(self.model, arr, minDepth=10, maxDepth=500, batch_size=1)

        # Resize and reshape output
        depth = result.reshape(result.shape[1], result.shape[2], 1)

        # Display depth
        # cv2.imshow("Result", depth)
        # cv2.waitKey(1)

        # Publish depth image
        depth = 255 * depth
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(depth.astype(np.uint8), "mono8"))

        #

def main():
    rospy.init_node("monodepth")

    depth = MonoDepth()

    rospy.spin()

if __name__ == "__main__":
    main()
