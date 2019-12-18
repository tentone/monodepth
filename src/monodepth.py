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
        self.debug = rospy.get_param("~debug", True)
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.topic_color = rospy.get_param("~topic_color", "/camera/image_raw")
        self.topic_depth = rospy.get_param("~topic_depth", "/camera/image_depth")
        self.topic_pointcloud = rospy.get_param("~topic_pointcloud", "/pointcloud")

        # Read keras model
        self.rospack = rospkg.RosPack()
        self.model_path = self.rospack.get_path("monodepth") + "/models/nyu.h5"

        # Custom object needed for inference and training
        self.start = time.time()
        self.custom_objects = {"BilinearUpSampling2D": BilinearUpSampling2D, "depth_loss_function": depth_loss_function}

        # Load model into GPU / CPU
        self.model = keras.models.load_model(self.model_path, custom_objects=self.custom_objects, compile=False)
        self.model._make_predict_function()

        # Publishers
        self.pub_image_depth = rospy.Publisher(self.topic_depth, Image, queue_size=1)
        self.pub_pointcloud = rospy.Publisher(self.topic_pointcloud, PointCloud2, queue_size=1)
        self.counter = 0

        # Subscribers
        self.bridge = CvBridge()
        self.sub_image_raw = rospy.Subscriber(self.topic_color, Image, self.image_callback)

    # Create a sensor_msgs.PointCloud2 from the depth and color images provided
    #
    # It ignores are camera parameters and assumes the images to be rectified
    def create_pointcloud(self, depth, color):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.header.seq = self.counter

        depth_height, depth_width, depth_channels = depth.shape
        color_height, color_width, color_channels = color.shape

        # Point cloud data numpy array
        i = 0
        data = np.zeros((depth_height * depth_width * 3), dtype=np.float32)

        # Message data size
        msg.height = 1
        msg.width = depth_width * depth_height

        # Iterate images and build point cloud
        for y in range(depth_height):
            for x in range(depth_width):
                #yc = y * depth_height / color_height
                #xc = x * depth_width / color_width
                data[i] =  x
                data[i + 1] = y
                data[i + 2] = depth[y, x]

                i += 3

        # Fields of the point cloud
        msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            #PointField("r", 12, PointField.FLOAT32, 1),
            #PointField("g", 16, PointField.FLOAT32, 1),
            #PointField("b", 20, PointField.FLOAT32, 1)
        ]

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * depth_height * depth_width
        msg.is_dense = True
        msg.data = data.tostring()

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
        if self.debug:
            cv2.imshow("Image", image)
            cv2.waitKey(1)

        # Get image data as a numpy array to be passed for processing.
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        arr = np.clip(np.asarray(img, dtype=np.float32) / 255, 0, 1)

        # Predict depth image
        with self.session.as_default():
            with self.session.graph.as_default():
                result = predict(self.model, arr, batch_size=1)

        # Resize and reshape output
        depth = result.reshape(result.shape[1], result.shape[2], 1)

        # Display depth
        if self.debug:
            cv2.imshow("Result", depth)
            cv2.waitKey(1)

        # Publish depth image
        depth = 255 * depth
        self.pub_image_depth.publish(self.bridge.cv2_to_imgmsg(depth.astype(np.uint8), "mono8"))

        # Generate Point cloud
        cloud_msg = self.create_pointcloud(depth, image)
        self.pub_pointcloud.publish(cloud_msg)

        # Increment counter
        self.counter += 1

def main():
    rospy.init_node("monodepth")

    depth = MonoDepth()

    rospy.spin()

if __name__ == "__main__":
    main()
