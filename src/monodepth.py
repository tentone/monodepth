#!/usr/bin/env python2

import os
import time
import numpy as np
import cv2
from utils import scale_up, predict

from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
import rospkg
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class MonoDepth():
    def __init__(self):

        # Get parameters
        self.topic_color = rospy.get_param('~topic_color', '/camera/image_raw')
        self.topic_depth = rospy.get_param('~topic_depth', '/camera/image_depth')

        # Read keras model
        self.rospack = rospkg.RosPack()
        self.model_path = self.rospack.get_path("monodepth") + "/models/nyu.h5"

        # Custom object needed for inference and training
        self.start = time.time()
        self.custom_objects = {"BilinearUpSampling2D": BilinearUpSampling2D, "depth_loss_function": depth_loss_function}

        # Load model into GPU / CPU
        self.model = load_model(self.model_path, custom_objects=self.custom_objects, compile=False)
        self.model._make_predict_function()

        # Counter
        self.count = 0
        self.ret = True

        # Image publisher
        self.image_pub = rospy.Publisher(self.topic_depth, Image)

        # Image subscriber
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.topic_color, Image, self.image_callback)

    # Get image data as a numpy array to be passed for processing.
    def get_img_arr(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (640, 480))
        x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
        return x

    # Callback to receive and process image published.
    #
    # After processing it publishes back the estimated depth result
    def image_callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image window", image)
        cv2.waitKey(1)

        #self.count += 1
        img_arr = self.get_img_arr(image)
        output = scale_up(2, predict(self.model, img_arr, batch_size=1))
        pred = output.reshape(output.shape[1], output.shape[2], 1)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(pred, "bgr8"))




def main():
    rospy.init_node("monodepth")

    depth = MonoDepth()

    rospy.spin()

if __name__ == "__main__":
    main()
