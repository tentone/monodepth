#!/usr/bin/env python2

import os
import time
import numpy as np
import cv2
import rospkg
import rospy
import keras

#import tensorflow
#from tensorflow import keras
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils import scale_up, predict
from layers import BilinearUpSampling2D
from loss import depth_loss_function

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
        self.model = keras.models.load_model(self.model_path, custom_objects=self.custom_objects, compile=False)
        self.model._make_predict_function()

        # Image publisher
        self.image_pub = rospy.Publisher(self.topic_depth, Image, queue=1)

        # Image subscriber
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.topic_color, Image, self.image_callback)

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
        cv2.imshow("Image window", image)
        cv2.waitKey(1)

        # Get image data as a numpy array to be passed for processing.
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (640, 480))
        arr = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)

        # Predict depth image
        output = scale_up(2, predict(self.model, arr, batch_size=1))
        pred = output.reshape(output.shape[1], output.shape[2], 1)

        # Publish depth image
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(pred, "bgr8"))

def main():
    rospy.init_node("monodepth")

    depth = MonoDepth()

    rospy.spin()

if __name__ == "__main__":
    main()
