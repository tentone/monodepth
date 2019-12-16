import os
import glob
import argparse
import time
from PIL import Image
import numpy as np
import PIL
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, display_images, to_multichannel, scale_up
from matplotlib import pyplot as plt

import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class MonoDepth():
    def __init__(self):
        self.image_pub = rospy.Publisher("image_depth", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image_raw", Image, self.callback)

        # Argument Parser
        #parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
        #parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
        #parser.add_argument('--input', default='test/*.mov', type=str, help='Path to Video')
        #args = parser.parse_args()

        # Custom object needed for inference and training
        #start = time.time()
        #custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

        # Load model into GPU / CPU
        #model = load_model(args.model, custom_objects=custom_objects, compile=False)

        # Video output
        #video_name = args.input
        # cap = cv2.VideoCapture(video_name)

        # Video output
        # out_video_name = 'output.avi'
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (1280, 480))

        #count = 0
        #ret = True

        #while ret:
         #   ret, image = cap.read()
          #  if ret is False:
           #     break
            #img_arr = get_img_arr(image)
            #count += 1
            #output = scale_up(2, predict(model, img_arr, batch_size=1))
            #pred = output.reshape(output.shape[1], output.shape[2], 1)
            #img_set = display_single_image(pred, img_arr)
            #plt.figure(figsize=(20, 10))
            #plt.imshow(img_set)
            #filename = 'img_' + str(count).zfill(4) + '.png'
            #plt.savefig(os.path.join('image_results', filename), bbox_inches='tight')

    def get_img_arr(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (640, 480))
        x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
        return x

    def display_single_image(output, inputs=None, is_colormap=True):
        import matplotlib.pyplot as plt
        plasma = plt.get_cmap('plasma')
        imgs = []
        imgs.append(inputs)

        ##rescale output
        out_min = np.min(output)
        out_max = np.max(output)
        output = output - out_min
        outputs = output/out_max

        if is_colormap:
            rescaled = outputs[:, :, 0]
            pred_x = plasma(rescaled)[:, :, :3]
            imgs.append(pred_x)

        img_set = np.hstack(imgs)

        return img_set

    # Callback to receive and process image published.
    #
    # After processing it publishes back the estimated depth result
    def image_callback(self,data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image window", image)
        cv2.waitKey(1)

        #try:
        #    self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        #except CvBridgeError as e:
        #    print(e)

def main():
    rospy.init_node('monodepth')

    depth = MonoDepth()

    rospy.spin()

if __name__ == '__main__':
    main()
