#!/bin/bash

#sudo pip3 install keras pillow matplotlib scikit-learn scikit-image opencv-python pydot tensorflow

python3 test_video.py --model ./models/kitti.h5 --input ./test/dji.mov

