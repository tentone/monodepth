#!/bin/bash

echo "Install python and pip"
apt-get install python python-pip curl
pip install rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg

echo "Install dependencies"
pip install tensorflow keras pillow matplotlib scikit-learn scikit-image opencv-python pydot GraphViz tk

echo "Clone monodepth"
git clone https://github.com/tentone/monodepth.git

echo "Download pretrained model"
cd monodepth/models
curl –o nyu.h5 https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5