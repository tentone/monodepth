## Mono Depth ROS
 - ROS node used to estimated depth from monocular RGB data.
 - Should be used with Python 2.X and ROS
 - The original code is at the repo [Dense Depth Original Code](https://github.com/ialhashim/DenseDepth)
 - [High Quality Monocular Depth Estimation via Transfer Learning](https://arxiv.org/abs/1812.11941) by Ibraheem Alhashim and Peter Wonka



### Install

- Install Python 2 and ROS dependencies

```
apt-get install python python-pip
pip install rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg
```

- Install project dependencies

`pip install tensorflow keras pillow matplotlib scikit-learn scikit-image opencv-python pydot GraphViz` 




### Data
 - [NYU Depth V2 (50K)](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_data.zip) (4.1 GB): You don't need to extract the dataset since the code loads the entire zip file into memory when training.



### Pretrained models used
 - Pre-trained models can be downloaded and placed in the models folder.
 - [NYU Depth V2](https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5) (165 MB)

