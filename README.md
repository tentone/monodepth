## Mono Depth ROS
 - ROS node used to estimated depth from monocular RGB data.
 - Should be used with Python 2.X and ROS
 - The original code is at the repository [Dense Depth Original Code](https://github.com/ialhashim/DenseDepth)
 - [High Quality Monocular Depth Estimation via Transfer Learning](https://arxiv.org/abs/1812.11941) by Ibraheem Alhashim and Peter Wonka

![a](readme\a.jpg)



### Configuration

- Topics subscribed by the ROS node
  - /image/camera_raw - Input image from camera (can be changed on the parameter topic_color)
- Topics published by the ROS node, containing depth and point cloud data generated.
  - /image/depth - Image message containing the depth image estimated (can be changed on the parameter topic_depth).
  - /pointcloud - Pointcloud2 message containing a estimated point cloud (can be changed on the parameter topic_pointcloud).
- Parameters that can be configurated
  - frame_id - TF Frame id to be published in the output messages.
  - debug - If set true a window with the output result if displayed.
  - min_depth, max_depth - Min and max depth values considered for scaling.
  - batch_size - Batch size used when predicting the depth image using the model provided.
  - model_file - Keras model file used, relative to the monodepth package.



### Setup

- Install Python 2 and ROS dependencies

```bash
apt-get install python python-pip
pip install rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg
```

- Install project dependencies

```bash
pip install tensorflow keras pillow matplotlib scikit-learn scikit-image opencv-python pydot GraphViz tk
```



### Pretrained models

 - Pre-trained keras models can be downloaded and placed in the /models folder from the following links:
    - [NYU Depth V2 (165MB)](https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5) 
    - [KITTI (165MB)](https://s3-eu-west-1.amazonaws.com/densedepth/kitti.h5)




### Datasets for training
 - [NYU Depth V2 (50K)](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) 
    - The NYU-Depth V2 data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.
    - [Download dataset](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_data.zip) (4.1 GB)
 - [KITTI Dataset (80K)](http://www.cvlibs.net/datasets/kitti/) 
    - Datasets captured by driving around the mid-size city of [Karlsruhe](http://maps.google.com/?ie=UTF8&z=15&ll=49.010627,8.405871&spn=0.018381,0.029826&t=k&om=1), in rural areas and on highways. Up to 15 cars and 30 pedestrians are visible per image.



