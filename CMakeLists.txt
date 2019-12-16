cmake_minimum_required(VERSION 2.8.3)
project(monodepth)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    geometry_msgs
    image_transport
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
)

generate_messages(DEPENDENCIES geometry_msgs nav_msgs sensor_msgs std_msgs)
catkin_package(CATKIN_DEPENDS cv_bridge geometry_msgs image_transport message_runtime nav_msgs rospy sensor_msgs std_msgs)
include_directories(${catkin_INCLUDE_DIRS})
