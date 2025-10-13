cmake_minimum_required(VERSION 3.0.2)
project(ds_mid)

find_package(catkin REQUIRED
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

catkin_package(CATKIN_DEPENDS
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS 
  scripts/training.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)