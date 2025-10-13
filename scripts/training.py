#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import colorsys

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from Segment import segment

marker_pub = None


def gen_hsv_colors(i, total):
    """
    根據索引生成一個鮮豔的 HSV 顏色，並轉換為 RGB。
    """
    hue = float(i) / total
    saturation = 1.0
    value = 1.0
    # colorsys.hsv_to_rgb 回傳的是 0-1 範圍的 float
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return r, g, b


def scan_callback(scan: LaserScan):
    global marker_pub

    ranges = np.array(scan.ranges)
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

    # 將極座標轉換為直角座標
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.vstack((x, y)).T # N x 2 array

    # 呼叫分割函數
    segments, _, num_segments = segment(points)

    marker_array = MarkerArray()

    clear_marker = Marker()
    clear_marker.header.frame_id = scan.header.frame_id
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.ns = 'ds_mid'
    clear_marker.id = 0
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    marker_pub.publish(marker_array)
    
    marker_array = MarkerArray()

    for i, seg in enumerate(segments):
        marker = Marker()
        marker.header.frame_id = scan.header.frame_id
        marker.header.stamp = rospy.Time.now()

        marker.ns = 'laser_segments'
        marker.id = i + 1  # ID 從 1 開始，0 保留給 DELETEALL

        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05 # 不重要

        r, g, b = gen_hsv_colors(i, num_segments)
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0 # 不透明

        marker.points = [
            Point(points[idx, 0], points[idx, 1], 0) 
            for idx in seg
        ]

        marker_array.markers.append(marker)

        if marker_array.markers:
            marker_pub.publish(marker_array)

        rospy.loginfo(f'Found {num_segments} segments.')


def main():
    global marker_pub
    rospy.init_node('ds_mid_node')

    marker_pub = rospy.Publisher('/laser_segments', MarkerArray, queue_size=10)

    rospy.Subscriber('/scan', LaserScan, scan_callback)

    rospy.spin()

if __name__ == '__main__':
    main()