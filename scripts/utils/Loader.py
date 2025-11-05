import csv
import os
from typing import Dict, List

import rosbag
from sensor_msgs.msg import LaserScan


def load_bag_data(bag_file) -> List[LaserScan]:
    """
    從 ROS bag 檔案中讀取 LaserScan 資料。
    """
    if not os.path.exists(bag_file):
        raise FileNotFoundError(f'找不到 Bag 檔案 {bag_file}')

    bag = rosbag.Bag(bag_file)
    scan_msgs = []
    for _, msg, _ in bag.read_messages(topics=['/scan']):
        scan_msgs.append(msg)
    bag.close()
    print(f'{bag_file}: 讀取到 {len(scan_msgs)} 筆 LaserScan 訊息。')
    return scan_msgs


def load_label(label_file) -> List[Dict[str, int]]:
    """
    從標籤檔案中讀取標籤資料。
    """
    labels = []
    if not os.path.exists(label_file):
        print(f'找不到 label 檔案 {label_file}')
        return labels

    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_dict = {}
                if row['ball']:
                    label_dict['ball'] = int(row['ball'])
                if row['box']:
                    label_dict['box'] = int(row['box'])
                labels.append(label_dict)
        print(f'{label_file}: 讀取到 {len(labels)} 筆標籤資料。')
    except IOError as e:
        print(f'讀取標籤檔案 {label_file} 時發生錯誤: {e}')

    return labels
