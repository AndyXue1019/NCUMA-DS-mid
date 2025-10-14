#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
from typing import List

import numpy as np
import rosbag
from sensor_msgs.msg import LaserScan

from utils.Adaboost import adaboost_predict, adaboost_train
from utils.Segment import extract_features, segment

data_train = []
label_train = []
data_test = []
label_test = []


def load_labels(data_file):
    ball = []
    box = []
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳過標題列
        for x, y in reader:
            ball.append(x)
            box.append(y)
    return ball, box


def load_bag_data(bag_file) -> List[LaserScan]:
    """
    從 ROS bag 檔案中讀取 LaserScan 資料。
    """
    bag = rosbag.Bag(bag_file)
    scan_msgs = []
    for _, msg, _ in bag.read_messages(topics=['/scan']):
        scan_msgs.append(msg)
    bag.close()
    print(f'讀取到 {len(scan_msgs)} 筆 LaserScan 訊息。')
    return scan_msgs


def label_segments(t, num_segments, data_file):
    ball, box = load_labels(data_file)

    PN = ['O'] * num_segments  # 預設標籤為 'O' (Other)
    PN[int(box[t])] = 'B'
    PN[int(ball[t])] = 'C'

    return PN


def confusionmat(actual, predicted):
    return [
        [
            sum((a == 'O' and p == 'O') for a, p in zip(actual, predicted)),
            sum((a == 'O' and p == 'B') for a, p in zip(actual, predicted)),
            sum((a == 'O' and p == 'C') for a, p in zip(actual, predicted)),
        ],
        [
            sum((a == 'B' and p == 'O') for a, p in zip(actual, predicted)),
            sum((a == 'B' and p == 'B') for a, p in zip(actual, predicted)),
            sum((a == 'B' and p == 'C') for a, p in zip(actual, predicted)),
        ],
        [
            sum((a == 'C' and p == 'O') for a, p in zip(actual, predicted)),
            sum((a == 'C' and p == 'B') for a, p in zip(actual, predicted)),
            sum((a == 'C' and p == 'C') for a, p in zip(actual, predicted)),
        ],
    ]


def accuracy(cm):
    cm = np.array(cm)
    accuracy = np.trace(cm) / np.sum(cm)
    print(f'準確率: {accuracy * 100:.2f}%')
    return accuracy


def data_label_prepare(bag_file, label_file, mode='train'):
    global data_train, label_train, data_test, label_test

    mode = mode.lower()
    if mode not in ['train', 'test']:
        raise ValueError("mode 必須是 'train' 或 'test'")

    scan_msgs = load_bag_data(bag_file)

    for t, scan in enumerate(scan_msgs):
        ranges = np.array(scan.ranges)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T  # N x 2 array

        Seg, Si_n, S_n = segment(points)
        PN = label_segments(t, S_n, label_file)

        # # 提取每個片段的特徵
        for i in range(S_n):
            if Si_n[i] < 3:
                continue
            segment_points = np.array([points[idx] for idx in Seg[i]])

            # 點的數量
            # n = Si_n[i]

            features = extract_features(segment_points)

            if mode == 'train':
                data_train.append(features)
                label_train.append(PN[i])
            else:
                data_test.append(features)
                label_test.append(PN[i])


def main():
    global data_train, label_train, data_test, label_test
    bag_file = './data/data_{}.bag'
    label_file = './data/data_{}_label.csv'

    # 準備訓練資料 (可自訂)
    print('準備訓練資料...')
    data_label_prepare(bag_file.format('1'), label_file.format('1'), mode='train')
    print('準備測試資料...')
    data_label_prepare(bag_file.format('2'), label_file.format('2'), mode='test')
    data_label_prepare(bag_file.format('3'), label_file.format('3'), mode='test')

    # 將訓練和測試數據轉換為 NumPy 陣列
    data_train = np.array(data_train)
    label_train = np.array(label_train)
    data_test = np.array(data_test)
    label_test = np.array(label_test)
    if data_train.size == 0:
        raise ValueError('訓練資料集為空，請檢查資料準備過程。')
    elif data_test.size == 0:
        raise ValueError('測試資料集為空，請檢查資料準備過程。')
    print(f'訓練資料集大小: {data_train.shape}, 測試資料集大小: {data_test.shape}')

    # 儲存資料集為 .npz 檔案
    # np.savez('train_data.npz', data=data_train, label=label_train)
    # np.savez('test_data.npz', data=data_test, label=label_test)

    stumps, alphas = adaboost_train(data_train, label_train, T=50)

    # 儲存訓練好的模型
    np.savez('./model/adaboost_model.npz', stumps=stumps, alphas=alphas)
    print('模型已儲存至 ./model/adaboost_model.npz')

    train_pred = adaboost_predict(data_train, stumps, alphas)
    test_pred = adaboost_predict(data_test, stumps, alphas)

    print('--- 訓練資料集 ---')
    cm_train = confusionmat(label_train, train_pred)
    for row in cm_train:
        print(row)
    accuracy(cm_train)

    print('--- 測試資料集 ---')
    cm_test = confusionmat(label_test, test_pred)
    for row in cm_test:
        print(row)
    accuracy(cm_test)


if __name__ == '__main__':
    main()
