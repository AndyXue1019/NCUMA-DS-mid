#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA


def fit_circle(points: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """
    使用最小二乘法擬合一個圓。
    返回圓心 (xc, yc) 和半徑 r。
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([2 * x, 2 * y, np.ones(len(x))]).T
    b = x**2 + y**2
    # 解 Ac = b
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return (c[0], c[1]), np.sqrt(c[2] + c[0]**2 + c[1]**2)


def extract_features(points) -> list:

    # 特徵 1: 點的數量
    # n = Si_n

    # 特徵 2: 片段寬度 (第一個點到最後一個點的距離)
    width = np.linalg.norm(points[0] - points[-1])

    # 特徵 3: 線性度 (使用 PCA)
    # PCA 會找到數據變異最大的方向。
    # explained_variance_ratio_[1] 代表垂直於主方向的變異程度。
    # 對於直線，這個值應該非常小。
    pca = PCA(n_components=2)
    pca.fit(points)
    linearity_err = pca.explained_variance_ratio_[1]

    # 特徵 4: 圓度 (擬合圓後的誤差)
    # 計算所有點到擬合圓心的距離，然後取其標準差。
    # 對於圓弧，這個標準差應該很小。
    center, radius = fit_circle(points)
    distances_to_center = np.linalg.norm(points - center, axis=1)
    circularity_err = np.std(distances_to_center)

    # 特徵 5: 點到質心的距離標準差
    centroid = np.mean(points, axis=0)
    distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
    std_dev_dist = np.std(distances_to_centroid)

    # 特徵 6: 曲率估計
    # 擬合二次多項式 y = ax^2 + bx + c 來估計曲率
    # 為了旋轉不變性，我們先將點對齊到主軸
    points_transformed = pca.transform(points)
    x_transformed = points_transformed[:, 0]
    y_transformed = points_transformed[:, 1]
    # 擬合二次多項式，曲率約等於 |2a|
    poly_coeffs = np.polyfit(x_transformed, y_transformed, 2)
    curvature = np.abs(2 * poly_coeffs[0])


    return [width, linearity_err, circularity_err, std_dev_dist, curvature]
