# _*_ coding: utf-8 _*_
"""
Time:     2022-05-20 22:20
Author:   Haolin Yan(XiDian University)
File:     main.py
"""
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from DBSCAN import DBSCAN, get_k_distance
dataFile = '../data/moon.mat'
data = scio.loadmat(dataFile)
X, Y = data["a"][:, :2], data["a"][:, -1]

# 绘制K距离图，确定epsilon的范围
k_distence = get_k_distance(X, k=5)
search_range = input().split(',')
search_range = np.linspace(float(search_range[0]), float(search_range[1]), num=5)
dbscan = DBSCAN()
for i, eps in enumerate(search_range):
    dbscan.set_hyparams(eps, 3)
    C = dbscan.fit(X, visual=True)
    for cluster in C:
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.show()
