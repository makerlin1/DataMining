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

# dataFile = '../data/sizes5.mat'
# data = scio.loadmat(dataFile)
# print(data.keys())
# X, Y = data["sizes5"][:, :2], data["sizes5"][:, -1]
# eps, k = 2.0, 19

# dataFile = '../data/moon.mat'
# data = scio.loadmat(dataFile)
# print(data.keys())
# X, Y = data["a"][:, :2], data["a"][:, -1]
# eps, k = 0.11, 3

# dataFile = '../data/square1.mat'
# data = scio.loadmat(dataFile)
# print(data.keys())
# X = data["square1"][:, :2]
# eps, k = 1, 6

# dataFile = '../data/square4.mat'
# data = scio.loadmat(dataFile)
# print(data.keys())
# X, Y = data["b"][:, :2], data["b"][:, -1]
# eps, k = 1.5, 40

# dataFile = '../data/smile.mat'
# data = scio.loadmat(dataFile)
# print(data.keys())
# X, Y = data["smile"][:, :2], data["smile"][:, -1]
# eps, k = 0.035, 3

dataFile = 'data/spiral.mat'
data = scio.loadmat(dataFile)
print(data.keys())
X, Y = data["spiral"][:, :2], data["spiral"][:, -1]
eps, k = 0.75, 3

k_distence = get_k_distance(X, k=5)
dbscan = DBSCAN()
dbscan.set_hyparams(eps, k)
C = dbscan.fit(X, visual=True)
for cluster in C:
    plt.scatter(cluster[:, 0], cluster[:, 1])
plt.title("eps: {:.3f} k: {}".format(eps, k))
plt.grid()
plt.show()
