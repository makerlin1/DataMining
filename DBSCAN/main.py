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


dataFile = '../data/sizes5.mat'
data = scio.loadmat(dataFile)
print(data.keys())
X, Y = data["sizes5"][:, :2], data["sizes5"][:, -1]
k_distence = get_k_distance(X, k=5)
eps, k = 1., 4
dbscan = DBSCAN()
dbscan.set_hyparams(eps, k)
C = dbscan.fit(X, visual=False)
for cluster in C:
    plt.scatter(cluster[:, 0], cluster[:, 1])
plt.title("eps: %f k: %d" % (eps, k))
plt.show()



# dataFile = '../data/smile.mat'
# data = scio.loadmat(dataFile)
# print(data.keys())
# X = data["smile"][:, :2]


