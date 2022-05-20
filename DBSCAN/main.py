# _*_ coding: utf-8 _*_
"""
Time:     2022-05-20 22:20
Author:   Haolin Yan(XiDian University)
File:     main.py
"""
import scipy.io as scio
import matplotlib.pyplot as plt
from DBSCAN import DBSCAN, get_k_distance
dataFile = '../data/moon.mat'
data = scio.loadmat(dataFile)
X, Y = data["a"][:, :2], data["a"][:, -1]
k_distence = get_k_distance(X, k=5)
dbscan = DBSCAN()
dbscan.set_hyparams(0.11, 3)
C = dbscan.fit(X, visual=True)
print(len(C))
a = 0
for cluster in C:
    a += len(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1])
plt.show()
print(a)
