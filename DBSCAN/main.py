# _*_ coding: utf-8 _*_
"""
Time:     2022-05-20 22:20
Author:   Haolin Yan(XiDian University)
File:     main.py
"""
import scipy.io as scio
import matplotlib.pyplot as plt
from DBSCAN import DBSCAN
dataFile = '../data/moon.mat'
data = scio.loadmat(dataFile)
X, Y = data["a"][:, :2], data["a"][:, -1]
dbscan = DBSCAN()
dbscan.set_hyparams(0.45, 3)
print(len(dbscan.fit(X)))
