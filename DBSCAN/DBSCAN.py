# _*_ coding: utf-8 _*_
"""
Time:     2022-05-20 21:49
Author:   Haolin Yan(XiDian University)
File:     DBSCAN.py
"""
import numpy as np
from loguru import logger
import copy
import random

random.seed(2022)


class DBSCAN:
    def set_hyparams(self, epsilon, MinPts):
        self.eps = epsilon
        self.MinPts = MinPts

    def get_neighbor(self, i, data):
        epsilon_region = set()
        x = data[i]
        for j, d in enumerate(data):
            if d == x:
                continue
            dis = np.sum((d - x) ** 2) ** 0.5
            if dis <= self.eps:
                epsilon_region.add(j)
        return epsilon_region

    @logger.catch
    def fit(self, X, epsilon=None, MinPts=None):
        if epsilon is not None:
            self.eps = epsilon
        if MinPts is not None:
            self.MinPts = MinPts
        # Algorithm line 1~7
        M = len(X)
        C = []
        D = {i for i in range(M)}
        Omega = set()
        for i in range(M):
            epsilon_region = self.get_neighbor(i, X)
            if len(epsilon_region) >= self.MinPts:
                Omega.add(i)

        k = 0
        Gamma = copy.deepcopy(D)
        # line 10~24
        while len(Omega) != 0:
            Gamma_old = copy.deepcopy(Gamma)
            o = random.choice(list(Omega))
            Q = [o]
            Gamma.remove(o)
            # line 14~21
            while len(Q) != 0:
                q = Q.pop(0)
                epsilon_region = self.get_neighbor(q, X)
                if len(epsilon_region) >= self.MinPts:
                    delta = epsilon_region & Gamma
                    for d in delta:
                        Q.append(d)
                        Gamma.remove(d)
            # line 22~24
            k = k + 1
            C_k = Gamma_old - Gamma
            Omega = Omega - C_k
            C.append(np.array([X[idx] for idx in C_k]))
        return C
