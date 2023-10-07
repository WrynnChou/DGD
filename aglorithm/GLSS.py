import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity,NearestNeighbors
from Utils.utils import *







class Glss(object):

    def __init__(self, data: np.ndarray, v: np.ndarray = 0.6):

        self.data_ = data

        self.d = data.shape[1]

        self.N = data.shape[0]

        self.M = np.floor(np.exp(2 * ( self.d * v))).astype('int')# Suggested bt the parper

        self.design = None

        self.r_s = None

        self.n_ = None

        self.design_choosed = None

        self.subsample = None

    def kde(self, kernel: str = 'gaussian', bandwidth: str = 'silverman'):
        """Kernel Density Estimation of full data"""

        model = KernelDensity(bandwidth=bandwidth, algorithm="kd_tree", kernel=kernel)
        model.fit(self.data_)
        self.model_ = model

        return model

    def design_transform(self):

        design_ = glp(self.M, self.d)
        design_new = np.zeros_like(design_)
        for i in range(self.d):
            max_ = np.max(self.data_[:, i])
            min_ = np.min(self.data_[:, i])
            design_new[:, i] = (max_ - min_) * (design_[:, i]) + min_

        self.design = design_new
        return design_new

    def sampling_design(self, n):

        nbr = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean')
        nbr.fit(self.design)
        dist, indx = nbr.kneighbors(self.design)
        near_1 = dist[:, 1]
        r = np.min(near_1)
        self.r_s = r * 0.5
        self.n_ = n

        model = self.model_
        pred = np.exp(model.score_samples(self.design))
        a = 0
        cdf_ = np.zeros((self.M + 1))
        for i in range((self.M)):
            a += pred[i]
            cdf_[(i + 1)] = a

        design_choosed_ = []
        for j in range(self.n_):
            l = (2 * j) / (2 * self.n_)
            idx = self.find_point(l, cdf_)
            d_j = self.design[idx, :]
            design_choosed_.append(d_j)

        design_choosed_ = np.array(design_choosed_)
        self.design_choosed = design_choosed_

        return design_choosed_

    def sampling(self):

        design_uni, count = uni_count(self.design_choosed)
        n_u = design_uni.shape[0]
        subsample = []
        nbr = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
        nbr.fit(self.data_)
        for i in range(n_u):
            dist, idx = nbr.kneighbors([design_uni[i]], n_neighbors=count[i])
            mask_ = self.r_s >= dist
            C_q = np.sum(mask_)
            c_q = count[i]
            if C_q == 0:
                quo = 0
                rem = c_q
            else:
                quo = c_q // C_q
                rem = c_q % C_q
            if rem != 0:
                subsample.append(self.data_[idx[0][:rem], :])
            if quo != 0:
                h = 0
                while h < quo:
                    h += 1
                    subsample.append(self.data_[idx[0][:C_q], :])
        subsample_ = []
        for i in subsample:
            for j in i:
                subsample_.append(j)

        self.subsample = subsample_
        return np.array(subsample_)

    def find_point(self, l: np.ndarray, cdf: np.ndarray):

        sum_ = cdf[-1] * l
        mask = np.where(sum_ >= cdf)
        idx = mask[0][-1]

        return idx



































