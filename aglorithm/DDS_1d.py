import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors

class Dds(object):

    """This class is used only for 1 dimension data"""
    def __init__(self, data: np.ndarray):

        assert len(data.shape) == 1, "This function only support 1 dimension data."

        self.data = data

        self.n = data.shape[0]

        self.iecdf = None

        self.ud = None

        self.sampling = None


    def sample(self, num: int):

        assert num <= self.n, "Subsampling number can't larger than the number of data."

        ud_ = np.arange(0, num)
        ud = (ud_ * 2 + 1) / (2 * num)
        iecdf_ = self.IECDF_1D(self.data)
        ud_inv = iecdf_(ud)

        # Find the NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(self.data.reshape((self.n, 1)))
        idx = neigh.kneighbors(ud_inv.reshape(num, 1))
        sampling = self.data[idx[1]]
        self.sampling = sampling
        self.iecdf = iecdf_
        self.ud = ud

        return sampling

    def plot(self, points: int = 100):

        data = self.data
        ecdf = self.ecdf_1dimension(data)
        x_plt = np.linspace(np.min(data), np.max(data), points)
        y_plt = ecdf(x_plt)
        point = self.sampling
        y_point = ecdf(point)
        plt.figure()
        plt.plot(x_plt, y_plt)
        plt.scatter(point, y_point, c="r", marker="*")
        plt.title("ECDF of data and subsampling points")
        # if plot function does not show figure, please change "plt.show(block=True)" into "plt.show(block=False)"
        plt.show(block=True)

    def IECDF_1D(self, sample):
        '''
        Given the inverse empirical calculated density function of 1 dimension sample
        :param sample: samples
        :return: IECDF function (or PPF)
        '''

        def iecdf(x):
            x = torch.as_tensor(x)
            index = torch.zeros_like(x)
            fix = x == 1
            n = len(sample)
            sort = sorted(sample)
            index[~fix] = torch.floor(torch.tensor(n) * x[~fix])
            index[fix] = -1 * torch.ones_like(x)[fix]
            result = np.array(sort)[index.type(torch.int)]
            return result

        return iecdf

    def ecdf_1dimension(self, x: np.ndarray):
        n = x.shape[0]
        assert len(x.shape) == 1, "This function only support one dimension data."

        def ecdf(u: np.ndarray):
            u = np.array(u)
            if len(u.shape) == 0:
                u = u.reshape(1)
            res = np.zeros_like(u)
            for i in range(u.shape[0]):
                s = np.sum(x <= u[i])
                res[i] = s / n
            return res

        return ecdf