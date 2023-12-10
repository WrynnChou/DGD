import numpy as np
import pingouin as pg
from Utils.utils import *
import torch
from sklearn.cluster import KMeans
mean = np.array([1, 1])
mean2 = np.array([2, 2])
sigma = np.array([[1, 0.5], [0.5, 1]])
sigma2 = np.array([[3, 1], [1, 5]])
x = np.random.multivariate_normal(mean, sigma, 1000)
x2 = np.random.multivariate_normal(mean2, sigma2, 1000)
x_c = np.append(x, x2, 0)
h = pg.multivariate_normality(x_c, 0.05)
h1 = pg.multivariate_normality(x, 0.05)

dim = 2
for i in range(dim):
    quantile = IECDF_1D(x_c[:, i])
    mask = x_c[:, i] >= quantile(0.5)
    part_1 = x_c[mask, :]
    part_2 = x_c[~mask, :]
    h1 = pg.multivariate_normality(part_1)
    h2 = pg.multivariate_normality(part_2)


opt = torch.optim.SGD

print('Have a nice day!')