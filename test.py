import matplotlib.pyplot as plt
from Utils.utils import glp
from sklearn.neighbors import NearestNeighbors, KernelDensity
import numpy as np
from aglorithm.GLSS import Glss
from Utils.utils import *

data = np.random.multivariate_normal([0,0,0], np.eye(3), 1000)


glss = Glss(data, 0.8)
design = glss.design_transform()
model = glss.kde()
design_choose = glss.sampling_design(100)
subsample = glss.sampling()

plt.figure()
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.scatter(subsample[:, 0], subsample[:, 1], marker="*", c='r')
plt.show(block=True)




print("Have a nice day!")


