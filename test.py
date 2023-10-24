import json
import matplotlib.pyplot as plt
import numpy as np
from aglorithm.DDS import Dds
from Utils.utils import *

data = np.random.multivariate_normal([0, 0, 0], np.eye(3), 10000)
beta = np.array([0.5, 0.5, 0.5])
y1 = logit(np.matmul(data, beta).ravel() + np.random.normal(0, 1, 10000))
label1, mask1 = classify(y1)

with open('Utils/conf.json', 'r') as f:
    conf = json.load(f)

dds = Dds(conf, data, label1)
design = glp(400, 2)
dds.pca()
subsample, subsample_y = dds.sampling(design)

plt.figure()
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.scatter(subsample[:, 0], subsample[:, 1], marker="*", c='r')
plt.show(block=True)

print("Have a nice day!")


