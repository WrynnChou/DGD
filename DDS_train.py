import numpy as np
import json
from aglorithm.DDS import Dds
from Utils.utils import *


with open('Utils/conf.json', 'r') as f:
    conf = json.load(f)

feature = np.loadtxt("logs/cifar10train.txt")
label = np.loadtxt('logs/cifar10train_label.txt')

dds = Dds(conf, feature, label)
dds.pca()
ud = glp(256, 3)
dds.sampling(ud, 2)

seqsamp, seqy = dds.seqential_sampling(0.1)

orderedSample = np.concatenate(seqsamp, 0)
orderedlabel = np.concatenate(seqy, 0)

epoch = conf["epoch"]
batch = conf["batch"]
lr = conf["learning_rate"]
subsampling_number = conf["subsampling_number"]
assert batch == subsampling_number, "The batch must match subsampling number!"












print('Have a nice day!')