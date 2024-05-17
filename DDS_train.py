import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from aglorithm.DDS import Dds
from Utils.utils import *
from Utils.model import *

with open('Utils/conf.json', 'r') as f:
    conf = json.load(f)

feature = np.loadtxt("logs/cifar10train.txt")
label = np.loadtxt('logs/cifar10train_label.txt')
epoch = conf["epoch"]
batch = conf["batch"]
lr = conf["learning_rate"]
subsampling_number = conf["subsampling_number"]
def SequentialDDS(data, label, conf, ud, shuffle=True, partition=5, threshold=0.1):

    n = conf['subsampling_number']
    num = data.shape[0]
    d = data.shape[1]
    orderedSample, orderedlabel = [], []
    while data.shape[0] > n * partition and data.shape[0] > threshold * num:
        if shuffle:
            np.random.shuffle(data)
        data_split = []
        label_split = []
        del_idx = []
        for i in range(partition):
            split_n = int(data.shape[0] / partition)
            data_split.append(data[split_n * i:split_n * (i + 1), :])
            label_split.append(label[split_n * i:split_n * (i + 1)])

        for i in range(partition):
            dds = Dds(conf, data_split[i], label_split[i])
            dds.pca()
            subsamp, sublabel, idx = dds.sampling(ud, 2)
            del_idx.append((split_n * i + idx))
            orderedSample.append(subsamp)
            orderedlabel.append(sublabel)
        del_idx = np.concatenate(del_idx, 0)
        data = np.delete(data, del_idx, 0)
        label = np.delete(label, del_idx, 0)
    orderedSample.append(data)
    orderedlabel.append(label)
    return np.concatenate(orderedSample, 0), np.concatenate(orderedlabel, 0)






ud = np.loadtxt('Utils/glp/500_3.txt')



orderedSample, orderedlabel = SequentialDDS(feature, label, conf, ud)


assert batch == subsampling_number, "The batch must match subsampling number!"

orderedSample = torch.from_numpy(orderedSample).float()
orderedlabel = torch.from_numpy(orderedlabel)

data_ = torch.utils.data.TensorDataset(orderedSample, orderedlabel)
dataloader = DataLoader(data_, batch_size=batch)
model = MLP(768, 10)
opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5E-5)
loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
model.train()
batch = 2500
epoch = 100
y_p2 = []
for i in range(epoch):

    acc_loss = torch.zeros(1)
    acc_rightNumber = torch.zeros(1)
    sample_num = 0

    for para in dataloader:
        x, y = para
        sample_num += x.shape[0]
        output = model(x)
        pred_class = torch.max(output, dim=1)[1]
        acc_rightNumber += torch.eq(pred_class, y).sum()
        loss = loss_function(output, y.long())
        opt.zero_grad()
        loss.backward()
        acc_loss += loss.detach()

        opt.step()
        opt.zero_grad()
        y_p2.append(acc_loss/sample_num)

    print("train loss:{}, accuracy:{}".format(acc_loss/sample_num, acc_rightNumber/sample_num))


########## random
batch = 2500

feature = torch.from_numpy(feature).float()
label = torch.from_numpy(label)

data_ = torch.utils.data.TensorDataset(feature, label)
dataloader = DataLoader(data_, batch_size=batch)
model = MLP(768, 10)
opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5E-5)
loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
model.train()
y_p = []
for i in range(epoch):

    acc_loss = torch.zeros(1)
    acc_rightNumber = torch.zeros(1)
    sample_num = 0
    for para in dataloader:
        x, y = para
        sample_num += x.shape[0]
        output = model(x)
        pred_class = torch.max(output, dim=1)[1]
        acc_rightNumber += torch.eq(pred_class, y).sum()
        loss = loss_function(output, y.long())
        opt.zero_grad()
        loss.backward()
        acc_loss += loss.detach()
        opt.step()
        opt.zero_grad()
    y_p.append(acc_loss/sample_num)
    print("train loss:{}, accuracy:{}".format(acc_loss/sample_num, acc_rightNumber/sample_num))









print('Have a nice day!')