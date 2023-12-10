import json
import matplotlib.pyplot as plt
import numpy as np
from aglorithm.DDS import Dds
from aglorithm.URS import Urs
from Utils.utils import *
import torch
from Utils.model import LR
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(1)

sample_nums = 100
mean_value = 1.7
bias = 1

n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value*n_data, 1) + bias #类别0 数据shape=(100,2)
y0 = torch.zeros(sample_nums) #类别0，数据shape=(100,1)
x1 = torch.normal(-mean_value * n_data, 1) + bias #类别1，数据shape=(100,2)
y1 = torch.ones(sample_nums) #类别1，shape=(100,1)

train_x = torch.cat([x0, x1], 0)
train_y = torch.cat([y0, y1], 0)

idx = np.random.randint(0, 200, 50)
test_x = torch.tensor(train_x[idx, :])
test_y = torch.tensor(train_y[idx])

with open('Utils/conf.json', 'r') as f:
    conf = json.load(f)

dds = Dds(conf, train_x, train_y)
dds.pca()
seqSamp, seqSamp_y = dds.seqential_sampling(0, 2)
orderedSamp = torch.tensor(np.concatenate(seqSamp))
orderedSamp_y = torch.tensor(np.concatenate(seqSamp_y))
data_ = torch.utils.data.TensorDataset(orderedSamp, orderedSamp_y)
dataloader = torch.utils.data.DataLoader(data_, batch_size=50)

model = LR()
loss_func = torch.nn.BCELoss()

lr = 0.01
opt = torch.optim.SGD(model.parameters(), lr)
trainEpoch = 10

Loss1 = []
for epoch in range(trainEpoch):
    for x, y in dataloader:

        print(x.shape)
        y_p = model(x)
        loss = loss_func(y_p, torch.unsqueeze(y, 1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        y_t = model(test_x)
        loss_test = loss_func(y_t, torch.unsqueeze(test_y, 1))
        Loss1.append(loss_test)




data_ = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
dataloader = torch.utils.data.DataLoader(data_, batch_size=50)

model = LR()
loss_func = torch.nn.BCELoss()

lr = 0.01
opt = torch.optim.SGD(model.parameters(), lr)
trainEpoch = 10

Loss2 = []
for epoch in range(trainEpoch):
    for x, y in dataloader:

        y_p = model(x)
        loss = loss_func(y_p, torch.unsqueeze(y, 1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        y_t = model(test_x)
        loss_test = loss_func(y_t, torch.unsqueeze(test_y, 1))
        Loss2.append(loss_test)






print("Have a nice day!")


