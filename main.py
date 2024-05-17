import numpy as np
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import datetime
from Utils.model import *
import pandas as pd
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
import math
# install GraB first
# from orderedsampler import OrderedSampler

def load_data_mnist(batch_size, resize=None):
    num_workers = 4
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=True)
    return (DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))

class MyDataset(Dataset):
    def __init__(self, path='data/mnist_pp1.csv'):
        data = pd.read_csv(path)
        data_ = torch.tensor(data.values)[:, 2:]
        data_ = (data_.resize(data.shape[0], 1, 28, 28)).float()
        label = torch.tensor(data.values)[:, 1].long()
        self.label = label
        self.data = data_
    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        data, label = self.data[item], self.label[item]
        return data, label
    
class MyDataset3(Dataset):
    def __init__(self, path='data/cifar_rearranged.csv'):
        data = pd.read_csv(path)
        data_ = torch.tensor(data.values)[:, 2:].float()
        label = torch.tensor(data.values)[:, 1].long()
        self.label = label
        self.data = data_
    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        data, label = self.data[item], self.label[item]
        return data, label
    
class MyDataset2(Dataset):
    def __init__(self, path='data/mnist_pp1.csv'):
        data = np.loadtxt(path)
        data_ = torch.tensor(data)[:, :-1].float()
        label = torch.tensor(data)[:, -1].long()
        self.label = label
        self.data = data_
    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        data, label = self.data[item], self.label[item]
        return data, label
    
def load_data_DDS(batch_size, path='data/mnist_pp1.csv', path_t='data/mnist_test.csv'):
    num_workers = 4
    data = MyDataset(path)
    test = MyDataset(path_t)
    return (DataLoader(data, batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test, batch_size, shuffle=False, num_workers=num_workers))

def load_data_cifar(batch_size, path='data/cifar_feature.txt', path_t='data/cifar_test.csv'):
    num_workers = 4
    data = MyDataset2(path)
    test = MyDataset3(path_t)
    return (DataLoader(data, batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test, batch_size, shuffle=False, num_workers=num_workers))

def load_data_cifar2(batch_size, path='data/cifar_rearranged.csv', path_t='data/cifar_test.csv'):
    num_workers = 4
    data = MyDataset3(path)
    test = MyDataset3(path_t)
    return (DataLoader(data, batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test, batch_size, shuffle=False, num_workers=num_workers))

def load_data_cifar3(batch_size, path='data/cifar_feature.txt', path_t='data/cifar_test.csv'):
    num_workers = 4
    data = MyDataset2(path)
    test = MyDataset3(path_t)
    return (DataLoader(data, batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(test, batch_size, shuffle=True, num_workers=num_workers))

def load_data_cifar4(batch_size, path='data/cifar_feature.txt', path_t='data/cifar_test.csv'):
    num_workers = 4
    data = MyDataset2(path)
    test = MyDataset3(path_t)
    
    return (DataLoader(data, batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(test, batch_size, shuffle=True, num_workers=num_workers))

def train_one_epoch(net, train_iter, eval_iter, device, optimizer, loss, cosine_schedule, SimulateAneal=True):
    train_l = 0.0
    train_ac = 0.0
    cnt = 0
    for i, (X, y) in enumerate(train_iter):
        net.train()
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        if SimulateAneal:
            cosine_schedule.step()

    for i, (X, y) in enumerate(eval_iter):
        net.eval()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l2 = loss(y_hat, y)
        train_l += l2.item()
        train_ac += torch.sum((torch.argmax(y_hat, dim=1) == y)).item()
        cnt += len(y)

    return train_l/len(eval_iter), train_ac/cnt

def evaluate_accuracy(net, data_iter, loss, device=None):
    net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    ac, cnt = 0.0, 0
    test_l = 0.0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        test_l += l.item()
        Z = torch.argmax(net(X), dim=1)
        ac += torch.sum(Z == y).item()
        cnt += len(y)

    return test_l/len(data_iter), ac/cnt

def train(net, train_iter, test_iter, eval_iter, num_epochs, lr, device, init=False, SimulateAneal=True):

    def init_weights(m):
        torch.manual_seed(777)
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    if not init:
        net.apply(init_weights)
    else:
        net.load_state_dict(torch.load(init))

    print('training on', device)
    net.to(device)
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.00001)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0, momentum=0)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0.005)
    # cosine_schedule = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=1000)
    loss = nn.CrossEntropyLoss()

    results_file = "logs/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    trl, tra, tea = [], [], []
    for epoch in range(num_epochs):
        train_l, train_acc = train_one_epoch(net, train_iter, eval_iter, device, optimizer, loss,
                                             cosine_schedule, SimulateAneal=SimulateAneal)
        test_l, test_acc = evaluate_accuracy(net, test_iter, loss, device)
        trl.append(train_l), tra.append(train_acc), tea.append(test_acc)
        # print(epoch, train_l, train_acc, test_acc)
        with open(results_file, "a") as f:
            f.write(f'epoch {epoch}, train_loss,{train_l}, train_acc{train_acc}, test loss ,{test_l}, test_acc{test_acc} \n')
        print(f'epoch{epoch}, train loss, {train_l}, train acc {train_acc}, test loss {test_l}, test acc {test_acc} \n')
    return train_l, test_l
    # plt.xlabel = 'epoch'
    # xlim = range(0, num_epochs)
    # legend = ['train loss', 'train acc', 'test acc']
    # plt.plot(xlim, trl, linewidth=1, color='purple')
    # plt.plot(xlim, tra, linewidth=1, color='green', linestyle='--')
    # plt.plot(xlim, tea, linewidth=1, color='blue', linestyle='--')
    # plt.legend(legend, ncol=4)
    # plt.grid(axis='y', linewidth=0.3)
    # plt.xticks(range(0, num_epochs, 2))
    # plt.show(block=True)


def train_grab(net, path, path_t, batch_size, num_epochs, lr, device, init=False, SimulateAneal=True):
    
    def init_weights(m):
        torch.manual_seed(777)
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    if not init:
        net.apply(init_weights)
    else:
        net.load_state_dict(torch.load(init))

    print('training on', device)
    net.to(device)
    
    data = MyDataset2(path)
    test = MyDataset3(path_t)
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.00001)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0, momentum=0)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0.005)
    # cosine_schedule = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=1000)
    loss = nn.CrossEntropyLoss()
    ordered_sampler = OrderedSampler(data,
                            batch_size=batch_size,
                            order_level=2,
                            model=net,
                            lossfunc=loss,
                            balance_type='pair_balance')
    
    train_loader = torch.utils.data.DataLoader(data, batch_sampler=ordered_sampler, num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    eval_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    results_file = "logs/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    trl, tra, tea = [], [], []
    for epoch in range(num_epochs):
        train_l, train_acc = train_one_epoch_grab(net, train_loader, eval_loader, device, optimizer, loss,
                                             cosine_schedule, ordered_sampler, SimulateAneal=SimulateAneal)
        test_l, test_acc = evaluate_accuracy(net, test_loader, loss, device)
        trl.append(train_l), tra.append(train_acc), tea.append(test_acc)
        # print(epoch, train_l, train_acc, test_acc)
        with open(results_file, "a") as f:
            f.write(f'epoch {epoch}, train_loss,{train_l}, train_acc{train_acc}, test loss ,{test_l}, test_acc{test_acc} \n')
        print(f'epoch{epoch}, train loss, {train_l}, train acc {train_acc}, test loss {test_l}, test acc {test_acc} \n')
    return train_l, test_l

def train_one_epoch_grab(net, train_iter, eval_iter, device, optimizer, loss, cosine_schedule, ordered_sampler, SimulateAneal=True):
    train_l = 0.0
    train_ac = 0.0
    cnt = 0
    net.train()
    for i, (X, y) in enumerate(train_iter):

        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        ordered_sampler.step()
        optimizer.step()
        if SimulateAneal:
            cosine_schedule.step()
    
    net.eval()
    for i, (X, y) in enumerate(eval_iter):

        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l2 = loss(y_hat, y)
        train_l += l2.item()
        train_ac += torch.sum((torch.argmax(y_hat, dim=1) == y)).item()
        cnt += len(y)

    return train_l/len(eval_iter), train_ac/cnt
def try_gpu():
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda:0')
    return torch.device('cpu')

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

if __name__ == '__main__':
    print('Have a nice day!')




