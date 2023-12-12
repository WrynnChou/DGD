import os
import math
import argparse

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trans_train = transforms.Compose(
        [transforms.RandomResizedCrop(224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
         # （即先随机采集，然后对裁剪得到的图像缩放为同一大小） 默认scale=(0.08, 1.0)
         transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    trans_valid = transforms.Compose(
        [transforms.Resize(256),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
         transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=True, transform=trans_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=trans_valid)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    test = []
    test_l = []
    model.eval()
    for img, lab in testloader:
        feature = model(img.to(device))
        test.append(feature.cpu().detach().numpy())
        test_l.append(lab)

    dummy = np.zeros_like(test[0])
    for node in test:
        dummy = np.concatenate([dummy, node], 0)
    feature = dummy[8:, :]

    dummy2 = np.zeros_like(test_l[0])
    for node in test_l:
        dummy2 = np.concatenate([dummy2, node], 0)
    feature_l = dummy2[8:]
    np.savetxt('logs/cifar10test.txt', feature)
    np.savetxt('logs/cifar10test_label.txt', feature_l)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="D:/7/study/DGD/data/flower_photos")
    parser.add_argument('--model-name', default='vit_base_patch16_224_in21k', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

    print('Have a nice day!')