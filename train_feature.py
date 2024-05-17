import os
import math
import argparse

import pandas as pd
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
        [transforms.RandomCrop(32, padding=4),
         transforms.Resize(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                              std=[0.2023, 0.1994, 0.2010])])

    trans_valid = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                              std=[0.2023, 0.1994, 0.2010])])

    trainset = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=True, transform=trans_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=trans_valid)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # model = ViT('B_16_imagenet1k', pretrained=True).to(device)


    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)

        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))


    model.train()

    train = []
    train_l = []
    model.eval()
    for img, lab in trainloader:
        feature = model(img.to(device))
        train.append(feature.cpu().detach().numpy())
        train_l.append(lab)

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
    featuret = dummy[16:, :]

    dummy2 = np.zeros_like(test_l[0])
    for node in test_l:
        dummy2 = np.concatenate([dummy2, node], 0)
    feature_lt = dummy2[16:]

    feature_outt = np.concatenate((featuret, feature_lt.reshape(feature_lt.shape[0], 1)), 1)

    dummy = np.zeros_like(train[0])
    for node in train:
        dummy = np.concatenate([dummy, node], 0)
    feature = dummy[16:, :]

    dummy2 = np.zeros_like(train_l[0])
    for node in train_l:
        dummy2 = np.concatenate([dummy2, node], 0)
    feature_l = dummy2[16:]

    feature_out = np.concatenate((feature, feature_l.reshape(feature_l.shape[0], 1)), 1)

    np.savetxt('data/cifar_feature.txt', feature_out)

    df1 = pd.DataFrame(feature_out)
    df2 = pd.DataFrame(feature_outt)
    df1.to_csv('data/cifar_train.csv')
    df2.to_csv('data/cifar_test.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path', type=str)
    parser.add_argument('--model-name', default='vit_base_patch16_224_in21k', help='create model name')


    # vit_base_patch16_224_in21k.pth
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    args = opt
    main(opt)

    print('Have a nice day!')