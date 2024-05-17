import torch
from torch import nn
class LR(torch.nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)

        return x

class MLP(torch.nn.Module):

    def __init__(self, num_features, num_classes, dropout=0.1):
        super(MLP, self).__init__()
        self.linear1 = nn.Sequential(
            torch.nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
            nn.Dropout(dropout)
        )
        # self.linear1 = torch.nn.Linear(num_features, num_classes)


    def forward(self, x):
        x = self.linear1(x)

        return x

class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.leNet5 = nn.Sequential(
                    nn.Conv2d(1, 6, kernel_size=5, padding=2),
                    nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120),
                    nn.Sigmoid(),
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    nn.Linear(84, 10)
                )

    def forward(self, x):
        return self.leNet5(x)