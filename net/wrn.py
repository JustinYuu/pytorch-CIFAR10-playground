import torch.nn as nn
import torch
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.shortcut = None
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.left(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)


class WRN(nn.Module):
    def __init__(self, depth, k, num_classes=10):
        super(WRN, self).__init__()
        self.in_channel = 16
        self.block_num = (depth - 4) // 6
        self.pre = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.stage2 = self.make_layer(16*k, 1)
        self.stage3 = self.make_layer(32*k, 2)
        self.stage4 = self.make_layer(64*k, 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.in_channel, num_classes)

    def make_layer(self, out_channel, stride):
        layers = []
        layers.append(Block(self.in_channel, out_channel, stride))
        self.in_channel = out_channel
        for i in range(1, self.block_num):
            layers.append(Block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def WRN_16_4():
    return WRN(16, 4)


def WRN_28_10():
    return WRN(28, 10)


def WRN_40_8():
    return WRN(40, 8)
