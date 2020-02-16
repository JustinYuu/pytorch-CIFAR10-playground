import torch.nn as nn
import torch
import torch.nn.functional as F


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(BottleNeck, self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel * self.expansion, stride=1, kernel_size=1, bias=False)
        )
        self.shortcut = None
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride, bias=False)
                # without BN and ReLU(PreAct)
            )

    def forward(self, x):
        out = self.left(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.shortcut = None
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)  # without BN and ReLU(PreAct)
            )

    def forward(self, x):
        out = self.left(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_channel = 64
        self.pre = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channel, block_num, stride):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))
        self.in_channel = out_channel * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def PreActResNet18():
    return PreActResNet(BasicBlock, [2, 2, 2, 2])


def PreActResNet34():
    return PreActResNet(BasicBlock, [3, 4, 6, 3])


def PreActResNet50():
    return PreActResNet(BottleNeck, [3, 4, 6, 3])


def PreActResNet101():
    return PreActResNet(BottleNeck, [3, 4, 23, 3])


def PreActResNet152():
    return PreActResNet(BottleNeck, [3, 8, 36, 3])
