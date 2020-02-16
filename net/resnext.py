import torch.nn as nn
import torch
import torch.nn.functional as F


class BottleNeck(nn.Module):
    expansion = 2

    def __init__(self, in_channel, cardinality, bottleneck_width, stride=1):
        super(BottleNeck, self).__init__()
        group_width = bottleneck_width * cardinality
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, group_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(group_width),
            nn.ReLU(True),
            nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                      bias=False),
            nn.BatchNorm2d(group_width),
            nn.ReLU(True),
            nn.Conv2d(group_width, group_width * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(group_width * self.expansion)
            # without ReLU
        )
        self.shortcut = None
        if stride != 1 or in_channel != group_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, group_width * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(group_width * self.expansion)
            )

    def forward(self, x):
        out = self.left(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)


class ResNeXt(nn.Module):
    def __init__(self, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.in_channel = 64
        self.bottleneck_width = bottleneck_width
        self.cardinality = cardinality
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer1 = self.make_layer(3, stride=1)  # 32*32
        self.layer2 = self.make_layer(4, stride=2)  # 16*16
        self.layer3 = self.make_layer(6, stride=2)  # 8*8
        self.layer4 = self.make_layer(3, stride=2)  # 4*4
        self.fc = nn.Linear(cardinality * bottleneck_width * 16, num_classes)

    def make_layer(self, block_num, stride):
        layers = []
        layers.append(BottleNeck(self.in_channel, self.cardinality, self.bottleneck_width, stride))
        self.in_channel = BottleNeck.expansion * self.cardinality * self.bottleneck_width
        for i in range(1, block_num):
            layers.append(BottleNeck(self.in_channel, self.cardinality, self.bottleneck_width, 1))
            self.in_channel = BottleNeck.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width = self.bottleneck_width * 2
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


def ResNeXt50_32x4d():
    return ResNeXt(cardinality=32, bottleneck_width=4)


def ResNeXt50_8x14d():
    return ResNeXt(cardinality=8, bottleneck_width=14)


def ResNeXt50_4x24d():
    return ResNeXt(cardinality=4, bottleneck_width=24)


def ResNeXt50_2x40d():
    return ResNeXt(cardinality=2, bottleneck_width=40)


def ResNeXt50_1x64d():
    return ResNeXt(cardinality=1, bottleneck_width=64)
