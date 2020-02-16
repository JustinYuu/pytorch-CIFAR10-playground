import torch.nn as nn
import torch
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super(BottleNeck, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Transition, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        out = self.transition(x)
        return out


class DenseNet(nn.Module):  # DenseNet-BC
    def __init__(self, cfg, k, num_classes=10):
        super(DenseNet, self).__init__()
        self.k = k
        in_channel = 2*k    # if in_channel is k, then the transition will deprecate the 0.5(1/2) from the first batch k
        self.pre = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3, 2*k, kernel_size=3, stride=1, padding=1, bias=False)  # 32*32*2k
        )

        self.dense1 = self.make_layer(in_channel, cfg[0])  # 32*32*(cfg[0]+2)k
        in_channel += cfg[0] * k
        self.transition1 = Transition(in_channel, in_channel // 2)     # 16*16*((cfg[0]+2)/2)k
        in_channel = in_channel // 2

        self.dense2 = self.make_layer(in_channel, cfg[1])  # 16*16*((cfg[0]+2)/2+cfg[1])k
        in_channel += cfg[1] * k
        self.transition2 = Transition(in_channel, in_channel // 2)     # 8*8*(((cfg[0]+2)/2+cfg[1])/2)k
        in_channel = in_channel // 2

        self.dense3 = self.make_layer(in_channel, cfg[2])      # 8*8*(((cfg[0]+2)/2+cfg[1])/2+cfg[2])k
        in_channel += cfg[2] * k
        self.transition3 = Transition(in_channel, in_channel // 2)     # 4*4*((((cfg[0]+2)/2+cfg[1])/2+cfg[2])/2)k
        in_channel = in_channel // 2

        self.dense4 = self.make_layer(in_channel, cfg[3])      # 4*4*((((cfg[0]+2)/2+cfg[1])/2+cfg[2])/2+cfg[3])k
        in_channel += cfg[3] * k
        self.avg_pool = nn.AvgPool2d(4)     # 1*1*((((cfg[0]+2)/2+cfg[1])/2+cfg[2])/2+cfg[3])k
        self.fc = nn.Linear(in_channel, num_classes)

    def make_layer(self, in_channel, block_num):
        layers = []
        for i in range(block_num):
            layers.append((BottleNeck(in_channel, self.k)))
            in_channel += self.k
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.dense1(out)
        out = self.transition1(out)
        out = self.dense2(out)
        out = self.transition2(out)
        out = self.dense3(out)
        out = self.transition3(out)
        out = self.dense4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def DenseNet121(k=24):
    cfg = [6, 12, 24, 16]
    return DenseNet(cfg, k)


def DenseNet169(k=24):
    cfg = [6, 12, 32, 32]
    return DenseNet(cfg, k)


def DenseNet201(k=24):
    cfg = [6, 12, 48, 32]
    return DenseNet(cfg, k)


def DenseNet264(k=24):
    cfg = [6, 12, 64, 48]
    return DenseNet(cfg, k)
