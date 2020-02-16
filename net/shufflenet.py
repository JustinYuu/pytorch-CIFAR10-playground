import torch
import torch.nn as nn
import torch.nn.functional as F


class Shuffle(nn.Module):
    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # transpose
        out = x.reshape(N, C, H, W)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride, groups):
        super(BottleNeck, self).__init__()
        self.stride = stride
        # As the author mentioned in raw paper, group convolution won't be used in stage2's first pointwise layer
        real_group = 1 if in_channel == 24 else groups
        tmp_channel = out_channel // self.expansion
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, tmp_channel, kernel_size=1, stride=1, groups=real_group, bias=False),
            nn.BatchNorm2d(tmp_channel),
            nn.ReLU(True),
            Shuffle(real_group),
            nn.Conv2d(tmp_channel, tmp_channel, kernel_size=3, stride=stride, padding=1, groups=tmp_channel,
                      bias=False),
            nn.BatchNorm2d(tmp_channel),
            nn.Conv2d(tmp_channel, out_channel, kernel_size=1, stride=1, groups=real_group, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if stride == 2:
            self.right = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.right = nn.Sequential()

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([self.left(x), self.right(x)], 1)
        else:
            out = self.right(x) + self.left(x)
        return F.relu(out)


class ShuffleNetV1(nn.Module):
    def __init__(self, groups, num_classes=10):
        super(ShuffleNetV1, self).__init__()
        cfg = {1: [144, 288, 576], 2: [200, 400, 800], 3: [240, 480, 960], 4: [272, 544, 1088], 8: [384, 768, 1536]}
        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True)
        )
        self.in_channel = 24
        self.stage2 = self.make_layer(cfg[groups][0], 3, groups)
        self.stage3 = self.make_layer(cfg[groups][1], 7, groups)
        self.stage4 = self.make_layer(cfg[groups][2], 3, groups)
        self.fc = nn.Linear(cfg[groups][2], num_classes)

    def make_layer(self, out_channel, block_num, groups):
        layers = []
        layers.append(BottleNeck(self.in_channel, out_channel-self.in_channel, stride=2, groups=groups))
        self.in_channel = out_channel
        for i in range(block_num):
            layers.append(BottleNeck(self.in_channel, out_channel, stride=1, groups=groups))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        pre = self.pre(x)
        out = self.stage2(pre)
        out = self.stage3(out)
        out = self.stage4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ShuffleNetV1_g1():
    return ShuffleNetV1(groups=1)


def ShuffleNetV1_g2():
    return ShuffleNetV1(groups=2)


def ShuffleNetV1_g3():
    return ShuffleNetV1(groups=3)


def ShuffleNetV1_g4():
    return ShuffleNetV1(groups=4)


def ShuffleNetV1_g8():
    return ShuffleNetV1(groups=8)
