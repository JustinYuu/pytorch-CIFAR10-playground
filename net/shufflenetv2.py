import torch.nn as nn
import torch
import torch.nn.functional as F


class Shuffle(nn.Module):
    def __init__(self, groups=2):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        x = x.view(N, g, C // g, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        out = x.reshape(N, C, H, W)
        return out


class SplitChannel(nn.Module):
    def __init__(self):
        super(SplitChannel, self).__init__()
        self.split_size = 0.5

    def forward(self, x):
        pos = int(x.size(1) * self.split_size)
        return x[:, :pos, :, :], x[:, pos:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channel, split_size=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitChannel()
        in_channel = int(in_channel * split_size)
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True)
        )
        self.shuffle = Shuffle()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.block(x2)
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class SDSBlock(nn.Module):  # Spatial down sampling block
    expansion = 2

    def __init__(self, in_channel, out_channel):
        super(SDSBlock, self).__init__()
        mid_channel = out_channel // self.expansion
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=2, padding=1, groups=mid_channel, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True)
        )
        self.shuffle = Shuffle()

    def forward(self, x):
        out = torch.cat([self.left(x), self.right(x)], 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, zoom, num_classes=10):
        super(ShuffleNetV2, self).__init__()
        cfg = {0.5: (48, 96, 192), 1: (116, 232, 464), 1.5: (176, 352, 704), 2: (244, 488, 976)}
        self.in_channel = 24
        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
        )
        self.stage2 = self.make_layer(cfg[zoom][0], 3)
        self.stage3 = self.make_layer(cfg[zoom][1], 7)
        self.stage4 = self.make_layer(cfg[zoom][2], 3)
        self.after = nn.Sequential(
            nn.Conv2d(cfg[zoom][2], 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.AvgPool2d(4)
        )
        self.fc = nn.Linear(1024, num_classes)

    def make_layer(self, out_channel, block_num):
        layers = []
        layers.append(SDSBlock(self.in_channel, out_channel))
        for i in range(block_num):
            layers.append((BasicBlock(out_channel)))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.after(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ShuffleNetV2_Z05():
    return ShuffleNetV2(zoom=0.5)


def ShuffleNetV2_Z1():
    return ShuffleNetV2(zoom=1)


def ShuffleNetV2_Z15():
    return ShuffleNetV2(zoom=1.5)


def ShuffleNetV2_Z2():
    return ShuffleNetV2(zoom=2)
