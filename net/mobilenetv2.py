import torch.nn as nn
import torch
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expansion):
        super(BottleNeck, self).__init__()
        self.stride = stride
        mid_channel = in_channel * expansion
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU6(True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1, stride=stride, groups=mid_channel, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU6(True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        if self.stride == 1:
            return self.shortcut(x) + self.left(x)
        else:
            return self.left(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        cfg = [(16, 1, 1, 1), (24, 2, 1, 6), (32, 3, 2, 6), (64, 4, 2, 6), (96, 3, 1, 6), (160, 3, 2, 6), (320, 1, 1, 6)]
        self.in_channel = 32
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.stage2 = self.make_layer(cfg[0])   # 16*16*16
        self.stage3 = self.make_layer(cfg[1])   # 16*16*24
        self.stage4 = self.make_layer(cfg[2])   # 16*16*32
        self.stage5 = self.make_layer(cfg[3])   # 8*8*64
        self.stage6 = self.make_layer(cfg[4])   # 8*8*96
        self.stage7 = self.make_layer(cfg[5])   # 4*4*160
        self.stage8 = self.make_layer(cfg[6])   # 4*4*320
        self.after = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),    # 4*4*1280
            nn.BatchNorm2d(1280),
            nn.ReLU(True)
        )
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Conv2d(1280, num_classes, kernel_size=1)   # use 1x1 conv as linear layer

    def make_layer(self, cfg):
        (out_channel, block_num, stride, expansion)= cfg
        layers= []
        layers.append(BottleNeck(self.in_channel, out_channel, stride, expansion))
        self.in_channel = out_channel
        for i in range(1, block_num):
            layers.append((BottleNeck(self.in_channel, out_channel, 1, expansion)))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        pre = self.pre(x)
        out = self.stage2(pre)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.stage8(out)
        out = self.after(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        return out

