import torch.nn as nn
import torch
import torch.nn.functional as F


class DPNBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, addition_channel, stride, flag=True):
        super(DPNBlock, self).__init__()
        self.out_channel = out_channel
        self.mid = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, groups=32, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, out_channel + addition_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel + addition_channel)
        )
        self.shortcut = None
        if flag:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel + addition_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel + addition_channel)
            )

    def forward(self, x):
        out = self.mid(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        k = self.out_channel
        out = torch.cat([residual[:, :k, :, :] + out[:, :k, :, :], residual[:, k:, :, :], out[:, k:, :, :]], 1)
        return F.relu(out)


class DPN(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(DPN, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        mid_channel = cfg['mid_channel']
        out_channel = cfg['out_channel']
        block_num = cfg['block_num']
        add_channel = cfg['add_channel']
        self.in_channel = 64
        self.stage2 = self.make_layer(mid_channel[0], out_channel[0], block_num[0], add_channel[0], 1)
        self.stage3 = self.make_layer(mid_channel[1], out_channel[1], block_num[1], add_channel[1], 2)
        self.stage4 = self.make_layer(mid_channel[2], out_channel[2], block_num[2], add_channel[2], 2)
        self.stage5 = self.make_layer(mid_channel[3], out_channel[3], block_num[3], add_channel[3], 2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(out_channel[3]+(block_num[3]+1)*add_channel[3], num_classes)

    def make_layer(self, mid_channel, out_channel, block_num, add_channel, stride):
        layers = []
        layers.append(DPNBlock(self.in_channel, mid_channel, out_channel, add_channel, stride))
        self.in_channel = out_channel + add_channel * 2
        for i in range(1, block_num):
            layers.append(DPNBlock(self.in_channel, mid_channel, out_channel, add_channel, 1, flag=False))
            self.in_channel = out_channel + add_channel * (i+2)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def DPN92():
    cfg = {'mid_channel': [96, 192, 384, 768], 'out_channel': [256, 512, 1024, 2048],
           'block_num': [3, 4, 20, 3], 'add_channel': [16, 32, 24, 128]}
    return DPN(cfg)


def DPN98():
    cfg = {'mid_channel': [160, 320, 640, 1280], 'out_channel': [256, 512, 1024, 2048],
           'block_num': [3, 6, 20, 3], 'add_channel': [16, 32, 32, 128]}
    return DPN(cfg)
