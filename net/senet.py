import torch.nn as nn
import torch
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(SEBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.squeeze = nn.AvgPool2d(out_channel)
        self.excitation = nn.Sequential(
            nn.Conv2d(out_channel, out_channel//16, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(out_channel//16, out_channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.shortcut = None
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        pre = self.conv(x)
        w = self.squeeze(pre)
        w = self.excitation(w)
        out = pre * w
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)


class SEResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SEResNet, self).__init__()
        self.in_channel = 64
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.stage2 = self.make_layer(256, 3)
        self.stage3 = self.make_layer(512, 4)
        self.stage4 = self.make_layer(1024, 6)
        self.stage5 = self.make_layer(2048, 3)
        self.avg_pool = nn.AvgPool2d(2)
        self.linear = nn.Linear(2048, num_classes)

    def make_layer(self, out_channel, block_num):
        layers = []
        layers.append(SEBlock(self.in_channel, out_channel, stride=2))
        self.in_channel = out_channel
        for i in range(1, block_num):
            layers.append(SEBlock(self.in_channel, out_channel))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


