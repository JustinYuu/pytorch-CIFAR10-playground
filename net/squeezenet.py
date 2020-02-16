import torch.nn as nn
import torch
import torch.nn.functional as F


class FireBlock(nn.Module):
    def __init__(self, in_channel, s1x1, e1x1, e3x3):
        super(FireBlock, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, s1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(s1x1),
            nn.ReLU(True)
        )
        self.expand1 = nn.Sequential(
            nn.Conv2d(s1x1, e1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(e1x1),
            nn.ReLU(True)
        )
        self.expand2 = nn.Sequential(
            nn.Conv2d(s1x1, e3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(e3x3),
            nn.ReLU(True)
        )

    def forward(self, x):
        squeeze = self.squeeze(x)
        expand1 = self.expand1(squeeze)
        expand2 = self.expand2(squeeze)
        out = torch.cat([expand1, expand2], 1)
        return out


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True)
        )
        self.fire2 = FireBlock(96, 16, 64, 64)
        self.fire3 = FireBlock(128, 16, 64, 64)
        self.fire4 = FireBlock(128, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(3, 2)
        self.fire5 = FireBlock(256, 32, 128, 128)
        self.fire6 = FireBlock(256, 48, 192, 192)
        self.fire7 = FireBlock(384, 48, 192, 192)
        self.fire8 = FireBlock(384, 64, 256, 256)
        self.max_pool8 = nn.MaxPool2d(3, 2)
        self.fire9 = FireBlock(512, 64, 256, 256)
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.fire2(out)
        out = self.fire3(out)
        out = self.fire4(out)
        out = self.max_pool4(out)
        out = self.fire5(out)
        out = self.fire6(out)
        out = self.fire7(out)
        out = self.fire8(out)
        out = self.max_pool8(out)
        out = self.fire9(out)
        out = self.conv10(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
