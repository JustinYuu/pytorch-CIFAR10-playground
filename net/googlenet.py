import torch.nn as nn
import torch


class Inception(nn.Module):
    def __init__(self, in_channel, n1x1, n3x31, n3x32, n5x51, n5x52, pool):
        super(Inception, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, n1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channel, n3x31, kernel_size=1, bias=False),
            nn.BatchNorm2d(n3x31),
            nn.ReLU(True),
            nn.Conv2d(n3x31, n3x32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n3x32),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channel, n5x51, kernel_size=1, bias=False),
            nn.BatchNorm2d(n5x51),
            nn.ReLU(True),
            nn.Conv2d(n5x51, n5x52, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n5x52),
            nn.ReLU(True),
            nn.Conv2d(n5x52, n5x52, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n5x52),
            nn.ReLU(True)
        )
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, pool, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool),
            nn.ReLU(True)
        )

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(x)
        block3 = self.block3(x)
        block4 = self.block4(x)
        out = torch.cat([block1, block2, block3, block4], 1)
        return out


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False),   # 32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=1, stride=1, bias=False),    # 32*32*192
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 16*16*480
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)   # 8*8*832
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)   # 1*1*1024
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        pre = self.pre(x)
        out = self.a3(pre)
        out = self.b3(out)
        out = self.max_pool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.max_pool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

