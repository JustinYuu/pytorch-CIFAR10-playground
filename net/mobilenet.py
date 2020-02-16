import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),    # depthwise
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),  # pointwise
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class MobileNetV1(nn.Module):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512,
           512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.main_layer = self.make_layer(in_channel=32)
        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(1024, num_classes)

    def make_layer(self, in_channel):
        layer = []
        for lst in self.cfg:
            if isinstance(lst, int):
                out_channel = lst
                stride = 1
            else:
                out_channel = lst[0]
                stride = lst[1]
            layer.append(Block(in_channel, out_channel, stride))
            in_channel = out_channel
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.pre(x)
        out = self.main_layer(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


