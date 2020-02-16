import torch.nn as nn
import torch
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


def drop_connect(inputs, training, p=0.5):
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.cuda())
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, k, expansion=4):
        super(MBConvBlock, self).__init__()
        mid_channel = in_channel * expansion
        self.conv_exp = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv_dw = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=k, stride=stride, padding=k//2, groups=mid_channel,
                      bias=False),
            nn.BatchNorm2d(mid_channel)
        )
        self.shortcut = None
        self.squeeze = nn.AvgPool2d(mid_channel)
        self.excitation1 = nn.Conv2d(mid_channel, mid_channel // 16, kernel_size=1, bias=False)
        self.excitation2 = nn.Sequential(
            nn.Conv2d(mid_channel // 16, mid_channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_prj = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5)     # use dropout as a substitution of dropconnect
        )
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        pre = self.conv_exp(x)
        pre = swish(pre)
        pre = self.conv_dw(pre)
        pre = swish(pre)
        w = self.squeeze(pre)
        w = self.excitation1(w)
        w = swish(w)
        w = self.excitation2(w)
        out = pre * w
        out = self.conv_prj(out)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return out


class EfficientB0(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientB0, self).__init__()
        self.in_channel = 32
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.stage2 = self.make_layers(16, 1, 1, 3)
        self.stage3 = self.make_layers(24, 2, 6, 3)
        self.stage4 = self.make_layers(40, 2, 6, 5)
        self.stage5 = self.make_layers(80, 3, 6, 3, 2)  # 16*16
        self.stage6 = self.make_layers(112, 3, 6, 5)
        self.stage7 = self.make_layers(192, 4, 6, 5, 2)  # 8*8
        self.stage8 = self.make_layers(320, 1, 6, 3, 2)  # 4*4
        self.after = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280)
        )
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(1280, num_classes)

    def make_layers(self, out_channel, block_num, expansion, k, stride=1):
        layers = []
        layers.append(MBConvBlock(self.in_channel, out_channel, stride, k=k, expansion=expansion))
        self.in_channel = out_channel
        for i in range(1, block_num):
            layers.append(MBConvBlock(self.in_channel, out_channel, stride=1, k=k, expansion=expansion))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = swish(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.stage8(out)
        out = self.after(out)
        out = swish(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

