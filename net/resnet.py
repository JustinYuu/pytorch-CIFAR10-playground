import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        out = F.relu(out)
        return out


class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(BottleNeckBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, self.expansion * out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        self.in_channel = 64
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channel, block_num, stride=1):
        shortcut = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, shortcut))
        self.in_channel = out_channel * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(BottleNeckBlock, [3, 4, 6, 3])


def ResNet101():
    return ResNet(BottleNeckBlock, [3, 4, 23, 3])


def ResNet152():
    return ResNet(BottleNeckBlock, [3, 8, 36, 3])

