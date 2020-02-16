import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1, bias=False),   # 16*16*96
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 8*8*96
            nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1,bias=False),  # 4*4*256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2*2*256
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=False),     # 2*2*384
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),    # 2*2*384
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),    # 2*2*384
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),    # 2*2*256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)                       # 1*1*256
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
