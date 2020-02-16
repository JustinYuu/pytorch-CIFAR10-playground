import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),     # 28*28*6
            nn.ReLU(True),
            nn.MaxPool2d(2),    # 14*14*6
            nn.Conv2d(6, 16, kernel_size=5),    # 10*10*16
            nn.ReLU(True),
            nn.MaxPool2d(2),    # 5*5*16
            nn.Conv2d(16, 120, kernel_size=5),  # 1*1*120
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
