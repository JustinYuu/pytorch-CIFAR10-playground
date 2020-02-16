import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, vgg_type, num_classes=10):
        super(VGG, self).__init__()
        self.cfg = cfg
        self.features = self.make_layers(cfg[vgg_type])
        self.classifier = nn.Linear(512, num_classes)

    def make_layers(self, cfg):
        layer = []
        in_channel = 3
        for i in cfg:
            if i == 'M':
                layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                cur_layer = nn.Sequential(
                    nn.Conv2d(in_channel, i, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(i),
                    nn.ReLU(True)
                )
                layer.append(cur_layer)
                in_channel = i
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')
