import argparse
from trainer import Trainer
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Selection')
    parser.add_argument('--model', type=str, default='DPN92', choices=[
        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'AlexNet',
        'LeNet', 'GoogLeNet', 'MobileNetV1', 'MobileNetV2', 'ShuffleNetV1_g1', 'ShuffleNetV1_g2', 'ShuffleNetV1_g3',
        'ShuffleNetV1_g4', 'ShuffleNetV1_g8', 'ShuffleNetV2_Z05', 'ShuffleNetV2_Z1', 'ShuffleNetV2_Z15',
        'ShuffleNetV2_Z2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet264', 'PreActResNet18',
        'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152', 'ResNeXt50_32x4d', 'ResNeXt50_8x14d',
        'ResNeXt50_1x64d', 'ResNeXt50_4x24d', 'ResNeXt50_2x40d', 'WRN_16_4', 'WRN_28_10', 'WRN_40_8', 'SqueezeNet',
        'SEResNet', 'EfficientB0', 'DPN92', 'DPN98'])

    parser.add_argument('--gpu', default='1')
    parser.add_argument('--epoch', type=int, default=5)
    args = parser.parse_args()
    print(vars(args))

    cuda_device = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    trainer = Trainer(args)
    trainer.train()
    trainer.test()
