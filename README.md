# PyTorch-CIFAR10-Playground

I have built and trained various popular networks for classification task on CIFAR10 datasets. Details and results are as below.  

## Requirments  

* python 3.6.10  
* pytorch 1.1.0  
* torchvision 0.3.0  
* cuda 9.0  

## How to run  

Run the script as below after cloning this repository.  

`python main.py --model xx --gpu x --epoch xxx`  

## Implementation Details  

* learning rate:   
    + epoch:    \[0, 100) 0.1
    + epoch:    \[100, 200) 0.01
    + epoch:    \[200, 300) 0.001  
* weight decay:    0.0001  
* Momentum:    0.9  
* Optimizer:    SGD  
* Epoch:    300  
* GPU:    Nvidia GeForce GTX 1080 Ti

## Performance  

model | accuracy(×100%) | epoch 
:-: | :-: | :-: 
ResNet18 | 0.9461 | 200
ResNet34 | 0.9507 | 200
ResNet50 | 0.9418 | 200
ResNet101 | 0.9471 | 200
ResNet152 | 0.95 | 200
VGG11 | 0.9157 | 200
VGG13 | 0.9032 | 200
VGG16 | 0.9359 | 300
VGG19 | 0.9347 | 300
AlexNet | 0.8195 | 300
LeNet | 0.7155 | 300
GoogLeNet | 0.9477 | 300
MobileNetV1 | 0.9185 | 300
ShuffleNetV1_g1 | 0.9145 | 300
ShuffleNetV1_g2 | 0.9123 | 300
ShuffleNetV1_g3 | 0.9205 | 300
ShuffleNetV1_g4 | 0.9175 | 300
ShuffleNetV1_g8 | 0.916 | 300
MobileNetV2 | 0.9434 | 300
ShuffleNetV2_Z05 | 0.9009 | 300
ShuffleNetV2_Z1 | 0.926 | 300
ShuffleNetV2_Z15 | 0.9338 | 300
ShuffleNetV2_Z2 | 0.9382 | 300
DenseNet121 | 0.9476 | 300
DenseNet169 | 0.9486 | 300
DenseNet201 | 0.9476 | 300
DenseNet264 | 0.9502 | 300
PreActResNet18 | 0.94 | 300
PreActResNet34 | 0.9474 | 300
PreActResNet50 | 0.9525 | 300
WRN_16_4 | 0.8189 | 300
WRN_40_8 | 0.9119 | 300
WRN_28_10 | 0.912 | 300
ResNeXt50_8x14d | 0.9536 | 300
ResNeXt50_1x64d | 0.9447 | 300
ResNeXt50_32x4d, |0.9548 | 300
ResNeXt50_2x40d | 0.9505 | 300
ResNeXt50_4x24d | 0.9533 | 300
SEResNet | 0.9437 | 300
SqueezeNet | 0.9297 | 300
EfficientB0 | 0.948 | 300

Since all networks use the same training method, this may not be the optimal performance of some networks.
