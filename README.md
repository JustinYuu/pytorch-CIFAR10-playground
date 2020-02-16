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

* learning rate: epoch \[0, 100) 0.1, epoch \[100, 200) 0.01, epoch \[200, 300) 0.001  
* weight decay 0.0001  
* Momentum 0.9  
* Optimizer SGD  

## Performance  

