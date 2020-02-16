import torch.optim as optim
import torch
import os
from net.resnet import *
from net.vgg import *
from net.lenet import LeNet
from net.googlenet import GoogLeNet
from net.mobilenet import *
from net.mobilenetv2 import *
from net.shufflenetv2 import *
from net.shufflenet import *
from net.densenet import *
from net.preact_resnet import *
from net.resnext import *
from net.wrn import *
from net.squeezenet import *
from net.senet import *
from net.efficientnet import *
from net.dpn import *
from dataloader import CIFAR10_DataLoader


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_data = CIFAR10_DataLoader(train=True)
        self.test_data = CIFAR10_DataLoader(train=False)
        self.num_epochs = args.epoch
        self.model = eval(args.model+str('()'))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100)

    def train(self):
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            self.model.train()
            sum_loss = 0.0
            self.scheduler.step()
            for i, data in enumerate(self.train_data):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                if i % 100 == 99:
                    print('[%d %d] loss:%.03f' %
                          (epoch + 1, i + 1, sum_loss / 100))
                    sum_loss = 0.0

            acc = self.test()

            if acc > best_acc:
                best_acc = acc
                state = {
                    'net': self.model.state_dict(),
                    'acc': best_acc,
                    'epoch': self.num_epochs,
                }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{0}_{1}_{2}.pth'.format(self.args.model, best_acc, self.num_epochs))
        with open('accuracy.txt', 'a') as f:
            f.write('model={0}, acc={1}, epoch={2}\n'.format(self.args.model, best_acc, self.num_epochs))

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data_test in enumerate(self.test_data):
                images, labels = data_test
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda(0)
                outputs_test = self.model(images)
                _, predicted = outputs_test.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        return acc
