from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def CIFAR10_DataLoader(root='data_CIFAR10', train=True):
    if train is True:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    dataset = datasets.CIFAR10(root=root, download=False, train=train, transform=transform)
    dataloader = DataLoader(dataset=dataset, shuffle=train, batch_size=128, num_workers=4)
    return dataloader
