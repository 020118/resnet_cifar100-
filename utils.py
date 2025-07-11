import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


def load_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])
    
    transform_val = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])

    train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=12, pin_memory=True)
    val_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=12, pin_memory=True)

    return train_loader, val_loader


def imshow(img):
    img = img/2 + 0.5
    img = img.numpy()
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()


def Optimizer(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    return criterion, optimizer
