import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from loss import BanditLoss

parser = argparse.ArgumentParser(description='PyTorch Bandit Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--method', default='smooth', type=str)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--eps', default=0.01, type=float)
parser.add_argument('--n', default=10, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--model', default='ResNet18', type=str)


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.model == 'ResNet18':
    model_ft = models.resnet18()

num_ftrs = model_ft.fc.in_features


if args.dataset == 'CIFAR10':
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)


elif args.dataset == 'MNIST':
    num_classes = 10
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    trainset = torchvision.datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    testset = torchvision.datasets.MNIST('./data', train=False,
                       transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
        shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, 
        shuffle=False, num_workers=2)

elif args.dataset == 'FashionMNIST':
    num_classes = 10
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                       transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data', train=False,
                       transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
        shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, 
        shuffle=False, num_workers=2)

model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft = model_ft.to(device)


if args.method == 'smooth':
    criterion = BanditLoss(method='smooth', alpha=args.alpha, eps=args.eps, \
    gamma=args.gamma, verbose=args.verbose) 
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
def find_lr(epoch):
    if epoch * len(trainloader) < 32e3:
        return 1
    elif epoch * len(trainloader) < 48e3: 
        return 0.1
    else:
        return 0.01
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, find_lr)

# Training
def train(epoch):
    model_ft.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model_ft(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('\nEpoch: %d' % epoch, "train", 'Acc: %.3f%% (%d/%d)'
                % (100.*correct/total, correct, total))

# Testing
def test():
    model_ft.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_ft(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print("test", 'Acc: %.3f%% (%d/%d)'
                % (100.*correct/total, correct, total))

start_epoch = 0
for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    test()
