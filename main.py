'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import os
import argparse

from models import *
from utils import progress_bar
from loss import BanditLoss



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--method', default='vanilla', type=str)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)




args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

if args.dataset == 'CIFAR10':

    num_classes = 10
    net = ResNet18(num_classes=num_classes)

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
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
elif args.dataset == 'MNIST':
    num_classes = 10
    net = ResNet18(num_classes=num_classes, in_channels=1)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    trainset = torchvision.datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    testset = torchvision.datasets.MNIST('./data', train=False,
                       transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
        shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, 
        shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt-{}-{}.pth'.format(args.dataset, args.method))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# criterion = nn.CrossEntropyLoss()
criterion = BanditLoss(method=args.method, alpha=args.alpha)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch, verbose=False):
    if verbose: print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if verbose:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss, 100.*correct/total


def test(epoch, verbose=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if verbose:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        if verbose: print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt-{}-{}.pth'.format(args.dataset, args.method))
        best_acc = acc

    return test_loss, 100.*correct/total
    


train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(start_epoch, start_epoch+args.epochs):
    loss, acc = train(epoch, args.verbose)
    train_losses.append(loss)
    train_accs.append(acc)
    loss, acc = test(epoch, args.verbose)
    test_losses.append(loss)
    test_accs.append(acc)
    scheduler.step()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(train_losses)
axs[0, 0].set_title('train_losses')
axs[0, 1].plot(train_accs)
axs[0, 1].set_title('train_acc')
axs[1, 0].plot(test_losses)
axs[1, 0].set_title('test_loss')
axs[1, 1].plot(test_accs)
axs[1, 1].set_title('test_acc')

if args.plot: plt.show()
print("final acc", test_accs[-1])


# uniform final acc 43.09
