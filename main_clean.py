## Based on: https://github.com/kuangliu/pytorch-cifar
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import argparse
import fire
import inspect
from pathlib import Path
import json
import time
from contextlib import redirect_stdout, redirect_stderr

from models.resnet18k import make_resnet18k
from models.mcnn import make_cnn
from utils import progress_bar, add_label_noise, restrict_classes


import sys
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.terminal.flush()
        pass


def main(
        lr: float = 1e-4,
        label_noise: float = 0.2,
        k: int = 12,
        model: str = "resnet-nnooo",
        augmentation: bool = True,
        optimizer: str = "adam",
        dataset: str = "cifar10",
        resume: bool = False,
        output_folder: str = "checkpoint",
        epochs: int = 4_000,
        num_classes: int = 10,
        run: int = 0,
):
    print("timestamp main start", time.time())
    all_args = inspect.getcallargs(main, **locals())
    output_folder = Path(output_folder) / f"model-{model}_noise-{label_noise}_k-{k}_run-{run}"
    output_folder.mkdir(parents=True, exist_ok=True)

    sys.stdout = Logger(output_folder / "log.txt")
    sys.stderr = Logger(output_folder / "log_err.txt")

    with open(output_folder / "params.json", "w") as fp:
        json.dump(all_args, fp, indent=2)
    print(all_args)

    torch.manual_seed(1234 + run)
    np.random.seed(run)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("timestamp get data", time.time())

    from models.get_dataset import get_dataset
    trainset, trainloader, testset, testloader = get_dataset(dataset)
    print("timestamp get model", time.time())

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    #net = ResNet18()
    if model == "resnet":
        net = make_resnet18k(k, num_classes=num_classes)
    elif model == "cnn":
        net = make_cnn(c=k, num_classes=num_classes)
    elif model == "cnn-no-bn":
        net = make_cnn(c=k, num_classes=num_classes)
    else:
        raise
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    if optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
    elif optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr,)
    else:
        raise ValueError(f"optimizer {optimizer} is not defined")
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
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

            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def teest(epoch):
        nonlocal best_acc
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

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        print(acc)
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, output_folder / f'ckpt_best.pth')
            best_acc = acc
        # save 1, 3, 10, 300, ...
        if np.log10(epoch+1) % 1 == 0 or np.log10((epoch+1)/3) % 1 == 0:# in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]:
            print('Saving.. epoch', epoch)
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, output_folder / f'ckpt_{epoch}.pth')
        return [acc, time.time(), epoch]

    print("timestamp start train", time.time())
    accs = []
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        acc = teest(epoch)
        accs.append(acc)
        np.savetxt(output_folder / f"all.txt", accs)
        #scheduler.step()


if __name__ == '__main__':
    print("timestamp main", time.time())
    fire.Fire(main)
