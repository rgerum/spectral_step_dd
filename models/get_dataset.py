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
import re

labels_for_supercategory_cifar100 = [[4, 30, 55, 72, 95], [1, 32, 67, 73, 91], [54, 62, 70, 82, 92], [9, 10, 16, 28, 61], [0, 51, 53, 57, 83], [22, 39, 40, 86, 87], [5, 20, 25, 84, 94], [6, 7, 14, 18, 24], [3, 42, 43, 88, 97], [12, 17, 37, 68, 76]]
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_dataset(dataset, label_noise=None, augmentation=False):
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


    if transform_train == "default":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if "[" in dataset:
        name, sliced = dataset.split("[", 1)
        sliced = sliced.strip("[]")
        values = []
        for s in sliced.split(","):
            match = re.match(r"(\d*):(\d*)?(:(\d*))?", s)
            if match:
                start = None
                if match.groups()[0]:
                    start = int(match.groups()[0])
                end = None
                if match.groups()[1]:
                    end = int(match.groups()[1])
                step = None
                if match.groups()[3]:
                    step = int(match.groups()[3])
                values.append(slice(start, end, step))
                print(":::::::", slice(start, end, step).indices(100))
            else:
                values.append(int(s))
    else:
        name = dataset
        values = None

    print("name", name, "values", values)
    classdef = None
    if name.upper() == "CIFAR100":
        classdef = torchvision.datasets.CIFAR100
    if name.upper() == "CIFAR10":
        classdef = torchvision.datasets.CIFAR10

    trainset = classdef(root='./data', train=True, download=True, transform=transform_train)
    if label_noise is not None:
        trainset = add_label_noise(trainset, label_noise)

    trainset = restrict_classes_slice(trainset, values)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)


    testset = classdef(
        root='./data', train=False, download=True, transform=transform_test)
    testset = restrict_classes_slice(testset, values)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader

def restrict_classes_slice(trainset, values):
    if values is None:
        return trainset
    data = trainset.data
    targets = np.array(trainset.targets)

    indices = targets*0
    max_classes = np.max(targets)+1
    for value in values:
        if isinstance(value, int):
            print("include", value)
            indices |= targets == value
        else:
            for i in range(*value.indices(max_classes)):
                print("include slice", i, value.indices(max_classes))
                indices |= targets == i
    print(np.sum(indices))
    data = data[indices]
    targets = targets[indices]
    #print(targets)

    trainset.data = data
    trainset.targets = list(targets)
    return trainset

if 1:
    print(get_dataset("CIFAR100[0:10]", label_noise=20))
    print(get_dataset("CIFAR100[:10]"))
    print(get_dataset("CIFAR100[::2]"))
    print(get_dataset("CIFAR100[0,3,8:10]"))

def get_cifar100_supercategory():
    from cifarDataset import CIFAR100
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = CIFAR100(root='./data', train=True, download=True, transform=transform_test, coarse=True)
    #Basic EDA with labels for train_data
    demo_loader = torch.utils.data.DataLoader(train_data, batch_size=10)

    batch = next(iter(demo_loader))
    img, course_labels, fine_labels = batch
    print(type(img), type(course_labels), type(fine_labels))
    print(img.shape, course_labels.shape, fine_labels.shape)
    print(course_labels)
    print(fine_labels)

    fine_labels_in_coarse = []
    for i in range(10):
        train_coarse_labels = np.array(train_data.train_coarse_labels)
        train_labels = np.array(train_data.train_labels)
        print(train_coarse_labels == i)

        print(train_labels[train_coarse_labels==i])
        print(np.unique(train_labels[train_coarse_labels==i]))
        fine_labels_in_coarse.append(list(np.unique(train_labels[train_coarse_labels==i])))
    print(fine_labels_in_coarse)
    print(len([x for xx in fine_labels_in_coarse for x in xx]))
    print(len(np.unique([x for xx in fine_labels_in_coarse for x in xx])))