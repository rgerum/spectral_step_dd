import torch

import torchvision
import torchvision.transforms as transforms

import numpy as np
from utils import add_label_noise
import re
#                                                                                                                                                                                                              large carnivores
labels_for_supercategory_cifar100 = [[4, 30, 55, 72, 95], [1, 32, 67, 73, 91], [54, 62, 70, 82, 92], [9, 10, 16, 28, 61], [0, 51, 53, 57, 83], [22, 39, 40, 86, 87], [5, 20, 25, 84, 94], [6, 7, 14, 18, 24], [3, 42, 43, 88, 97], [12, 17, 37, 68, 76], [23, 33, 49, 60, 71], [15, 19, 21, 31, 38], [34, 63, 64, 66, 75], [26, 45, 77, 79, 99], [2, 11, 35, 46, 98], [27, 29, 44, 78, 93], [36, 50, 65, 74, 80], [47, 52, 56, 59, 96], [8, 13, 48, 58, 90], [41, 69, 81, 85, 89]]
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_dataset(dataset, label_noise=None, augmentation=False, batch_size_train=128, batch_size_test=100):
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
            else:
                values.append(int(s))
    else:
        name = dataset
        values = None

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
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)


    testset = classdef(
        root='./data', train=False, download=True, transform=transform_test)
    testset = restrict_classes_slice(testset, values)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader

def restrict_classes_slice(trainset, values):
    if values is None:
        return trainset
    data = trainset.data
    targets = np.array(trainset.targets)

    indices = targets*0
    max_classes = np.max(targets)+1

    new_values = []
    for value in values:
        if isinstance(value, int):
            indices |= targets == value
            new_values.append(value)
        else:
            for i in range(*value.indices(max_classes)):
                indices |= targets == i
                new_values.append(i)

    indices = indices.astype(bool)
    data = data[indices]
    targets = targets[indices]

    targets2 = np.zeros_like(targets)
    for index, value in enumerate(new_values):
        targets2[targets == value] = index
    #print(targets)

    trainset.data = data
    trainset.targets = list(targets2)
    return trainset

if 0:
    #print(get_dataset("CIFAR100[0:10]", label_noise=20))
    #print(get_dataset("CIFAR100[:10]"))
    #print(get_dataset("CIFAR100[::2]"))
    #print(get_dataset("CIFAR100[0,3,8:10]"))
    trainset, trainloader, testset, testloader = get_dataset("cifar100[4, 30, 55, 72, 95, 1, 32, 67, 73, 91]")
    trainset, trainloader, testset, testloader = get_dataset("cifar100[12, 17, 37, 68, 76]")

    for id, group in enumerate(labels_for_supercategory_cifar100):
        group = ",".join([str(i) for i in group])
        trainset, trainloader, testset, testloader = get_dataset(f"cifar100[{group}]")
        print(trainset)
        print(np.unique(trainset.targets), np.max(trainset.targets))
        for batch_x, batch_y in trainloader:
            print(batch_x.shape)
            import matplotlib.pyplot as plt
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow(batch_x[i].numpy().transpose(1, 2, 0))
                plt.title(batch_y[i].numpy())
            print(batch_y.shape)
            plt.savefig(f"group_{id}_{group}.png")
            plt.clf()
            break
    exit()

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
    for i in range(20):
        train_coarse_labels = np.array(train_data.train_coarse_labels)
        train_labels = np.array(train_data.train_labels)
        print(train_coarse_labels == i)

        print(train_labels[train_coarse_labels==i])
        print(np.unique(train_labels[train_coarse_labels==i]))
        fine_labels_in_coarse.append(list(np.unique(train_labels[train_coarse_labels==i])))
    print(fine_labels_in_coarse)
    print(len([x for xx in fine_labels_in_coarse for x in xx]))
    print(len(np.unique([x for xx in fine_labels_in_coarse for x in xx])))

#get_cifar100_supercategory()