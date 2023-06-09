## Based on: https://github.com/kuangliu/pytorch-cifar
'''Train CIFAR10 with PyTorch.'''
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import json

# need to first call it once before importing torch
X = np.random.rand(300, 160).astype(np.float32)
pca = PCA(n_components=2)
pca.fit(X)


import pandas as pd
import torch

import numpy as np
import fire
import inspect
import time
from pathlib import Path

from models.get_logging import get_logging
from models.get_dataset import get_dataset
from models.get_model import get_model
from models.get_loss import get_loss
from models.get_train import get_train

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(
        lr: float = 1e-4,
        label_noise: float = 0.2,
        k: int = 12,
        model: str = "resnet",
        augmentation: bool = True,
        optimizer: str = "adam",
        dataset: str = "cifar10",
        resume: bool = False,
        output_folder: str = "checkpoint",
        epochs: int = 4_000,
        num_classes: int = 10,
        run: int = 0,
        checkpoint_file: str = None,
        train_layers: str = "all",
        subspace_to_train=None,
):
    print("timestamp main start", time.time())
    all_args = inspect.getcallargs(main, **locals())
    output_folder = get_logging(all_args)

    torch.manual_seed(1234 + run)
    np.random.seed(run)

    # Data
    print('==> Preparing data..')
    trainset, trainloader, testset, testloader = get_dataset(dataset, label_noise, augmentation)
    print("timestamp get model", time.time())

    # Model
    print('==> Building model..')
    net, trainable_params = get_model(model, k, num_classes, device, checkpoint_file, train_layers, subspace_to_train)
    criterion, optimizer = get_loss(trainable_params, optimizer, lr)

    print(net)
    # Training
    train, test = get_train(net, trainloader, testloader, device, optimizer, criterion, output_folder)

    print("timestamp start train", time.time())
    epoch_data = []
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        data = test(epoch)
        epoch_data.append(data)
        pd.DataFrame(epoch_data).to_csv(Path(output_folder) / f"all.csv")


if __name__ == '__main__':
    print("timestamp main", time.time())
    fire.Fire(main)
