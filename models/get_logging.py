## Based on: https://github.com/kuangliu/pytorch-cifar
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import fire
import inspect
from pathlib import Path
import json
import time

from models.get_model import get_model
from models.get_dataset import get_dataset


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


def get_logging(all_args):
    output_folder = all_args["output_folder"]
    model = all_args["model"]
    label_noise = all_args["label_noise"]
    k = all_args["k"]
    run = all_args["run"]

    output_folder = Path(output_folder) / f"modelx-{model}_noise-{label_noise}_k-{k}_run-{run}"
    output_folder.mkdir(parents=True, exist_ok=True)

    sys.stdout = Logger(output_folder / "log.txt")
    sys.stderr = Logger(output_folder / "log_err.txt")

    with open(output_folder / "params.json", "w") as fp:
        json.dump(all_args, fp, indent=2)
    print(all_args)
