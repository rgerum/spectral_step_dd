import numpy as np
from sklearn.decomposition import PCA
import json

# need to first call it once before importing torch
X = np.random.rand(300, 160).astype(np.float32)
pca = PCA(n_components=2)
pca.fit(X)

from matplotlib import pyplot as plt

from rgerum_utils.format_glob import format_glob_pd
from rgerum_utils.cache_decorator import cache

import torch
from pathlib import Path

from models.get_logging import get_logging
from models.get_dataset import get_dataset
from models.get_model import get_model
from models.get_loss import get_loss
from models.get_train import get_train

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_alpha_sklearn(x, n_comp=100):
    x = x.reshape(x.shape[0], -1)
    pca = PCA(n_components=min(n_comp, x.shape[1]))
    pca.fit(x)
    y = np.log10(pca.explained_variance_)
    x = np.log10(np.arange(1, y.shape[0]+1, dtype=y.dtype))
    return {"spectrum": {"x": x, "y": y}}


@cache("{stem}_spectrum.npz", version=5, force_write=True)
def get_spectrum(filename, stem):
    with open(Path(filename) / "params.json", "r") as fp:
        params = json.load(fp)

    trainset, trainloader, testset, testloader = get_dataset(params["dataset"], params["label_noise"],
                                                             params["augmentation"], batch_size_test=10000)

    net, trainable_params = get_model(params["model"], params["k"], params["num_classes"], device, filename / (stem + ".pth"))

    out = net.module.get_last_repr(next(iter(testloader)), device)
    d = get_alpha_sklearn(out)
    return d

data = format_glob_pd("logs_2023-06-05/resnet*/classes-10/*/*/ckpt_{epoch:d}.pth")

print(data)
for i, row in data.iterrows():
    print(i)
    get_spectrum(Path(row.filename).parent, Path(row.filename).stem)
