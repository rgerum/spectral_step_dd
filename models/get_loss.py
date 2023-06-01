import torch.nn as nn
import torch.optim as optim

def get_loss(trainable_params, optimizer, lr):
    criterion = nn.CrossEntropyLoss()
    if optimizer == "sgd":
        optimizer = optim.SGD(trainable_params, lr=lr,
                              momentum=0.9, weight_decay=5e-4)
    elif optimizer == "adam":
        optimizer = optim.Adam(trainable_params, lr=lr, )
    else:
        raise ValueError(f"optimizer {optimizer} is not defined")

    return criterion, optimizer
