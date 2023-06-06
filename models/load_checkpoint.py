import torch
from collections import OrderedDict


def load_checkpoint(net, file, strip_module=True):
    print("load_checkpoint", file)
    checkpoint = torch.load(file)

    checkpoint2 = OrderedDict()
    for name in checkpoint["net"]:
        name2 = name
        if name.startswith("module.") and strip_module:
            name2 = name[len("module."):]
        checkpoint2[name2] = checkpoint["net"][name]
    net.load_state_dict(checkpoint2)
    return checkpoint["acc"]