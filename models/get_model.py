import torch
import torch.backends.cudnn as cudnn

from models.resnet18k import make_resnet18k
from models.mcnn import make_cnn
from models.load_checkpoint import load_checkpoint


def get_model(model, k, num_classes, device, checkpoint_file):
    if model == "resnet":
        net = make_resnet18k(k, num_classes=num_classes)
    elif model == "cnn":
        net = make_cnn(c=k, num_classes=num_classes)
    elif model == "cnn-no-bn":
        net = make_cnn(c=k, num_classes=num_classes)
    else:
        raise ValueError("model name not defined", model)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if checkpoint_file is not None:
        print("checkpoint_file", checkpoint_file)
        load_checkpoint(net, checkpoint_file, strip_module=False)

    trainable_params = net.parameters()
    return net, trainable_params
