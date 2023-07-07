import torch
import torch.backends.cudnn as cudnn

from models.resnet18k import make_resnet18k
from models.mcnn import make_cnn
from models.load_checkpoint import load_checkpoint


def get_model(model, k, num_classes, device, checkpoint_file, train_layers="all", subspace_to_train=None):
    if model == "resnet":
        net = make_resnet18k(k, num_classes=num_classes)
    elif model == "cnn":
        net = make_cnn(c=k, num_classes=num_classes)
    elif model == "cnn-no-bn":
        net = make_cnn(c=k, num_classes=num_classes)
    else:
        raise ValueError("model name not defined", model)

    if checkpoint_file is not None:
        print("checkpoint_file", checkpoint_file)
        load_checkpoint(net, checkpoint_file, strip_module=True)

    if subspace_to_train is not None:
        def get_subspace():
            from sklearn.decomposition import PCA
            from models.get_dataset import get_dataset
            trainset, trainloader, testset, testloader = get_dataset(subspace_to_train, 0, None, batch_size_test=10000)
            net, trainable_params = get_model(model, k, num_classes, device, checkpoint_file, train_layers)
            x = net.module.get_last_repr(next(iter(testloader)), device)
            pca = PCA(n_components=10)
            pca.fit(x)
            return pca.mean_, pca.components_
        mean, components = get_subspace()
        net.project(mean, components, device)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if train_layers == "all":
        trainable_params = net.parameters()
    elif train_layers == "last":
        trainable_params = net.module.get_last_layer().parameters()

    return net, trainable_params
