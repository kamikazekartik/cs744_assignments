import time
import os
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import logging

import models.vgg as vgg
import models.resnet as resnet
import models.lenet as lenet

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def seed_experiment(seed=0):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything with seed: {}".format(seed))


class ToHalfTensor(object):
    """Convert Tensors to HalfTensors"""

    def __init__(self, use_half):
        self.use_half = use_half

    def __call__(self, img):
        """
        Args:
            Tensor, use_half

        Returns:
            Half precision typecast tensor if use_half=True
              else: do nothing
        """
        if self.use_half:
            img = img.half()

        return img


def get_dataloader(dataset="MNIST", use_half=False, PRELOAD=False, batch_size=64, test_batch_size=1000):
    if dataset == 'MNIST':
        train_set = torchvision.datasets.MNIST('./data', train=True, download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(
                                                       (0.1307,), (0.3081,)),
                                                   ToHalfTensor(use_half),
                                               ]))
        test_set = torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ToHalfTensor(use_half),
        ]))
    elif dataset == 'EMNIST':
        train_set = torchvision.datasets.EMNIST('./data', split="digits", train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        (0.1307,), (0.3081,)),
                                                    ToHalfTensor(use_half),
                                                ]))
        test_set = torchvision.datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ToHalfTensor(use_half),
        ]))

    elif dataset == 'Cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            ToHalfTensor(use_half),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            ToHalfTensor(use_half),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                              shuffle=False, num_workers=2)

    preloaded_train_loader = []
    if PRELOAD:
        # nothing fancy. Just preload into a list
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            preloaded_train_loader.append((inputs, labels))
        train_loader = preloaded_train_loader

    return (train_loader, test_loader)


def get_model(model_name='lenet', device=None, pruned_model_path=None, num_classes = 1000):
    if model_name == 'lenet':
        model = lenet.Net().to(device)
    elif model_name == 'vgg11':
        model = vgg.vgg11().to(device)
    elif model_name == 'vgg13':
        model = vgg.vgg13().to(device)
    elif model_name == 'vgg16':
        model = vgg.vgg16().to(device)
    elif model_name == 'vgg19':
        model = vgg.vgg19().to(device)
    elif model_name == 'vgg_pruned':
        model = vgg.vgg_pruned(pruned_model_path, False, num_classes = num_classes).to(device)
    elif model_name == 'resnet18':
        model = resnet.ResNet18().to(device)
    elif model_name == 'resnet50':
        model = resnet.ResNet50().to(device)
    elif model_name =='resnet152':
        model = resnet.ResNet152().to(device)
    else:
        logger.info("Unknown Model Name!!!")
        model -1
    return model
