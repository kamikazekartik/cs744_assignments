import time
import os
import random
import copy
import argparse
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
import models.lenet as lenet

import utils

from qtorch.quant import *
# from qtorch.optim import OptimLP
# using our custom OptimLP since qtorch has an issue
from custom_optim_low import OptimLP
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower

num_types = ["weight", "activate", "grad", "error", "momentum"]

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# test function:
def test(dataset, model, device, test_loader, criterion, test_batch_size=1000):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    if dataset in ['EMNIST', 'MNIST']:
        classes = [str(i) for i in range(10)]
    elif dataset == 'Cifar10':
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                   'horse', 'ship', 'truck')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            test_loss = criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            for image_index in range(test_batch_size):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)

    for i in range(10):
      logger.info('Accuracy of %5s : %2d %%' % (
          classes[i], 100 * class_correct[i] / class_total[i]))

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

    return 100.0 * correct/len(test_loader.dataset)


# train method
def train(model, optimizer, criterion, scaler, train_loader, use_amp, epoch=0):

    # Iterating through the train loader
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()            # Reset the gradient in every iteration
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Loss forward pass
        scaler.scale(loss).backward()      # Loss backward pass
        scaler.step(optimizer)
        # Update all the parameters by the given learning rule
        scaler.update()
        # optimizer.zero_grad()              # set_to_none=True here can modestly improve performance

        if batch_idx % 500 == 0:
            logger.info('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(inputs),
              100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()


def run_experiment(args, number_dict, quant_dict):
    epoch_list = [0]
    loss_epoch_list = [-1]
    epoch_train_time_list = [-1]
    total_train_time_list = [-1]
    lr_list = [-1]
    test_acc_list = []

    # get data 
    train_loader, test_loader = utils.get_dataloader(args.dataset, use_half=args.use_half, PRELOAD=args.preload_data,
            batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    if args.pruned_model_path == None:
        model = utils.get_model(args.model, device)
    else:
        model = utils.get_model(args.model, device, args.pruned_model_path, args.num_classes)
    if args.low_prec:
        # automatically insert quantization modules
        model = sequential_lower(model, layer_types=["conv", "linear"],
                                 forward_number=number_dict["activate"], backward_number=number_dict["error"],
                                 forward_rounding=args.rounding, backward_rounding=args.rounding)

        if args.model in ('vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg_pruned'):
        # removing the final quantization module
            model.classifier = model.classifier[0]
        elif args.model in ('resnet50', 'resnet101'):
            model.linear = model.linear[0]
        elif args.model == 'resnet50_pruned':
            model.fc = model.fc[0]
        elif args.model == 'preresnet':
            # TODO: might need to put something in here
            model.fc = model.fc[0]

    if args.use_half:
        model.half()

    criterion = nn.CrossEntropyLoss()
    curr_lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=curr_lr, momentum=0.9)
    use_scaler = args.use_amp or args.low_prec or args.use_half
    # use scaler unless you are in full precicion mode
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    total_training_time = 0

    if args.low_prec:
        optimizer = OptimLP(optimizer,
                    weight_quant=quant_dict["weight"],
                    grad_quant=quant_dict["grad"],
                    momentum_quant=quant_dict["momentum"])

    # check accuracy before training
    test_acc = test(args.dataset, model, args.device, test_loader, criterion)
    test_acc_list.append(test_acc)
    curr_lr = args.lr

    for epoch in range(1, args.max_epochs):
        start_time = time.time()
        last_epoch_loss = train(model, optimizer, criterion, scaler, train_loader, args.use_amp, epoch)
        end_time = time.time()
        epoch_training_time = end_time - start_time
        total_training_time += epoch_training_time
        epoch_list.append(epoch)
        epoch_train_time_list.append(epoch_training_time)
        total_train_time_list.append(total_training_time)
        lr_list.append(curr_lr)
        if epoch%10 == 0:
            test_acc = test(args.dataset, model, args.device, test_loader, criterion)
        else:
            test_acc = -1
        test_acc_list.append(test_acc)
        loss_epoch_list.append(last_epoch_loss)

        # learning rate decay
        # curr_lr = args.gamma * curr_lr
        if epoch >= 80:
            curr_lr = 0.01
        if epoch >= 120:
            curr_lr = 0.001
        for g in optimizer.param_groups:
            g['lr'] = curr_lr


    # (OPTIONAL) Save trained model:
    # PATH = './cifar_net.pt'
    # torch.save(net.state_dict(), PATH)

    # (OPTIONAL) Load saved model
    # net.load_state_dict(torch.load(PATH))
    # net.to(device)

    results_df = pd.DataFrame({"epoch": epoch_list, "training_loss": loss_epoch_list, "test_acc": test_acc_list, "epoch_train_time": epoch_train_time_list, "total_train_time": total_train_time_list, "lr": lr_list, })
    return results_df


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch Mixed-Precision Experiments')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--max-epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='dataset to use MNIST|EMNIST|Cifar10')
    parser.add_argument('--model', type=str, default='lenet',
                        help='model to use lenet|vgg11|vgg13|vgg16|vgg19|vgg_pruned')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                       help='For Saving the current Model')
    parser.add_argument('--use-amp', type=bool_string, default=False,
                        help='use AMP')
    parser.add_argument('--use-half', type=bool_string, default=False,
                        help='use half precision for training')
    parser.add_argument('--preload-data', type=bool_string, default=False,
                        help='preload data into a list')
    parser.add_argument('--results-filename', type=str, default='results.csv',
                        help='specify csv file to write results')
    parser.add_argument('--low-prec', type=bool_string, default=False,
                        help='use qPytorch for custom low precision training')
    parser.add_argument('--pruned-model-path', type=str, default=None,
                        help='path to pruned model checkpoint when using pruned models')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes to predict for dataset')

    # quantization arguments
    for num in num_types:
        parser.add_argument('--nbits-{}'.format(num), type=int, default=-1, metavar='N',
                            help='word length in bits for {}; -1 if full precision.'.format(num))
    
    parser.add_argument('--rounding'.format(num), type=str, default='stochastic', metavar='S',
                        choices=["stochastic","nearest"],
                        help='rounding method for {}, stochastic or nearest'.format(num))
    
    args = parser.parse_args()
    # seed experiment
    utils.seed_experiment(args.seed)

    # Make sure we're using a GPU, and report what GPU it is.
    # (Otherwise this would run **forever**)
    if torch.cuda.is_available():
      logger.info("using "+torch.cuda.get_device_name(0))
    else:
      logger.info('No GPU available (enable it?), quitting.')
      exit()

    device = torch.device(args.device)

    # prepare quantization functions
    # using block floating point, allocating shared exponent along the first dimension
    number_dict = dict()
    for num in num_types:
        num_wl = getattr(args, "nbits_{}".format(num))
        number_dict[num] = BlockFloatingPoint(wl=num_wl, dim=0)
        logger.info("{:10}: {}".format(num, number_dict[num]))
    quant_dict = dict()
    for num in ["weight", "momentum", "grad"]:
        quant_dict[num] = quantizer(forward_number=number_dict[num],
                                    forward_rounding=args.rounding)
    
    logger.info("Running experiment with args: {}".format(args))
    results_df = run_experiment(args, number_dict, quant_dict)
    if args.results_filename == 'results.csv':
        # default filename, change it
        results_filename = "results_{}_{}_batchsize{}_".format(args.model, args.dataset, args.batch_size)
        if args.use_half:
            results_filename += "Half"
        elif args.use_amp:
            results_filename += "AMP"
        elif args.low_prec:
            results_filename += "Lowprec"
        else:
            results_filename += "Full"
        results_filename += ".csv"
    else:
        results_filename = args.results_filename

    results_df.to_csv(results_filename)
    logger.info("Results: ")
    logger.info(results_df)

