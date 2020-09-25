import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import argparse
import torch.distributed as dist
import time

device = "cpu"
torch.set_num_threads(4)
logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.INFO)

batch_size = 256 # batch for one node
RAND_SEED = 46

# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def train_model(model, train_loader, optimizer, criterion, epoch, args=None):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    model.train()
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        # Your code goes here!
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        w = list(model.parameters())
        
        if args.distributed:
            for layer in range(len(w)):
                grad_vec = w[layer].grad.data
                # need to do this per layer (#TODO: there might be a better way)
                dist.all_reduce(grad_vec, op=dist.ReduceOp.SUM)
                mean_grad = grad_vec/args.num_nodes
                w[layer].grad.data = mean_grad

        optimizer.step()
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        
        if batch_idx % 20 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTimeTaken: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), elapsed_time))

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    torch.manual_seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    parser = argparse.ArgumentParser(description='PyTorch Assignment')
    parser.add_argument('--distributed', type=bool_string, default=True,
                        help='flag to run in distributed mode')
    parser.add_argument('--master-ip', type=str, default='10.10.1.1',
                        help='flag to run in distributed mode')
    parser.add_argument('--node-rank', type=int, default=0, metavar='N',
                        help='rank-0 = Master')
    parser.add_argument('--num-nodes', type=int, default=4, metavar='N',
                        help='Number of nodes')

    args = parser.parse_args()

    if args.distributed:
        os.environ['MASTER_ADDR'] = args.master_ip
        os.environ['MASTER_PORT'] = '29500'  # hardcoding this because it doesn't really matter
        dist.init_process_group("gloo", rank=args.node_rank, world_size=args.num_nodes)
        global batch_size
        batch_size = batch_size/args.num_nodes

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="../data", train=True,
                                                download=True, transform=transform_train)
    
    training_sampler = \
        torch.utils.data.distributed.DistributedSampler(training_set,
                                                        num_replicas=args.num_nodes,
                                                        rank=args.node_rank,
                                                        shuffle=True,)

    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=training_sampler,
                                                    shuffle=False,
                                                    pin_memory=True)
    
    test_set = datasets.CIFAR10(root="../data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)

    logger.info("Testing model before training")
    test_model(model, test_loader, training_criterion)
    # running training for one epoch
    for epoch in range(1):
        training_sampler.set_epoch(epoch)
        train_model(model, train_loader, optimizer, training_criterion, epoch, args)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    main()
