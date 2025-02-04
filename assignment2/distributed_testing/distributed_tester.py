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

device = "cpu"
torch.set_num_threads(4)
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Assignment')
    parser.add_argument('--node-rank', type=int, default=0, metavar='N',
                        help='rank-0 = Master')
    parser.add_argument('--world-size', type=int, default=3, metavar='N',
                        help='Number of nodes')

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '10.10.1.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("gloo", rank=args.node_rank, world_size=args.world_size)

    

    logger.info("EXITING")


