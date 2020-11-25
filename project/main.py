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
import models.lenet as lenet

import utils

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# seed experiment
seed_experiment(42)

# Options:
use_amp = False
use_half_all = True
use_half_conv = False
use_half_lin = False
dataset = 'MNIST'
PRELOAD = True # decides if we should use pytorch's dataloader or just preload into a python list

# Make sure we're using a GPU, and report what GPU it is.
# (Otherwise this would run **forever**)
if torch.cuda.is_available():
  print("using "+torch.cuda.get_device_name(0))
else:
  print('No GPU available (enable it?), quitting.')
  exit()
device = torch.device("cuda:0")

# Set up dataset:
batch_size = 64
test_batch_size = 1000

# test function:

def test(dataset, model, device, test_loader, criterion):
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


 def run_experiment(MAX_EPOCHS=3):
	epoch_list = [0]
	loss_epoch_list = [-1]
	epoch_train_time_list = [-1]
	total_train_time_list = [-1]
	lr_list = [-1]
	test_acc_list = []

	# get data
	train_loader, test_loader = get_dataloader(dataset, use_half=use_half_all, PRELOAD=PRELOAD)

	if dataset in ['MNIST', 'EMNIST']:
		model = Net().to(device)
	elif dataset == 'Cifar10':
		model = vgg11().to(device)
	else:
		logger.info("BAD DATASET!!!")
		exit()

	if use_half_all:
		model.half()
	
	criterion = nn.CrossEntropyLoss()
	curr_lr = 0.01
	optimizer = optim.SGD(model.parameters(), lr=curr_lr, momentum=0.9)
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
	total_training_time = 0

	# check accuracy before training
	test_acc = test(dataset, model, device, test_loader, criterion)
	test_acc_list.append(test_acc)

	for epoch in range(1, MAX_EPOCHS):
		start_time = time.time()
		last_epoch_loss = train(model, optimizer, criterion, scaler, train_loader, use_amp, epoch)
		end_time = time.time()
		epoch_training_time = end_time - start_time
		total_training_time += epoch_training_time
		epoch_list.append(epoch)
		epoch_train_time_list.append(epoch_training_time)
		total_train_time_list.append(total_training_time)
		lr_list.append(curr_lr)
		test_acc = test(dataset, model, device, test_loader, criterion)
		test_acc_list.append(test_acc)
		loss_epoch_list.append(last_epoch_loss)

		# cut learning rate in half every 20 epochs
		if epoch % 20 == 19:
		  curr_lr = 0.5 * curr_lr
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
