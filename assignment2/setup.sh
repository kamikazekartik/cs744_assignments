#!/bin/bash

# Instructions:
# - SSH into each of the four machines in the cluster (opening separate terminal tabs to do this helps)
# - Choose one machine to be the master. Copy-paste this script onto the master, then run it with `./setup.sh`

echo "Howdy. This script is part of our CS744 Big Data Systems group's code for Assignment 2."


# Collect cluster IP addresses
export CS744_MASTER="10.10.1.1"
export CS744_SLAVE_1="10.10.1.2"
export CS744_SLAVE_2="10.10.1.3"
export CS744_SLAVE_3="10.10.1.4"


# Update software
echo "Most of the cluster configuration is done automatically by this script, but a few things must be done manually first."
echo "Run the following command on the three slave machines (this script will run them here on the master):"
echo "> sudo apt-get -y update && sudo apt-get -y install python-pip"
echo "Hit enter to continue (and run the commands here on the master)..."
read msg
sudo apt-get -y update && sudo apt-get -y install python-pip
echo "Hit enter to continue..."
read msg


# Generate SSH key on master and distribute to slaves (manually)
echo "Generating SSH key..."
ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
echo "Copy the following key file to ~/.ssh/authorized_keys all machines (already done for master):"
cat ~/.ssh/id_rsa.pub
echo ""
echo "Hit enter to continue..."
read msg


# Install/configure parallel-ssh
echo "Installing parallel-ssh..."
sudo apt -y install python-pip python-setuptools
sudo pip install --upgrade pip
sudo pip install parallel-ssh
echo "Creating parallel-ssh hosts file..."
echo "$CS744_MASTER" > ~/cs744_hosts
echo "$CS744_SLAVE_1" >> ~/cs744_hosts
echo "$CS744_SLAVE_2" >> ~/cs744_hosts
echo "$CS744_SLAVE_3" >> ~/cs744_hosts
echo "parallel-ssh installed. You can run it manually after this script finishes here on the master with"
echo "> sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no \"df -h\""
echo "(Now you can close your SSH connections to the two slaves if you want.)"
echo "Hit enter to continue..."
read msg


# Set up numpy and PyTorch on each machine
echo "Setting up numpy and PyTorch on each machine..."
sudo parallel-ssh -t 30 -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo pip install intel-numpy"
sudo parallel-ssh -t 30 -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
sudo parallel-ssh -t 30 -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo pip install future"
echo "Hit enter to continue if the above looks good..."
read msg
echo "Great! all done."

# install dstat on all nodes
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo apt -y install dstat"
# now you can log network activity on any node with something like
# > rm -f ~/net_log.csv && dstat -n --output ~/net_log.csv > /dev/null &
# > time python main.py --distributed=True --master-ip=10.10.1.1 --node-rank=0 --num-nodes=4
# (make sure to kill <procid> when done to stop dstat)


# Pull down our repo onto each machine
echo "Cloning code repo..."
cd ~/
git clone https://github.com/kamikazekartik/cs744_assignments.git
echo "Done."
