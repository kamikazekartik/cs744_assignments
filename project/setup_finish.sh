#!/bin/bash

# install CUDA
sudo apt install -y software-properties-common
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /"
sudo add-apt-repository contrib
sudo apt-get -y update
sudo apt-get -y install cuda

# check that NVIDIA drivers are present
nvidia-smi

# install pip
sudo apt install -y python3-distutils
sudo apt install -y python3-apt
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

# install required Python packages
pip install numpy pandas matplotlib seaborn torch torchvision

# install some really esoteric things for qtorch
pip3 install qtorch
sudo apt install -y ninja-build
sudo apt install -y python3-dev

# allocate swap memory to prevent runtime errors
sudo fallocate -l 10G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# check swap memory is available
free -h

# now you should be able to put your Python code in main.py and run it with `python3 main.py`
