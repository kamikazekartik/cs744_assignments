#!/bin/bash

# install wget
sudo apt install -y wget

# upgrade Linux kernel
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v4.20/linux-headers-4.20.0-042000_4.20.0-042000.201812232030_all.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v4.20/linux-headers-4.20.0-042000-generic_4.20.0-042000.201812232030_amd64.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v4.20/linux-image-unsigned-4.20.0-042000-generic_4.20.0-042000.201812232030_amd64.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v4.20/linux-modules-4.20.0-042000-generic_4.20.0-042000.201812232030_amd64.deb
sudo dpkg -i *.deb

# clean up .deb files
rm -f *.deb

# reboot so kernel upgrade takes effect
sudo reboot
