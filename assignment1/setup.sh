#!/bin/bash

# Instructions:
# - SSH into each of the three machines in the cluster (opening separate terminal tabs to do this helps)
# - Choose one machine to be the master. Copy-paste this script onto the master, then run it with `./setup.sh`

echo "Howdy. This script is part of our CS744 Big Data Systems group's code for Assignment 1."


# Collect cluster IP addresses
echo "Most of the cluster configuration is done automatically by this script, but a few things must be done manually first."
echo "On the other two machines (where you're not running this script), run"
echo "> ifconfig"
echo "to get their local IP addresses (10.0.0.x). Enter the IP addresses at the prompt below."
echo "Slave 1 IP address:"
export CS744_SLAVE_1="$(read msg && echo $msg)"
echo "Slave 2 IP address:"
export CS744_SLAVE_2="$(read msg && echo $msg)"
export CS744_MASTER="$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | grep -v '172.*')"
# (from https://stackoverflow.com/questions/13322485/how-to-get-the-primary-ip-address-of-the-local-machine-on-linux-and-os-x)


# Update software
echo "Run the following two commands on the two slave machines (this script will run them here on the master):"
echo "> sudo apt-get update"
echo "> sudo apt-get install openjdk-8-jdk"
echo "Hit enter to continue (and run the commands here on the master)..."
read msg
sudo apt-get update
sudo apt-get install openjdk-8-jdk
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
sudo apt install python-pip python-setuptools
sudo pip install --upgrade pip
sudo pip install parallel-ssh
echo "Creating parallel-ssh hosts file..."
echo "$CS744_MASTER" > ~/cs744_hosts
echo "$CS744_SLAVE_1" >> ~/cs744_hosts
echo "$CS744_SLAVE_2" >> ~/cs744_hosts
echo "parallel-ssh installed. You can run it manually after this script finishes here on the master with"
echo "> sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no \"df -h\""
echo "(Now you can close your SSH connections to the two slaves if you want.)"
echo "Hit enter to continue..."
read msg


# Set up HDFS on each machine
echo "Setting up HDFS on each machine..."
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo mkfs.ext4 /dev/xvda4"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo mkdir -p /mnt/data"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo mount /dev/xvda4 /mnt/data"
echo "Testing HDFS setup..."
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "df -h | grep \"data\""
echo "Hit enter to continue if the above looks good..."
read msg
echo "Great! Now your files can be stored at /mnt/data"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "cd /mnt/data/"
cd /mnt/data/


# Install Apache Hadoop on each machine
echo "Installing Hadoop..."
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "cd /mnt/data/ && wget http://apache.mirrors.hoobly.com/hadoop/common/hadoop-3.1.4/hadoop-3.1.4.tar.gz"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "cd /mnt/data/ && tar zvxf hadoop-3.1.4.tar.gz"
echo "Configuring Hadoop..."
# update hadoop-3.1.4/etc/hadoop/core-site.xml:
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" > /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "<?xml-stylesheet type=\"text/xsl\" href=\"configuration.xsl\"?>" >> /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "<configuration>" >> /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "<property>" >> /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "<name>fs.default.name</name>" >> /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"<value>hdfs://$CS744_MASTER:9000</value>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "</property>" >> /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "</configuration>" >> /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml'
# make namenode and datanode dirs:
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "mkdir /mnt/data/hadoop-3.1.4/data/"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "mkdir /mnt/data/hadoop-3.1.4/data/namenode/"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "mkdir /mnt/data/hadoop-3.1.4/data/datanode/"
# update hadoop-3.1.4/etc/hadoop/hdfs-site.xml
#sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'x=$(head -n 18 /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml.bkp)  && echo $x > /mnt/data/hadoop-3.1.4/etc/hadoop/core-site.xml.bkp'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" > /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no 'echo "<?xml-stylesheet type=\"text/xsl\" href=\"configuration.xsl\"?>" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml'
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"<configuration>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"<property>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"<name>dfs.namenode.name.dir</name>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"<value>/mnt/data/hadoop-3.1.4/data/namenode/</value>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"</property>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"<property>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"<name>dfs.datanode.data.dir</name>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"<value>/mnt/data/hadoop-3.1.4/data/datanode/</value>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"</property>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"</configuration>\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hdfs-site.xml"
# update JAVA_HOME environment variable:
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"export JAVA_HOME=/$(update-alternatives --display java | head -n 2 | tail -n 1 | cut -d / -f2- | sed 's/\/bin\/java//')\" >> /mnt/data/hadoop-3.1.4/etc/hadoop/hadoop-env.sh"
# update /mnt/data/hadoop-3.1.4/etc/hadoop/workers:
#sudo echo "$CS744_MASTER" >> /mnt/data/hadoop-3.1.4/etc/hadoop/workers <----- not needed, since /mnt/data/hadoop-3.1.4/etc/hadoop/workers already contains "localhost"
echo "$CS744_SLAVE_1" | sudo tee -a /mnt/data/hadoop-3.1.4/etc/hadoop/workers
echo "$CS744_SLAVE_2" | sudo tee -a /mnt/data/hadoop-3.1.4/etc/hadoop/workers
# update PATH to include Hadoop:
export PATH="/mnt/data/hadoop-3.1.4/bin:$PATH"
export PATH="/mnt/data/hadoop-3.1.4/sbin:$PATH"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo chmod -R 777 /mnt/data/hadoop-3.1.4/"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo chown -R treitz /mnt/data/hadoop-3.1.4/"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo chmod +x /mnt/data/"
echo "Starting HDFS..."
/mnt/data/hadoop-3.1.4/bin/hdfs namenode -format
/mnt/data/hadoop-3.1.4/sbin/start-dfs.sh
echo "Done. Checking HDFS is up..."
#jps
/mnt/data/hadoop-3.1.4/bin/hdfs dfsadmin -report
echo "Hit enter to continue if the above looks good..."
read msg


# Install Apache Spark on each machine
echo "Installing Spark..."
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "cd /mnt/data/ && wget http://mirror.metrocast.net/apache/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "cd /mnt/data/ && tar zvxf spark-2.4.7-bin-hadoop2.7.tgz"
echo "Configuring Spark..."
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"$CS744_SLAVE_1\" > /mnt/data/spark-2.4.7-bin-hadoop2.7/conf/slaves"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "echo \"$CS744_SLAVE_2\" >> /mnt/data/spark-2.4.7-bin-hadoop2.7/conf/slaves"
# ^ verify that it's ok to do this on master too (should it be on slaves but not master?)
echo "Starting Spark..."
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo chmod -R 777 /mnt/data/spark-2.4.7-bin-hadoop2.7/"
sudo parallel-ssh -i -h ~/cs744_hosts -O StrictHostKeyChecking=no "sudo chown -R treitz /mnt/data/spark-2.4.7-bin-hadoop2.7/"
/mnt/data/spark-2.4.7-bin-hadoop2.7/sbin/start-all.sh
echo "Done. Checking Spark is up..."
jps
echo "Hit enter to continue if the above looks good..."
read msg
# From (https://spark.apache.org/docs/latest/spark-standalone.html):
#   "To access Hadoop data from Spark, just use an hdfs:// URL (typically hdfs://<namenode>:9000/path)"

# run spark apps like this:
# ./spark-submit --master spark://c220g5-111230vm-1.wisc.cloudlab.us:7077 ../examples/src/main/python/pi.py
# spark://... comes from `wget http://localhost:8080` --> look at output index.html


# Pull down our repo onto each machine
echo "Cloning code repo..."
cd /mnt/data/
sudo mkdir code
sudo chown treitz code
cd code
git clone https://github.com/kamikazekartik/cs744_assignments.git
echo "Done."

echo "Setup is complete! You can now run Hadoop/Spark workloads on this cluster with something like"
echo "> /mnt/data/spark-2.4.7-bin-hadoop2.7/spark-submit path/to/pyspark.py"
echo "Remember to put your data at /mnt/data"
echo "You can stop the cluster at any time with"
echo "> /mnt/data/spark-2.4.6-bin-hadoop2.7/sbin/stop-all.sh"
echo "> stop-dfs.sh"
echo "Good luck!"
