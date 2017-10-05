#!/bin/bash

sudo apt-get update
sudo apt-get -y install git
sudo apt-get -y install vim
sudo apt-get -y install python3 python3-pip python3-dev
sudo apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo apt-get -y install libpq-dev

sudo -H pip3 install --upgrade pip
sudo -H pip3 install jupyter
sudo -H pip3 install awscli

sudo apt-get -y install libhdf5-10 libhdf5-dev
sudo apt-get -y install python3-h5py

mkdir -p ~/.jupyter
echo "c.NotebookApp.allow_origin = '*'
c.NotebookApp.ip = '0.0.0.0'" | sudo tee /home/ubuntu/.jupyter/jupyter_notebook_config.py

mkdir -p ~/.aws
mv /tmp/credentials ~/.aws/credentials

mkdir -p ~/dev && cd ~/dev || exit 1
aws s3 cp s3://fachung/archives/fachung.tar.gz . --profile='fachung'
tar -xzvf fachung.tar.gz

cp /tmp/fachung.cfg ~/dev/fachung.cfg

echo "Installing fachung"
ls
sudo make setup
sudo make install
