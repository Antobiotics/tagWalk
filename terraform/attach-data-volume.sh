#!/bin/bash

devpath=$(readlink -f /dev/xvdh)

sudo file -s "$devpath" | grep -q ext4
if [[ 1 == "$?" && -b "$devpath" ]]; then
  sudo mkfs -t ext4 "$devpath"
fi

sudo mkdir /data
sudo chown -R ubuntu:ubuntu /data
sudo chmod 0775 /data

echo "$devpath /data ext4 defaults,nofail,noatime,nodiratime,barrier=0,data=writeback 0 2" | sudo tee -a /etc/fstab > /dev/null
sudo mount -t ext4 "$devpath" /data

sudo chown -R ubuntu:ubuntu /data
sudo chmod 0775 /data

sudo mkdir -p /home/ubuntu/dev
sudo chown -R ubuntu:ubuntu /home/ubuntu/dev
sudo ln -sf /data ~/dev/data