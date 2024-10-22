#!/bin/bash

sudo apt-get update
sudo apt-get install redis-server
sudo apt install python3-pip python3-venv

# download repo
git clone https://github.com/elmorenox/CNN_deploy.git

cd ~/CNN_DEPLOY/pneumonia_api
python3 -m venv venv
