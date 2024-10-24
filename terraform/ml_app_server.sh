#!/bin/bash

sudo apt-get update
sudo apt-get install redis-server
sudo apt install python3-pip python3-venv

# download repo
git clone https://github.com/elmorenox/CNN_deploy.git

cd ~/CNN_DEPLOY/pneumonia_api
python3 -m venv venv

source venv/bin/active

pip install --upgrade pip
pip install -r requirements.txt

# allow access from the ML training server
# vim /etc/redis/redis.conf
# bind 0.0.0.0 
# protected-mode no
# sudo systemctl restart redis

#start gunicorn
gunicorn --config gunicorn_config.py app:app

