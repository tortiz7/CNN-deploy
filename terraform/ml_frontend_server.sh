#!/bin/bash

sudo apt-get update

sudo apt install python3-pip python3-venv


# download repo
git clone https://github.com/elmorenox/CNN_deploy.git

cd ~/CNN_DEPLOY/pneumonia_web

python3 -m venv venv

source venv/bin/active

pip install --upgrade pip
pip install -r requirements.txt

gunicorn --config gunicorn.conf.py app:app
