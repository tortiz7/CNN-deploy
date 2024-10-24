# Pneumonia Detection ML System

A distributed system for training and serving a pneumonia detection model using chest X-rays. The system consists of three main components: ML training server, FLask API/Redis server, and a Flask/HTMX frontend.

## Infrastructure Setup
The project uses Terraform to provision AWS resources:
* 1 VPC
* 2 Availability Zones (us-east-1a for API/Frontend, us-east-1b for ML training)
* 3 Subnets (1 public for frontend, 2 private for API and ML)
* 3 EC2 Instances:
  - t3.micro (Frontend/HTMX in public subnet)
  - t3.medium (API/Redis in private subnet)
  - p3.2xlarge (ML training in private subnet)
* 2 Route Tables (public and private)
* Internet Gateway
* NAT Gateway
* Elastic IP
* Security Groups for each service

For detailed Terraform configuration, see the terraform directory.

## Server Setup Instructions

THE SCRIPTS BELOW IDEALLY CAN BE RUN AS PART OF THE TF DEPLOY BUT THEY WILL NOT RUN TO COMPLETION IN THEIR CURRENT STATE. USE THEM AS GUIDES FOR MANUAL SET UP.
MANUAL STEPS ARE WRITTEN IN THE COMMENTS

### ML Training Server Setup

```bash
# System packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv awscli wget git build-essential

# NVIDIA/CUDA setup
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers cuda

# Environment setup
cd ~/CNN_deploy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure inference.py with Redis server private IP
# Set REDIS_HOST in inference.py
```

### API/Redis Server Setup

```bash
# System packages
sudo apt-get update
sudo apt-get install redis-server python3-pip python3-venv

# Redis Configuration
sudo vi /etc/redis/redis.conf

# Add/modify these lines:
# bind 0.0.0.0
# protected-mode no

sudo systemctl restart redis

# Application setup
cd ~/CNN_deploy/pneumonia_api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start API server
gunicorn --config gunicorn_config.py app:app
```

### Frontend Server Setup

```bash
# System packages
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Application setup
cd ~/CNN_deploy/pneumonia_web
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API endpoint in app.py
# Set API_URL to private IP of API server

# Start frontend server
gunicorn --config gunicorn_config.py app:app
```

## Key Files

- `cnn.py`: Model training script
- `inference.py`: Runs predictions and stores in Redis
- `pneumonia_api/app.py`: Flask API endpoint
- `pneumonia_web/app.py`: Frontend Flask application