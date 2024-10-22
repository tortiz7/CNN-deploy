#!/bin/bash

# Exit on any error
set -e

echo "Starting ML training server setup..."

# Update and install basic packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    awscli \
    wget \
    git \
    build-essential

# Install NVIDIA drivers and CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
sudo apt-get install -y cuda

# Set up CUDA environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify NVIDIA setup
nvidia-smi

# Download dataset from S3
# aws cli must be set up before this step
aws s3 cp s3://x-raysbucket/chest_xray/ ~/chest_xray --recursive

# Clone repository
git clone <your-repo-url> ~/CNN_deploy
cd ~/CNN_deploy

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Run training
echo "Starting model training..."
python3 cnn.py

# Run inference test
echo "Running inference tests..."
python3 inference.py

echo "Setup complete. Rebooting..."
sudo reboot