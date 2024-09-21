# Update system
sudo apt update && sudo apt upgrade -y
# Install pyton, pip and AWS cli
sudo apt install python3-pip awscli -y
# Install tensorflow framework and other dependencies
pip3 install tensorflow boto3 pandas Flask
# Run script that trains, tests, and loads CNN model
python3 cnn.py
# Run Flask API
python3 app.py
