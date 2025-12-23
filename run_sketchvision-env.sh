#!/bin/bash
set -e  # exit on error

IMAGE_NAME="sketchvision-env"
WORKDIR=$(pwd)/SketchVision  # current project folder
ULIMIT_NOFILE=65536  # soft/hard file limit

# -----------------------------
# 1. Check if Docker is installed
# -----------------------------
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    exit 1
else
  echo "Docker not found. Installing Docker..."
    sudo apt update
    sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io
fi

if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "NVIDIA Container Toolkit already installed."
else
    echo "Installing NVIDIA Container Toolkit..."
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi
# -----------------------------
# 2. Build Docker image
# -----------------------------
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# -----------------------------
# 3. Run Docker container
# -----------------------------
echo "Running Docker container..."
docker run --rm -it \
    --gpus all \
    --ulimit nofile=$ULIMIT_NOFILE:$ULIMIT_NOFILE \
    -v "$WORKDIR":/SketchVision \
    -w /SketchVision \
    $IMAGE_NAME "$@"
