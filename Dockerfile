#FROM python:3.11-slim
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Optional system packages you need (add more as needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        vim \
        bash \
        cmake \
        ninja-build \
        pkg-config \
        libpcap-dev \
        python3-dev \
        python3-pip \
        python3-opencv \
        libboost-all-dev \
        libopencv-dev \
        libgflags-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Install PcapPlusPlus from source
RUN apt-get update && apt-get install -y build-essential cmake git pkg-config libpcap-dev && \
    git clone --branch v21.05 https://github.com/seladb/PcapPlusPlus.git /tmp/pcpp && \
    cd /tmp/pcpp && \
    ./configure-linux.sh --default && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf /tmp/pcpp

# Create a working directory inside the container
WORKDIR /SketchVision

# Copy your entire project
#COPY SketchVision/ /workspace/



# Install Python dependencies (if any)
#RUN if [ -f requirements.txt ]; then python3 -m pip install --no-cache-dir -r requirements.txt; fi
#RUN python3 -m pip install invoke

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install "numpy<2" matplotlib invoke tqdm scikit-image scikit-learn numba

# Make entrypoint script executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
