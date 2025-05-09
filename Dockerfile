# Base image with CUDA Toolkit 12.1 and Ubuntu 20.04
FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install necessary dependencies and add the deadsnakes PPA for Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \ 
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a virtual environment with Python 3.10
RUN python3 -m venv /app/venv

# Upgrade pip inside the virtual environment
RUN /app/venv/bin/pip install --upgrade pip

# Install PyTorch with the appropriate CUDA version
RUN /app/venv/bin/pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install additional libraries one by one
RUN /app/venv/bin/pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
RUN /app/venv/bin/pip install git+https://github.com/pyt-team/TopoNetX.git
RUN /app/venv/bin/pip install git+https://github.com/pyt-team/TopoModelX.git
RUN /app/venv/bin/pip install networkx ogb
RUN /app/venv/bin/pip install networkx torchinfo


# Set environment variables to ensure Python uses the correct environment
ENV PATH="/app/venv/bin:$PATH"

# Run a simple command to confirm the setup
CMD ["python3", "-c", "import torch; print(torch.__version__)"]
