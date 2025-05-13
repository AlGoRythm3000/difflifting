# use base image of Ubuntu 20.04
FROM ubuntu:20.04

# Define workdir
WORKDIR /workdir

# change according to time zone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo

# requirements
RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    libboost-all-dev \
    libcairomm-1.0-dev \
    libgtk-3-dev \
    tzdata \
    gnupg2 \
    curl

# install Miniconda (conda) on Ubuntu
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda init bash

# create and activate conda environment with Python 3.10
RUN conda create -y -n difflifting python=3.10
RUN conda install -n difflifting -c conda-forge graph-tool
# Instale as dependências do PyTorch com CUDA 12.1
RUN conda install -n difflifting -y -c pytorch -c nvidia \
    pytorch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    pytorch-cuda=12.1


RUN conda run -n difflifting pip install --upgrade pip

# more requirements
RUN conda run -n difflifting pip install torch_geometric
RUN conda run -n difflifting pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
RUN conda run -n difflifting pip install git+https://github.com/pyt-team/TopoNetX.git
RUN conda run -n difflifting pip install git+https://github.com/pyt-team/TopoModelX.git
RUN conda run -n difflifting pip install ogb colorama networkx torchinfo entmax



# default command
CMD ["/bin/bash"]