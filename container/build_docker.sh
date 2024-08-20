#!/usr/bin/env bash

# WARNING: This script must be executed from project root directory

# -- Create a Dockerfile from the HPCCM recipe
python3 container/hpccm-cuda-gcc-openmpi-hdf-conda.py --format docker > container/Dockerfile

# -- Build a Docker Image
docker build -t gitregistry.cnpem.br/gcd/data-science/segmentation/harpia -f container/Dockerfile .
