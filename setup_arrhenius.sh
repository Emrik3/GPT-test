#!/bin/bash
set -e

# Load the cluster environment
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
module load GCC/14.3.0
module load OpenMPI/5.0.8
module load PyTorch/2.9.1-CUDA-12.9.1

# Create the venv using the cluster Python
python -m venv --system-site-packages .venv

source .venv/bin/activate

# Make venv packages take precedence over cluster packages
export PYTHONPATH="$PWD/.venv/lib/python3.13/site-packages"

# Install project dependencies, but don't reinstall PyTorch
python -m pip install --upgrade pip
python -m pip install \
    pandas \
    matplotlib \
    scikit-learn \
    requests \
    transformers \
    datasets \
    accelerate \
    tiktoken \
    zstandard \
    wandb \
    hydra-core \
    regex

python -m pip install -e . --no-deps

echo
echo "Environment setup complete."
echo
