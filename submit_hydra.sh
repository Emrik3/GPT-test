#!/bin/bash

CONFIG_NAME=$1

mkdir -p outputs/slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o outputs/slurm_logs/${CONFIG_NAME}_%j.log
#SBATCH -A naiss2026-4-1701-gpu
#SBATCH -p gpu

set -e

echo "=== Loading modules ==="

module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
module load GCC/14.3.0
module load OpenMPI/5.0.8
module load PyTorch/2.9.1-CUDA-12.9.1

echo "=== Setting up Python environment ==="

if [ ! -d .venv ]; then
    python -m venv --system-site-packages .venv
fi

source .venv/bin/activate

export PYTHONPATH="\$PWD/.venv/lib/python3.13/site-packages"

echo "=== Installing dependencies ==="

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

echo "=== Running ==="

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

time torchrun --standalone --nproc_per_node=1 run_hydra.py -cn $@
EOF
