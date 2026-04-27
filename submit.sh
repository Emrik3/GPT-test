#!/bin/bash

CONFIG_NAME=$(basename "$1" .yaml)

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus-per-node=A100:4
#SBATCH --time=05:00:00
#SBATCH -o outputs/slurm_logs/${CONFIG_NAME}_%j.log
#SBATCH --account=naiss2025-22-762

export OMP_NUM_THREADS=1

# Activate environment
source .venv/bin/activate

# Install the necessary packages
python -m pip install -e .

# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=4 run.py --config $1
EOF
