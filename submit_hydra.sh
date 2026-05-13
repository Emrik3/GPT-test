#!/bin/bash

CONFIG_NAME=$1

mkdir -p outputs/slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus-per-node=T4:1 # need to recofig login to correct one for wandb
#SBATCH --time=00:15:00
#SBATCH -o outputs/slurm_logs/${CONFIG_NAME}_%j.log
#SBATCH --account=naiss2025-22-762

module load python
export OMP_NUM_THREADS=1

# Activate environment
source .venv/bin/activate

# Install the necessary packages
python3 -m pip install -e .

export PYTHONUNBUFFERED=1

# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=1 run_hydra.py -cn $@
#### srun -u python3 -u run_hydra.py -cn $@
EOF
