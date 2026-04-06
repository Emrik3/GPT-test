#!/bin/bash
CONFIG_NAME=$1
mkdir -p outputs/slurm_logs

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

time torchrun --standalone --nproc_per_node=1 run_hydra.py -cn $@
