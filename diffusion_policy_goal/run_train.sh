#!/bin/bash

# Training script for goal-conditioned diffusion policy

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate human_demon

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Config file
CONFIG=${1:-"config.yaml"}

echo "Starting training with config: $CONFIG"
echo "Number of GPUs: $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"

# Run training
python train_ddp.py --config $CONFIG

echo "Training completed!"

