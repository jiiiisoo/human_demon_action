#!/bin/bash

# Compute action statistics before training
# This avoids OOM during training initialization

cd /home/jisookim/human_demon_action/diffusion_policy_goal

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate human_demon

# Run computation
python compute_action_stats.py

echo "Action statistics computed successfully!"

