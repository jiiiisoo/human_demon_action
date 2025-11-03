#!/bin/bash
#SBATCH --job-name=compute_stats
#SBATCH --output=/home/jisookim/human_demon_action/diffusion_policy_goal/logs/compute_stats_%j.out
#SBATCH --error=/home/jisookim/human_demon_action/diffusion_policy_goal/logs/compute_stats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH -q main

# Compute action statistics before training

cd /home/jisookim/human_demon_action/diffusion_policy_goal

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate human_demon

# Run computation
python compute_action_stats.py

echo "Action statistics computed successfully!"

