#!/bin/bash
#SBATCH --job-name=diffusion_goal
#SBATCH --output=/home/jisookim/human_demon_action/diffusion_policy_goal/logs/diffusion_%j.out
#SBATCH --error=/home/jisookim/human_demon_action/diffusion_policy_goal/logs/diffusion_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=500:00:00
#SBATCH --mem=200G
#SBATCH -q main
#SBATCH --open-mode=append

# Create log directory
mkdir -p /home/jisookim/human_demon_action/diffusion_policy_goal/logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate human_demon

# Change to project directory
cd /home/jisookim/human_demon_action/diffusion_policy_goal

# Config file (default to config.yaml, can override with argument)
CONFIG=${1:-"config.yaml"}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG"
echo "=========================================="

# Run training with unbuffered output for real-time logging
python -u train_ddp.py --config $CONFIG

echo "Training completed!"

