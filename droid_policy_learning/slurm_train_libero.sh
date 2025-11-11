#!/bin/bash
#SBATCH --job-name=libero_finetune
#SBATCH --output=/home/jisookim/human_demon_action/droid_policy_learning/logs/libero_finetune_%j.out
#SBATCH --error=/home/jisookim/human_demon_action/droid_policy_learning/logs/libero_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --mem=1500G
#SBATCH --cpus-per-task=128
#SBATCH -q main
#SBATCH --open-mode=append

export PYTHONPATH=/home/jisookim/human_demon_action/droid_policy_learning:$PYTHONPATH

set -euo pipefail

CONFIG_PATH=${1:-"/home/jisookim/human_demon_action/droid_policy_learning/configs/libero_spatial_finetune.json"}

mkdir -p /home/jisookim/human_demon_action/droid_policy_learning/logs

echo "=========================================="
echo "Job ID    : ${SLURM_JOB_ID:-N/A}"
echo "Node list : ${SLURM_NODELIST:-N/A}"
echo "Config    : ${CONFIG_PATH}"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate human_demon

cd /home/jisookim/human_demon_action

python -u droid_policy_learning/robomimic/scripts/train.py --config "${CONFIG_PATH}"

echo "Training finished."

