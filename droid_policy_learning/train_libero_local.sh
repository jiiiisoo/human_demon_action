#!/bin/bash
# Local training script for LIBERO finetuning (without SLURM)

export PYTHONPATH=/home/jisookim/human_demon_action/droid_policy_learning:$PYTHONPATH

CONFIG_PATH=${1:-"/home/jisookim/human_demon_action/droid_policy_learning/configs/libero_spatial_finetune.json"}

echo "=========================================="
echo "Training LIBERO Finetuning"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

cd /home/jisookim/human_demon_action

python -u droid_policy_learning/robomimic/scripts/train.py --config "${CONFIG_PATH}"

echo "Training finished."

