#!/bin/bash
#SBATCH --job-name=droid_diffusion
#SBATCH --output=/home/jisookim/human_demon_action/droid_policy_learning/logs/droid_diffusion_%j.out
#SBATCH --error=/home/jisookim/human_demon_action/droid_policy_learning/logs/droid_diffusion_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=168:00:00
#SBATCH --mem=1500G
#SBATCH --cpus-per-task=128
#SBATCH -q main
#SBATCH --open-mode=append

export PYTHONPATH=/home/jisookim/human_demon_action/droid_policy_learning:$PYTHONPATH

set -euo pipefail

CONFIG_PATH=${1:-"/home/jisookim/human_demon_action/droid_policy_learning/log/tmp/autogen_configs/ril/diffusion_policy/droid/im/11-05-None/11-05-25-12-25-39/json/bz_128_noise_samples_8_sample_weights_1_dataset_names_droid_cams_2cams_goalcams_2cams_goal_mode_offset_truncated_geom_factor_0.3_ldkeys_proprio-lang_visenc_VisualCore_fuser_None.json"}

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
