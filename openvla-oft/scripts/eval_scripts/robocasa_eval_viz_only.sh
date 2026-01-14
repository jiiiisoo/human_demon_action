#!/bin/bash
#SBATCH --job-name=eval-viz-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0  # Use your GPU ID
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/jisoo_kim/human_demon_action/LIBERO:$PYTHONPATH

# Checkpoint and dataset paths
CHECKPOINT_PATH="/workspace/exp_results/openvla-oft/robocasa/ours/openvla-7b+robocasa+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--robocasa_single_stage_regenerate_3image--proprio_state_b16_aug--12000_chkpt"
STATS_PATH="/weka/jisookim/dataset/robocasa/datasets/single_stage_regenerate/robocasa_statistics.json"
LOG_DIR="/home/jisoo_kim/human_demon_action/openvla-oft/viz_only_eval"
ROLLOUT_DIR="${LOG_DIR}/rollout"

# Evaluate without pointcloud input to VLA, but visualize pointcloud for sanity check
python experiments/robot/robocasa/run_robocasa_eval_pointcloud.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --task_suite_name robocasa \
    --num_episodes 50 \
    --max_episode_steps 720 \
    \
    --use_pointcloud_input False \
    --pointcloud_num_points 1024 \
    --pointcloud_dim 3 \
    --pointcloud_cube_half 0.6 \
    --include_table False \
    --normalize_pointcloud True \
    --precomputed_statistics_path ${STATS_PATH} \
    \
    --use_proprio True \
    --num_images_in_input 3 \
    --action_chunk_size 8 \
    \
    --visualize_pc_image True \
    --save_pc_ply True \
    --pc_viz_freq 20 \
    \
    --point_visualize False \
    --save_pc_debug False \
    \
    --local_log_dir ${LOG_DIR} \
    --rollout_dir ${ROLLOUT_DIR}
