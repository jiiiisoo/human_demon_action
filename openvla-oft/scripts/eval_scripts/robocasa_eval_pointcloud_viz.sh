#!/bin/bash
#SBATCH --job-name=eval-pc-viz
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
CHECKPOINT_PATH="/weka/jisookim/experiment/openvla/robocasa/ckpt/YOUR_CHECKPOINT_DIR"
STATS_PATH="/weka/jisookim/dataset/robocasa/datasets/single_stage_regenerate/robocasa_statistics.json"
LOG_DIR="/weka/jisookim/experiment/openvla/robocasa/exp_results/pointcloud_viz_eval"
ROLLOUT_DIR="${LOG_DIR}/rollout"

python experiments/robot/robocasa/run_robocasa_eval_pointcloud.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --task_suite_name robocasa \
    --num_episodes 50 \
    --max_episode_steps 720 \
    \
    --use_pointcloud_input True \
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
    --pc_viz_freq 10 \
    --pc_viz_max_points 2000 \
    \
    --point_visualize False \
    --save_pc_debug False \
    \
    --local_log_dir ${LOG_DIR} \
    --rollout_dir ${ROLLOUT_DIR}
