#!/bin/bash
#SBATCH --job-name=evpt-uni-80000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export CUDA_VISIBLE_DEVICES=3
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=3  # Use your GPU ID
export PYOPENGL_PLATFORM=egl
# export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/jisoo_kim/human_demon_action/LIBERO:$PYTHONPATH

python experiments/robot/robocasa/run_robocasa_eval_pointcloud.py \
    --pretrained_checkpoint /weka/jisookim/experiment/openvla/robocasa/ckpt/robocasa_single_stage_regenerate_3image--proprio_state_b16_aug/openvla-7b+robocasa+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--robocasa_single_stage_regenerate_3image--proprio_state_b16_aug--120000_chkpt \
    --task_suite_name robocasa \
    --num_trials_per_task 50 \
    --pointcloud_num_points 1024 \
    --pointcloud_dim 3 \
    --pointcloud_cube_half 1 \
    --use_pointcloud_input False \
    --use_proprio True \
    --num_images_in_input 3 \
    --save_pc_debug False \
    --local_log_dir /weka/jisookim/experiment/openvla/robocasa/exp_results/openvla-oft/robocasa/robocasa_single_stage_regenerate_3image--proprio_state_b16_aug/evaluate/120k_ckpt \
    --rollout_dir /weka/jisookim/experiment/openvla/robocasa/exp_results/openvla-oft/robocasa/robocasa_single_stage_regenerate_3image--proprio_state_b16_aug/evaluate/rollout/120k_ckpt \
    --point_visualize False \
    --tracking_num_points 1024 \
    --tracking_dim 3 \
    --action_chunk_size 8 \
    --normalize_pointcloud True \
    --normalize_tracking True \
    --precomputed_statistics_path /weka/jisookim/dataset/robocasa/datasets/single_stage_regenerate/robocasa_statistics.json