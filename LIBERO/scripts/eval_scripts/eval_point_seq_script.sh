#!/bin/bash
#SBATCH --job-name=evrseq-uni-40000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export PYTHONPATH=/home/jisookim/openvla-oft/LIBERO:$PYTHONPATH

python -u experiments/robot/libero/run_libero_eval_pointcloud.py \
    --pretrained_checkpoint /mnt/data/jisookim/openvla_finetune/re_point_512_use_input/openvla-7b+libero_goal_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--point_512_re_use_input-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state--90000_chkpt \
    --task_suite_name libero_goal \
    --num_trials_per_task 50 \
    --pointcloud_num_points 512 \
    --pointcloud_dim 3 \
    --pointcloud_cube_half 0.5 \
    --use_pointcloud_input True \
    --use_proprio True \
    --num_images_in_input 1 \
    --save_pc_debug False \
    --local_log_dir /home/jisookim/openvla-oft/experiments/512_point_re_seq_uniform_50_trials/logs_90000_chkpt \
    --rollout_dir /home/jisookim/openvla-oft/rollouts/512_point_re_seq_uniform_50_trials_90000_chkpt \
    --point_visualize True \
    --tracking_num_points 512 \
    --tracking_dim 3