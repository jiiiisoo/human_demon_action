#!/bin/bash
#SBATCH --job-name=evpt-uni-80000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export PYTHONPATH=/home/jisookim/openvla-oft/LIBERO:$PYTHONPATH

python -u experiments/robot/libero/run_libero_eval_pointcloud.py \
    --pretrained_checkpoint /mnt/data/jisookim/openvla_finetune/track_5000_w10_new_data_hidden8192_blocks3/openvla-7b+libero_goal_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--track_5000_w10_new_data_hidden8192_blocks3-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state--50000_chkpt \
    --task_suite_name libero_goal \
    --num_trials_per_task 50 \
    --pointcloud_num_points 5000 \
    --pointcloud_dim 3 \
    --pointcloud_cube_half 1000000000 \
    --use_pointcloud_input True \
    --use_proprio True \
    --num_images_in_input 1 \
    --save_pc_debug False \
    --local_log_dir /home/jisookim/openvla-oft/experiments/track_5000_w10_new_data_hidden8192_blocks3/logs_50000_chkpt \
    --rollout_dir /home/jisookim/openvla-oft/rollouts/track_5000_w10_new_data_hidden8192_blocks3_50000_chkpt \
    --point_visualize True \
    --tracking_num_points 5000 \
    --tracking_dim 3 \
    --tracking_hidden_dim 8192 \
    --tracking_num_blocks 3 \
    --include_table True \
    --include_wall False \
    --table_weight 0.3