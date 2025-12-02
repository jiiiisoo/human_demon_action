#!/bin/bash
#SBATCH --job-name=eval-512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


python scripts/infer_pointcloud.py \
  --checkpoint_dir /mnt/data/jisookim/openvla_finetune/point_512_use_input/openvla-7b+libero_goal_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--point_512_use_input-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state--20000_chkpt \
  --base_model_path openvla/openvla-7b \
  --data_root_dir /mnt/data/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --pointcloud_root /mnt/data/libero/modified_libero_wotable_mesh/1.0.0 \
  --pointcloud_subdir pointclouds_512 --pointcloud_ext .ply \
  --tracking_num_points 256 --tracking_dim 3 \
  --pointcloud_input_num_points 256 --pointcloud_input_dim 3 \
  --use_proprio \
  --output_ply /home/jisookim/openvla-oft/point_infer/512_point/predicted_pointcloud.ply \
  --static_video_path /home/jisookim/openvla-oft/point_infer/512_point/predicted_pointcloud_static.mp4 \
  --video_frames 120 --video_fps 30 --video_elev 20 --video_azim 45