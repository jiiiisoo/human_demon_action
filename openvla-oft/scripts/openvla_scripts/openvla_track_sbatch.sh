#!/bin/bash
#SBATCH --job-name=ptw1024_new_data_8192_blocks3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=main
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Optional: load modules or activate conda env
# module load cuda/11.8
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate openvla-oft

cd /home/jisookim/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --dataset_name libero_goal_no_noops \
  --data_root_dir /mnt/data/modified_libero_rlds \
  --use_tracking_head True \
  --tracking_dim 3 \
  --tracking_num_points 1024 \
  --tracking_label_key '' \
  --use_pointcloud_input True \
  --pointcloud_input_num_points 1024 \
  --pointcloud_input_dim 3 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --use_wandb False \
  --run_id_note track_1024_w10_new_data_hidden8192_blocks3-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state \
  --use_tensorboard True \
  --tensorboard_log_dir /home/jisookim/openvla-oft/runs/tensorboard_track_1024_w10_new_data_hidden8192_blocks3 \
  --run_root_dir /mnt/data/jisookim/openvla_finetune/track_1024_w10_new_data_hidden8192_blocks3 \
  --tracking_tracks_root /mnt/data/libero/modified_libero_meshes_tracks_hash_final/1.0.0 \
  --tracking_tracks_filename vertex_tracks_resampled_1024.npy \
  --use_pointcloud_from_tracks True \
  --tracking_loss_weight 10.0 \
  --tracking_hidden_dim 8192 \
  --tracking_num_blocks 3 \
  --save_tracking_viz True \
  --tracking_viz_freq 100 \
  --tracking_viz_dir /mnt/data/jisookim/openvla_finetune/track_1024_w10_new_data_hidden8192_blocks3/videos

# torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
#   --vla_path openvla/openvla-7b \
#   --dataset_name libero_goal_no_noops \
#   --data_root_dir /mnt/data/modified_libero_rlds \
#   --use_tracking_head False \
#   --use_pointcloud_input True \
#   --pointcloud_input_num_points 5000 \
#   --pointcloud_input_dim 3 \
#   --use_l1_regression True \
#   --use_diffusion False \
#   --use_film False \
#   --num_images_in_input 1 \
#   --use_proprio True \
#   --batch_size 32 \
#   --learning_rate 5e-4 \
#   --num_steps_before_decay 100000 \
#   --max_steps 150005 \
#   --save_freq 10000 \
#   --save_latest_checkpoint_only False \
#   --image_aug True \
#   --lora_rank 32 \
#   --use_wandb False \
#   --run_id_note track_5000_w10_notracking_new_data_hidden8192_blocks3-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state \
#   --use_tensorboard True \
#   --tensorboard_log_dir /home/jisookim/openvla-oft/runs/tensorboard_track_5000_w10_notracking_new_data_hidden8192_blocks3 \
#   --run_root_dir /mnt/data/jisookim/openvla_finetune/track_5000_w10_notracking_new_data_hidden8192_blocks3 \
#   --tracking_tracks_root /mnt/data/libero/modified_libero_meshes_tracks_hash_final/1.0.0 \
#   --tracking_tracks_filename vertex_tracks_face_uniform.npy \
#   --use_pointcloud_from_tracks True \