set -euo pipefail

# Optional: load modules or activate conda env
# module load cuda/11.8
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate openvla-oft

# tracking_head_type: mlp, with_point_input, parallel

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_libero.py \
  --vla_path openvla/openvla-7b \
  --libero_task_suite libero_10 \
  --libero_data_dir /mnt/local/jisookim/data/libero/libero_10_no_noops \
  --precomputed_statistics_path /mnt/local/jisookim/data/libero/libero_10_no_noops/libero_long_statistics_1024.json \
  --use_tracking_head True \
  --use_pointcloud_input False \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --use_wandb False \
  --run_id_note libero_long_track_1024_input_head_onlytrack_transformer_8_stride1_hidden512_blocks3_aug-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state \
  --use_tensorboard True \
  --tensorboard_log_dir /mnt/local/jisookim/experiment/libero/runs/tensorboard_libero_long_track_1024_input_head_onlytrack_transformer_8_stride1_hidden512_blocks3_aug \
  --run_root_dir /mnt/local/jisookim/experiment/libero/ckpt/libero_long_track_1024_input_head_onlytrack_transformer_8_stride1_hidden512_blocks3_aug \
  --tracking_tracks_root /mnt/local/jisookim/data/pointrack/libero_10 \
  --tracking_tracks_filename vertex_tracks_resampled_1024.npy \
  --tracking_use_pointcloud_input True \
  --tracking_use_point_features True \
  --tracking_head_type parallel \
  --tracking_hidden_dim 512 \
  --tracking_num_blocks 3 \
  --save_tracking_viz True \
  --tracking_viz_freq 4000 \
  --tracking_viz_dir /mnt/local/jisookim/experiment/libero/ckpt/libero_long_track_1024_input_head_onlytrack_transformer_8_stride1_hidden512_blocks3_aug/videos \
  --tracking_dim 3 \
  --tracking_num_points 1024 \
  --pointcloud_input_num_points 1024 \
  --pointcloud_input_dim 3 \
  --grad_accumulation_steps 2 \
  --use_wandb False \
  --run_id_note libero_long_track_1024_input_head_onlytrack_transformer_8_stride1_hidden512_blocks3_aug-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state \
  --window_stride 1 \
  # --action_chunk_size 16 \
