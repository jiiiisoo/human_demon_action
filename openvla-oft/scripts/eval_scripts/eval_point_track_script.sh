#!/bin/bash
#SBATCH --job-name=evpt-uni-80000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export CUDA_VISIBLE_DEVICES=7
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=7  # Use your GPU ID
export PYOPENGL_PLATFORM=egl
# export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/jisoo_kim/human_demon_action/LIBERO:$PYTHONPATH

python experiments/robot/libero/run_libero_eval_pointcloud.py \
    --pretrained_checkpoint /weka/jisookim/experiment/openvla/libero/ckpt/libero_long_track_1024_input_head_onlytrack_nomask_transformer_8_stride1_hidden512_blocks3_b16_aug/openvla-7b+libero_10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_long_track_1024_input_head_onlytrack_nomask_transformer_8_stride1_hidden512_blocks3_b16_aug-8_acts_chunk--3rd_person_img--proprio_state--90000_chkpt \
    --task_suite_name libero_10 \
    --num_trials_per_task 50 \
    --pointcloud_num_points 1024 \
    --pointcloud_dim 2 \
    --pointcloud_cube_half 1 \
    --use_pointcloud_input False \
    --use_proprio True \
    --num_images_in_input 1 \
    --save_pc_debug False \
    --local_log_dir /weka/jisookim/experiment/openvla/libero/exp_results/openvla-oft/libero_10/libero_long_track_1024_input_head_onlytrack_nomask_transformer_8_stride1_hidden512_blocks3_b16_aug/evaluate/90k_ckpt \
    --rollout_dir /weka/jisookim/experiment/openvla/libero/exp_results/openvla-oft/libero_10/libero_long_track_1024_input_head_onlytrack_nomask_transformer_8_stride1_hidden512_blocks3_b16_aug/evaluate/rollout/90k_ckpt \
    --point_visualize False \
    --tracking_num_points 1024 \
    --tracking_dim 3 \
    --action_chunk_size 8 \
    --normalize_pointcloud True \
    --normalize_tracking True \
    --precomputed_statistics_path /weka/jisookim/dataset/libero/tfds/libero_10_no_noops/libero_long_statistics_1024.json