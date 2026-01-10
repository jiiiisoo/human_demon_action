#!/bin/bash
#SBATCH --job-name=robo_6
#SBATCH --qos=big_qos
#SBATCH --partition=big_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch2/jisoo6687/robocasa/datasets/pointrack/kitchen_pnp/PnPStoveToCounter/2024-05-01/generate_tracks.out
#SBATCH --error=/scratch2/jisoo6687/robocasa/datasets/pointrack/kitchen_pnp/PnPStoveToCounter/2024-05-01/generate_tracks.err

# python robocasa/scripts/generate_tracking_data.py \
#     --dataset /scratch2/jisoo6687/robocasa/datasets/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams.hdf5 \
#     --point_cloud_dir /scratch2/jisoo6687/robocasa/datasets/pointrack/kitchen_pnp/PnPStoveToCounter/2024-05-01 \
#     --cube_half 0.6 \
#     --anchor_body robot0_link0 \
#     --max_track_points 1024 \
#     --exclude_wall \
#     --skip_mesh_save \
#     --save_first_ply \
#     --keyword obj distr \
#     --direction_anchor_body robot0_link0 \
#     --direction_target_body gripper0_right_right_gripper \
#     --direction_offset 0.5 \
#     --recenter_points \
#     --align_forward_to_neg_x

python robocasa/scripts/generate_tracking_data.py \
    --dataset /scratch2/jisoo6687/robocasa/datasets/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams.hdf5 \
    --point_cloud_dir /scratch2/jisoo6687/robocasa/datasets/pointrack/kitchen_pnp/PnPStoveToCounter/2024-05-01 \
    --cube_half 0.6 \
    --anchor_body robot0_link0 \
    --max_track_points 1024 \
    --exclude_wall \
    --skip_mesh_save \
    --save_first_ply \
    --direction_anchor_body robot0_link0 \
    --direction_target_body gripper0_right_right_gripper \
    --direction_offset 0.5 \
    --recenter_points \
    --align_forward_to_neg_x