# export CUDA_VISIBLE_DEVICES=0,7
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
# export MUJOCO_GL=osmesa  # OSMesa not available on this system
export MUJOCO_EGL_DEVICE_ID=1  # Use GPU 0 for rendering
export PYOPENGL_PLATFORM=egl
# export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH  # For OSMesa
export PYTHONPATH=/workspace/human_demon_action/LIBERO:$PYTHONPATH
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80
export OPENBLAS_NUM_THREADS=80
export NUMEXPR_NUM_THREADS=80
# export DISPLAY=  # Only needed for OSMesa


python experiments/robot/robocasa/generate_tracking_data_inference.py \
    --pretrained_checkpoint /workspace/exp_results/openvla-oft/robocasa/vla_input_pointnet_120k \
    --task_suite_name robocasa \
    --num_trials_per_task 50 \
    --pointcloud_num_points 1024 \
    --pointcloud_dim 3 \
    --visualize_pc_image True \
    --save_pc_ply True \
    --pc_viz_freq 200 \
    --save_pc_debug False \
    --use_pointcloud_input True \
    --use_proprio True \
    --num_images_in_input 3 \
    --local_log_dir /workspace/exp_results/openvla-oft/robocasa/vla_input_pointnet_120k/evaluate/120k_ckpt_seed7 \
    --rollout_dir /workspace/exp_results/openvla-oft/robocasa/vla_input_pointnet_120k/evaluate/rollout/120k_ckpt_seed7 \
    --seed 7 \
    --point_visualize False \
    --tracking_num_points 1024 \
    --tracking_dim 3 \
    --action_chunk_size 8 \
    --normalize_pointcloud True \
    --normalize_tracking True \
    --
    --precomputed_statistics_path /workspace/dataset/robocasa/robocasa_statistics.json \
    --task_slice_start 3 \
    --task_slice_end 6