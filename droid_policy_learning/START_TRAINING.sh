#!/bin/bash

echo "================================================================"
echo "LIBERO Spatial Finetuning - Quick Start"
echo "================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "configs/libero_spatial_finetune.json" ]; then
    echo "ERROR: Please run this script from the droid_policy_learning directory"
    echo "cd /home/jisookim/human_demon_action/droid_policy_learning"
    exit 1
fi

echo "Step 1: Verify config loads correctly..."
conda run -n human_demon python test_config_loading.py
if [ $? -ne 0 ]; then
    echo "ERROR: Config loading failed. Please check test_config_loading.py output"
    exit 1
fi
echo "✓ Config loaded successfully"
echo ""

echo "Step 2: Check if checkpoint exists..."
CKPT_PATH="/home/jisookim/human_demon_action/droid_policy_learning/log/droid/im/diffusion_policy/11-05-None/bz_2048_noise_samples_8_sample_weights_1_dataset_names_droid_cams_2cams_goalcams_2cams_goal_mode_offset_truncated_geom_factor_0.3_ldkeys_proprio-lang_visenc_VisualCore_fuser_None/20251106125113/models/model_epoch_100.pth"
if [ ! -f "$CKPT_PATH" ]; then
    echo "WARNING: Checkpoint not found at:"
    echo "  $CKPT_PATH"
    echo ""
    echo "Please update the checkpoint path in configs/libero_spatial_finetune.json"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Checkpoint found"
fi
echo ""

echo "Step 3: Check if LIBERO datasets exist..."
SAMPLE_DATASET="/mnt/data/libero/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5"
if [ ! -f "$SAMPLE_DATASET" ]; then
    echo "WARNING: LIBERO dataset not found at:"
    echo "  $SAMPLE_DATASET"
    echo ""
    echo "Please ensure LIBERO datasets are downloaded to /mnt/data/libero/libero_spatial/"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ LIBERO datasets found"
fi
echo ""

echo "================================================================"
echo "Ready to start training!"
echo "================================================================"
echo ""
echo "Choose training mode:"
echo "  1) Local training (current terminal)"
echo "  2) SLURM cluster training"
echo "  3) Cancel"
echo ""
read -p "Enter choice (1/2/3): " -n 1 -r
echo
echo ""

case $REPLY in
    1)
        echo "Starting local training..."
        echo "Press Ctrl+C to stop training"
        echo ""
        sleep 2
        bash train_libero_local.sh
        ;;
    2)
        echo "Submitting SLURM job..."
        sbatch slurm_train_libero.sh
        echo ""
        echo "Monitor job status with: squeue -u $(whoami)"
        echo "View logs in: log/libero/spatial/diffusion_policy/<timestamp>/"
        ;;
    3)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run again and enter 1, 2, or 3"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Training started!"
echo "================================================================"
echo ""
echo "Monitor progress:"
echo "  TensorBoard: tensorboard --logdir=log/libero/spatial/diffusion_policy"
echo "  Logs: tail -f log/libero/spatial/diffusion_policy/<timestamp>/logs/*.txt"
echo ""
echo "Good luck! 🚀"



