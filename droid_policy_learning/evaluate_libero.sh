#!/bin/bash

# Evaluation script for LIBERO finetuned diffusion policy

# Set environment
source /home/jisookim/miniconda3/bin/activate human_demon

# Default parameters
CHECKPOINT="log/libero/spatial/diffusion_policy/libero_spatial_finetune_from_droid/20251110093831/models/model_epoch_50.pth"
BENCHMARK="libero_spatial"
TASK_ID=""  # Empty means evaluate all tasks
NUM_EPISODES=20
MAX_STEPS=300
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --task_id)
            TASK_ID="--task_id $2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "==========================================="
echo "LIBERO Evaluation"
echo "==========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Benchmark: $BENCHMARK"
echo "Episodes per task: $NUM_EPISODES"
echo "Max steps: $MAX_STEPS"
echo "==========================================="

python evaluate_libero.py \
    --checkpoint "$CHECKPOINT" \
    --benchmark "$BENCHMARK" \
    $TASK_ID \
    --num_episodes $NUM_EPISODES \
    --max_steps $MAX_STEPS \
    --device $DEVICE

echo "Evaluation complete!"

