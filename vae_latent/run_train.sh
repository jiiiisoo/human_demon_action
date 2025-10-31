#!/bin/bash

# Training script for Vanilla VAE on SomethingToSomething v2
# Adjust parameters as needed

# Default parameters
CONFIG="config_vae_droid.yaml"
MODE="multi"  # "multi" or "single"
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG] [--mode multi|single] [--resume CHECKPOINT]"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "VAE Training Script"
echo "======================================"
echo "Config: $CONFIG"
echo "Mode: $MODE"
if [ -n "$RESUME" ]; then
    echo "Resume from: $RESUME"
fi
echo "======================================"
echo ""

# Run training based on mode
if [ "$MODE" == "multi" ]; then
    echo "Starting multi-GPU training..."
    if [ -n "$RESUME" ]; then
        python train_ddp.py --config "$CONFIG" --resume "$RESUME"
    else
        python train_ddp.py --config "$CONFIG"
    fi
elif [ "$MODE" == "single" ]; then
    echo "Starting single-GPU training..."
    if [ -n "$RESUME" ]; then
        python train_single_gpu.py --config "$CONFIG" --resume "$RESUME"
    else
        python train_single_gpu.py --config "$CONFIG"
    fi
else
    echo "Invalid mode: $MODE"
    echo "Use 'multi' or 'single'"
    exit 1
fi

