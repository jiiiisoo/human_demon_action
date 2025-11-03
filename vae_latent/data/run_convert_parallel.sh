#!/bin/bash

# Parallel DROID dataset conversion script
# Run 8 processes in parallel to speed up conversion

WORLD_SIZE=8
DATA_PATH="gs://gresearch/robotics/droid/1.0.0"
OUTPUT_DIR="/mnt/data/droid/droid_local"
MAX_EPISODES=${1:-""}  # Optional: pass max episodes as first argument
IMAGE_KEYS="exterior_image_1_left"

echo "======================================================================"
echo "Parallel DROID Dataset Conversion"
echo "======================================================================"
echo "World size: $WORLD_SIZE"
echo "Output: $OUTPUT_DIR"
echo "Max episodes per process: ${MAX_EPISODES:-All}"
echo ""

# Activate conda environment
source /home/jisookim/miniconda3/etc/profile.d/conda.sh
conda activate human_demon

# Create log directory
LOG_DIR="logs_conversion"
mkdir -p $LOG_DIR

# Launch parallel processes
for rank in $(seq 0 $((WORLD_SIZE-1))); do
    echo "Launching rank $rank..."
    
    if [ -z "$MAX_EPISODES" ]; then
        # No max episodes limit
        python data/convert_droid_to_local.py \
            --data_path $DATA_PATH \
            --output $OUTPUT_DIR \
            --splits train \
            --image_keys $IMAGE_KEYS \
            --rank $rank \
            --world_size $WORLD_SIZE \
            > $LOG_DIR/rank_${rank}.log 2>&1 &
    else
        # With max episodes limit
        python data/convert_droid_to_local.py \
            --data_path $DATA_PATH \
            --output $OUTPUT_DIR \
            --max_episodes $MAX_EPISODES \
            --splits train \
            --image_keys $IMAGE_KEYS \
            --rank $rank \
            --world_size $WORLD_SIZE \
            > $LOG_DIR/rank_${rank}.log 2>&1 &
    fi
    
    PID=$!
    echo "  Rank $rank started (PID: $PID)"
    sleep 2  # Stagger launches slightly
done

echo ""
echo "======================================================================"
echo "All $WORLD_SIZE processes launched!"
echo "======================================================================"
echo "Monitor progress with:"
echo "  tail -f $LOG_DIR/rank_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep convert_droid_to_local"
echo ""
echo "Wait for all processes:"
echo "  wait"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background jobs
wait

echo ""
echo "======================================================================"
echo "✅ All conversions complete!"
echo "======================================================================"

# Aggregate statistics
echo ""
echo "Aggregating statistics..."
python - <<EOF
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR") / "train"
total_episodes = 0
total_frames = 0

for rank in range($WORLD_SIZE):
    info_file = output_dir / f"split_info_rank{rank}.json"
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
            total_episodes += info['num_episodes']
            total_frames += info['num_frames']
            print(f"  Rank {rank}: {info['num_episodes']} episodes, {info['num_frames']} frames")

print(f"\nTotal: {total_episodes} episodes, {total_frames} frames")
print(f"Output directory: $OUTPUT_DIR")

# Save combined info
combined_info = {
    'total_episodes': total_episodes,
    'total_frames': total_frames,
    'world_size': $WORLD_SIZE,
    'image_keys': ['$IMAGE_KEYS']
}
with open(output_dir / 'split_info_combined.json', 'w') as f:
    json.dump(combined_info, f, indent=2)
EOF

echo ""
echo "Done! Dataset saved to: $OUTPUT_DIR"

