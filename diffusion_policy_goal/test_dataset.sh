#!/bin/bash

# Test dataset loading

source ~/miniconda3/etc/profile.d/conda.sh
conda activate human_demon

cd /home/jisookim/human_demon_action/diffusion_policy_goal

echo "Testing dataset..."
python dataset_droid.py


