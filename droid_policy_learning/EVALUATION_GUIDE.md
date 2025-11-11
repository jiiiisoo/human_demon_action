# LIBERO Evaluation Guide

This guide explains how to evaluate your finetuned diffusion policy on LIBERO environments.

## Prerequisites

1. **Install LIBERO**:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

2. **Download LIBERO datasets and assets**:
```bash
python libero/scripts/download_libero_datasets.py
```

## Evaluation Script

The evaluation script `evaluate_libero.py` loads your trained diffusion policy checkpoint and rolls it out in LIBERO environments.

### Key Features

- **6D Rotation Conversion**: Automatically converts policy's 6D rotation output to Euler angles for LIBERO
- **Skill Conditioning**: Loads goal images from demonstrations for skill-conditioned policies
- **Observation History**: Maintains observation history for temporal policies (observation_horizon=2)
- **Vectorized Evaluation**: Runs multiple episodes in parallel for faster evaluation

## Usage

### Evaluate on a Single Task

```bash
bash evaluate_libero.sh \
    --checkpoint log/libero/spatial/diffusion_policy/libero_spatial_finetune_from_droid/20251110093831/models/model_epoch_50.pth \
    --task_id 0
```

### Evaluate on All Tasks

```bash
bash evaluate_libero.sh \
    --checkpoint log/libero/spatial/diffusion_policy/libero_spatial_finetune_from_droid/20251110093831/models/model_epoch_50.pth
```

### Custom Evaluation

```bash
python evaluate_libero.py \
    --checkpoint <path_to_checkpoint> \
    --benchmark libero_spatial \
    --task_id 0 \
    --num_episodes 20 \
    --max_steps 300 \
    --device cuda
```

## Parameters

- `--checkpoint`: Path to the trained model checkpoint (.pth file)
- `--benchmark`: LIBERO benchmark to evaluate on
  - `libero_spatial`: LIBERO Spatial (10 tasks)
  - `libero_10`: LIBERO-10 (10 tasks)
  - `libero_object`: LIBERO Object (10 tasks)
  - `libero_goal`: LIBERO Goal (10 tasks)
- `--task_id`: Specific task ID (0-9) to evaluate, or omit to evaluate all tasks
- `--num_episodes`: Number of episodes per task (default: 20)
- `--max_steps`: Maximum steps per episode (default: 300)
- `--device`: Device to use (default: cuda)

## Output

### Console Output

```
[Evaluation] Task 0: Pick up the black bowl...
  Episodes: 20, Max steps: 300
  Loaded goal observation with keys: ['agentview_image', 'robot0_eye_in_hand_image']
  Step 50/300: 5/20 succeeded
  Step 100/300: 12/20 succeeded
  Step 150/300: 18/20 succeeded
  Result: 18/20 succeeded (90.0%)
```

### Results File

Evaluation results are automatically saved to:
```
<checkpoint_path>_eval_results.json
```

Example:
```json
{
  "benchmark": "libero_spatial",
  "num_episodes": 20,
  "results": {
    "0": 0.90,
    "1": 0.85,
    ...
    "9": 0.75
  },
  "average": 0.82
}
```

## Expected Success Rates

For LIBERO Spatial (from DROID finetuning):
- **Good performance**: 60-80% average success rate
- **Excellent performance**: >80% average success rate

Note: Success rates depend on:
- Amount of finetuning data
- Number of training epochs
- Quality of DROID pretraining
- Task difficulty

## Troubleshooting

### Import Errors

If you see `ImportError: No module named 'libero'`:
```bash
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

### CUDA Out of Memory

Reduce batch size by evaluating fewer episodes in parallel:
```bash
python evaluate_libero.py --checkpoint <path> --num_episodes 10
```

### Action Conversion Issues

If the robot behaves erratically, check:
1. Action normalization in config (`min_max` to [-1, 1])
2. 6D rotation conversion (should match training)
3. Gripper action range (0 = closed, 1 = open)

## Visualization

To visualize rollouts (requires additional setup):
```bash
python evaluate_libero.py \
    --checkpoint <path> \
    --task_id 0 \
    --save_videos \
    --video_folder eval_videos/
```

## Comparison with LIBERO Baselines

LIBERO baseline success rates (BC-RNN):
- LIBERO Spatial: ~40-60% (depending on task)
- LIBERO-10: ~50-70%

Your finetuned diffusion policy should aim to match or exceed these baselines.

## Next Steps

1. **Hyperparameter Tuning**: Adjust learning rate, batch size for better performance
2. **More Training Data**: Increase number of demonstrations per task
3. **Longer Training**: Train for more epochs (50-100)
4. **Ensemble**: Evaluate multiple checkpoints and ensemble predictions


