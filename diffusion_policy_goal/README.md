# Goal-Conditioned Diffusion Policy

Goal-conditioned diffusion policy using VAE-encoded goal latents as conditioning.

## 📋 Overview

This implementation follows the UniSkill paper's approach:
- **Goal Conditioning**: Uses VAE encoder to extract latent representation from goal images
- **Diffusion Policy**: 1D UNet for action trajectory generation
- **DDIM Scheduler**: 20 denoising steps for fast inference
- **Action Horizon**: Predicts 16 timesteps, executes 8 in open-loop

## 🏗️ Architecture

```
Goal Image (128x128)
    ↓
VAE Encoder (frozen)
    ↓
Goal Latent (64-dim)
    ↓ (concatenated with timestep embedding)
Conditional UNet 1D
    ↓
Action Trajectory (16 timesteps, 7-dim)
```

### Key Components

1. **VAE Goal Encoder** (`models/vae_encoder.py`)
   - Loads pretrained VAE from `vae_uniskill`
   - Extracts 64-dim latent representation
   - Frozen during diffusion policy training

2. **Conditional UNet 1D** (`models/diffusion_nets.py`)
   - Based on robomimic's implementation
   - FiLM conditioning with goal latent + timestep
   - Down dims: [256, 512, 1024]

3. **Dataset** (`dataset_droid.py`)
   - Loads from local DROID dataset
   - Returns: goal image, observation sequence, action trajectory
   - 90/10 train/val split

## 📂 Project Structure

```
diffusion_policy_goal/
├── models/
│   ├── __init__.py
│   ├── vae_encoder.py          # VAE encoder wrapper
│   └── diffusion_nets.py       # UNet architecture
├── dataset_droid.py             # Dataset loader
├── train_ddp.py                 # Training script (DDP)
├── config.yaml                  # Full training config
├── config_debug.yaml            # Debug config (50 episodes)
├── run_train.sh                 # Manual training script
├── slurm_train.sh              # SLURM job script
├── test_dataset.sh             # Test dataset loading
└── README.md
```

## 🚀 Quick Start

### 1. Test Dataset

```bash
./test_dataset.sh
```

Expected output:
```
Found 76000 total episode directories
Using 68400 episodes for train split
Indexed 68400 episodes, XXXXXX samples
Dataset size: XXXXXX

Sample 0:
  Goal image: torch.Size([3, 128, 128])
  Obs images: torch.Size([2, 3, 128, 128])
  Actions: torch.Size([16, 7])
```

### 2. Precompute Action Statistics (IMPORTANT!)

**Before training**, you must compute action statistics to avoid OOM:

```bash
# Manual (single GPU)
./run_compute_stats.sh

# SLURM (recommended)
sbatch slurm_compute_stats.sh
```

This will create `outputs/action_stats.pt` which will be loaded during training.

**Why?** Computing action statistics requires iterating through the entire dataset, which can cause OOM during multi-GPU training initialization. By precomputing, we:
- Avoid OOM errors during training startup
- Speed up training initialization
- Only need to compute once

### 3. Debug Training (50 episodes)

```bash
./run_train.sh config_debug.yaml
```

### 4. Full Training

```bash
# Manual
./run_train.sh config.yaml

# SLURM
sbatch slurm_train.sh config.yaml
```

## ⚙️ Configuration

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `image_size` | 128 | Image resolution (as in UniSkill) |
| `obs_horizon` | 2 | Number of observation frames |
| `action_horizon` | 16 | Number of actions to predict |
| `latent_dim` | 64 | VAE latent dimension |
| `num_train_timesteps` | 100 | DDPM training steps |
| `num_inference_timesteps` | 20 | DDIM inference steps |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `ema_power` | 0.75 | EMA decay rate |

### Config Files

- **`config.yaml`**: Full training (all episodes)
- **`config_debug.yaml`**: Debug mode (50 episodes, 20 epochs)

## 📊 Training Details

### Data Preprocessing

1. **Images**: Resized to 128×128
2. **Actions**: Normalized to [-1, 1] using dataset statistics
3. **Goal**: Last frame of each episode

### Training Process

1. **Goal Encoding**: VAE encoder (frozen) → 64-dim latent
2. **Forward Diffusion**: Add noise to clean actions
3. **Noise Prediction**: UNet predicts noise conditioned on goal latent
4. **Loss**: MSE between predicted and true noise
5. **EMA**: Exponential moving average for stable inference

### Inference

```python
# Pseudocode
goal_latent = vae_encoder(goal_image)  # (1, 64)
noisy_action = randn(1, 16, 7)         # Random noise

for t in reversed(range(20)):  # 20 DDIM steps
    noise_pred = unet(noisy_action, t, goal_latent)
    noisy_action = ddim_step(noisy_action, noise_pred, t)

clean_action = noisy_action  # (1, 16, 7)
```

## 📈 Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/logs --port 6010
```

Metrics:
- `train/loss`: Training MSE loss
- `val/loss`: Validation MSE loss

### Check Logs

```bash
# SLURM logs
tail -f logs/diffusion_*.out
tail -f logs/diffusion_*.err

# Find latest job
ls -lt logs/ | head
```

## 🔧 Troubleshooting

### Dataset Issues

**Problem**: "No valid samples found"
```bash
# Check episode structure
ls /mnt/data/droid/droid_local/train/episode_0/
# Should see: actions.npy, frames/, metadata.json, etc.
```

**Problem**: Action dimension mismatch
```bash
# Check action shape
python -c "import numpy as np; print(np.load('/mnt/data/droid/droid_local/train/episode_0/actions.npy').shape)"
```

### VAE Checkpoint

**Problem**: VAE checkpoint not found
```bash
# Check if checkpoint exists
ls /home/jisookim/human_demon_action/vae_uniskill/checkpoints/

# Update config.yaml with correct path
```

### Memory Issues

**Problem**: OOM during action stats computation
```bash
# Make sure you precompute action stats BEFORE training
sbatch slurm_compute_stats.sh
```

**Problem**: OOM during training
- Reduce `batch_size` in config
- Reduce `down_dims` in model config
- Use fewer GPUs
- Make sure action stats are precomputed (see step 2 above)

## 📝 Implementation Notes

### Differences from UniSkill Paper

1. **Observation Encoding**: Currently only uses goal latent
   - UniSkill: ResNet visual encoder + MLP for observations
   - Ours: Only goal latent (can be extended)

2. **Action Execution**: Paper executes 8 steps open-loop
   - Our implementation predicts 16, can execute first 8

3. **Image Size**: 128×128 (same as paper)

### Future Extensions

1. **Add Observation Encoding**:
   ```python
   # In train_ddp.py
   obs_features = obs_encoder(obs_images)  # (B, obs_horizon, obs_dim)
   obs_features = obs_features.flatten(1)   # (B, obs_horizon * obs_dim)
   global_cond = torch.cat([goal_latents, obs_features], dim=-1)
   ```

2. **Add Language Conditioning**:
   ```python
   language_emb = language_encoder(language_text)  # (B, lang_dim)
   global_cond = torch.cat([goal_latents, obs_features, language_emb], dim=-1)
   ```

3. **Receding Horizon Control**:
   - Implement action queue (as in robomimic)
   - Execute 8 actions, then re-plan

## 📚 References

- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [UniSkill Paper](https://arxiv.org/abs/2410.00534)
- [Robomimic](https://robomimic.github.io/)
- [DROID Dataset](https://droid-dataset.github.io/)

## 🎯 Expected Results

After training:
- Train loss should decrease to ~0.01-0.05
- Val loss should be similar to train loss
- EMA model should have lower loss than non-EMA

Checkpoints saved to:
- `outputs/checkpoints/diffusion_epoch_X.pt`
- `outputs/checkpoints/best_model.pt`


