# LIBERO Spatial Dataset Finetuning Guide

This guide explains how to finetune your DROID-trained diffusion policy model on the LIBERO Spatial dataset.

## Overview

The configuration has been set up to:
- Load your pretrained DROID model from epoch 100
- Finetune on all 10 LIBERO Spatial tasks
- Use appropriate observation and action spaces for LIBERO
- Apply lower learning rate for finetuning (1e-5 instead of 1e-4)

## Key Configuration Changes

### From DROID to LIBERO:

1. **Data Format**: Changed from `droid_rlds` to HDF5 format
2. **Observations**:
   - RGB: `agentview_rgb`, `eye_in_hand_rgb` (128x128 → auto-resized to 256x256)
   - Low-dim: `ee_pos`, `ee_ori`, `gripper_states`
3. **Actions**: 7-DOF actions in range [-1, 1] (down from DROID's 10-DOF)
4. **Checkpoint**: Loads from epoch 100 of your DROID training with `strict=False`
5. **Batch Size**: Reduced to 64 (from 2048) for smaller LIBERO dataset
6. **Sequence Length**: 10 (LIBERO standard)
7. **Learning Rate**: 1e-5 (finetuning rate)
8. **Skill Conditioning**: Enabled using DROID's pre-trained VAE

## Files Created

1. **Config**: `/home/jisookim/human_demon_action/droid_policy_learning/configs/libero_spatial_finetune.json`
2. **SLURM Script**: `/home/jisookim/human_demon_action/droid_policy_learning/slurm_train_libero.sh`
3. **Local Script**: `/home/jisookim/human_demon_action/droid_policy_learning/train_libero_local.sh`

## Dataset Structure

The config uses all 10 LIBERO Spatial tasks:
- pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
- pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
- pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
- pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
- pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
- pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
- pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
- pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
- pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
- pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate

Total transitions: ~59,000 across 500 demonstrations

## Usage

### Option 1: Using SLURM (Recommended for cluster)

```bash
cd /home/jisookim/human_demon_action/droid_policy_learning
sbatch slurm_train_libero.sh
```

Or with a custom config:
```bash
sbatch slurm_train_libero.sh /path/to/custom/config.json
```

### Option 2: Local Training (For testing or local GPU)

```bash
cd /home/jisookim/human_demon_action/droid_policy_learning
bash train_libero_local.sh
```

Or with a custom config:
```bash
bash train_libero_local.sh /path/to/custom/config.json
```

### Option 3: Direct Python Execution

```bash
cd /home/jisookim/human_demon_action
export PYTHONPATH=/home/jisookim/human_demon_action/droid_policy_learning:$PYTHONPATH
python droid_policy_learning/robomimic/scripts/train.py \
    --config droid_policy_learning/configs/libero_spatial_finetune.json
```

## Monitoring Training

### TensorBoard Logs
```bash
tensorboard --logdir=/home/jisookim/human_demon_action/droid_policy_learning/log/libero/spatial/diffusion_policy
```

### Output Directories
- **Models**: `log/libero/spatial/diffusion_policy/<timestamp>/models/`
- **Logs**: `log/libero/spatial/diffusion_policy/<timestamp>/logs/`
- **TensorBoard**: `log/libero/spatial/diffusion_policy/<timestamp>/logs/tb/`
- **Videos** (if enabled): `log/libero/spatial/diffusion_policy/<timestamp>/videos/`

## Customization Options

### To finetune on a subset of tasks:
Edit the `train.data` array in the config to include only desired HDF5 files.

### To change the checkpoint:
Update `experiment.ckpt_path` in the config to point to a different epoch:
```json
"ckpt_path": "/path/to/model_epoch_X.pth"
```

### To adjust training hyperparameters:
- `train.batch_size`: Adjust based on GPU memory
- `train.num_epochs`: Number of training epochs
- `algo.optim_params.policy.learning_rate.initial`: Finetuning learning rate
- `experiment.save.every_n_epochs`: Checkpoint frequency

### To enable rollouts:
Set `experiment.rollout.enabled` to `true` and configure LIBERO environment (requires additional setup)

## Expected Training Time

With 1 GPU:
- ~3-5 minutes per epoch
- 500 epochs: ~25-40 hours

## Troubleshooting

### Issue: Checkpoint loading with key mismatches
**What it means**: When loading the DROID checkpoint, you'll see messages about "Missing keys" and "Unexpected keys". This is **expected and normal** when finetuning across different observation keys and action dimensions.

**What happens**:
- ✅ Visual encoder weights (ResNet50) are loaded successfully
- ✅ Most of the model architecture transfers over
- ⚠️ Observation encoder layers for LIBERO keys are randomly initialized
- ⚠️ Action prediction head is randomly initialized (due to 10→7 DOF change)

**Verification**: Check the training log for:
```
Loading checkpoint with strict=False

[Checkpoint Loading] strict=False mode:
  Missing keys (will be randomly initialized): XX keys
  Unexpected keys (ignored from checkpoint): YY keys
```

See `CHECKPOINT_LOADING_FIX.md` for detailed explanation.

### Issue: Checkpoint path not found
**Solution**: Verify the checkpoint path exists and is accessible:
```bash
ls -lh /home/jisookim/human_demon_action/droid_policy_learning/log/droid/im/diffusion_policy/11-05-None/bz_2048_noise_samples_8_sample_weights_1_dataset_names_droid_cams_2cams_goalcams_2cams_goal_mode_offset_truncated_geom_factor_0.3_ldkeys_proprio-lang_visenc_VisualCore_fuser_None/20251106125113/models/model_epoch_100.pth
```

### Issue: Out of memory
**Solution**: Reduce `train.batch_size` in the config (try 32 or 16)

### Issue: Dataset not found
**Solution**: Verify LIBERO datasets exist:
```bash
ls -lh /mnt/data/libero/libero_spatial/
```

### Issue: Observation key mismatch
**Solution**: The training script will automatically detect observation keys from the HDF5 file. If there are issues, check the keys in your dataset using:
```bash
cd /home/jisookim/human_demon_action/LIBERO
python scripts/get_dataset_info.py --dataset /mnt/data/libero/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5
```

## Architecture Compatibility

The finetuning config preserves the following from your DROID model:
- ✅ Visual encoder architecture (ResNet50)
- ✅ Diffusion policy UNet architecture
- ✅ EMA settings
- ✅ DDIM scheduler settings
- ✅ Horizon settings (obs: 2, action: 8, pred: 16)

**Note**: Skill conditioning (VAE) is disabled for LIBERO finetuning as LIBERO doesn't use the same goal conditioning setup.

## Next Steps

After training completes:
1. Evaluate checkpoints on LIBERO environments
2. Compare performance across different finetuning epochs
3. Optionally train on other LIBERO suites (Object, Goal, 90, 10)

## Additional LIBERO Suites

To finetune on other LIBERO suites, create similar configs pointing to:
- **LIBERO Object**: `/mnt/data/libero/libero_object/`
- **LIBERO Goal**: `/mnt/data/libero/libero_goal/`
- **LIBERO 90**: `/mnt/data/libero/libero_90/`
- **LIBERO 10**: `/mnt/data/libero/libero_10/`

## Questions or Issues?

Check the main training script for debugging info:
- `/home/jisookim/human_demon_action/droid_policy_learning/robomimic/scripts/train.py`
- Training logs will show observation shapes and other details at startup

Good luck with your finetuning! 🚀


