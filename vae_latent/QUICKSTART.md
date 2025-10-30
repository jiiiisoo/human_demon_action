# Quick Start Guide - Vanilla VAE Training

## Overview

This folder contains a complete training setup for Vanilla VAE on SomethingToSomething v2 dataset, configured for fair comparison with LAQ encoder.

## Key Features

✅ **Multi-GPU Training**: Full DDP support for efficient multi-GPU training  
✅ **LAQ-Matched Configuration**: Similar dimensions and settings to LAQ for fair comparison  
✅ **Comprehensive Logging**: TensorBoard logging with reconstruction visualizations  
✅ **Checkpoint Management**: Automatic saving and resuming from checkpoints  
✅ **Easy Configuration**: YAML-based configuration system  

## Files Created

### Core Files
- `vanilla_vae_model.py` - VAE model architecture
- `dataset_sthv2.py` - Dataset loader for SthV2 frames
- `config_vae_sthv2.yaml` - Training configuration

### Training Scripts
- `train_ddp.py` - Multi-GPU training with DDP (recommended)
- `train_single_gpu.py` - Single-GPU training
- `run_train.sh` - Bash script for easy launching

### Utility Scripts
- `test_setup.py` - Test script to verify setup
- `compare_with_laq.py` - Compare VAE and LAQ encoders
- `requirements.txt` - Python dependencies

### Documentation
- `README.md` - Comprehensive documentation
- `QUICKSTART.md` - This file

## Installation

```bash
cd /workspace/human_demon_action/vae_latent
pip install -r requirements.txt
```

## Configuration

Before training, update the data path in `config_vae_sthv2.yaml`:

```yaml
data_params:
  data_path: "/workspace/dataset/something_to_something/frames_full"  # Update this
```

## Testing Setup

Verify everything is working:

```bash
python test_setup.py
```

This will test:
- Model instantiation and forward pass
- Dataset loading
- CUDA availability
- Configuration file

## Training

### Multi-GPU Training (Recommended)

```bash
# Using all available GPUs
python train_ddp.py --config config_vae_sthv2.yaml

# Or use the bash script
bash run_train.sh --mode multi
```

### Single-GPU Training

```bash
python train_single_gpu.py --config config_vae_sthv2.yaml

# Or use the bash script
bash run_train.sh --mode single
```

### Resume from Checkpoint

```bash
python train_ddp.py --config config_vae_sthv2.yaml --resume logs_vae/VanillaVAE_sthv2/checkpoints/vae_epoch_50.pt
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs_vae/VanillaVAE_sthv2
```

Open http://localhost:6006 in your browser.

### Metrics Tracked

- Training loss, reconstruction loss, KLD loss
- Validation loss, reconstruction loss, KLD loss
- Learning rate
- Sample reconstructions (images)

## Output Structure

```
logs_vae/VanillaVAE_sthv2/
├── events.out.tfevents.*          # TensorBoard logs
├── checkpoints/
│   ├── vae_step_5000.pt           # Checkpoint every N steps
│   ├── vae_epoch_0.pt             # Checkpoint every epoch
│   ├── vae_epoch_1.pt
│   └── vae_final.pt               # Final model
└── reconstructions/
    ├── recon_epoch_0_step_5000.png
    └── recon_epoch_1_step_10000.png
```

## Comparing with LAQ

After training both models:

```bash
python compare_with_laq.py \
    --vae logs_vae/VanillaVAE_sthv2/checkpoints/vae_final.pt \
    --laq ../laq/results/model.pt \
    --image /path/to/test/image.jpg \
    --output comparison_results
```

This will generate a comparison visualization showing:
- Original image
- VAE reconstruction
- LAQ reconstruction
- MSE metrics for each

## Configuration Matching LAQ

| Parameter | LAQ | VAE |
|-----------|-----|-----|
| Feature Dimension | 1024 | 1024 (hidden_dims[-1]) |
| Image Size | 256x256 | 256x256 |
| Learning Rate | 1e-4 | 1e-4 |
| Batch Size | 100 | 64 (adjustable) |
| Optimizer | AdamW | Adam |
| Dataset | SthV2 Frames | SthV2 Frames |

## Hyperparameter Tuning

### If reconstructions are blurry:
- Decrease `kld_weight` (e.g., 0.00005 instead of 0.0001)
- Increase model capacity (add more hidden dims)

### If KLD is too large:
- Increase `kld_weight` (e.g., 0.0005 instead of 0.0001)

### If training is slow:
- Increase batch size (if GPU memory allows)
- Use gradient accumulation
- Reduce validation frequency

### If overfitting:
- Add weight decay
- Use data augmentation
- Early stopping

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size in config
- Use gradient checkpointing
- Reduce model size (fewer hidden dims)

### Dataset not found
- Check data path in config
- Ensure SthV2 frames are extracted

### Multi-GPU issues
- Check `nvidia-smi` for GPU availability
- Ensure NCCL is properly installed
- Try single-GPU training first

## Tips for Best Results

1. **Start with pretrained weights**: If available, initialize from pretrained weights
2. **Monitor reconstructions**: Check reconstruction quality regularly
3. **Compare metrics**: Compare with LAQ at similar training steps
4. **Use validation set**: Monitor validation metrics to avoid overfitting
5. **Learning rate**: Use learning rate warmup for stable training

## Expected Training Time

On 8x A100 GPUs with batch size 64 per GPU:
- ~5 hours for 200 epochs on SthV2 dataset
- Checkpoint saves every 5000 steps
- Validation runs every epoch

## Next Steps

After training:

1. **Evaluate reconstruction quality**
   - Visual inspection of reconstructions
   - Quantitative metrics (MSE, SSIM, LPIPS)

2. **Compare with LAQ**
   - Use `compare_with_laq.py` script
   - Compare latent space quality

3. **Use for downstream tasks**
   - Action recognition
   - Video prediction
   - Representation learning

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Verify setup with test_setup.py
3. Check TensorBoard logs for training issues
4. Review config file for parameter mismatches

## Citation

If you use this code, please cite:
- PyTorch-VAE: https://github.com/AntixK/PyTorch-VAE
- Original LAQ paper/code

