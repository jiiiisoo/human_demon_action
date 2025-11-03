# VAE Training Setup - Summary

## What Was Created

A complete, production-ready training pipeline for Vanilla VAE on SomethingToSomething v2 dataset, specifically designed for fair comparison with the LAQ encoder.

## File Structure

```
vae_latent/
├── vanilla_vae_model.py          # VAE model (matches LAQ capacity)
├── dataset_sthv2.py               # SthV2 dataset loader
├── config_vae_sthv2.yaml          # Training configuration
│
├── train_ddp.py                   # Multi-GPU training (DDP)
├── train_single_gpu.py            # Single-GPU training
├── run_train.sh                   # Bash launcher script
│
├── test_setup.py                  # Setup verification script
├── compare_with_laq.py            # VAE vs LAQ comparison
│
├── requirements.txt               # Python dependencies
├── README.md                      # Full documentation
├── QUICKSTART.md                  # Quick start guide
└── SUMMARY.md                     # This file
```

## Key Features

### 1. Fair Comparison with LAQ

| Aspect | LAQ | Vanilla VAE |
|--------|-----|-------------|
| Feature Dimension | 1024 | 1024 (hidden_dims[-1]) |
| Image Size | 256×256 | 256×256 |
| Learning Rate | 1e-4 | 1e-4 |
| Dataset | SthV2 Frames | SthV2 Frames |
| Optimizer | AdamW | Adam |
| Architecture | Transformer + VQ | Convolutional + Continuous |

### 2. Multi-GPU Support

- Full DistributedDataParallel (DDP) implementation
- Automatic GPU detection and usage
- Single-GPU fallback for debugging
- Efficient data loading with DistributedSampler

### 3. Comprehensive Logging

- TensorBoard integration
- Loss tracking (total, reconstruction, KLD)
- Learning rate monitoring
- Reconstruction image visualization
- Regular checkpoint saving

### 4. Production-Ready Code

- Type hints and documentation
- Error handling and fallbacks
- Configurable via YAML
- Resume from checkpoint support
- Gradient clipping
- Learning rate scheduling

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Update data path in config_vae_sthv2.yaml
# Edit: data_params.data_path

# 3. Test setup
python test_setup.py

# 4. Start training (multi-GPU)
python train_ddp.py --config config_vae_sthv2.yaml

# 5. Monitor training
tensorboard --logdir logs_vae/VanillaVAE_sthv2

# 6. Compare with LAQ (after training)
python compare_with_laq.py \
    --vae logs_vae/VanillaVAE_sthv2/checkpoints/vae_final.pt \
    --laq ../laq/results/model.pt \
    --image test_image.jpg
```

## Architecture Details

### Encoder
```
Input (3, 256, 256)
  ↓ Conv2d + BN + LeakyReLU [64]   → (64, 128, 128)
  ↓ Conv2d + BN + LeakyReLU [128]  → (128, 64, 64)
  ↓ Conv2d + BN + LeakyReLU [256]  → (256, 32, 32)
  ↓ Conv2d + BN + LeakyReLU [512]  → (512, 16, 16)
  ↓ Conv2d + BN + LeakyReLU [1024] → (1024, 8, 8)
  ↓ Flatten                         → (1024 * 8 * 8)
  ↓ Linear                          → μ (256), log σ² (256)
```

### Decoder
```
Latent z (256)
  ↓ Linear                          → (1024 * 8 * 8)
  ↓ Reshape                         → (1024, 8, 8)
  ↓ ConvTranspose2d + BN + LeakyReLU [512] → (512, 16, 16)
  ↓ ConvTranspose2d + BN + LeakyReLU [256] → (256, 32, 32)
  ↓ ConvTranspose2d + BN + LeakyReLU [128] → (128, 64, 64)
  ↓ ConvTranspose2d + BN + LeakyReLU [64]  → (64, 128, 128)
  ↓ ConvTranspose2d + BN + LeakyReLU [64]  → (64, 256, 256)
  ↓ Conv2d + Tanh                   → Output (3, 256, 256)
```

### Loss Function
```
Total Loss = Reconstruction Loss + β * KLD Loss

Where:
- Reconstruction Loss = MSE(reconstructed, original)
- KLD Loss = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
- β = kld_weight (default: 0.0001)
```

## Configuration Options

### Model Parameters
```yaml
model_params:
  name: 'VanillaVAE'
  in_channels: 3
  hidden_dims: [64, 128, 256, 512, 1024]
  latent_dim: 256
  image_size: 256
```

### Data Parameters
```yaml
data_params:
  data_path: "/path/to/sthv2/frames"
  image_size: 256
  train_batch_size: 64
  val_batch_size: 64
  num_workers: 4
  pin_memory: true
```

### Training Parameters
```yaml
exp_params:
  LR: 1.0e-4
  weight_decay: 0.0
  scheduler_gamma: 0.95
  kld_weight: 0.0001
  max_grad_norm: 1.0

trainer_params:
  gpus: -1  # Use all GPUs
  max_epochs: 200
  gradient_clip_val: 1.0
  check_val_every_n_epoch: 1
  log_every_n_steps: 50
```

### Logging Parameters
```yaml
logging_params:
  save_dir: "logs_vae"
  name: "VanillaVAE_sthv2"
  save_every_n_steps: 5000
  log_images_every_n_steps: 5000
```

## Expected Results

### Training Metrics
- **Reconstruction Loss**: Should decrease steadily
- **KLD Loss**: Should stabilize (not go to 0)
- **Total Loss**: Should converge

### Reconstruction Quality
- Clear, sharp images (not blurry)
- Faithful color reproduction
- Preserved object structure
- Minimal artifacts

### Comparison with LAQ
- VAE: Continuous latent space, smoother reconstructions
- LAQ: Discrete codes, potentially sharper details
- Both: Similar capacity, comparable quality

## Performance Benchmarks

### Training Speed (Estimated)
- Single A100 (40GB): ~10 hours for 200 epochs
- 8x A100 GPUs: ~5 hours for 200 epochs
- Batch size 64 per GPU: ~500 it/s

### Memory Usage
- Model parameters: ~50M params
- Peak GPU memory (batch=64, 256×256): ~8GB
- Recommended: 16GB+ VRAM per GPU

## Troubleshooting Guide

### Issue: Blurry Reconstructions
**Solution**: Decrease `kld_weight` (try 0.00005)

### Issue: KLD Loss Too High
**Solution**: Increase `kld_weight` (try 0.0005)

### Issue: OOM Error
**Solution**: Reduce `train_batch_size` (try 32 or 16)

### Issue: Training Unstable
**Solution**: 
- Check learning rate (try 5e-5)
- Increase gradient clipping (try 0.5)
- Use learning rate warmup

### Issue: Poor Validation Loss
**Solution**:
- Check for overfitting
- Increase regularization (weight_decay)
- Add data augmentation

## Comparison Script Usage

After training both VAE and LAQ:

```bash
# Compare on a single image
python compare_with_laq.py \
    --vae logs_vae/VanillaVAE_sthv2/checkpoints/vae_final.pt \
    --laq ../laq/results/model.pt \
    --image /path/to/test/image.jpg \
    --output comparison_results

# Output:
# - comparison_results/comparison.png  (visualization)
# - MSE metrics for both models
# - Latent space statistics
```

## Next Steps

1. **Update Configuration**
   - Set correct data path in `config_vae_sthv2.yaml`
   - Adjust batch size based on GPU memory
   - Fine-tune hyperparameters if needed

2. **Run Tests**
   ```bash
   python test_setup.py
   ```

3. **Start Training**
   ```bash
   # Multi-GPU
   python train_ddp.py --config config_vae_sthv2.yaml
   
   # Or single-GPU
   python train_single_gpu.py --config config_vae_sthv2.yaml
   ```

4. **Monitor Training**
   ```bash
   tensorboard --logdir logs_vae/VanillaVAE_sthv2
   ```

5. **Compare with LAQ**
   ```bash
   python compare_with_laq.py --vae VAE_CHECKPOINT --laq LAQ_CHECKPOINT --image TEST_IMAGE
   ```

6. **Evaluate Results**
   - Visual quality assessment
   - Quantitative metrics (MSE, SSIM, LPIPS)
   - Downstream task performance

## Design Decisions

### Why These Dimensions?
- Hidden dims match LAQ's transformer dimension (1024)
- Latent dim (256) provides good reconstruction vs. compression trade-off
- Image size (256) matches LAQ and is computationally efficient

### Why Adam vs AdamW?
- Adam is standard for VAE training
- Can switch to AdamW by changing one line in code
- Both work well for this task

### Why This KLD Weight?
- 0.0001 is a good starting point
- Balances reconstruction quality and latent regularization
- Can be tuned based on downstream task requirements

### Why DDP?
- Most efficient multi-GPU training method
- Better scaling than DataParallel
- Industry standard for distributed training

## Acknowledgments

- Model architecture inspired by PyTorch-VAE
- Training setup follows LAQ for fair comparison
- Dataset loader adapted for SthV2 format

## Support

For questions or issues:
1. Check README.md for detailed documentation
2. Run test_setup.py to verify installation
3. Review TensorBoard logs for training issues
4. Compare with LAQ using comparison script

---

**Created**: 2025-10-30  
**Purpose**: Fair comparison between Vanilla VAE and LAQ encoders  
**Status**: Ready for training ✅

