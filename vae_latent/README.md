# Vanilla VAE Training for SomethingToSomething v2

This folder contains code for training a Vanilla VAE on the SomethingToSomething v2 dataset for fair comparison with the LAQ style encoder.

## Architecture Configuration

The VAE is configured to match LAQ's capacity for fair comparison:
- **Hidden dims**: [64, 128, 256, 512, 1024] (final dim matches LAQ's 1024)
- **Latent dim**: 256
- **Image size**: 256x256 (same as LAQ)
- **Learning rate**: 1e-4 (same as LAQ)
- **Batch size**: 64 per GPU

## Files

- `vanilla_vae_model.py`: VAE model architecture
- `dataset_sthv2.py`: Dataset loader for SthV2 frames
- `train_ddp.py`: Multi-GPU training script with DDP
- `train_single_gpu.py`: Simple single-GPU training script
- `config_vae_sthv2.yaml`: Training configuration
- `requirements.txt`: Python dependencies

## Installation

```bash
cd /workspace/human_demon_action/vae_latent
pip install -r requirements.txt
```

## Usage

### Multi-GPU Training (Recommended)

Train with all available GPUs using DistributedDataParallel:

```bash
python train_ddp.py --config config_vae_sthv2.yaml
```

Resume from a checkpoint:

```bash
python train_ddp.py --config config_vae_sthv2.yaml --resume logs_vae/VanillaVAE_sthv2/checkpoints/vae_epoch_50.pt
```

### Single-GPU Training

For debugging or single-GPU training:

```bash
python train_single_gpu.py --config config_vae_sthv2.yaml
```

## Configuration

Edit `config_vae_sthv2.yaml` to customize:

- **Data path**: Update `data_params.data_path` to point to your SthV2 frames folder
- **Batch size**: Adjust `data_params.train_batch_size` based on GPU memory
- **Learning rate**: Modify `exp_params.LR`
- **KLD weight**: Adjust `exp_params.kld_weight` for reconstruction vs. regularization trade-off

## Monitoring

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir logs_vae/VanillaVAE_sthv2
```

Metrics logged:
- Training loss, reconstruction loss, KLD loss
- Validation loss, reconstruction loss, KLD loss
- Learning rate
- Sample reconstructions every N steps

## Checkpoints

Checkpoints are saved in `logs_vae/VanillaVAE_sthv2/checkpoints/`:
- Every N steps (configurable in config)
- At the end of each epoch
- Final model as `vae_final.pt`

## Comparison with LAQ

This VAE uses similar architecture capacity and training settings as LAQ:

| Parameter | LAQ | Vanilla VAE |
|-----------|-----|-------------|
| Dimension | 1024 | 1024 (hidden_dims[-1]) |
| Image Size | 256 | 256 |
| Learning Rate | 1e-4 | 1e-4 |
| Optimizer | AdamW | Adam |
| Dataset | SthV2 Frames | SthV2 Frames |

The main difference is the architecture: LAQ uses a transformer-based approach with VQ codes, while this VAE uses a convolutional encoder-decoder with continuous latent space.

## Expected Performance

The VAE should achieve reasonable reconstruction quality on natural images from SthV2. You can compare:
1. Reconstruction quality (visual inspection)
2. Latent space quality (e.g., for downstream tasks)
3. Training efficiency (time, memory)

## Tips

1. **Batch size**: Start with 64 and adjust based on GPU memory
2. **KLD weight**: Start with 0.0001 and adjust if reconstructions are blurry (decrease) or if KLD is too large (increase)
3. **Multi-GPU**: DDP scales efficiently to multiple GPUs
4. **Validation**: Check reconstructions regularly to ensure quality

