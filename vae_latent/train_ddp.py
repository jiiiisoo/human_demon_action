"""
Multi-GPU training script for Vanilla VAE on SomethingToSomething v2 dataset.
Supports DDP (DistributedDataParallel) for efficient multi-GPU training.
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from vanilla_vae_model import VanillaVAE
from dataset_sthv2 import SthV2FrameDataset


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scheduler, step, epoch, save_path):
    """Save model checkpoint."""
    # If using DDP, save only the module state dict
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'step': step,
        'epoch': epoch,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('step', 0), checkpoint.get('epoch', 0)


def train_one_epoch(
    model, train_loader, optimizer, scheduler, epoch, config, 
    rank, world_size, writer, global_step, log_dir
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.cuda(rank)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss_dict = model.module.loss_function(*outputs, kld_weight=config['exp_params']['kld_weight']) \
            if isinstance(model, DDP) else model.loss_function(*outputs, kld_weight=config['exp_params']['kld_weight'])
        
        loss = loss_dict['loss']
        recon_loss = loss_dict['Reconstruction_Loss']
        kld_loss = loss_dict['KLD']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['exp_params'].get('max_grad_norm', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['exp_params']['max_grad_norm'])
        
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kld_loss += kld_loss.item()
        
        global_step += 1
        
        # Logging
        if rank == 0:
            if isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kld': f'{kld_loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            if global_step % config['trainer_params']['log_every_n_steps'] == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
                writer.add_scalar('train/kld_loss', kld_loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            # Save checkpoint
            if global_step % config['logging_params']['save_every_n_steps'] == 0:
                checkpoint_path = os.path.join(log_dir, 'checkpoints', f'vae_step_{global_step}.pt')
                save_checkpoint(model, optimizer, scheduler, global_step, epoch, checkpoint_path)
                print(f"\nCheckpoint saved at step {global_step}")
    
    # Average losses
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kld_loss = total_kld_loss / num_batches
    
    return global_step, avg_loss, avg_recon_loss, avg_kld_loss


@torch.no_grad()
def validate(model, val_loader, epoch, config, rank, writer, global_step, log_dir):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    
    if rank == 0:
        pbar = tqdm(val_loader, desc=f"Validation")
    else:
        pbar = val_loader
    
    # Store some samples for visualization
    sample_images = []
    sample_recons = []
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.cuda(rank)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss_dict = model.module.loss_function(*outputs, kld_weight=config['exp_params']['kld_weight']) \
            if isinstance(model, DDP) else model.loss_function(*outputs, kld_weight=config['exp_params']['kld_weight'])
        
        loss = loss_dict['loss']
        recon_loss = loss_dict['Reconstruction_Loss']
        kld_loss = loss_dict['KLD']
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kld_loss += kld_loss.item()
        
        # Collect samples for visualization
        if batch_idx == 0 and rank == 0:
            sample_images = images[:8].cpu()
            sample_recons = outputs[0][:8].cpu()
    
    # Average losses
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kld_loss = total_kld_loss / num_batches
    
    if rank == 0:
        writer.add_scalar('val/loss', avg_loss, global_step)
        writer.add_scalar('val/recon_loss', avg_recon_loss, global_step)
        writer.add_scalar('val/kld_loss', avg_kld_loss, global_step)
        
        # Save reconstruction images
        if len(sample_images) > 0:
            # Denormalize from [-1, 1] to [0, 1]
            sample_images = (sample_images + 1) / 2
            sample_recons = (sample_recons + 1) / 2
            
            comparison = torch.cat([sample_images, sample_recons])
            img_grid = vutils.make_grid(comparison, nrow=8, normalize=False)
            
            recon_dir = os.path.join(log_dir, 'reconstructions')
            os.makedirs(recon_dir, exist_ok=True)
            vutils.save_image(img_grid, os.path.join(recon_dir, f'recon_epoch_{epoch}_step_{global_step}.png'))
            writer.add_image('val/reconstructions', img_grid, global_step)
        
        print(f"\nValidation - Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KLD: {avg_kld_loss:.4f}")
    
    return avg_loss, avg_recon_loss, avg_kld_loss


def train(rank, world_size, config, resume_from=None):
    """Main training function for each process."""
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Create log directory (only on rank 0)
    if rank == 0:
        log_dir = os.path.join(
            config['logging_params']['save_dir'],
            config['logging_params']['name']
        )
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'reconstructions'), exist_ok=True)
        
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Logging to {log_dir}")
    else:
        log_dir = None
        writer = None
    
    # Create datasets
    train_dataset = SthV2FrameDataset(
        config['data_params']['data_path'],
        image_size=config['data_params']['image_size'],
        split='train',
        val_split=0.05
    )
    
    val_dataset = SthV2FrameDataset(
        config['data_params']['data_path'],
        image_size=config['data_params']['image_size'],
        split='val',
        val_split=0.05
    )
    
    # Create samplers for DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data_params']['train_batch_size'],
        sampler=train_sampler,
        num_workers=config['data_params']['num_workers'],
        pin_memory=config['data_params']['pin_memory'],
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data_params']['val_batch_size'],
        sampler=val_sampler,
        num_workers=config['data_params']['num_workers'],
        pin_memory=config['data_params']['pin_memory'],
        drop_last=False
    )
    
    # Create model
    model = VanillaVAE(**config['model_params'])
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['exp_params']['LR'],
        weight_decay=config['exp_params']['weight_decay']
    )
    
    # Create scheduler
    scheduler = None
    if config['exp_params'].get('scheduler_gamma', None):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config['exp_params']['scheduler_gamma']
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if resume_from is not None:
        start_epoch, global_step = load_checkpoint(resume_from, model, optimizer, scheduler)
        if rank == 0:
            print(f"Resumed from checkpoint: {resume_from}")
            print(f"Starting from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    for epoch in range(start_epoch, config['trainer_params']['max_epochs']):
        train_sampler.set_epoch(epoch)
        
        # Train
        global_step, avg_loss, avg_recon_loss, avg_kld_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch,
            config, rank, world_size, writer, global_step, log_dir
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch} - Train Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KLD: {avg_kld_loss:.4f}")
        
        # Validate
        if (epoch + 1) % config['trainer_params']['check_val_every_n_epoch'] == 0:
            val_loss, val_recon_loss, val_kld_loss = validate(
                model, val_loader, epoch, config, rank, writer, global_step, log_dir
            )
        
        # Save checkpoint at end of epoch (only on rank 0)
        if rank == 0:
            checkpoint_path = os.path.join(log_dir, 'checkpoints', f'vae_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, scheduler, global_step, epoch, checkpoint_path)
    
    # Save final checkpoint
    if rank == 0:
        final_path = os.path.join(log_dir, 'checkpoints', 'vae_final.pt')
        save_checkpoint(model, optimizer, scheduler, global_step, config['trainer_params']['max_epochs'], final_path)
        writer.close()
        print(f"\nTraining completed! Final model saved to {final_path}")
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='Train Vanilla VAE on SthV2 with multi-GPU support')
    parser.add_argument('--config', type=str, default='config_vae_sthv2.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices available. Please run on GPU.")
        sys.exit(1)
    
    print(f"Training on {world_size} GPUs")
    print(f"Config: {args.config}")
    print(f"Model: {config['model_params']}")
    print(f"Data: {config['data_params']['data_path']}")
    
    # Launch training
    if world_size == 1:
        # Single GPU training
        train(0, 1, config, args.resume)
    else:
        # Multi-GPU training with spawn
        import torch.multiprocessing as mp
        mp.spawn(
            train,
            args=(world_size, config, args.resume),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    main()

