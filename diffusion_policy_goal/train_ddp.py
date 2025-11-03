"""
Training script for goal-conditioned diffusion policy with DDP
Uses VAE-encoded goal latents as conditioning
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Diffusers for noise scheduling
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

from dataset_droid import DroidGoalDataset
from models.vae_encoder import VAEGoalEncoder
from models.obs_encoder import ResNetObsEncoder
from models.diffusion_nets import ConditionalUnet1D


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def normalize_actions(actions, action_stats):
    """Normalize actions to [-1, 1]"""
    mean = action_stats['mean']
    std = action_stats['std']
    return (actions - mean) / (std + 1e-8)


def denormalize_actions(actions, action_stats):
    """Denormalize actions from [-1, 1]"""
    mean = action_stats['mean']
    std = action_stats['std']
    return actions * (std + 1e-8) + mean


def compute_action_stats(dataloader, device):
    """Compute action statistics for normalization - memory efficient version"""
    # Use online/incremental statistics computation to avoid OOM
    n_samples = 0
    sum_actions = None
    sum_sq_actions = None
    action_dim = None
    
    for batch in tqdm(dataloader, desc="Computing action stats"):
        actions = batch['actions']  # (B, T, action_dim)
        actions = actions.reshape(-1, actions.shape[-1])  # (B*T, action_dim)
        
        if action_dim is None:
            action_dim = actions.shape[-1]
            sum_actions = torch.zeros(action_dim, device=device)
            sum_sq_actions = torch.zeros(action_dim, device=device)
        
        # Move to device and accumulate
        actions = actions.to(device)
        sum_actions += actions.sum(dim=0)
        sum_sq_actions += (actions ** 2).sum(dim=0)
        n_samples += actions.shape[0]
        
        # Free memory
        del actions
    
    # Compute mean and std
    mean = sum_actions / n_samples
    std = torch.sqrt(sum_sq_actions / n_samples - mean ** 2)
    
    return {'mean': mean, 'std': std}


def save_action_stats(action_stats, save_path):
    """Save action statistics to file"""
    torch.save({
        'mean': action_stats['mean'].cpu(),
        'std': action_stats['std'].cpu(),
    }, save_path)
    print(f"Action stats saved to {save_path}")


def load_action_stats(load_path, device):
    """Load action statistics from file"""
    stats = torch.load(load_path, map_location=device)
    print(f"Action stats loaded from {load_path}")
    return {
        'mean': stats['mean'].to(device),
        'std': stats['std'].to(device),
    }


def train_one_epoch(
    rank,
    epoch,
    model,
    goal_encoder,
    obs_encoder,
    dataloader,
    optimizer,
    noise_scheduler,
    ema,
    action_stats,
    config,
    writer=None,
    global_step=0,
):
    """Train for one epoch"""
    model.train()
    goal_encoder.eval()  # Keep goal encoder frozen
    obs_encoder.train()  # Train observation encoder
    
    total_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        goal_images = batch['goal_image'].cuda(rank)  # (B, 3, H, W)
        obs_images = batch['obs_image'].cuda(rank)  # (B, 3, H, W)
        actions = batch['actions'].cuda(rank)  # (B, action_horizon, action_dim)
        
        B = actions.shape[0]
        
        # Normalize actions to [-1, 1]
        actions = normalize_actions(actions, action_stats)
        
        # Encode goal image to latent (frozen VAE) - this is the skill representation zt
        with torch.no_grad():
            goal_latents = goal_encoder(goal_images)  # (B, latent_dim)
        
        # Encode observation image (trainable ResNet) - this is ot
        obs_features = obs_encoder(obs_images)  # (B, feature_dim)
        
        # Concatenate: [ot, zt] as global conditioning
        global_cond = torch.cat([obs_features, goal_latents], dim=1)  # (B, feature_dim + latent_dim)
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=actions.device
        ).long()
        
        # Add noise (forward diffusion)
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
        
        # Predict noise
        noise_pred = model(noisy_actions, timesteps, global_cond=global_cond)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.step(model.module if isinstance(model, DDP) else model)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})
            
            if writer is not None and global_step % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
        
        global_step += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, global_step


@torch.no_grad()
def validate(
    rank,
    epoch,
    model,
    goal_encoder,
    obs_encoder,
    dataloader,
    noise_scheduler,
    ema,
    action_stats,
    config,
    writer=None,
    global_step=0,
):
    """Validation"""
    model.eval()
    goal_encoder.eval()
    obs_encoder.eval()
    
    # Use EMA model if available
    eval_model = ema.averaged_model if ema is not None else model
    
    total_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Validation")
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for batch in pbar:
            goal_images = batch['goal_image'].cuda(rank)
            obs_images = batch['obs_image'].cuda(rank)
            actions = batch['actions'].cuda(rank)
            
            B = actions.shape[0]
            
            # Normalize actions
            actions = normalize_actions(actions, action_stats)
            
            # Encode goal and observation
            goal_latents = goal_encoder(goal_images)
            obs_features = obs_encoder(obs_images)
            global_cond = torch.cat([obs_features, goal_latents], dim=1)
            
            # Sample noise and timesteps
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=actions.device
            ).long()
            
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
            
            # Predict noise
            noise_pred = eval_model(noisy_actions, timesteps, global_cond=global_cond)
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if rank == 0 and writer is not None:
        writer.add_scalar('val/loss', avg_loss, global_step)
    
    return avg_loss


def train(rank, world_size, config):
    """Main training function"""
    setup_ddp(rank, world_size)
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    # TensorBoard
    writer = None
    if rank == 0:
        log_dir = output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    
    # Create datasets - rank 0 builds index, then broadcasts to all ranks
    if rank == 0:
        print("Rank 0: Building dataset index...")
        train_dataset_tmp = DroidGoalDataset(
            data_path=config['data']['data_path'],
            split='train',
            image_size=config['data']['image_size'],
            obs_horizon=config['model']['obs_horizon'],
            action_horizon=config['model']['action_horizon'],
            debug_max_episodes=config['data'].get('debug_max_episodes', None),
        )
        val_dataset_tmp = DroidGoalDataset(
            data_path=config['data']['data_path'],
            split='val',
            image_size=config['data']['image_size'],
            obs_horizon=config['model']['obs_horizon'],
            action_horizon=config['model']['action_horizon'],
            debug_max_episodes=config['data'].get('debug_max_episodes', None),
        )
        train_index = train_dataset_tmp.samples
        val_index = val_dataset_tmp.samples
    else:
        train_index = None
        val_index = None
    
    # Broadcast indices to all ranks
    if world_size > 1:
        print(f"Rank {rank}: Waiting for dataset index broadcast...")
        index_list = [train_index, val_index]
        dist.broadcast_object_list(index_list, src=0)
        train_index, val_index = index_list
        print(f"Rank {rank}: Received dataset index")
    
    # Create datasets with prebuilt index
    train_dataset = DroidGoalDataset(
        data_path=config['data']['data_path'],
        split='train',
        image_size=config['data']['image_size'],
        obs_horizon=config['model']['obs_horizon'],
        action_horizon=config['model']['action_horizon'],
        debug_max_episodes=config['data'].get('debug_max_episodes', None),
        prebuilt_index=train_index,
    )
    
    val_dataset = DroidGoalDataset(
        data_path=config['data']['data_path'],
        split='val',
        image_size=config['data']['image_size'],
        obs_horizon=config['model']['obs_horizon'],
        action_horizon=config['model']['action_horizon'],
        debug_max_episodes=config['data'].get('debug_max_episodes', None),
        prebuilt_index=val_index,
    )
    
    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    if rank == 0:
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
    
    # Load or compute action statistics (only on rank 0, then broadcast)
    action_stats_path = output_dir / 'action_stats_train.pt'
    
    if rank == 0:
        print(f"Loading action statistics from {action_stats_path}...")
        action_stats = load_action_stats(action_stats_path, torch.device(f'cuda:{rank}'))
        
        print(f"Action mean: {action_stats['mean']}")
        print(f"Action std: {action_stats['std']}")
    else:
        action_stats = {'mean': None, 'std': None}
    
    # Broadcast action stats to all ranks
    if world_size > 1:
        if rank == 0:
            # Move to CPU for broadcasting
            mean_std_list = [action_stats['mean'].cpu(), action_stats['std'].cpu()]
        else:
            mean_std_list = [None, None]
        
        dist.broadcast_object_list(mean_std_list, src=0)
        
        if rank != 0:
            action_stats = {'mean': mean_std_list[0], 'std': mean_std_list[1]}
    
    # Move action stats to each rank's device
    action_stats['mean'] = action_stats['mean'].cuda(rank)
    action_stats['std'] = action_stats['std'].cuda(rank)
    
    if rank == 0:
        print(f"Action stats moved to cuda:{rank}")
    
    # Create goal encoder (frozen VAE)
    goal_encoder = VAEGoalEncoder(
        vae_checkpoint_path=config['model']['vae_checkpoint_path'],
        latent_dim=config['model']['latent_dim'],
        freeze_encoder=True,
    ).cuda(rank)
    
    # Create observation encoder (trainable ResNet with SpatialSoftmax - robomimic style)
    obs_encoder = ResNetObsEncoder(
        image_channels=3,
        image_size=config['data']['image_size'],
        num_kp=config['model'].get('num_kp', 32),  # SpatialSoftmax keypoints
        feature_dim=config['model']['obs_feature_dim'],
        pretrained=True,
    ).cuda(rank)
    
    # Wrap obs_encoder with DDP
    obs_encoder = DDP(obs_encoder, device_ids=[rank], find_unused_parameters=False)
    
    # Calculate total global conditioning dimension: [ot, zt]
    global_cond_dim = config['model']['obs_feature_dim'] + config['model']['latent_dim']
    
    if rank == 0:
        print(f"Global conditioning dimension: {global_cond_dim}")
        print(f"  - Obs features (ot): {config['model']['obs_feature_dim']}")
        print(f"  - Goal latent (zt): {config['model']['latent_dim']}")
    
    # Create diffusion model
    model = ConditionalUnet1D(
        input_dim=config['model']['action_dim'],
        global_cond_dim=global_cond_dim,  # Goal latent + obs features
        diffusion_step_embed_dim=config['model']['diffusion_step_embed_dim'],
        down_dims=config['model']['down_dims'],
        kernel_size=config['model']['kernel_size'],
        n_groups=config['model']['n_groups'],
    ).cuda(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Create noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config['diffusion']['num_train_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        clip_sample=config['diffusion']['clip_sample'],
        set_alpha_to_one=config['diffusion']['set_alpha_to_one'],
        steps_offset=config['diffusion']['steps_offset'],
        prediction_type=config['diffusion']['prediction_type'],
    )
    
    # Create EMA
    ema = None
    if config['training']['use_ema']:
        ema = EMAModel(
            model.module if isinstance(model, DDP) else model,
            power=config['training']['ema_power'],
        )
    
    # Create optimizer (train both diffusion model and obs encoder)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(obs_encoder.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, global_step = train_one_epoch(
            rank=rank,
            epoch=epoch,
            model=model,
            goal_encoder=goal_encoder,
            obs_encoder=obs_encoder,
            dataloader=train_loader,
            optimizer=optimizer,
            noise_scheduler=noise_scheduler,
            ema=ema,
            action_stats=action_stats,
            config=config,
            writer=writer,
            global_step=global_step,
        )
        
        # Validate
        val_loss = validate(
            rank=rank,
            epoch=epoch,
            model=model,
            goal_encoder=goal_encoder,
            obs_encoder=obs_encoder,
            dataloader=val_loader,
            noise_scheduler=noise_scheduler,
            ema=ema,
            action_stats=action_stats,
            config=config,
            writer=writer,
            global_step=global_step,
        )
        
        if rank == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % config['training']['save_every'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'obs_encoder_state_dict': obs_encoder.module.state_dict() if isinstance(obs_encoder, DDP) else obs_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_state_dict': ema.averaged_model.state_dict() if ema is not None else None,
                    'action_stats': action_stats,
                    'config': config,
                }
                
                checkpoint_path = output_dir / 'checkpoints' / f'diffusion_epoch_{epoch+1}.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_state_dict': ema.averaged_model.state_dict() if ema is not None else None,
                    'action_stats': action_stats,
                    'config': config,
                }
                
                best_path = output_dir / 'checkpoints' / 'best_model.pt'
                torch.save(checkpoint, best_path)
                print(f"Saved best model to {best_path}")
    
    if rank == 0 and writer is not None:
        writer.close()
    
    cleanup_ddp()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get world size
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    if world_size > 1:
        mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train(0, 1, config)


if __name__ == '__main__':
    main()

