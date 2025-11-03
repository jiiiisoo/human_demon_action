"""
Precompute action statistics for normalization
This avoids OOM during training initialization
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob


def compute_action_stats_from_files(data_path, split='train'):
    """Compute action statistics directly from .npy files - memory efficient
    
    Note: We use ONLY train split to compute statistics to avoid data leakage.
    This is the standard practice in ML.
    """
    # Find all action files (we need to split manually like the dataset does)
    train_dir = Path(data_path) / 'train'
    all_action_files = sorted(glob(str(train_dir / 'episode_*' / 'actions.npy')))
    
    # Apply 90/10 split (same as dataset)
    num_total = len(all_action_files)
    split_idx = int(num_total * 0.9)
    
    if split == 'train':
        action_files = all_action_files[:split_idx]
    elif split == 'val':
        action_files = all_action_files[split_idx:]
    else:
        raise ValueError(f"Invalid split: {split}")
    
    print(f"Found {num_total} total episodes")
    print(f"Using {len(action_files)} episodes for {split} split")
    
    # Use online/incremental statistics computation to avoid OOM
    n_samples = 0
    sum_actions = None
    sum_sq_actions = None
    action_dim = None
    
    for action_file in tqdm(action_files, desc=f"Computing action stats ({split})"):
        # Load actions from file
        actions = np.load(action_file)  # (T, action_dim)
        
        if action_dim is None:
            action_dim = actions.shape[-1]
            sum_actions = np.zeros(action_dim, dtype=np.float64)
            sum_sq_actions = np.zeros(action_dim, dtype=np.float64)
        
        # Accumulate statistics
        sum_actions += actions.sum(axis=0)
        sum_sq_actions += (actions ** 2).sum(axis=0)
        n_samples += actions.shape[0]
    
    # Compute mean and std
    mean = sum_actions / n_samples
    std = np.sqrt(sum_sq_actions / n_samples - mean ** 2)
    
    # Convert to torch tensors
    return {
        'mean': torch.from_numpy(mean).float(),
        'std': torch.from_numpy(std).float()
    }


def save_action_stats(action_stats, save_path):
    """Save action statistics to file"""
    torch.save({
        'mean': action_stats['mean'].cpu(),
        'std': action_stats['std'].cpu(),
    }, save_path)
    print(f"Action stats saved to {save_path}")


def main():
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['data_path']
    print(f"Data path: {data_path}")
    
    # Compute action statistics directly from .npy files
    # IMPORTANT: Always use 'train' split to avoid data leakage
    # Val/test will use the same train statistics for normalization
    print("Computing action statistics from .npy files (train split only)...")
    action_stats = compute_action_stats_from_files(data_path, split='train')
    
    print(f"\nAction statistics (from train split):")
    print(f"  Mean: {action_stats['mean']}")
    print(f"  Std:  {action_stats['std']}")
    
    # Save to output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'action_stats.pt'
    save_action_stats(action_stats, save_path)
    
    print(f"\nNote: This statistics will be used for BOTH train and val normalization.")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

