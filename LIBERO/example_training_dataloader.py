"""
Example script showing how to use the extracted training data.

This demonstrates:
1. Loading the extracted data
2. Creating a PyTorch DataLoader
3. Basic data preprocessing
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os


class LIBERODataset(Dataset):
    """PyTorch Dataset for LIBERO training data."""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing extracted data (all_images.npy, all_actions.npy, metadata.json)
            transform: Optional transform to apply to images
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loading dataset: {self.metadata['language_instruction']}")
        print(f"Total frames: {self.metadata['total_frames']}")
        
        # Load images and actions
        self.images = np.load(os.path.join(data_dir, 'all_images.npy'))
        self.actions = np.load(os.path.join(data_dir, 'all_actions.npy'))
        
        print(f"Images shape: {self.images.shape}")
        print(f"Actions shape: {self.actions.shape}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and action
        image = self.images[idx]  # Shape: (H, W, 3)
        action = self.actions[idx]  # Shape: (7,)
        
        # Convert to PyTorch tensors
        # Normalize image to [0, 1]
        image = torch.from_numpy(image).float() / 255.0
        
        # Rearrange image from (H, W, C) to (C, H, W) for PyTorch
        image = image.permute(2, 0, 1)
        
        action = torch.from_numpy(action).float()
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'action': action,
            'language_instruction': self.metadata['language_instruction']
        }
    
    def get_demo_info(self):
        """Get information about individual demonstrations."""
        return self.metadata['demo_info']


def main():
    # Example usage
    data_dir = "./extracted_data"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run extract_training_data.py first to extract the data.")
        return
    
    # Create dataset
    dataset = LIBERODataset(data_dir)
    
    # Create dataloader
    batch_size = 32
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nDataLoader created with batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Example: iterate through one batch
    print("\n" + "="*80)
    print("Example batch:")
    print("="*80)
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image']  # Shape: (batch_size, 3, 128, 128)
        actions = batch['action']  # Shape: (batch_size, 7)
        
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"  Task: {batch['language_instruction'][0]}")
        
        # Only show first batch for demo
        break
    
    print("\n" + "="*80)
    print("Demo information:")
    print("="*80)
    demo_info = dataset.get_demo_info()
    print(f"Number of demonstrations: {len(demo_info)}")
    print(f"First demo: {demo_info[0]['demo_name']}")
    print(f"  Frames: {demo_info[0]['num_frames']}")
    print(f"  Frame indices: {demo_info[0]['start_idx']} to {demo_info[0]['end_idx']}")


if __name__ == "__main__":
    main()




























