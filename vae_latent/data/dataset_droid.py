"""
Dataset loader for Droid frames.
Compatible with PyTorch DataLoader and DDP training.
Based on uniskill BaseDataset structure but adapted for VAE single-frame training.
"""
import os
import random
from pathlib import Path
from typing import Union, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm


class DroidFrameDataset(Dataset):
    """
    Load single images from Droid dataset using TFDS for VAE reconstruction.
    Returns a tensor of shape [C, H, W] and a dummy label for compatibility.
    
    Based on uniskill's BaseDataset structure:
    - Uses 90:10 train/val split (same as uniskill)
    - Loads all individual frames for VAE training
    """

    def __init__(
        self,
        data_path: str,
        image_size: Union[int, Tuple[int, int]] = 256,
        train: bool = True,
        image_key: str = 'exterior_image_1_left'
    ):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.image_key = image_key
        
        # Load TFDS builder
        builder = tfds.builder_from_directory(builder_dir=data_path)
        
        # Determine split (90:10 same as uniskill)
        if "val" not in builder.info.splits:
            if train:
                tfds_split = "train[:90%]"
            else:
                tfds_split = "train[90%:]"
        else:
            tfds_split = "train" if train else "val"
        
        # Load dataset
        ds = builder.as_dataset(split=tfds_split)
        
        # Collect all frames from all episodes
        split_name = "train" if train else "val"
        self.frames = []
        for episode in tqdm(ds, desc=f"Loading {split_name} split"):
            steps = episode['steps']
            for step in steps:
                # Store the image tensor directly
                self.frames.append(step['observation'][self.image_key])
        
        print(f"Loaded {len(self.frames)} frames for {split_name} split")
        
        # Image transforms (similar to uniskill's image_transforms + fdm_normalize)
        # Normalize to [-1, 1] to match Tanh output
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: torch.Tensor of shape [C, H, W]
            label: dummy label (0) for compatibility with training loop
        """
        try:
            # Get the tensorflow tensor
            img_tensor = self.frames[index]
            
            # Convert to numpy and then PIL Image
            img_np = img_tensor.numpy()
            img = Image.fromarray(img_np)
            
            # Apply transforms
            img_tensor = self.transform(img)
            
            return img_tensor, 0  # Return dummy label for compatibility
            
        except Exception as e:
            # Fallback to another sample if this one fails
            print(f"Error loading frame {index}: {e}")
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


def get_dataloaders(
    data_path: str,
    image_size: int = 256,
    train_batch_size: int = 64,
    val_batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Create train and validation dataloaders.
    
    Args:
        data_path: Path to the TFDS droid dataset
        image_size: Image size to resize to
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = DroidFrameDataset(
        data_path=data_path,
        image_size=image_size,
        train=True,
    )
    
    val_dataset = DroidFrameDataset(
        data_path=data_path,
        image_size=image_size,
        train=False,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader

