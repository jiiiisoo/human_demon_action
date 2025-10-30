"""
Dataset loader for SomethingToSomething v2 frames.
Compatible with PyTorch DataLoader and DDP training.
"""
import os
import random
from pathlib import Path
from typing import Union, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class SthV2FrameDataset(Dataset):
    """
    Load single images from SthV2-style frame folders for VAE reconstruction.
    Returns a tensor of shape [C, H, W] and a dummy label for compatibility.
    """

    def __init__(
        self,
        folder: str,
        image_size: Union[int, Tuple[int, int]] = 256,
        split: str = 'train',
        val_split: float = 0.05
    ):
        super().__init__()
        self.folder = folder
        self.split = split
        
        # Get all video folders
        all_folders = sorted([f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])
        
        # Split into train/val
        num_val = int(len(all_folders) * val_split)
        if split == 'val':
            self.folder_list = all_folders[:num_val]
        else:
            self.folder_list = all_folders[num_val:]
        
        print(f"{split.upper()} set: {len(self.folder_list)} videos")
        
        # Normalize to [-1, 1] to match Tanh output
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: torch.Tensor of shape [C, H, W]
            label: dummy label (0) for compatibility with training loop
        """
        try:
            video_id = self.folder_list[index]
            frame_dir = os.path.join(self.folder, video_id)
            img_list = os.listdir(frame_dir)

            # Sort by numeric suffix if possible
            def get_frame_number(filename):
                try:
                    if filename.startswith('frame') and filename.endswith('.jpg'):
                        return int(filename[5:-4])
                    return int(''.join(filter(str.isdigit, filename.split('.')[0])))
                except:
                    return 0

            img_list = sorted(img_list, key=get_frame_number)
            if not img_list:
                raise RuntimeError('empty folder')

            # Pick a random frame for VAE training
            idx = random.randint(0, len(img_list) - 1)
            path = os.path.join(frame_dir, img_list[idx])
            img = Image.open(path)
            img_tensor = self.transform(img)
            
            return img_tensor, 0  # Return dummy label for compatibility
            
        except Exception as e:
            # Fallback to another sample if this one fails
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
    val_split: float = 0.05
):
    """
    Create train and validation dataloaders.
    
    Args:
        data_path: Path to the frame folders
        image_size: Image size to resize to
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        val_split: Fraction of data to use for validation
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = SthV2FrameDataset(
        data_path,
        image_size=image_size,
        split='train',
        val_split=val_split
    )
    
    val_dataset = SthV2FrameDataset(
        data_path,
        image_size=image_size,
        split='val',
        val_split=val_split
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

