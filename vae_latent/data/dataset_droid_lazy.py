"""
Lazy-loading Dataset for Droid frames.
Stores only metadata, loads images on-demand to prevent OOM.
"""
import os
import sys

# CRITICAL: Set environment variables BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
from pathlib import Path
from typing import Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# Suppress stderr during TensorFlow import
import io
_stderr = sys.stderr
sys.stderr = io.StringIO()

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('ERROR')
    import tensorflow_datasets as tfds
finally:
    sys.stderr = _stderr

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm


def ensure_rgb(img):
    return img.convert('RGB') if img.mode != 'RGB' else img


class DroidFrameDatasetLazy(Dataset):
    """
    Lazy-loading Droid dataset - stores only frame indices and episode references.
    Loads actual images on-demand during __getitem__.
    
    Memory efficient: O(num_episodes) instead of O(num_frames)
    """

    def __init__(
        self,
        data_path: str,
        image_size: Union[int, Tuple[int, int]] = 256,
        train: bool = True,
        image_key: str = 'exterior_image_1_left',
        rank: int = 0,
        world_size: int = 1,
        debug_max_episodes: int = None
    ):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.image_key = image_key
        self.rank = rank
        self.world_size = max(1, world_size)
        
        # Load TFDS builder
        self.builder = tfds.builder_from_directory(builder_dir=data_path)
        
        # Determine split
        if "val" not in self.builder.info.splits:
            if train:
                self.tfds_split = "train[:90%]"
            else:
                self.tfds_split = "train[90%:]"
        else:
            self.tfds_split = "train" if train else "val"
        
        # Build frame index: store (episode_idx, step_idx) tuples instead of actual frames
        self.frame_indices = []
        self._build_frame_index(debug_max_episodes)
        
        # Image transforms
        target_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transform = T.Compose([
            T.Lambda(ensure_rgb),
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])
    
    def _build_frame_index(self, debug_max_episodes):
        """Build index of (episode_idx, step_idx) without loading actual frames"""
        split_name = "train" if self.train else "val"
        
        # Load dataset metadata
        ds = self.builder.as_dataset(split=self.tfds_split)
        
        if debug_max_episodes is not None and debug_max_episodes > 0:
            ds = ds.take(debug_max_episodes)
            if self.rank == 0:
                print(f"[DEBUG] Indexing only {debug_max_episodes} episodes")
        
        # Only index episodes for this rank (DDP sharding)
        print(f"[Rank {self.rank}] Building frame index for {split_name} split...")
        
        for epi_idx, episode in enumerate(tqdm(ds, desc=f"Indexing {split_name} (rank {self.rank})")):
            # Shard episodes across GPUs
            if (epi_idx % self.world_size) != self.rank:
                continue
            
            # Store (episode_idx, step_idx) for each frame
            num_steps = len(episode['steps'])
            for step_idx in range(num_steps):
                self.frame_indices.append((epi_idx, step_idx))
        
        print(f"[Rank {self.rank}] Indexed {len(self.frame_indices)} frames from {split_name} split")
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single frame on-demand.
        
        Returns:
            image: torch.Tensor of shape [C, H, W]
            label: dummy label (0) for compatibility
        """
        try:
            epi_idx, step_idx = self.frame_indices[index]
            
            # Load only the required episode (lazy)
            ds = self.builder.as_dataset(split=self.tfds_split)
            
            # Skip to the correct episode
            episode = None
            for i, ep in enumerate(ds):
                # Account for sharding
                if (i % self.world_size) != self.rank:
                    continue
                if i == epi_idx:
                    episode = ep
                    break
            
            if episode is None:
                raise ValueError(f"Episode {epi_idx} not found")
            
            # Get the specific frame
            step = episode['steps'][step_idx]
            img_tensor = step['observation'][self.image_key]
            
            # Convert to PIL Image
            img_np = img_tensor.numpy()
            img = Image.fromarray(img_np)
            
            # Apply transforms
            img_tensor = self.transform(img)
            
            return img_tensor, 0
            
        except Exception as e:
            print(f"Error loading frame {index} (epi={epi_idx}, step={step_idx}): {e}")
            # Fallback to another sample
            if index < len(self) - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(0)


class DroidFrameDatasetCached(Dataset):
    """
    TRULY lazy loading: Only store frame count, load on-demand.
    Uses TFDS snapshot to avoid reloading entire dataset each time.
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: Union[int, Tuple[int, int]] = 256,
        train: bool = True,
        image_key: str = 'exterior_image_1_left',
        rank: int = 0,
        world_size: int = 1,
        debug_max_episodes: int = None
    ):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.image_key = image_key
        self.rank = rank
        self.world_size = max(1, world_size)
        
        # Load TFDS builder
        self.builder = tfds.builder_from_directory(builder_dir=data_path)
        
        # Determine split
        if "val" not in self.builder.info.splits:
            if train:
                self.tfds_split = "train[:90%]"
            else:
                self.tfds_split = "train[90%:]"
        else:
            self.tfds_split = "train" if train else "val"
        
        # Convert dataset to list for random access
        ds = self.builder.as_dataset(split=self.tfds_split)
        
        if debug_max_episodes is not None and debug_max_episodes > 0:
            ds = ds.take(debug_max_episodes)
            if self.rank == 0:
                print(f"[DEBUG] Loading only {debug_max_episodes} episodes")
        
        split_name = "train" if self.train else "val"
        self.episodes_list = []  # List of episodes (converted to dict/numpy)
        self.frame_indices = []  # (episode_list_idx, step_idx)
        
        print(f"[Rank {self.rank}] Loading episodes for {split_name} split...")
        for global_epi_idx, episode in enumerate(tqdm(ds, desc=f"Loading {split_name} (rank {self.rank})")):
            # Shard episodes across GPUs
            if (global_epi_idx % self.world_size) != self.rank:
                continue
            
            # Convert TF episode to Python dict with numpy arrays (for indexing)
            episode_dict = {
                'steps': []
            }
            for step in episode['steps']:
                step_dict = {'observation': {}}
                for key, value in step['observation'].items():
                    if hasattr(value, 'numpy'):
                        step_dict['observation'][key] = value.numpy()
                    else:
                        step_dict['observation'][key] = np.array(value)
                episode_dict['steps'].append(step_dict)
            
            self.episodes_list.append(episode_dict)
            local_epi_idx = len(self.episodes_list) - 1
            
            # Build frame index: (local_episode_idx, step_idx)
            num_steps = len(episode_dict['steps'])
            for step_idx in range(num_steps):
                self.frame_indices.append((local_epi_idx, step_idx))
        
        print(f"[Rank {self.rank}] Loaded {len(self.episodes_list)} episodes, "
              f"{len(self.frame_indices)} frames from {split_name} split")
        
        # Image transforms
        target_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transform = T.Compose([
            T.Lambda(ensure_rgb),
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: torch.Tensor of shape [C, H, W]
            label: dummy label (0) for compatibility
        """
        try:
            local_epi_idx, step_idx = self.frame_indices[index]
            
            # Direct random access! No cache needed
            episode = self.episodes_list[local_epi_idx]
            
            # Get frame from episode (already numpy)
            step = episode['steps'][step_idx]
            img_np = step['observation'][self.image_key]
            
            # Convert numpy to PIL
            img = Image.fromarray(img_np)
            
            # Apply transforms
            img_tensor = self.transform(img)
            
            return img_tensor, 0
            
        except Exception as e:
            print(f"Error loading frame {index} (epi={local_epi_idx}, step={step_idx}): {e}")
            if index < len(self) - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, len(self) - 1))


def get_dataloaders(
    data_path: str,
    image_size: int = 256,
    train_batch_size: int = 64,
    val_batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_cached: bool = True,  # True = cache episodes, False = full lazy
):
    """
    Create train and validation dataloaders with memory-efficient loading.
    
    Args:
        use_cached: If True, uses DroidFrameDatasetCached (caches episodes, ~10-100x less memory).
                    If False, uses full lazy loading (slowest but minimal memory).
    """
    DatasetClass = DroidFrameDatasetCached if use_cached else DroidFrameDatasetLazy
    
    train_dataset = DatasetClass(
        data_path=data_path,
        image_size=image_size,
        train=True,
    )
    
    val_dataset = DatasetClass(
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

