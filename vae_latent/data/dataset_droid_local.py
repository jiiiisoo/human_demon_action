"""
DROID dataset loader for local file system structure.
No TensorFlow dependencies - uses PIL for image loading.

Expected directory structure:
data/droid_local/
├── train/
│   ├── episode_000000/
│   │   ├── metadata.json
│   │   ├── frames/
│   │   │   └── exterior_image_1_left/
│   │   │       ├── 000000.jpg
│   │   │       ├── 000001.jpg
│   │   │       └── ...
│   │   ├── actions.npy
│   │   └── observations.npz
│   ├── episode_000001/
│   └── ...
└── val/
    └── ...
"""

import os
import json
from pathlib import Path
from typing import Union, Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm


def ensure_rgb(img):
    """Ensure image is in RGB format."""
    return img.convert('RGB') if img.mode != 'RGB' else img


class DroidFrameDatasetLocal(Dataset):
    """
    DROID dataset loader for local file system.
    
    Memory efficient: Only stores file paths, loads images on-demand.
    No TensorFlow dependencies.
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
        """
        Args:
            data_path: Path to local droid dataset (e.g., '/mnt/data/droid/droid_local')
            image_size: Target image size (int or (H, W))
            train: Whether to load train or val split
            image_key: Which camera view to use
            rank: Process rank for DDP
            world_size: Total number of processes for DDP
            debug_max_episodes: Limit number of episodes for debugging
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.train = train
        self.image_key = image_key
        self.rank = rank
        self.world_size = max(1, world_size)
        
        # Note: We expect data to be in data_path/train/ directory
        # We'll split it into 90% train, 10% val ourselves
        self.data_dir = self.data_path / 'train'
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        print(f"[Rank {self.rank}] Loading DROID dataset from {self.data_dir}")
        print(f"[Rank {self.rank}] Split: {'train (90%)' if train else 'val (10%)'}")
        
        # Build frame index: list of (episode_dir, frame_idx) tuples
        self.frame_paths = []
        self._build_index(debug_max_episodes)
        
        # Image transforms
        target_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transform = T.Compose([
            T.Lambda(ensure_rgb),
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _build_index(self, debug_max_episodes: int = None):
        """
        Build index of all frame paths.
        Stores (episode_dir, frame_idx) for each frame.
        Splits episodes into 90% train, 10% val.
        """
        # Get all episode directories (sorted for reproducibility)
        all_episode_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])
        
        print(f"[Rank {self.rank}] Found {len(all_episode_dirs)} total episode directories")
        
        # Split into train (90%) and val (10%)
        num_total = len(all_episode_dirs)
        num_train = int(num_total * 0.9)
        
        if self.train:
            episode_dirs = all_episode_dirs[:num_train]
        else:
            episode_dirs = all_episode_dirs[num_train:]
        
        print(f"[Rank {self.rank}] Using {len(episode_dirs)} episodes for {'train' if self.train else 'val'} split")
        
        if debug_max_episodes is not None:
            # For debugging with DDP, ensure each GPU gets some data
            total_episodes = min(debug_max_episodes, len(episode_dirs))
            # Adjust to ensure divisibility by world_size
            total_episodes = max(self.world_size, (total_episodes // self.world_size) * self.world_size)
            episode_dirs = episode_dirs[:total_episodes]
            print(f"[Rank {self.rank}] Debug mode: limiting to {total_episodes} episodes")
        
        print(f"[Rank {self.rank}] Processing {len(episode_dirs)} episode directories")
        
        # Build frame index with DDP sharding
        num_episodes_processed = 0
        for episode_dir in tqdm(episode_dirs, desc=f"[Rank {self.rank}] Indexing episodes"):
            # Extract episode index from directory name
            episode_idx = int(episode_dir.name.split('_')[-1])
            
            # Shard episodes across GPUs
            if (episode_idx % self.world_size) != self.rank:
                continue
            
            # Get image directory
            image_dir = episode_dir / 'frames' / self.image_key
            
            if not image_dir.exists():
                print(f"[Rank {self.rank}] Warning: Image directory not found: {image_dir}")
                continue
            
            # Get all frame files
            frame_files = sorted([f for f in image_dir.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']])
            
            if len(frame_files) == 0:
                print(f"[Rank {self.rank}] Warning: No frames found in {image_dir}")
                continue
            
            # Add frame paths
            for frame_file in frame_files:
                self.frame_paths.append(frame_file)
            
            num_episodes_processed += 1
        
        print(f"[Rank {self.rank}] Indexed {num_episodes_processed} episodes, "
              f"{len(self.frame_paths)} frames")
        
        if len(self.frame_paths) == 0:
            raise ValueError(f"[Rank {self.rank}] No frames found! Check data path and image_key.")
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: torch.Tensor of shape [C, H, W]
            label: dummy label (0) for compatibility
        """
        try:
            frame_path = self.frame_paths[index]
            
            # Load image
            img = Image.open(frame_path)
            
            # Apply transforms
            img_tensor = self.transform(img)
            
            return img_tensor, 0  # Return dummy label for compatibility
            
        except Exception as e:
            print(f"[Rank {self.rank}] Error loading frame {index}: {e}")
            print(f"  Frame path: {frame_path}")
            # Return a dummy image on error
            target_size = self.transform.transforms[1].size
            dummy_img = torch.zeros(3, target_size[0], target_size[1])
            return dummy_img, 0


class DroidEpisodeDatasetLocal(Dataset):
    """
    DROID dataset loader that returns full episodes (for evaluation/visualization).
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
        self.data_path = Path(data_path)
        self.train = train
        self.image_key = image_key
        self.rank = rank
        self.world_size = max(1, world_size)
        
        # Note: We expect data to be in data_path/train/ directory
        # We'll split it into 90% train, 10% val ourselves
        self.data_dir = self.data_path / 'train'
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        print(f"[Rank {self.rank}] Loading DROID episodes from {self.data_dir}")
        print(f"[Rank {self.rank}] Split: {'train (90%)' if train else 'val (10%)'}")
        
        # Get all episode directories (sorted for reproducibility)
        all_episode_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])
        
        # Split into train (90%) and val (10%)
        num_total = len(all_episode_dirs)
        num_train = int(num_total * 0.9)
        
        if self.train:
            episode_dirs = all_episode_dirs[:num_train]
        else:
            episode_dirs = all_episode_dirs[num_train:]
        
        print(f"[Rank {self.rank}] Using {len(episode_dirs)} episodes for {'train' if self.train else 'val'} split")
        
        if debug_max_episodes is not None:
            total_episodes = min(debug_max_episodes, len(episode_dirs))
            total_episodes = max(self.world_size, (total_episodes // self.world_size) * self.world_size)
            episode_dirs = episode_dirs[:total_episodes]
            print(f"[Rank {self.rank}] Debug mode: limiting to {total_episodes} episodes")
        
        # Shard episodes
        self.episode_dirs = []
        for episode_dir in episode_dirs:
            episode_idx = int(episode_dir.name.split('_')[-1])
            if (episode_idx % self.world_size) == self.rank:
                self.episode_dirs.append(episode_dir)
        
        print(f"[Rank {self.rank}] Loaded {len(self.episode_dirs)} episodes")
        
        # Image transforms
        target_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transform = T.Compose([
            T.Lambda(ensure_rgb),
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.episode_dirs)
    
    def __getitem__(self, index: int) -> dict:
        """
        Returns:
            dict with keys:
                - images: List[torch.Tensor] of shape [C, H, W]
                - actions: np.ndarray of shape [T, action_dim]
                - metadata: dict
        """
        episode_dir = self.episode_dirs[index]
        
        # Load metadata
        metadata = {}
        metadata_file = episode_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        
        # Load actions
        actions = None
        actions_file = episode_dir / 'actions.npy'
        if actions_file.exists():
            actions = np.load(actions_file)
        
        # Load images
        image_dir = episode_dir / 'frames' / self.image_key
        frame_files = sorted([f for f in image_dir.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']])
        
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            img_tensor = self.transform(img)
            images.append(img_tensor)
        
        return {
            'images': images,
            'actions': actions,
            'metadata': metadata,
            'episode_dir': str(episode_dir)
        }


def test_dataset():
    """Test the dataset loader."""
    print("Testing DroidFrameDatasetLocal...")
    
    dataset = DroidFrameDatasetLocal(
        data_path='/mnt/data/droid/droid_local',
        image_size=256,
        train=True,
        image_key='exterior_image_1_left',
        rank=0,
        world_size=1,
        debug_max_episodes=5
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test loading a few samples
    print("\nTesting sample loading...")
    for i in range(min(3, len(dataset))):
        img, label = dataset[i]
        print(f"  Sample {i}: image shape={img.shape}, label={label}")
    
    print("\n✅ Dataset test passed!")


if __name__ == '__main__':
    test_dataset()

