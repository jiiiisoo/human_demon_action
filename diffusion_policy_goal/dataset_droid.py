"""
Dataset for goal-conditioned diffusion policy training
Loads goal images, current observations, and action trajectories from DROID local dataset
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


def ensure_rgb(img):
    """Ensure image is RGB"""
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img


class DroidGoalDataset(Dataset):
    """
    Dataset for goal-conditioned diffusion policy
    
    Returns:
        goal_image: Goal image (last frame of episode)
        obs_images: Observation images (sequence of frames)
        actions: Action trajectory
    """
    def __init__(
        self,
        data_path,
        split='train',
        image_size=128,
        obs_horizon=2,
        action_horizon=16,
        debug_max_episodes=None,
        prebuilt_index=None,  # For distributed training: rank 0 builds, others receive
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        
        # Expect data in data_path/train/ or data_path/val/
        self.data_dir = self.data_path / 'train'
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Build or use prebuilt index
        if prebuilt_index is not None:
            # Use prebuilt index (from rank 0)
            self.samples = prebuilt_index
            print(f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] "
                  f"Loaded {len(self.samples)} samples from prebuilt index ({split} split)")
        else:
            # Build index (only rank 0 should do this)
            print(f"Loading DROID goal-conditioned dataset from {self.data_dir}")
            print(f"Split: {split}")
            
            self.samples = []
            self._build_index(debug_max_episodes)
            
            print(f"Loaded {len(self.samples)} samples from {split} split")
        
        # Image transforms
        self.transform = T.Compose([
            T.Lambda(ensure_rgb),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
    
    def _build_index(self, debug_max_episodes):
        """
        Build index of all valid samples.
        Each sample = (episode_dir, start_idx)
        """
        # Get all episode directories
        all_episode_dirs = sorted([d for d in self.data_dir.iterdir() 
                                   if d.is_dir() and d.name.startswith('episode_')])
        
        print(f"Found {len(all_episode_dirs)} total episode directories")
        
        # Split into train (90%) and val (10%)
        num_total = len(all_episode_dirs)
        num_train = int(num_total * 0.9)
        
        if self.split == 'train':
            episode_dirs = all_episode_dirs[:num_train]
        else:
            episode_dirs = all_episode_dirs[num_train:]
        
        print(f"Using {len(episode_dirs)} episodes for {self.split} split")
        
        if debug_max_episodes is not None:
            episode_dirs = episode_dirs[:debug_max_episodes]
            print(f"Debug mode: limiting to {debug_max_episodes} episodes")
        
        # Build sample index with progress bar
        num_episodes_processed = 0
        print(f"Indexing episodes...")
        for episode_dir in tqdm(episode_dirs, desc=f"Building {self.split} index", unit="ep"):
            # Check if episode has required data
            actions_file = episode_dir / 'actions.npy'
            image_dir = episode_dir / 'frames' / 'exterior_image_1_left'
            
            if not actions_file.exists() or not image_dir.exists():
                continue
            
            # Load actions to get episode length
            try:
                actions = np.load(actions_file)
                episode_len = len(actions)
            except:
                continue
            
            # Get all frame files
            frame_files = sorted([f for f in image_dir.iterdir() 
                                 if f.suffix in ['.jpg', '.jpeg', '.png']])
            
            if len(frame_files) == 0 or len(frame_files) != episode_len:
                continue
            
            # Create samples: each valid starting index in the episode
            # We need obs_horizon frames + action_horizon actions
            max_start_idx = episode_len - self.action_horizon
            
            for start_idx in range(max(0, max_start_idx)):
                # Make sure we have enough observation frames
                if start_idx >= self.obs_horizon - 1:
                    self.samples.append({
                        'episode_dir': episode_dir,
                        'start_idx': start_idx,
                        'episode_len': episode_len,
                    })
            
            num_episodes_processed += 1
        
        print(f"Indexed {num_episodes_processed} episodes, {len(self.samples)} samples")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found! Check data path.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Get a single sample
        
        Returns:
            dict with keys:
                'goal_image': (3, H, W) - goal image (last frame)
                'obs_image': (3, H, W) - current observation image
                'actions': (action_horizon, action_dim) - action trajectory
        """
        try:
            sample_info = self.samples[index]
            episode_dir = sample_info['episode_dir']
            start_idx = sample_info['start_idx']
            episode_len = sample_info['episode_len']
            
            image_dir = episode_dir / 'frames' / 'exterior_image_1_left'
            
            # Load goal image (last frame action_horizon)
            goal_frame_path = image_dir / f'{start_idx + self.action_horizon - 1:06d}.jpg'
            goal_img = Image.open(goal_frame_path)
            goal_img_tensor = self.transform(goal_img)
            
            # Load observation image (current frame at start_idx)
            obs_frame_path = image_dir / f'{start_idx:06d}.jpg'
            obs_img = Image.open(obs_frame_path)
            obs_img_tensor = self.transform(obs_img)  # (3, H, W)
            
            # Load actions
            actions_file = episode_dir / 'actions.npy'
            actions = np.load(actions_file)
            
            # Extract action trajectory (action_horizon actions starting from start_idx)
            action_traj = actions[start_idx:start_idx + self.action_horizon]
            action_traj = torch.from_numpy(action_traj).float()
            
            return {
                'goal_image': goal_img_tensor,  # (3, H, W)
                'obs_image': obs_img_tensor,    # (3, H, W)
                'actions': action_traj,         # (action_horizon, action_dim)
            }
            
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            print(f"  Episode: {episode_dir}")
            print(f"  Start idx: {start_idx}")
            # Return a dummy sample on error
            dummy_goal = torch.zeros(3, self.image_size, self.image_size)
            dummy_obs = torch.zeros(3, self.image_size, self.image_size)
            dummy_actions = torch.zeros(self.action_horizon, 7)  # assume 7-dim actions
            return {
                'goal_image': dummy_goal,
                'obs_image': dummy_obs,
                'actions': dummy_actions,
            }


if __name__ == '__main__':
    # Test dataset
    dataset = DroidGoalDataset(
        data_path='/mnt/data/droid/droid_local',
        split='train',
        debug_max_episodes=10,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Goal image: {sample['goal_image'].shape}")
    print(f"  Obs image: {sample['obs_image'].shape}")
    print(f"  Actions: {sample['actions'].shape}")
    print(f"  Action range: [{sample['actions'].min():.3f}, {sample['actions'].max():.3f}]")

