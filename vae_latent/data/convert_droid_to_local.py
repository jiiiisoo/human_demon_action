"""
Convert DROID dataset from TFDS to local file structure.

Output structure:
data/droid_local/
├── train/
│   ├── episode_000000/
│   │   ├── metadata.json
│   │   ├── frames/
│   │   │   ├── exterior_image_1_left/
│   │   │   │   ├── 000000.jpg
│   │   │   │   ├── 000001.jpg
│   │   │   │   └── ...
│   │   │   ├── exterior_image_2_left/
│   │   │   └── wrist_image_left/
│   │   └── actions.npy  # All actions for the episode
│   ├── episode_000001/
│   └── ...
└── val/
    └── ...
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

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


def convert_episode(episode, output_dir, episode_idx, image_keys):
    """
    Convert a single episode to local files.
    
    Args:
        episode: TFDS episode
        output_dir: Base output directory
        episode_idx: Episode index for naming
        image_keys: List of image keys to save
    """
    episode_dir = output_dir / f"episode_{episode_idx:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Save episode metadata
    metadata = {
        'episode_idx': episode_idx,
        'num_steps': len(episode['steps']),
    }
    
    if 'episode_metadata' in episode:
        for key, value in episode['episode_metadata'].items():
            if hasattr(value, 'numpy'):
                val = value.numpy()
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                metadata[key] = val
            else:
                metadata[key] = str(value)
    
    with open(episode_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create frames directory
    frames_dir = episode_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)
    
    for image_key in image_keys:
        (frames_dir / image_key).mkdir(exist_ok=True)
    
    # Lists to accumulate data
    actions = []
    observations_numeric = []
    language_instructions = []
    
    # Process each step
    for step_idx, step in enumerate(episode['steps']):
        # Save images
        obs = step['observation']
        for image_key in image_keys:
            if image_key in obs:
                img_np = obs[image_key].numpy()
                img = Image.fromarray(img_np)
                img.save(frames_dir / image_key / f"{step_idx:06d}.jpg", quality=95)
        
        # Collect actions
        action = step['action'].numpy()
        actions.append(action)
        
        # Collect numeric observations
        obs_numeric = {}
        for key in ['cartesian_position', 'cartesian_velocity', 'gripper_position', 'gripper_velocity', 'joint_position', 'joint_velocity']:
            if key in obs:
                obs_numeric[key] = obs[key].numpy()
        observations_numeric.append(obs_numeric)
        
        # Collect language instructions
        if 'language_instruction' in step:
            lang = step['language_instruction'].numpy()
            lang_2 = step['language_instruction_2'].numpy()
            lang_3 = step['language_instruction_3'].numpy()
            if isinstance(lang, bytes):
                lang = lang.decode('utf-8')
            language_instructions.append(lang)
            language_instructions.append(lang_2)
            language_instructions.append(lang_3)
    
    # Save actions as numpy array
    actions = np.array(actions)
    np.save(episode_dir / 'actions.npy', actions)
    
    # Save numeric observations
    if observations_numeric:
        obs_dict = {}
        for key in observations_numeric[0].keys():
            obs_dict[key] = np.array([obs[key] for obs in observations_numeric])
        np.savez(episode_dir / 'observations.npz', **obs_dict)
    
    # Save language instructions
    if language_instructions:
        with open(episode_dir / 'language.txt', 'w') as f:
            f.write(language_instructions[0])  # Usually same for all steps
    
    return len(episode['steps'])


def convert_split(data_path, output_base, split_name, max_episodes=None, 
                  image_keys=['exterior_image_1_left'], rank=0, world_size=1):
    """
    Convert a data split (train/val) to local files.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of parallel processes
    """
    print(f"\n{'='*70}")
    print(f"[Rank {rank}/{world_size}] Converting {split_name} split")
    print(f"{'='*70}")
    
    # Load TFDS
    builder = tfds.builder_from_directory(builder_dir=data_path)
    
    # Determine split
    if "val" not in builder.info.splits:
        if split_name == "train":
            tfds_split = "train[:90%]"
        else:
            tfds_split = "train[90%:]"
    else:
        tfds_split = "train" if split_name == "train" else "val"
    
    ds = builder.as_dataset(split=tfds_split)
    
    if max_episodes is not None:
        ds = ds.take(max_episodes)
        print(f"[Rank {rank}] Limiting to {max_episodes} episodes")
    
    # Output directory
    output_dir = Path(output_base) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert episodes (with sharding)
    total_frames = 0
    converted_episodes = 0
    for global_episode_idx, episode in enumerate(tqdm(ds, desc=f"[Rank {rank}] Converting", position=rank)):
        # Shard episodes across processes
        if (global_episode_idx % world_size) != rank:
            continue
        
        num_frames = convert_episode(episode, output_dir, global_episode_idx, image_keys)
        total_frames += num_frames
        converted_episodes += 1
    
    print(f"\n✅ [Rank {rank}] Converted {converted_episodes} episodes, {total_frames} total frames")
    print(f"   Output: {output_dir}")
    
    # Save rank-specific split info
    split_info = {
        'rank': rank,
        'world_size': world_size,
        'num_episodes': converted_episodes,
        'num_frames': total_frames,
        'image_keys': image_keys,
    }
    with open(output_dir / f'split_info_rank{rank}.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return converted_episodes, total_frames


def main():
    parser = argparse.ArgumentParser(description='Convert DROID TFDS to local files')
    parser.add_argument('--data_path', type=str, 
                       default='gs://gresearch/robotics/droid/1.0.0',
                       help='Path to TFDS DROID dataset')
    parser.add_argument('--output', type=str,
                       default='data/droid_local',
                       help='Output directory')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Max episodes to convert (None = all)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help='Splits to convert')
    parser.add_argument('--image_keys', nargs='+',
                       default=['exterior_image_1_left'],
                       help='Image keys to save')
    parser.add_argument('--rank', type=int, default=0,
                       help='Process rank (0 to world_size-1)')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Total number of parallel processes')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"DROID Dataset Conversion [Rank {args.rank}/{args.world_size}]")
    print("="*70)
    print(f"Input:  {args.data_path}")
    print(f"Output: {args.output}")
    print(f"Image keys: {args.image_keys}")
    print(f"Max episodes: {args.max_episodes or 'All'}")
    
    for split_name in args.splits:
        convert_split(
            args.data_path,
            args.output,
            split_name,
            args.max_episodes,
            args.image_keys,
            args.rank,
            args.world_size
        )
    
    print("\n" + "="*70)
    print(f"✅ [Rank {args.rank}] Conversion complete!")
    print("="*70)
    print(f"\nYou can now use the local dataset at: {args.output}")


if __name__ == '__main__':
    main()

