"""
PyTorch DataLoader example that directly loads data from HDF5 files using JSON index.

Loads data from HDF5 on-the-fly during training without pre-extracting all data.

Usage:
    # First create the index
    python create_dataset_index.py --data-dir /mnt/data/libero/libero_10 --output ./dataset_index.json
    
    # Test the DataLoader
    python example_hdf5_dataloader.py --index ./dataset_index.json
"""

import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import os


class LIBEROHdf5Dataset(Dataset):
    """
    Dataset that loads data directly from HDF5 using JSON index.
    Memory-efficient and enables fast training.
    """
    
    def __init__(self, index_path, obs_key="agentview_rgb", cache_hdf5=True, 
                 suites=None, tasks=None):
        """
        Args:
            index_path: Path to JSON index file
            obs_key: Observation key to use (default: "agentview_rgb")
            cache_hdf5: Whether to cache HDF5 file handles (memory vs speed tradeoff)
            suites: List of suites to use (None = use all)
            tasks: List of tasks to use (None = use all)
        """
        self.obs_key = obs_key
        self.cache_hdf5 = cache_hdf5
        self.hdf5_cache = {}
        
        # Load JSON index
        with open(index_path, 'r') as f:
            self.dataset_index = json.load(f)
        
        # Build full data index
        self.data_index = []  # [(hdf5_path, demo_name, frame_idx), ...]
        
        # Structure: suite_name -> {task_name -> task_info}
        for suite_name, suite_tasks in self.dataset_index.items():
            # Filter by suite
            if suites is not None and suite_name not in suites:
                continue
            
            for task_name, task_info in suite_tasks.items():
                # Filter by task
                if tasks is not None and task_name not in tasks:
                    continue
                
                hdf5_path = task_info["file_path"]
                
                if not os.path.exists(hdf5_path):
                    print(f"Warning: HDF5 file not found: {hdf5_path}")
                    continue
                
                for demo_name, demo_meta in task_info["demos"].items():
                    num_frames = demo_meta["num_frames"]
                    
                    # Check if observation key exists
                    if obs_key not in demo_meta["obs_keys"]:
                        print(f"Warning: {obs_key} not found in {suite_name}/{task_name}/{demo_name}")
                        continue
                    
                    # Add index for each frame
                    for frame_idx in range(num_frames):
                        self.data_index.append({
                            "hdf5_path": hdf5_path,
                            "demo_name": demo_name,
                            "frame_idx": frame_idx,
                            "language_instruction": demo_meta["language_instruction"],
                            "task_name": task_name,
                            "suite_name": suite_name
                        })
        
        print(f"\nDataset loaded from index: {index_path}")
        print(f"Total samples: {len(self.data_index)}")
        print(f"Observation key: {obs_key}")
        print(f"HDF5 caching: {'Enabled' if cache_hdf5 else 'Disabled'}")
        
        # Print statistics
        suites_used = set(item["suite_name"] for item in self.data_index)
        tasks_used = set(item["task_name"] for item in self.data_index)
        print(f"Suites used: {', '.join(sorted(suites_used))}")
        print(f"Tasks used: {len(tasks_used)}")
    
    def _get_hdf5_file(self, hdf5_path):
        """Get HDF5 file handle (with caching option)"""
        if self.cache_hdf5:
            if hdf5_path not in self.hdf5_cache:
                self.hdf5_cache[hdf5_path] = h5py.File(hdf5_path, 'r')
            return self.hdf5_cache[hdf5_path]
        else:
            return h5py.File(hdf5_path, 'r')
    
    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - image: torch.Tensor, shape (C, H, W), normalized to [0, 1]
                - action: torch.Tensor, shape (7,)
                - language_instruction: str
                - task_name: str
                - suite_name: str
                - demo_name: str
                - frame_idx: int
        """
        item = self.data_index[idx]
        hdf5_path = item["hdf5_path"]
        demo_name = item["demo_name"]
        frame_idx = item["frame_idx"]
        
        # Read data from HDF5
        f = self._get_hdf5_file(hdf5_path)
        
        try:
            # Read image
            image = f[f"data/{demo_name}/obs/{self.obs_key}"][frame_idx]  # (H, W, 3)
            
            # Read action
            action = f[f"data/{demo_name}/actions"][frame_idx]  # (7,)
            
            # Convert to PyTorch tensors
            image = torch.from_numpy(image.copy()).float() / 255.0  # Normalize to [0, 1]
            image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            action = torch.from_numpy(action.copy()).float()
            
            return {
                "image": image,
                "action": action,
                "language_instruction": item["language_instruction"],
                "task_name": item["task_name"],
                "suite_name": item["suite_name"],
                "demo_name": demo_name,
                "frame_idx": frame_idx,
                "hdf5_path": hdf5_path
            }
        
        finally:
            # Close file if not caching
            if not self.cache_hdf5:
                f.close()
    
    def get_dataset_info(self):
        """Return dataset information"""
        return self.dataset_index
    
    def get_suites(self):
        """Return list of suites"""
        return list(self.dataset_index.keys())
    
    def get_tasks(self, suite_name=None):
        """Return list of tasks"""
        tasks = {}
        
        if suite_name:
            # Tasks from specific suite only
            if suite_name in self.dataset_index:
                for task_name, task_info in self.dataset_index[suite_name].items():
                    tasks[task_name] = {
                        "suite_name": suite_name,
                        "num_demos": task_info["num_demos"],
                        "total_frames": task_info["total_frames"],
                        "hdf5_path": task_info["file_path"]
                    }
        else:
            # All tasks from all suites
            for suite_name, suite_tasks in self.dataset_index.items():
                for task_name, task_info in suite_tasks.items():
                    full_task_name = f"{suite_name}/{task_name}"
                    tasks[full_task_name] = {
                        "suite_name": suite_name,
                        "task_name": task_name,
                        "num_demos": task_info["num_demos"],
                        "total_frames": task_info["total_frames"],
                        "hdf5_path": task_info["file_path"]
                    }
        
        return tasks
    
    def __del__(self):
        """Close cached HDF5 files"""
        for f in self.hdf5_cache.values():
            try:
                f.close()
            except:
                pass


class LIBEROMultiTaskDataset(LIBEROHdf5Dataset):
    """
    Dataset for multi-task learning.
    Allows adjusting sampling ratios per task.
    """
    
    def __init__(self, index_path, obs_key="agentview_rgb", 
                 task_weights=None, cache_hdf5=True):
        """
        Args:
            index_path: Path to JSON index file
            obs_key: Observation key to use
            task_weights: Sampling weights per task, dict {task_name: weight}
            cache_hdf5: Whether to cache HDF5 file handles
        """
        super().__init__(index_path, obs_key, cache_hdf5)
        
        # Apply task weights
        if task_weights is not None:
            self._apply_task_weights(task_weights)
    
    def _apply_task_weights(self, task_weights):
        """Adjust data sampling based on task weights"""
        weighted_index = []
        
        for item in self.data_index:
            task_name = item["task_name"]
            weight = task_weights.get(task_name, 1.0)
            
            # Repeat by weight
            num_repeats = int(weight)
            weighted_index.extend([item] * num_repeats)
        
        self.data_index = weighted_index
        print(f"Applied task weights. New dataset size: {len(self.data_index)}")


def main():
    parser = argparse.ArgumentParser(description='Test HDF5 DataLoader')
    parser.add_argument('--index', type=str, default='./dataset_index.json',
                        help='Path to dataset index JSON file')
    parser.add_argument('--obs-key', type=str, default='agentview_rgb',
                        help='Observation key to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.index):
        print(f"Error: Index file not found: {args.index}")
        print(f"Please run create_dataset_index.py first to create the index file.")
        return
    
    # Dataset 생성
    print("="*80)
    print("Creating Dataset...")
    print("="*80)
    
    dataset = LIBEROHdf5Dataset(
        index_path=args.index,
        obs_key=args.obs_key,
        cache_hdf5=True
    )
    
    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoader created:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    print(f"  Number of workers: {args.num_workers}")
    
    # Suite 정보 출력
    print("\n" + "="*80)
    print("Suites in dataset:")
    print("="*80)
    suites = dataset.get_suites()
    for suite_name in suites:
        print(f"\n[{suite_name}]")
        tasks = dataset.get_tasks(suite_name)
        total_demos = sum(t['num_demos'] for t in tasks.values())
        total_frames = sum(t['total_frames'] for t in tasks.values())
        print(f"  Tasks: {len(tasks)}")
        print(f"  Total demos: {total_demos}")
        print(f"  Total frames: {total_frames}")
        print(f"  Sample tasks: {', '.join(list(tasks.keys())[:3])}...")
    
    # 첫 배치 테스트
    print("\n" + "="*80)
    print("Testing first batch...")
    print("="*80)
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image']
        actions = batch['action']
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"  Sample suite: {batch['suite_name'][0]}")
        print(f"  Sample task: {batch['task_name'][0]}")
        print(f"  Sample instruction: {batch['language_instruction'][0]}")
        print(f"  Sample demo: {batch['demo_name'][0]}, frame {batch['frame_idx'][0].item()}")
        
        # 첫 배치만 테스트
        break
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)
    print("\nYou can now use this DataLoader for training:")
    print("  from example_hdf5_dataloader import LIBEROHdf5Dataset")
    print("  dataset = LIBEROHdf5Dataset('./dataset_index.json')")
    print("  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)")


if __name__ == "__main__":
    main()

