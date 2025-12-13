"""
Compute and save dataset statistics for LIBERO HDF5 datasets.

This script computes normalization statistics for:
- Actions
- Proprioception
- Pointcloud (from tracking files)
- Tracking deltas (from tracking files)

Usage:
    python scripts/compute_libero_statistics.py \
        --libero_data_dir /scratch2/jisoo6687/libero/libero_goal_no_noops_track \
        --tracking_tracks_root /path/to/tracking_tracks \
        --output_path ./dataset_statistics.json
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from tqdm import tqdm


def compute_statistics(libero_data_dir: Path, tracking_tracks_root: Path) -> Dict:
    """
    Compute dataset statistics from LIBERO HDF5 files and tracking data.
    
    Args:
        libero_data_dir: Directory containing LIBERO HDF5 files
        tracking_tracks_root: Root directory for tracking data
    
    Returns:
        Dictionary containing all statistics
    """
    print("="*80)
    print("Computing LIBERO Dataset Statistics")
    print("="*80)
    print(f"Data directory: {libero_data_dir}")
    print(f"Tracking root: {tracking_tracks_root}")
    print()
    
    # Find all HDF5 files
    hdf5_files = sorted(glob.glob(str(libero_data_dir / "*_demo.hdf5")))
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Lists to accumulate data
    actions_list = []
    proprio_list = []
    pointcloud_list = []
    tracking_list = []
    
    num_episodes = 0
    num_frames = 0
    
    # Iterate through all HDF5 files
    for hdf5_path in tqdm(hdf5_files, desc="Processing HDF5 files"):
        task_name = Path(hdf5_path).stem.replace("_demo", "")
        
        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            
            for demo_name in sorted(data_grp.keys()):
                demo_grp = data_grp[demo_name]
                
                # Get point cloud identifier
                point_cloud_id = None
                if "obs" in demo_grp and "point_meta" in demo_grp["obs"]:
                    point_cloud_id = demo_grp["obs"]["point_meta"][()].decode("utf-8")
                
                if point_cloud_id is None:
                    # Create fallback identifier
                    demo_idx = demo_name.replace("demo_", "")
                    # Try to find matching tracking directory
                    matching_dirs = list(tracking_tracks_root.glob(f"*_{demo_idx}_{task_name}"))
                    if matching_dirs:
                        point_cloud_id = matching_dirs[0].name
                    else:
                        print(f"Warning: No tracking data found for {task_name}/{demo_name}")
                        continue
                
                # Load actions
                actions = demo_grp["actions"][()]
                actions_list.append(actions)
                num_frames += len(actions)
                
                # Load proprioception
                if "obs" in demo_grp:
                    gripper_states = demo_grp["obs"]["gripper_states"][()]
                    ee_states = demo_grp["obs"]["ee_states"][()]
                    proprio = np.concatenate([gripper_states, ee_states], axis=1)
                    proprio_list.append(proprio)
                
                # Load tracking data
                if tracking_tracks_root and point_cloud_id:
                    track_file = tracking_tracks_root / point_cloud_id / "vertex_tracks_face_uniform.npy"
                    
                    if track_file.exists():
                        try:
                            tracks = np.load(track_file)  # (T, num_points, 3)
                            
                            if len(tracks) > 0:
                                # Add ALL pointclouds (all timesteps, since any can be used as VLA input)
                                for t in range(len(tracks)):
                                    pointcloud_list.append(tracks[t])  # (num_points, 3)
                                
                                # Compute tracking deltas for all timesteps
                                for t in range(1, len(tracks)):
                                    delta = tracks[t] - tracks[t-1]  # (num_points, 3)
                                    tracking_list.append(delta)
                        except Exception as e:
                            print(f"Error loading {track_file}: {e}")
                
                num_episodes += 1
    
    print(f"\nProcessed {num_episodes} episodes, {num_frames} frames")
    
    # Compute action statistics
    print("\nComputing action statistics...")
    all_actions = np.concatenate(actions_list, axis=0)
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    action_min = np.min(all_actions, axis=0)
    action_max = np.max(all_actions, axis=0)
    
    statistics = {
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "min": action_min.tolist(),
            "max": action_max.tolist(),
            "q01": np.percentile(all_actions, 1, axis=0).tolist(),
            "q99": np.percentile(all_actions, 99, axis=0).tolist(),
        },
        "num_transitions": int(len(all_actions)),
        "num_trajectories": int(num_episodes),
    }
    
    print(f"  Shape: {all_actions.shape}")
    print(f"  Mean: {action_mean}")
    print(f"  Std: {action_std}")
    
    # Compute proprio statistics
    if proprio_list:
        print("\nComputing proprioception statistics...")
        all_proprio = np.concatenate(proprio_list, axis=0)
        proprio_mean = np.mean(all_proprio, axis=0)
        proprio_std = np.std(all_proprio, axis=0)
        proprio_min = np.min(all_proprio, axis=0)
        proprio_max = np.max(all_proprio, axis=0)
        
        statistics["proprio"] = {
            "mean": proprio_mean.tolist(),
            "std": proprio_std.tolist(),
            "min": proprio_min.tolist(),
            "max": proprio_max.tolist(),
            "q01": np.percentile(all_proprio, 1, axis=0).tolist(),
            "q99": np.percentile(all_proprio, 99, axis=0).tolist(),
        }
        
        print(f"  Shape: {all_proprio.shape}")
        print(f"  Mean: {proprio_mean}")
        print(f"  Std: {proprio_std}")
    
    # Compute pointcloud statistics (x, y, z separately)
    if pointcloud_list:
        print("\nComputing pointcloud statistics...")
        all_pointclouds = np.concatenate(pointcloud_list, axis=0)  # (total_points, 3)
        
        pc_mean = np.mean(all_pointclouds, axis=0)  # (3,)
        pc_std = np.std(all_pointclouds, axis=0)  # (3,)
        pc_min = np.min(all_pointclouds, axis=0)
        pc_max = np.max(all_pointclouds, axis=0)
        
        statistics["pointcloud"] = {
            "mean": pc_mean.tolist(),
            "std": pc_std.tolist(),
            "min": pc_min.tolist(),
            "max": pc_max.tolist(),
            "q01": np.percentile(all_pointclouds, 1, axis=0).tolist(),
            "q99": np.percentile(all_pointclouds, 99, axis=0).tolist(),
        }
        
        print(f"  Shape: {all_pointclouds.shape}")
        print(f"  Mean (x,y,z): {pc_mean}")
        print(f"  Std (x,y,z): {pc_std}")
    
    # Compute tracking statistics (x, y, z separately)
    if tracking_list:
        print("\nComputing tracking statistics...")
        all_tracking = np.concatenate(tracking_list, axis=0)  # (total_points, 3)
        
        track_mean = np.mean(all_tracking, axis=0)  # (3,)
        track_std = np.std(all_tracking, axis=0)  # (3,)
        track_min = np.min(all_tracking, axis=0)
        track_max = np.max(all_tracking, axis=0)
        
        statistics["tracking"] = {
            "mean": track_mean.tolist(),
            "std": track_std.tolist(),
            "min": track_min.tolist(),
            "max": track_max.tolist(),
            "q01": np.percentile(all_tracking, 1, axis=0).tolist(),
            "q99": np.percentile(all_tracking, 99, axis=0).tolist(),
        }
        
        print(f"  Shape: {all_tracking.shape}")
        print(f"  Mean (x,y,z): {track_mean}")
        print(f"  Std (x,y,z): {track_std}")
    
    return statistics


def main():
    parser = argparse.ArgumentParser(description="Compute LIBERO dataset statistics")
    parser.add_argument(
        "--libero_data_dir",
        type=str,
        required=True,
        help="Directory containing LIBERO HDF5 files"
    )
    parser.add_argument(
        "--tracking_tracks_root",
        type=str,
        required=True,
        help="Root directory for tracking data"
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_goal",
        help="Task suite name (e.g., libero_goal, libero_spatial)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dataset_statistics.json",
        help="Output path for statistics JSON file"
    )
    
    args = parser.parse_args()
    
    libero_data_dir = Path(args.libero_data_dir)
    tracking_tracks_root = Path(args.tracking_tracks_root)
    output_path = Path(args.output_path)
    
    if not libero_data_dir.exists():
        print(f"Error: Data directory not found: {libero_data_dir}")
        return 1
    
    if not tracking_tracks_root.exists():
        print(f"Error: Tracking root not found: {tracking_tracks_root}")
        return 1
    
    # Compute statistics
    statistics = compute_statistics(libero_data_dir, tracking_tracks_root)
    
    # Wrap in RLDS format: {dataset_name: {action: {...}, ...}}
    statistics_rlds = {
        args.task_suite: statistics
    }
    
    # Save to JSON
    print(f"\nSaving statistics to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(statistics_rlds, f, indent=2)
    
    print("Done!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())

