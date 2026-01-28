"""
Compute and save dataset statistics for RoboCasa HDF5 datasets with ablation mode support.

This script computes normalization statistics for:
- Actions (first 7 dimensions only)
- Proprioception
- Pointcloud (from tracking files, with optional ablation padding/truncation)
- Tracking deltas (from tracking files, with optional ablation padding/truncation)

Usage:
    # Without ablation (same as compute_robocasa_statistics.py):
    python scripts/compute_robocasa_statistics_ablation.py \
        --robocasa_data_dir /weka/jisookim/dataset/robocasa/datasets/regenerate_single/regenerate_single \
        --tracking_tracks_root /path/to/tracking_tracks \
        --output_path ./robocasa_statistics.json

    # With ablation mode (pad/truncate points to mean):
    python scripts/compute_robocasa_statistics_ablation.py \
        --robocasa_data_dir /weka/jisookim/dataset/robocasa/datasets/regenerate_single/regenerate_single \
        --tracking_tracks_root /weka/jisookim/dataset/robocasa/datasets/scene_pointrack \
        --tracking_filename scene_points.npy \
        --ablation_mean_points_path ./ablation_mean_points.json \
        --output_path ./robocasa_statistics_ablation.json
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from tqdm import tqdm


def pad_or_truncate_points(points: np.ndarray, target_n: int) -> np.ndarray:
    """
    Pad or truncate points to match target number of points.

    Args:
        points: np.ndarray of shape (N, D) or (T, N, D)
            where N is number of points, D is dimension (usually 3)
        target_n: Target number of points

    Returns:
        Padded/truncated array with N = target_n
    """
    if points.ndim == 2:
        # Shape: (N, D)
        N, D = points.shape
        if N > target_n:
            return points[:target_n]
        elif N < target_n:
            pad_shape = (target_n - N, D)
            padding = np.zeros(pad_shape, dtype=points.dtype)
            return np.concatenate([points, padding], axis=0)
        else:
            return points

    elif points.ndim == 3:
        # Shape: (T, N, D)
        T, N, D = points.shape
        if N > target_n:
            return points[:, :target_n, :]
        elif N < target_n:
            pad_shape = (T, target_n - N, D)
            padding = np.zeros(pad_shape, dtype=points.dtype)
            return np.concatenate([points, padding], axis=1)
        else:
            return points

    else:
        return points


def compute_statistics(
    robocasa_data_dir: Path,
    tracking_tracks_root: Optional[Path] = None,
    tracking_filename: str = "vertex_tracks_face_uniform.npy",
    ablation_mean_points: Optional[int] = None,
) -> Dict:
    """
    Compute dataset statistics from RoboCasa HDF5 files and optional tracking data.

    Args:
        robocasa_data_dir: Root directory containing RoboCasa HDF5 files
        tracking_tracks_root: Root directory for tracking data (optional)
        tracking_filename: Filename for tracking data
        ablation_mean_points: If set, pad/truncate points to this value

    Returns:
        Dictionary containing all statistics
    """
    print("=" * 80)
    print("Computing RoboCasa Dataset Statistics (Ablation Mode)")
    print("=" * 80)
    print(f"Data directory: {robocasa_data_dir}")
    if tracking_tracks_root:
        print(f"Tracking root: {tracking_tracks_root}")
        print(f"Tracking filename: {tracking_filename}")
    else:
        print("Tracking root: None (skipping pointcloud/tracking statistics)")
    if ablation_mean_points:
        print(f"[Ablation Mode] Padding/truncating points to {ablation_mean_points}")
    print()

    # Find all HDF5 files (recursive search for RoboCasa structure)
    # Pattern: data_dir/*/*/*/*.hdf5 (e.g., kitchen_coffee/CoffeePressButton/2024-04-25/*.hdf5)
    hdf5_files = sorted(glob.glob(str(robocasa_data_dir / "*" / "*" / "*" / "*.hdf5")))
    print(f"Found {len(hdf5_files)} HDF5 files")

    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 files found in {robocasa_data_dir}/*/*/*/*.hdf5")

    # Lists to accumulate data
    actions_list = []
    proprio_list = []
    pointcloud_list = []
    tracking_list = []

    num_episodes = 0
    num_frames = 0

    # Iterate through all HDF5 files
    for hdf5_path in tqdm(hdf5_files, desc="Processing HDF5 files"):
        # Extract task name from path structure
        path_parts = Path(hdf5_path).parts
        task_name = path_parts[-3] if len(path_parts) >= 3 else Path(hdf5_path).stem

        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]

            # Sort demo names numerically (demo_1, demo_2, ..., demo_10, ...)
            demo_names = sorted(data_grp.keys(), key=lambda x: int(x.split("_")[1]))

            for demo_name in demo_names:
                demo_grp = data_grp[demo_name]

                # Build tracking path from HDF5 path structure
                # HDF5: .../single_stage_regenerate/kitchen_coffee/CoffeePressButton/2024-04-25/demo.hdf5
                # Track: .../pointrack/kitchen_coffee/CoffeePressButton/2024-04-25/demo_1/vertex_tracks.npy
                # Extract: kitchen_coffee/CoffeePressButton/2024-04-25 from HDF5 path
                tracking_subpath = None
                hdf5_parts = Path(hdf5_path).parts
                for marker in ["single_stage_regenerate", "regenerate_single"]:
                    if marker in hdf5_parts:
                        marker_idx = hdf5_parts.index(marker)
                        # Get parts after marker, excluding the filename
                        tracking_subpath = "/".join(hdf5_parts[marker_idx + 1 : -1])
                        break
                # Fallback: use last 3 directories before filename
                if tracking_subpath is None:
                    tracking_subpath = "/".join(hdf5_parts[-4:-1])

                # Load actions (only first 7 dimensions)
                actions = demo_grp["actions"][()][:, :7]
                actions_list.append(actions)
                num_frames += len(actions)

                # Load proprioception
                if "obs" in demo_grp:
                    gripper_states = demo_grp["obs"]["gripper_states"][()]
                    ee_states = demo_grp["obs"]["ee_states"][()]
                    proprio = np.concatenate([gripper_states, ee_states], axis=1)
                    proprio_list.append(proprio)

                # Load tracking data (optional)
                # Path: tracking_tracks_root / tracking_subpath / demo_name / filename
                # e.g., pointrack/kitchen_coffee/CoffeePressButton/2024-04-25/demo_1/vertex_tracks_face_uniform.npy
                if tracking_tracks_root and tracking_subpath:
                    track_file = tracking_tracks_root / tracking_subpath / demo_name / tracking_filename

                    if track_file.exists():
                        try:
                            tracks = np.load(track_file)  # (T, num_points, 3)

                            # Apply ablation padding/truncation if enabled
                            if ablation_mean_points is not None:
                                tracks = pad_or_truncate_points(tracks, ablation_mean_points)

                            if len(tracks) > 0:
                                # Add ALL pointclouds (all timesteps, since any can be used as VLA input)
                                for t in range(len(tracks)):
                                    pointcloud_list.append(tracks[t])  # (num_points, 3)

                                # Compute tracking deltas for all timesteps
                                for t in range(1, len(tracks)):
                                    delta = tracks[t] - tracks[t - 1]  # (num_points, 3)
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

    # Add ablation info to statistics
    if ablation_mean_points is not None:
        statistics["ablation_mean_points"] = ablation_mean_points

    return statistics


def main():
    parser = argparse.ArgumentParser(description="Compute RoboCasa dataset statistics with ablation support")
    parser.add_argument(
        "--robocasa_data_dir",
        type=str,
        required=True,
        help="Root directory containing RoboCasa HDF5 files",
    )
    parser.add_argument(
        "--tracking_tracks_root",
        type=str,
        default=None,
        help="Root directory for tracking data (optional)",
    )
    parser.add_argument(
        "--tracking_filename",
        type=str,
        default="vertex_tracks_face_uniform.npy",
        help="Filename for tracking data",
    )
    parser.add_argument(
        "--ablation_mean_points_path",
        type=str,
        default=None,
        help="Path to JSON file with precomputed mean point count (from compute_ablation_mean_points.py)",
    )
    parser.add_argument(
        "--ablation_mean_points",
        type=int,
        default=None,
        help="Directly specify mean points value (alternative to ablation_mean_points_path)",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="robocasa",
        help="Task suite name (e.g., robocasa)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="robocasa_statistics.json",
        help="Output path for statistics JSON file",
    )

    args = parser.parse_args()

    robocasa_data_dir = Path(args.robocasa_data_dir)
    tracking_tracks_root = Path(args.tracking_tracks_root) if args.tracking_tracks_root else None
    output_path = Path(args.output_path)

    if not robocasa_data_dir.exists():
        print(f"Error: Data directory not found: {robocasa_data_dir}")
        return 1

    if tracking_tracks_root and not tracking_tracks_root.exists():
        print(f"Warning: Tracking root not found: {tracking_tracks_root}")
        print("Continuing without tracking data...")
        tracking_tracks_root = None

    # Load or use ablation mean points
    ablation_mean_points = args.ablation_mean_points
    if args.ablation_mean_points_path:
        ablation_path = Path(args.ablation_mean_points_path)
        if not ablation_path.exists():
            print(f"Error: Ablation mean points file not found: {ablation_path}")
            return 1
        with open(ablation_path, "r") as f:
            ablation_stats = json.load(f)
        ablation_mean_points = ablation_stats["mean_points"]
        print(f"Loaded ablation_mean_points={ablation_mean_points} from {ablation_path}")

    # Compute statistics
    statistics = compute_statistics(
        robocasa_data_dir,
        tracking_tracks_root,
        args.tracking_filename,
        ablation_mean_points,
    )

    # Wrap in RLDS format: {dataset_name: {action: {...}, ...}}
    statistics_rlds = {args.task_suite: statistics}

    # Save to JSON
    print(f"\nSaving statistics to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(statistics_rlds, f, indent=2)

    print("Done!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
