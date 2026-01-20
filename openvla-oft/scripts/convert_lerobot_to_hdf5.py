#!/usr/bin/env python3
"""
LeRobot v3.0 → HDF5 변환 스크립트

LeRobot v3.0 형식의 데이터셋을 robocasa_hdf5_dataset.py가 읽을 수 있는
HDF5 형식으로 변환합니다.

Usage:
    python scripts/convert_lerobot_to_hdf5.py \
        --input_dir /workspace/human_demon_action/robot_sample_data/omy_f3m_simple_santa_pri \
        --output_dir /workspace/human_demon_action/robot_sample_data_hdf5 \
        --language_instruction "pick up the santa and place it in the basket"

Input (LeRobot v3.0):
    input_dir/
    ├── meta/
    │   ├── info.json
    │   └── tasks.parquet
    ├── data/
    │   └── chunk-XXX/file-000.parquet
    └── videos/
        ├── observation.images.cam_third/chunk-XXX/file-XXX.mp4
        ├── observation.images.cam_top/chunk-XXX/file-XXX.mp4
        └── observation.images.cam_wrist/chunk-XXX/file-XXX.mp4

Output (HDF5 format):
    output_dir/
    └── task_name/
        └── dataset_name/
            └── date/
                └── demo.hdf5
                    └── data/
                        ├── demo_1/
                        │   ├── obs/
                        │   │   ├── cam_third: (T, H, W, 3)    # primary camera
                        │   │   ├── cam_top: (T, H, W, 3)      # secondary camera
                        │   │   ├── cam_wrist: (T, H, W, 3)    # wrist camera
                        │   │   ├── gripper_states: (T, 2)
                        │   │   ├── ee_states: (T, 6)
                        │   │   ├── joint_states: (T, 7)
                        │   │   └── language: str
                        │   ├── actions: (T-1, 7)
                        │   ├── rewards: (T-1,)
                        │   └── dones: (T-1,)
                        ├── demo_2/
                        └── ...
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_video_frames(video_path: str) -> np.ndarray:
    """Read all frames from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        np.ndarray of shape (T, H, W, 3) in RGB format
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return np.zeros((1, 480, 640, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(f"Warning: No frames read from: {video_path}")
        return np.zeros((1, 480, 640, 3), dtype=np.uint8)

    return np.array(frames, dtype=np.uint8)


def load_lerobot_data(input_dir: Path) -> Tuple[Dict, pd.DataFrame, Dict[str, str]]:
    """Load LeRobot v3.0 dataset metadata and parquet data.

    Args:
        input_dir: Path to LeRobot dataset root

    Returns:
        (info_dict, all_data_df, tasks_dict)
    """
    # Load info.json
    with open(input_dir / "meta" / "info.json", "r") as f:
        info = json.load(f)

    # Load tasks.parquet
    tasks = {}
    tasks_parquet_path = input_dir / "meta" / "tasks.parquet"
    if tasks_parquet_path.exists():
        tasks_df = pd.read_parquet(tasks_parquet_path)
        # v3 format: index is task description, column is task_index
        for task_desc, row in tasks_df.iterrows():
            tasks[row["task_index"]] = task_desc

    # Load all parquet data
    all_dfs = []
    chunk_idx = 0
    while True:
        parquet_path = input_dir / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet"
        if not parquet_path.exists():
            break
        df = pd.read_parquet(parquet_path)
        all_dfs.append(df)
        chunk_idx += 1

    if all_dfs:
        all_data = pd.concat(all_dfs, ignore_index=True)
    else:
        all_data = pd.DataFrame()

    return info, all_data, tasks


def convert_episode_to_hdf5(
    episode_idx: int,
    episode_df: pd.DataFrame,
    input_dir: Path,
    info: Dict,
    tasks: Dict[int, str],
    language_instruction: Optional[str] = None,
) -> Dict:
    """Convert a single episode to HDF5-compatible format.

    Args:
        episode_idx: Episode index
        episode_df: DataFrame containing episode data
        input_dir: LeRobot dataset root
        info: Dataset info dict
        tasks: Task index to description mapping
        language_instruction: Override language instruction (optional)

    Returns:
        Dictionary with episode data ready for HDF5
    """
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode_idx // chunks_size

    # Sort by frame index
    episode_df = episode_df.sort_values("frame_index").reset_index(drop=True)
    num_frames = len(episode_df)

    # Get task description
    task_idx = int(episode_df["task_index"].iloc[0])
    if language_instruction:
        lang = language_instruction
    else:
        lang = tasks.get(task_idx, "perform the task")

    # Load video frames for all 3 cameras
    # LeRobot camera names → HDF5 camera names mapping:
    # We keep the same names as LeRobot for consistency
    # cam_third → cam_third (primary, used as agentview_left in robocasa)
    # cam_top → cam_top (secondary, used as agentview_right in robocasa)
    # cam_wrist → cam_wrist (wrist, used as eye_in_hand in robocasa)
    camera_mapping = {
        "cam_third": "observation.images.cam_third",
        "cam_top": "observation.images.cam_top",
        "cam_wrist": "observation.images.cam_wrist",
    }

    frames = {}
    for hdf5_name, lerobot_name in camera_mapping.items():
        video_path = (
            input_dir / "videos" / lerobot_name /
            f"chunk-{chunk_idx:03d}" / f"file-{episode_idx:03d}.mp4"
        )
        frames[hdf5_name] = read_video_frames(str(video_path))

    # Get states and actions
    # observation.state: [joint1-6, rh_r1_joint] = 7D
    states = np.stack(episode_df["observation.state"].values).astype(np.float32)
    actions = np.stack(episode_df["action"].values).astype(np.float32)

    # Create gripper_states (2D) - use last joint as gripper, duplicate for 2D
    # RoboCasa expects (T, 2) gripper states
    gripper_states = np.column_stack([
        states[:, 6],  # gripper position
        states[:, 6],  # duplicate (RoboCasa has 2-finger gripper)
    ]).astype(np.float32)

    # Create ee_states (6D) - placeholder since LeRobot doesn't have EE pose
    # We'll use zeros or derive from joint states if needed
    # For now, using placeholder zeros
    ee_states = np.zeros((num_frames, 6), dtype=np.float32)

    # Create joint_states (7D) - same as observation.state
    joint_states = states.astype(np.float32)

    # Create rewards and dones
    # Reward: 0 for all except last frame (1.0)
    # Done: False for all except last frame (True)
    rewards = np.zeros(num_frames - 1, dtype=np.float32)
    rewards[-1] = 1.0
    dones = np.zeros(num_frames - 1, dtype=np.bool_)
    dones[-1] = True

    return {
        "obs": {
            "cam_third": frames["cam_third"],      # primary camera
            "cam_top": frames["cam_top"],          # secondary camera
            "cam_wrist": frames["cam_wrist"],      # wrist camera
            "gripper_states": gripper_states,
            "ee_states": ee_states,
            "joint_states": joint_states,
            "language": lang,
        },
        "actions": actions[:-1] if len(actions) > 1 else actions,  # T-1 actions
        "rewards": rewards,
        "dones": dones,
    }


def convert_statistics(input_dir: Path, output_dir: Path, task_suite: str) -> None:
    """Convert LeRobot stats.json to omy_f3m_hdf5_dataset.py compatible format.

    LeRobot stats.json format:
        - observation.state: proprio (7D)
        - action: action (7D)

    Output stats.json format (wrapped with task_suite name):
        {
            "task_suite": {
                "action": {"min": [...], "max": [...], "mean": [...], "std": [...], "q01": [...], "q99": [...]},
                "proprio": {"min": [...], "max": [...], "mean": [...], "std": [...], "q01": [...], "q99": [...]}
            }
        }
    """
    input_stats_path = input_dir / "meta" / "stats.json"
    if not input_stats_path.exists():
        print(f"  Warning: stats.json not found at {input_stats_path}")
        return

    with open(input_stats_path, "r") as f:
        lerobot_stats = json.load(f)

    # Extract action and proprio (observation.state) statistics
    output_stats = {}

    # Action statistics
    if "action" in lerobot_stats:
        output_stats["action"] = {
            "min": lerobot_stats["action"]["min"],
            "max": lerobot_stats["action"]["max"],
            "mean": lerobot_stats["action"]["mean"],
            "std": lerobot_stats["action"]["std"],
            "q01": lerobot_stats["action"]["q01"],
            "q99": lerobot_stats["action"]["q99"],
        }

    # Proprio statistics (from observation.state)
    if "observation.state" in lerobot_stats:
        output_stats["proprio"] = {
            "min": lerobot_stats["observation.state"]["min"],
            "max": lerobot_stats["observation.state"]["max"],
            "mean": lerobot_stats["observation.state"]["mean"],
            "std": lerobot_stats["observation.state"]["std"],
            "q01": lerobot_stats["observation.state"]["q01"],
            "q99": lerobot_stats["observation.state"]["q99"],
        }

    # Wrap with task_suite name for compatibility
    wrapped_stats = {task_suite: output_stats}

    # Save to output directory
    output_stats_path = output_dir / "stats.json"
    with open(output_stats_path, "w") as f:
        json.dump(wrapped_stats, f, indent=4)

    print(f"  Statistics saved to: {output_stats_path}")


def convert_lerobot_to_hdf5(
    input_dir: Path,
    output_dir: Path,
    language_instruction: Optional[str] = None,
    task_name: str = "real_world",
    dataset_name: Optional[str] = None,
) -> None:
    """Convert LeRobot v3.0 dataset to HDF5 format.

    Args:
        input_dir: Path to LeRobot dataset
        output_dir: Path to output HDF5 directory
        language_instruction: Language instruction for all episodes
        task_name: Task category name (e.g., "real_world", "kitchen_coffee")
        dataset_name: Dataset name (defaults to input directory name)
    """
    print(f"\n{'='*60}")
    print(f"LeRobot v3.0 → HDF5 Converter")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Load LeRobot data
    print("\n[1/3] Loading LeRobot data...")
    info, all_data, tasks = load_lerobot_data(input_dir)

    total_episodes = info["total_episodes"]
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")

    # Determine output path structure
    # robocasa_hdf5_dataset.py expects: data_dir/*/*/*/*.hdf5
    if dataset_name is None:
        dataset_name = input_dir.name

    date_str = datetime.now().strftime("%Y-%m-%d")
    hdf5_dir = output_dir / task_name / dataset_name / date_str
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = hdf5_dir / "demo.hdf5"

    print(f"\n[2/3] Converting episodes to HDF5...")
    print(f"  Output file: {hdf5_path}")

    with h5py.File(hdf5_path, "w") as f:
        # Create data group
        data_grp = f.create_group("data")

        for episode_idx in tqdm(range(total_episodes), desc="Converting"):
            # Filter data for this episode
            episode_df = all_data[all_data["episode_index"] == episode_idx].copy()

            if len(episode_df) == 0:
                print(f"  Warning: No data for episode {episode_idx}")
                continue

            # Convert episode
            episode_data = convert_episode_to_hdf5(
                episode_idx=episode_idx,
                episode_df=episode_df,
                input_dir=input_dir,
                info=info,
                tasks=tasks,
                language_instruction=language_instruction,
            )

            # Create demo group (demo_1, demo_2, ... - 1-indexed)
            demo_name = f"demo_{episode_idx + 1}"
            demo_grp = data_grp.create_group(demo_name)

            # Create obs group
            obs_grp = demo_grp.create_group("obs")

            # Store observations
            for key, value in episode_data["obs"].items():
                if key == "language":
                    # Store language as string
                    obs_grp.create_dataset(
                        key,
                        data=np.array(value, dtype=h5py.string_dtype())
                    )
                else:
                    obs_grp.create_dataset(key, data=value, compression="gzip")

            # Store actions, rewards, dones
            demo_grp.create_dataset("actions", data=episode_data["actions"], compression="gzip")
            demo_grp.create_dataset("rewards", data=episode_data["rewards"])
            demo_grp.create_dataset("dones", data=episode_data["dones"])

    print(f"\n[3/4] Converting statistics...")
    convert_statistics(input_dir, hdf5_dir, task_suite=dataset_name)

    print(f"\n[4/4] Conversion complete!")
    print(f"{'='*60}")
    print(f"Output HDF5: {hdf5_path}")
    print(f"Output stats: {hdf5_dir / 'stats.json'}")
    print(f"")
    print(f"To use with omy_f3m_hdf5_dataset.py:")
    print(f"  data_dir = '{output_dir}'")
    print(f"  task_suite = '{dataset_name}'")
    print(f"  precomputed_statistics_path = '{hdf5_dir / 'stats.json'}'")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3.0 dataset to HDF5 format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to LeRobot v3.0 dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output HDF5 directory",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default=None,
        help="Language instruction for all episodes (overrides tasks.parquet)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="real_world",
        help="Task category name (default: real_world)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name (default: input directory name)",
    )

    args = parser.parse_args()

    convert_lerobot_to_hdf5(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        language_instruction=args.language_instruction,
        task_name=args.task_name,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
