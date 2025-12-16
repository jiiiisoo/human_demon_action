"""
LIBERO HDF5 Dataset Loader for OpenVLA Fine-tuning

This module provides a PyTorch dataset that loads LIBERO demonstrations from HDF5 files
created by regenerate_libero_dataset_with_tracks.py and formats them for OpenVLA training.

HDF5 Structure (created by regenerate_libero_dataset_with_tracks.py):
    data/
        demo_0/
            obs/
                point_meta: str (point cloud identifier)
                gripper_states: (T, 2)
                joint_states: (T, 7)
                ee_states: (T, 6) [pos(3) + ori(3)]
                ee_pos: (T, 3)
                ee_ori: (T, 3)
                agentview_rgb: (T, H, W, 3)
                eye_in_hand_rgb: (T, H, W, 3)
            actions: (T, 7)
            states: (T, state_dim)
            robot_states: (T, robot_state_dim)
            rewards: (T,)
            dones: (T,)
        demo_1/
            ...
"""

import glob
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import h5py
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX
from dataclasses import dataclass


@dataclass
class LIBEROBatchTransform:
    """
    Batch transform for LIBERO HDF5 dataset.
    
    Similar to RLDSBatchTransform but adapted for LIBERO data format:
    - Images are already single frames (no [0] indexing needed)
    - Actions are already chunked
    - Language instructions are already strings
    - Pointcloud and tracking are loaded from tracking files in _load_frame
    """
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_pointcloud_input: bool = False
    use_tracking_head: bool = False
    
    def __call__(self, libero_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a LIBERO batch to the format expected by the OpenVLA collator/models.
        
        LIBERO batch format:
            observation:
                image_primary: (H, W, 3) - single frame
                image_wrist: (H, W, 3) - single frame
                proprio: (8,) - single timestep
                episode_name: str
            action: (action_chunk_size, 7) - already chunked
            task:
                language_instruction: str - already string
            dataset_name: str
        """
        dataset_name = libero_batch["dataset_name"]
        current_action = libero_batch["action"][0]  # First action in chunk
        actions = libero_batch["action"]  # Full action chunk
        
        # Image is already a single frame (H, W, 3)
        img = Image.fromarray(libero_batch["observation"]["image_primary"])
        
        # Language instruction is already a string
        lang = libero_batch["task"]["language_instruction"].lower()
        
        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        
        # Get future action chunk
        future_actions = libero_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        
        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        
        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        
        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        
        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions
        )
        
        # Add wrist image if needed
        if self.use_wrist_image:
            img_wrist = Image.fromarray(libero_batch["observation"]["image_wrist"])
            pixel_values_wrist = self.image_transform(img_wrist)
            return_dict["pixel_values_wrist"] = pixel_values_wrist.unsqueeze(0)  # Add batch-like dimension
        
        # Add proprioception if needed
        if self.use_proprio and "proprio" in libero_batch["observation"]:
            proprio = libero_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio
        
        # Add pointcloud if available and requested
        # Note: pointcloud is needed for BOTH VLA input and tracking head input
        if "pointcloud" in libero_batch:
            pointcloud = libero_batch["pointcloud"]
            if pointcloud is not None:
                if isinstance(pointcloud, torch.Tensor):
                    pointcloud_tensor = pointcloud.float()
                else:
                    pointcloud_tensor = torch.as_tensor(np.array(pointcloud), dtype=torch.float32)
                return_dict["pointcloud"] = pointcloud_tensor
        
        # Add tracking deltas if available and requested
        if "tracking" in libero_batch:
            tracking = libero_batch["tracking"]
            if tracking is not None:
                if isinstance(tracking, torch.Tensor):
                    tracking_tensor = tracking.float()
                else:
                    tracking_tensor = torch.as_tensor(np.array(tracking), dtype=torch.float32)
                return_dict["tracking"] = tracking_tensor
        
        return return_dict


class LIBEROHdf5Dataset(Dataset):
    """
    PyTorch Dataset that loads LIBERO demonstrations from HDF5 files
    and yields them in a format compatible with RLDSBatchTransform.
    
    This is a map-style dataset that supports random access via __getitem__,
    enabling proper shuffling across epochs via DataLoader.
    """

    def __init__(
        self,
        data_dir: Path,
        task_suite: str,
        batch_transform: Callable,
        resize_resolution: Tuple[int, int] = (256, 256),
        shuffle_buffer_size: int = 1000,
        train: bool = True,
        image_aug: bool = False,
        use_wrist_image: bool = False,
        tracking_tracks_root: Optional[Path] = None,
        action_chunk_size: int = 1,
        window_stride: int = 1,
        seed: int = 42,
        normalize_pointcloud: bool = True,
        normalize_tracking: bool = True,
        precomputed_statistics_path: Optional[Path] = None,
        filename: str = "vertex_tracks_face_uniform.npy",
    ) -> None:
        """
        Args:
            data_dir: Directory containing LIBERO HDF5 files (e.g., /path/to/libero_goal_no_noops_track/)
            task_suite: Name of the task suite (e.g., "libero_goal")
            batch_transform: Transform to apply to batches (RLDSBatchTransform)
            resize_resolution: Target image resolution (H, W)
            shuffle_buffer_size: Size of shuffle buffer for randomizing samples
            train: Whether this is training dataset (affects shuffling)
            image_aug: Whether to apply image augmentations
            use_wrist_image: Whether to use wrist camera images
            tracking_tracks_root: Root directory for tracking data (pointcloud + tracking deltas)
            action_chunk_size: Number of future actions to include (for action chunking)
            window_stride: Stride for sampling frames (1 = all frames, >1 = skip frames for less overlap)
            seed: Random seed for shuffling
            normalize_pointcloud: If True, normalize pointcloud input using dataset statistics (x, y, z separately)
            normalize_tracking: If True, normalize tracking data using dataset statistics (x, y, z separately)
            precomputed_statistics_path: Path to precomputed statistics JSON file (if None, compute on-the-fly)
        """
        self.data_dir = Path(data_dir)
        self.task_suite = task_suite
        self.batch_transform = batch_transform
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.image_aug = image_aug
        self.use_wrist_image = use_wrist_image
        self.tracking_tracks_root = Path(tracking_tracks_root) if tracking_tracks_root else None
        self.action_chunk_size = action_chunk_size
        self.window_stride = window_stride
        self.seed = seed
        self.should_normalize_pointcloud = normalize_pointcloud
        self.should_normalize_tracking = normalize_tracking
        self.precomputed_statistics_path = Path(precomputed_statistics_path) if precomputed_statistics_path else None
        self.filename = filename
        
        # Find all HDF5 files in the directory
        self.hdf5_files = sorted(glob.glob(str(self.data_dir / "*_demo.hdf5")))
        assert len(self.hdf5_files) > 0, f"No HDF5 files found in {self.data_dir}"
        
        print(f"\n[LIBERO HDF5 Dataset]")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Found {len(self.hdf5_files)} HDF5 files")
        print(f"  Task suite: {self.task_suite}")
        print(f"  Image resolution: {resize_resolution}")
        print(f"  Wrist camera: {use_wrist_image}")
        print(f"  Action chunk size: {action_chunk_size}")
        print(f"  Window stride: {window_stride}")
        
        # Build index of all episodes
        self._build_episode_index()
        
        # Build flat frame index for random access
        self._build_frame_index()
        
        # Load or compute dataset statistics
        if self.precomputed_statistics_path and self.precomputed_statistics_path.exists():
            print(f"  Loading precomputed statistics from {self.precomputed_statistics_path}")
            with open(self.precomputed_statistics_path, "r") as f:
                loaded_stats = json.load(f)
            
            # Check if it's in RLDS format {dataset_name: {action: ...}} or flat format {action: ...}
            if self.task_suite in loaded_stats:
                # RLDS format
                self.dataset_statistics = loaded_stats
                self._statistics_data = loaded_stats[self.task_suite]
            elif "action" in loaded_stats:
                # Flat format - wrap it
                self._statistics_data = loaded_stats
                self.dataset_statistics = {self.task_suite: loaded_stats}
            else:
                raise ValueError(f"Invalid precomputed statistics format in {self.precomputed_statistics_path}")
            
            print(f"  ✓ Statistics loaded successfully")
        else:
            print("  Computing dataset statistics (this may take a while)...")
            self._compute_statistics()
        
    def _build_episode_index(self):
        """Build an index of all episodes across all HDF5 files."""
        self.episode_index = []  # List of (hdf5_path, demo_name, num_frames, language_instruction)
        
        for hdf5_path in self.hdf5_files:
            # Extract task name from filename (e.g., "open_the_middle_drawer_of_the_cabinet_demo.hdf5")
            task_name = Path(hdf5_path).stem.replace("_demo", "")
            language_instruction = task_name.replace("_", " ")
            
            with h5py.File(hdf5_path, "r") as f:
                data_grp = f["data"]
                
                for demo_name in sorted(data_grp.keys()):
                    demo_grp = data_grp[demo_name]
                    
                    # Get number of frames (actions)
                    num_frames = len(demo_grp["actions"])
                    
                    # Get point cloud identifier if available
                    point_cloud_id = None
                    if "obs" in demo_grp and "point_meta" in demo_grp["obs"]:
                        point_cloud_id = demo_grp["obs"]["point_meta"][()].decode("utf-8")
                    
                    self.episode_index.append({
                        "hdf5_path": hdf5_path,
                        "demo_name": demo_name,
                        "num_frames": num_frames,
                        "language_instruction": language_instruction,
                        "task_name": task_name,
                        "point_cloud_id": point_cloud_id,
                    })
        
        print(f"  Total episodes: {len(self.episode_index)}")
        total_frames = sum(ep["num_frames"] for ep in self.episode_index)
        print(f"  Total frames: {total_frames}")
        
        self.dataset_length = total_frames
    
    def _build_frame_index(self):
        """
        Build a flat index mapping global frame index to (episode_idx, frame_idx).
        
        Note: We exclude the last (action_chunk_size - 1) frames from each episode
        to ensure we can always load a complete action chunk without padding.
        Window stride controls the sampling density (1 = all frames, >1 = skip frames).
        """
        self.frame_index = []  # List of (episode_idx, frame_idx) tuples
        
        for ep_idx, ep_info in enumerate(self.episode_index):
            # Ensure we can load action_chunk_size actions starting from frame_idx
            # Valid frame_idx: 0 <= frame_idx <= num_frames - action_chunk_size
            max_valid_frame_idx = max(0, ep_info["num_frames"] - self.action_chunk_size)
            
            # Sample frames with stride
            for frame_idx in range(0, max_valid_frame_idx, self.window_stride):
                self.frame_index.append((ep_idx, frame_idx))
        
        # Update dataset_length to actual number of valid samples
        self.dataset_length = len(self.frame_index)
        
        print(f"  Built frame index: {len(self.frame_index)} frames (stride={self.window_stride})")
        print(f"  (Excluded last {self.action_chunk_size - 1} frames per episode for action chunking)")
        
    def _compute_statistics(self):
        """Compute dataset statistics for normalization from ALL data."""
        print("  Computing dataset statistics from all episodes...")
        print("  This may take a while for large datasets...")
        
        actions_list = []
        proprio_list = []
        pointcloud_list = []
        tracking_list = []
        
        for ep_info in tqdm.tqdm(self.episode_index, desc="Computing statistics"):
            with h5py.File(ep_info["hdf5_path"], "r") as f:
                demo_grp = f["data"][ep_info["demo_name"]]
                actions = demo_grp["actions"][()]
                actions_list.append(actions)
                
                # Proprioception: gripper_states (2D) + ee_pos (3D) + ee_ori (3D) = 8D
                if "obs" in demo_grp:
                    gripper_states = demo_grp["obs"]["gripper_states"][()]
                    ee_states = demo_grp["obs"]["ee_states"][()]
                    proprio = np.concatenate([gripper_states, ee_states], axis=1)
                    proprio_list.append(proprio)
            
            # Load ALL tracking data from tracking file
            if self.tracking_tracks_root and ep_info["point_cloud_id"]:
                track_file = self.tracking_tracks_root / ep_info["point_cloud_id"] / self.filename
                if track_file.exists():
                    try:
                        tracks = np.load(track_file)  # (T, num_points, 3)
                        if len(tracks) > 0:
                            # Add ALL pointclouds (all timesteps, since any can be used as VLA input)
                            for t in range(len(tracks)):
                                pointcloud_list.append(tracks[t])  # (num_points, 3)
                            
                            # Compute ALL tracking deltas
                            for t in range(1, len(tracks)):
                                delta = tracks[t] - tracks[t-1]  # (num_points, 3)
                                tracking_list.append(delta)
                    except Exception as e:
                        print(f"  Warning: Error loading {track_file}: {e}")
        
        all_actions = np.concatenate(actions_list, axis=0)
        
        # Compute action statistics
        action_mean = np.mean(all_actions, axis=0)
        action_std = np.std(all_actions, axis=0)
        action_min = np.min(all_actions, axis=0)
        action_max = np.max(all_actions, axis=0)
        
        # Wrap in RLDS format: {dataset_name: {action: {...}, ...}}
        stats_data = {
            "action": {
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "min": action_min.tolist(),
                "max": action_max.tolist(),
                "q01": np.percentile(all_actions, 1, axis=0).tolist(),
                "q99": np.percentile(all_actions, 99, axis=0).tolist(),
            },
            "num_transitions": len(all_actions),
            "num_trajectories": len(self.episode_index),
        }
        
        # Keep internal reference without dataset wrapper for easy access
        self._statistics_data = stats_data.copy()
        
        # Add proprio statistics if available
        if proprio_list:
            all_proprio = np.concatenate(proprio_list, axis=0)
            proprio_mean = np.mean(all_proprio, axis=0)
            proprio_std = np.std(all_proprio, axis=0)
            proprio_min = np.min(all_proprio, axis=0)
            proprio_max = np.max(all_proprio, axis=0)
            
            stats_data["proprio"] = {
                "mean": proprio_mean.tolist(),
                "std": proprio_std.tolist(),
                "min": proprio_min.tolist(),
                "max": proprio_max.tolist(),
                "q01": np.percentile(all_proprio, 1, axis=0).tolist(),
                "q99": np.percentile(all_proprio, 99, axis=0).tolist(),
            }
            self._statistics_data["proprio"] = stats_data["proprio"].copy()
        
        # Compute pointcloud statistics if available (x, y, z separately)
        if pointcloud_list:
            print("  Computing pointcloud statistics...")
            all_pointclouds = np.concatenate(pointcloud_list, axis=0)  # (total_points, 3)
            
            # Compute mean and std for each dimension (x, y, z)
            pc_mean = np.mean(all_pointclouds, axis=0)  # (3,)
            pc_std = np.std(all_pointclouds, axis=0)  # (3,)
            pc_min = np.min(all_pointclouds, axis=0)
            pc_max = np.max(all_pointclouds, axis=0)
            
            stats_data["pointcloud"] = {
                "mean": pc_mean.tolist(),
                "std": pc_std.tolist(),
                "min": pc_min.tolist(),
                "max": pc_max.tolist(),
                "q01": np.percentile(all_pointclouds, 1, axis=0).tolist(),
                "q99": np.percentile(all_pointclouds, 99, axis=0).tolist(),
            }
            self._statistics_data["pointcloud"] = stats_data["pointcloud"].copy()
            
            print(f"  Pointcloud statistics computed:")
            print(f"    Mean (x,y,z): {pc_mean}")
            print(f"    Std (x,y,z): {pc_std}")
        
        # Compute tracking statistics if available (x, y, z separately)
        if tracking_list:
            print("  Computing tracking statistics...")
            all_tracking = np.concatenate(tracking_list, axis=0)  # (total_points, 3)
            
            # Compute mean and std for each dimension (x, y, z) of deltas
            track_mean = np.mean(all_tracking, axis=0)  # (3,)
            track_std = np.std(all_tracking, axis=0)  # (3,)
            track_min = np.min(all_tracking, axis=0)
            track_max = np.max(all_tracking, axis=0)
            
            stats_data["tracking"] = {
                "mean": track_mean.tolist(),
                "std": track_std.tolist(),
                "min": track_min.tolist(),
                "max": track_max.tolist(),
                "q01": np.percentile(all_tracking, 1, axis=0).tolist(),
                "q99": np.percentile(all_tracking, 99, axis=0).tolist(),
            }
            self._statistics_data["tracking"] = stats_data["tracking"].copy()
            
            print(f"  Tracking statistics computed:")
            print(f"    Mean (x,y,z): {track_mean}")
            print(f"    Std (x,y,z): {track_std}")
        
        print(f"  Action statistics computed:")
        print(f"    Mean: {action_mean}")
        print(f"    Std: {action_std}")
        
        # Wrap in RLDS format: {dataset_name: {action: {...}, ...}}
        self.dataset_statistics = {
            self.task_suite: stats_data
        }
        
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action using BOUNDS_Q99 method (same as RLDS).
        Maps [q01, q99] -> [-1, 1]
        
        Args:
            action: np.ndarray of shape (action_dim,) or (chunk_size, action_dim)
        
        Returns:
            Normalized action of same shape
        """
        if "action" not in self._statistics_data:
            return action
        
        q01 = np.array(self._statistics_data["action"]["q01"])
        q99 = np.array(self._statistics_data["action"]["q99"])
        
        # BOUNDS_Q99: [q01, q99] -> [-1, 1]
        # normalized = 2 * (action - q01) / (q99 - q01) - 1
        normalized = 2.0 * (action - q01) / (q99 - q01 + 1e-8) - 1.0
        
        # Clip to [-1, 1] for safety
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Denormalize action from [-1, 1] back to original scale.
        
        Args:
            action: np.ndarray of shape (action_dim,) or (chunk_size, action_dim)
        
        Returns:
            Denormalized action of same shape
        """
        if "action" not in self._statistics_data:
            return action
        
        q01 = np.array(self._statistics_data["action"]["q01"])
        q99 = np.array(self._statistics_data["action"]["q99"])
        
        # Inverse of BOUNDS_Q99
        # denormalized = (normalized + 1) * (q99 - q01) / 2 + q01
        denormalized = (action + 1.0) * (q99 - q01) / 2.0 + q01
        
        return denormalized
        
    def normalize_pointcloud(self, pointcloud: np.ndarray) -> np.ndarray:
        """
        Normalize pointcloud using dataset statistics (x, y, z separately).
        
        Args:
            pointcloud: np.ndarray of shape (N, 3) or (N, D) where D >= 3
        
        Returns:
            Normalized pointcloud of same shape
        """
        if "pointcloud" not in self._statistics_data:
            return pointcloud
        
        pc_mean = np.array(self._statistics_data["pointcloud"]["mean"])
        pc_std = np.array(self._statistics_data["pointcloud"]["std"])
        
        # Avoid division by zero
        pc_std = np.where(pc_std < 1e-6, 1.0, pc_std)
        
        # Normalize (only first 3 dimensions if D > 3)
        normalized = pointcloud.copy()
        normalized[:, :3] = (pointcloud[:, :3] - pc_mean) / pc_std
        
        return normalized
    
    def normalize_tracking(self, tracking: np.ndarray) -> np.ndarray:
        """
        Normalize tracking data using dataset statistics (x, y, z separately).
        
        Args:
            tracking: np.ndarray of shape (num_points, 3) or (T, num_points, 3)
        
        Returns:
            Normalized tracking of same shape
        """
        if "tracking" not in self._statistics_data:
            return tracking
        
        track_mean = np.array(self._statistics_data["tracking"]["mean"])
        track_std = np.array(self._statistics_data["tracking"]["std"])
        
        # Avoid division by zero
        track_std = np.where(track_std < 1e-6, 1.0, track_std)
        
        # Normalize
        if len(tracking.shape) == 2:
            # (num_points, 3)
            normalized = (tracking - track_mean) / track_std
        elif len(tracking.shape) == 3:
            # (T, num_points, 3)
            normalized = (tracking - track_mean[None, None, :]) / track_std[None, None, :]
        else:
            normalized = tracking
        
        return normalized
    
    def denormalize_pointcloud(self, pointcloud: np.ndarray) -> np.ndarray:
        """
        Denormalize pointcloud back to original scale.
        
        Args:
            pointcloud: Normalized pointcloud of shape (N, 3) or (N, D)
        
        Returns:
            Denormalized pointcloud of same shape
        """
        if "pointcloud" not in self._statistics_data:
            return pointcloud
        
        pc_mean = np.array(self._statistics_data["pointcloud"]["mean"])
        pc_std = np.array(self._statistics_data["pointcloud"]["std"])

        pc_std = np.where(pc_std < 1e-6, 1.0, pc_std)
        
        denormalized = pointcloud.copy()
        denormalized[:, :3] = pointcloud[:, :3] * pc_std + pc_mean
        
        return denormalized
    
    def normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """
        Normalize proprioceptive state using BOUNDS_Q99 method (same as action normalization).
        Maps [q01, q99] -> [-1, 1]
        
        Args:
            proprio: np.ndarray of shape (proprio_dim,)
        
        Returns:
            Normalized proprio of same shape
        """
        if "proprio" not in self._statistics_data:
            return proprio
        
        q01 = np.array(self._statistics_data["proprio"]["q01"])
        q99 = np.array(self._statistics_data["proprio"]["q99"])
        
        # BOUNDS_Q99: [q01, q99] -> [-1, 1]
        normalized = 2.0 * (proprio - q01) / (q99 - q01 + 1e-8) - 1.0
        
        # Clip to [-1, 1] for safety
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def denormalize_tracking(self, tracking: np.ndarray) -> np.ndarray:
        """
        Denormalize tracking data back to original scale.
        
        Args:
            tracking: Normalized tracking of shape (num_points, 3) or (T, num_points, 3)
        
        Returns:
            Denormalized tracking of same shape
        """
        if "tracking" not in self._statistics_data:
            return tracking
        
        track_mean = np.array(self._statistics_data["tracking"]["mean"])
        track_std = np.array(self._statistics_data["tracking"]["std"])

        track_std = np.where(track_std < 1e-6, 1.0, track_std)
        
        if len(tracking.shape) == 2:
            denormalized = tracking * track_std + track_mean
        elif len(tracking.shape) == 3:
            denormalized = tracking * track_std[None, None, :] + track_mean[None, None, :]
        else:
            denormalized = tracking
        
        return denormalized
    
    def _load_frame(self, hdf5_file, demo_grp, frame_idx: int, ep_info: Dict) -> Dict[str, Any]:
        """
        Load a single frame from HDF5 and format it to match LIBERO format.
        
        Returns a dictionary with keys:
            - observation: dict with image_primary, image_wrist, proprio, timestep, pointcloud
            - action: np.ndarray of shape (action_chunk_size, action_dim)
            - task: dict with language_instruction
            - dataset_name: str
            - tracking: np.ndarray of shape (action_chunk_size, num_points, 3) - tracking deltas
        
        Note: The same pointcloud (tracks[frame_idx]) is used for both VLA input and tracking head.
        """
        num_frames = ep_info["num_frames"]
        
        # Load primary image (agentview)
        image_primary = demo_grp["obs"]["agentview_rgb"][frame_idx]  # (H, W, 3)
        image_primary = image_primary[::-1, ::-1]  # 180 degree rotation
        
        # Load wrist image if needed
        if self.use_wrist_image and "eye_in_hand_rgb" in demo_grp["obs"]:
            image_wrist = demo_grp["obs"]["eye_in_hand_rgb"][frame_idx]  # (H, W, 3)
            image_wrist = image_wrist[::-1, ::-1]  # 180 degree rotation

        # Load proprioception: [gripper_qpos(2), ee_pos(3), ee_ori(3)] = 8D
        gripper_states = demo_grp["obs"]["gripper_states"][frame_idx]  # (2,)
        ee_states = demo_grp["obs"]["ee_states"][frame_idx]  # (6,)
        proprio = np.concatenate([gripper_states, ee_states], axis=0)  # (8,)
        
        # Load actions with chunking
        # Note: frame_idx is guaranteed to be valid for action chunking by _build_frame_index()
        actions = []
        for offset in range(self.action_chunk_size):
            action_idx = frame_idx + offset
            action = demo_grp["actions"][action_idx]
            actions.append(action)
        actions = np.stack(actions, axis=0)  # (action_chunk_size, action_dim)
        
        # Build observation dict
        if self.use_wrist_image:
            observation = {
                "image_primary": image_primary,
                "image_wrist": image_wrist,
                "proprio": proprio,
                "timestep": np.array([frame_idx], dtype=np.int32),
            }
        else:
            observation = {
                "image_primary": image_primary,
                "proprio": proprio,
                "timestep": np.array([frame_idx], dtype=np.int32),
            }
        
        # Add episode_name for external data loading (point clouds, tracks)
        observation["episode_name"] = ep_info["point_cloud_id"]
        
        # Load tracking data from tracking file if available
        pointcloud = None
        tracking_deltas = None
        
        if self.tracking_tracks_root and ep_info["point_cloud_id"]:
            track_file = self.tracking_tracks_root / ep_info["point_cloud_id"] / self.filename
            if track_file.exists():
                tracks = np.load(track_file)  # (T, num_points, 3)
                
                # Initial pointcloud (for both VLA input and tracking head)
                # Use tracks[frame_idx] as the current state
                # Note: frame_idx is guaranteed to be valid by _build_frame_index()
                pointcloud = tracks[frame_idx]  # (num_points, 3)
                
                # Tracking deltas for action chunk
                # If actions are frame_idx to frame_idx+chunk_size-1,
                # tracking deltas are tracks[frame_idx+1] - tracks[frame_idx], 
                #                    tracks[frame_idx+2] - tracks[frame_idx+1], ...
                tracking_deltas = []
                for offset in range(self.action_chunk_size):
                    t_curr = frame_idx + offset
                    t_next = t_curr + 1
                    
                    # Both t_curr and t_next are guaranteed to be in bounds
                    # because frame_idx + action_chunk_size <= num_frames < len(tracks)
                    delta = tracks[t_next] - tracks[t_curr]  # (num_points, 3)
                    tracking_deltas.append(delta)
                
                tracking_deltas = np.stack(tracking_deltas, axis=0)  # (action_chunk_size, num_points, 3)
        
        # Build task dict
        task = {
            "language_instruction": ep_info["language_instruction"],
        }
        
        # Build return dict
        return_dict = {
            "observation": observation,
            "action": actions,
            "task": task,
            "dataset_name": self.task_suite,
            "pointcloud": pointcloud,
            "tracking": tracking_deltas,
        }
        
        return return_dict
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single frame by index.
        
        Args:
            idx: Global frame index (0 to len(dataset)-1)
        
        Returns:
            Transformed batch dictionary
        """
        # Get (episode_idx, frame_idx) from global index
        ep_idx, frame_idx = self.frame_index[idx]
        ep_info = self.episode_index[ep_idx]
        
        # Load frame from HDF5
        with h5py.File(ep_info["hdf5_path"], "r") as f:
            demo_grp = f["data"][ep_info["demo_name"]]
            frame_data = self._load_frame(f, demo_grp, frame_idx, ep_info)
        
        # Apply normalization BEFORE batch transform
        # This way we normalize the raw numpy data before tensorization
        
        # Normalize actions using BOUNDS_Q99 (same as RLDS)
        frame_data["action"] = self.normalize_action(frame_data["action"])
        
        # Normalize proprioceptive state using BOUNDS_Q99
        proprio = frame_data["observation"]["proprio"]
        frame_data["observation"]["proprio"] = self.normalize_proprio(proprio)
        
        pc = frame_data["pointcloud"]
        frame_data["pointcloud"] = self.normalize_pointcloud(pc)
        
        tracking = frame_data["tracking"]
        frame_data["tracking"] = self.normalize_tracking(tracking)
        
        # Apply batch transform (converts to tensors)
        transformed = self.batch_transform(frame_data)
        
        if transformed is None:
            # If transform returns None, return empty dict (will be filtered by collator)
            return {}
        
        return transformed
    
    def __len__(self) -> int:
        # Return the actual number of valid samples (after excluding last frames for chunking)
        return len(self.frame_index)


def make_libero_hdf5_datasets(
    data_dir: Path,
    task_suite: str,
    batch_transform: Callable,
    resize_resolution: Tuple[int, int] = (256, 256),
    shuffle_buffer_size: int = 1000,
    image_aug: bool = False,
    use_wrist_image: bool = False,
    tracking_tracks_root: Optional[Path] = None,
    action_chunk_size: int = 1,
    window_stride: int = 1,
    use_val_set: bool = False,
    val_ratio: float = 0.1,
    normalize_pointcloud: bool = True,
    normalize_tracking: bool = True,
    precomputed_statistics_path: Optional[Path] = None,
    filename: str = "vertex_tracks_face_uniform.npy",
) -> Tuple[LIBEROHdf5Dataset, Optional[LIBEROHdf5Dataset]]:
    """
    Create train and (optionally) validation datasets from LIBERO HDF5 files.
    
    Args:
        data_dir: Directory containing LIBERO HDF5 files
        task_suite: Name of the task suite
        batch_transform: Transform to apply to batches
        resize_resolution: Target image resolution
        shuffle_buffer_size: Size of shuffle buffer
        image_aug: Whether to apply image augmentations (train only)
        use_wrist_image: Whether to use wrist camera
        tracking_tracks_root: Root directory for tracking data (pointcloud + tracking deltas)
        action_chunk_size: Number of future actions to include
        window_stride: Stride for sampling frames (1 = all frames, >1 = skip frames)
        use_val_set: Whether to create validation set
        val_ratio: Ratio of data to use for validation
        normalize_pointcloud: Whether to normalize pointcloud input (x, y, z separately)
        normalize_tracking: Whether to normalize tracking data (x, y, z separately)
        precomputed_statistics_path: Path to precomputed statistics JSON (recommended for large datasets)
    
    Returns:
        (train_dataset, val_dataset) where val_dataset is None if use_val_set=False
    """
    train_dataset = LIBEROHdf5Dataset(
        data_dir=data_dir,
        task_suite=task_suite,
        batch_transform=batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
        use_wrist_image=use_wrist_image,
        tracking_tracks_root=tracking_tracks_root,
        action_chunk_size=action_chunk_size,
        window_stride=window_stride,
        seed=42,
        normalize_pointcloud=normalize_pointcloud,
        normalize_tracking=normalize_tracking,
        precomputed_statistics_path=precomputed_statistics_path,
        filename=filename,
    )
    
    val_dataset = None
    if use_val_set:
        val_dataset = LIBEROHdf5Dataset(
            data_dir=data_dir,
            task_suite=task_suite,
            batch_transform=batch_transform,
            resize_resolution=resize_resolution,
            shuffle_buffer_size=shuffle_buffer_size // 10,
            train=False,
            image_aug=False,  # No augmentation for validation
            use_wrist_image=use_wrist_image,
            tracking_tracks_root=tracking_tracks_root,
            action_chunk_size=action_chunk_size,
            window_stride=window_stride,
            seed=123,
            normalize_pointcloud=normalize_pointcloud,
            normalize_tracking=normalize_tracking,
            precomputed_statistics_path=precomputed_statistics_path,
            filename=filename,
        )
    
    return train_dataset, val_dataset


# Example usage
if __name__ == "__main__":
    from prismatic.vla.datasets import RLDSBatchTransform
    
    # This is just a placeholder - in actual usage, you'd get these from finetune.py
    class DummyBatchTransform:
        def __call__(self, batch):
            return batch
    
    data_dir = Path("/scratch2/jisoo6687/libero/libero_goal_no_noops_track/")
    
    dataset = LIBEROHdf5Dataset(
        data_dir=data_dir,
        task_suite="libero_goal",
        batch_transform=DummyBatchTransform(),
        resize_resolution=(256, 256),
        shuffle_buffer_size=10000,
        train=True,
        image_aug=False,
    )
    
    print("\nTesting dataset iteration...")
    for i, batch in enumerate(dataset):
        if i == 0:
            print(f"Sample batch keys: {batch.keys()}")
            print(f"  observation keys: {batch['observation'].keys()}")
            print(f"  image_primary shape: {batch['observation']['image_primary'].shape}")
            print(f"  action shape: {batch['action'].shape}")
            print(f"  language_instruction: {batch['task']['language_instruction']}")
        if i >= 5:
            break
    
    print("\nDataset test complete!")

