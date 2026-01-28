"""
RoboCasa HDF5 Dataset Loader for OpenVLA Fine-tuning

This module provides a PyTorch dataset that loads RoboCasa demonstrations from HDF5 files
and formats them for OpenVLA training.

HDF5 Structure (RoboCasa format):
    data/
        demo_1/  (Note: starts from demo_1, not demo_0)
            obs/
                gripper_states: (T+1, 2)
                joint_states: (T+1, 7)
                ee_states: (T+1, 6) [pos(3) + ori(3)]
                ee_pos: (T+1, 3)
                ee_ori: (T+1, 3)
                agentview_left_rgb: (T+1, H, W, 3)
                agentview_center_rgb: (T+1, H, W, 3)
                agentview_right_rgb: (T+1, H, W, 3)
                eye_in_hand_rgb: (T+1, H, W, 3)
            actions: (T, 12)  # RoboCasa uses 12-dim actions
            states: (T+1, state_dim)
            robot_states: (T+1, robot_state_dim)
            rewards: (T,)
            dones: (T,)
        demo_2/
            ...

Data directory structure:
    data_root/
        kitchen_coffee/
            CoffeePressButton/
                2024-04-25/
                    demo_gentex_im128_randcams_im256.hdf5
        kitchen_doors/
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
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX
from dataclasses import dataclass


@dataclass
class RoboCasaBatchTransform:
    """
    Batch transform for RoboCasa HDF5 dataset.

    Adapted for RoboCasa data format:
    - Supports multiple camera views:
      - num_images_in_input=1: agentview_left only (primary)
      - num_images_in_input=3: agentview_left + agentview_right + eye_in_hand
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
    num_images_in_input: int = 1  # 1=left only, 3=left+right+wrist
    use_proprio: bool = False
    use_pointcloud_input: bool = False
    use_tracking_head: bool = False
    tracking_use_pointcloud_input: bool = False
    
    def __call__(self, robocasa_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a RoboCasa batch to the format expected by the OpenVLA collator/models.

        RoboCasa batch format:
            observation:
                image_primary: (H, W, 3) - agentview_left (single frame)
                image_secondary: (H, W, 3) - agentview_right (single frame, optional)
                image_wrist: (H, W, 3) - eye_in_hand (single frame, optional)
                proprio: (8,) - single timestep
                episode_name: str
            action: (action_chunk_size, 12) - already chunked (RoboCasa uses 12-dim actions)
            task:
                language_instruction: str - already string
            dataset_name: str
        """
        dataset_name = robocasa_batch["dataset_name"]
        current_action = robocasa_batch["action"][0]  # First action in chunk
        actions = robocasa_batch["action"]  # Full action chunk

        # Primary image is agentview_left (H, W, 3)
        img = Image.fromarray(robocasa_batch["observation"]["image_primary"])

        # Language instruction is already a string
        lang = robocasa_batch["task"]["language_instruction"].lower()

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = robocasa_batch["action"][1:]
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

        # Add secondary (right) and wrist images if num_images_in_input=3
        if self.num_images_in_input == 3:
            # Secondary image: agentview_right
            if "image_secondary" in robocasa_batch["observation"]:
                img_secondary = Image.fromarray(robocasa_batch["observation"]["image_secondary"])
                pixel_values_secondary = self.image_transform(img_secondary)
                return_dict["pixel_values_secondary"] = pixel_values_secondary

            # Wrist image: eye_in_hand
            if "image_wrist" in robocasa_batch["observation"]:
                img_wrist = Image.fromarray(robocasa_batch["observation"]["image_wrist"])
                pixel_values_wrist = self.image_transform(img_wrist)
                return_dict["pixel_values_wrist"] = pixel_values_wrist

        # Add proprioception if needed
        if self.use_proprio and "proprio" in robocasa_batch["observation"]:
            proprio = robocasa_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        # Add pointcloud if available and requested
        # Note: pointcloud is needed for BOTH VLA input and tracking head input
        if "pointcloud" in robocasa_batch:
            pointcloud = robocasa_batch["pointcloud"]
            if pointcloud is not None:
                if isinstance(pointcloud, torch.Tensor):
                    pointcloud_tensor = pointcloud.float()
                else:
                    pointcloud_tensor = torch.as_tensor(np.array(pointcloud), dtype=torch.float32)
                return_dict["pointcloud"] = pointcloud_tensor

        # Add tracking deltas if available and requested
        if "tracking" in robocasa_batch:
            tracking = robocasa_batch["tracking"]
            if tracking is not None:
                if isinstance(tracking, torch.Tensor):
                    tracking_tensor = tracking.float()
                else:
                    tracking_tensor = torch.as_tensor(np.array(tracking), dtype=torch.float32)
                return_dict["tracking"] = tracking_tensor

        return return_dict


class RoboCasaHdf5Dataset(Dataset):
    """
    PyTorch Dataset that loads RoboCasa demonstrations from HDF5 files
    and yields them in a format compatible with RLDSBatchTransform.

    This is a map-style dataset that supports random access via __getitem__,
    enabling proper shuffling across epochs via DataLoader.

    Data directory structure:
        data_root/
            kitchen_coffee/
                CoffeePressButton/
                    2024-04-25/
                        demo_gentex_im128_randcams_im256.hdf5
            kitchen_doors/
                ...
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
        num_images_in_input: int = 1,  # 1=left only, 3=left+right+wrist
        tracking_tracks_root: Optional[Path] = None,
        action_chunk_size: int = 1,
        window_stride: int = 1,
        seed: int = 42,
        normalize_pointcloud: bool = True,
        normalize_tracking: bool = True,
        precomputed_statistics_path: Optional[Path] = None,
        filename: str = "vertex_tracks_face_uniform.npy",
        use_last_pointcloud_target: bool = False,  # If True, returns final pointcloud instead of tracking deltas
        ablation_dataset: bool = False,  # If True, pad/truncate points to ablation_mean_points
        ablation_mean_points: Optional[int] = None,  # Target number of points for ablation mode
    ) -> None:
        """
        Args:
            data_dir: Root directory containing RoboCasa HDF5 files
                      (e.g., /path/to/regenerate_single/regenerate_single/)
            task_suite: Name of the task suite (e.g., "robocasa")
            batch_transform: Transform to apply to batches (RoboCasaBatchTransform)
            resize_resolution: Target image resolution (H, W)
            shuffle_buffer_size: Size of shuffle buffer for randomizing samples
            train: Whether this is training dataset (affects shuffling)
            image_aug: Whether to apply image augmentations
            num_images_in_input: Number of camera views (1=left only, 3=left+right+wrist)
            tracking_tracks_root: Root directory for tracking data (pointcloud + tracking deltas)
            action_chunk_size: Number of future actions to include (for action chunking)
            window_stride: Stride for sampling frames (1 = all frames, >1 = skip frames for less overlap)
            seed: Random seed for shuffling
            normalize_pointcloud: If True, normalize pointcloud input using dataset statistics (x, y, z separately)
            normalize_tracking: If True, normalize tracking data using dataset statistics (x, y, z separately)
            precomputed_statistics_path: Path to precomputed statistics JSON file (if None, compute on-the-fly)
            use_last_pointcloud_target: If True, returns final pointcloud as target instead of tracking deltas
        """
        self.data_dir = Path(data_dir)
        self.task_suite = task_suite
        self.batch_transform = batch_transform
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.image_aug = image_aug
        self.num_images_in_input = num_images_in_input
        self.tracking_tracks_root = Path(tracking_tracks_root) if tracking_tracks_root else None
        self.action_chunk_size = action_chunk_size
        self.window_stride = window_stride
        self.seed = seed
        self.should_normalize_pointcloud = normalize_pointcloud
        self.should_normalize_tracking = normalize_tracking
        self.precomputed_statistics_path = Path(precomputed_statistics_path) if precomputed_statistics_path else None
        self.filename = filename
        self.use_last_pointcloud_target = use_last_pointcloud_target
        self.ablation_dataset = ablation_dataset
        self.ablation_mean_points = ablation_mean_points

        if self.ablation_dataset:
            assert self.ablation_mean_points is not None and self.ablation_mean_points > 0, (
                "ablation_mean_points must be provided and > 0 when ablation_dataset=True"
            )

        # Find all HDF5 files in the directory (recursive search for RoboCasa structure)
        # Pattern: data_dir/*/*/*/*.hdf5 (e.g., kitchen_coffee/CoffeePressButton/2024-04-25/*.hdf5)
        self.hdf5_files = sorted(glob.glob(str(self.data_dir / "*" / "*" / "*" / "*.hdf5")))
        assert len(self.hdf5_files) > 0, f"No HDF5 files found in {self.data_dir}/*/*/*/*.hdf5"

        print(f"\n[RoboCasa HDF5 Dataset]")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Found {len(self.hdf5_files)} HDF5 files")
        print(f"  Task suite: {self.task_suite}")
        print(f"  Image resolution: {resize_resolution}")
        print(f"  Num images in input: {num_images_in_input}")
        print(f"  Action chunk size: {action_chunk_size}")
        print(f"  Window stride: {window_stride}")
        if self.ablation_dataset:
            print(f"  [Ablation Mode] Padding/truncating points to {self.ablation_mean_points}")

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

    def _augment_image_np(self, img: np.ndarray, seed: int) -> np.ndarray:
        """
        Apply augmentations with deterministic RNG (per-sample seed) to match RLDS order:
        resize -> random_resized_crop -> color jitter.
        """
        rng = np.random.RandomState(seed)
        pil_img = Image.fromarray(img)

        # 1) Resize to target resolution (RLDS decode_and_resize)
        pil_img = pil_img.resize(self.resize_resolution[::-1], resample=Image.BILINEAR)

        # 2) RandomResizedCrop with scale=0.9, ratio=1.0
        w, h = pil_img.size
        scale = 0.9
        crop_size = int(round(scale * min(w, h)))
        crop_h = crop_w = max(1, crop_size)
        if w == crop_w:
            j = 0
        else:
            j = rng.randint(0, w - crop_w + 1)
        if h == crop_h:
            i = 0
        else:
            i = rng.randint(0, h - crop_h + 1)
        pil_img = TF.resized_crop(pil_img, top=i, left=j, height=crop_h, width=crop_w, size=(h, w), antialias=True)

        # 3) Color jitter
        b_factor = 1.0 + rng.uniform(-0.2, 0.2)  # brightness delta 0.2
        c_factor = rng.uniform(0.8, 1.2)          # contrast
        s_factor = rng.uniform(0.8, 1.2)          # saturation
        h_factor = rng.uniform(-0.05, 0.05)       # hue
        pil_img = TF.adjust_brightness(pil_img, b_factor)
        pil_img = TF.adjust_contrast(pil_img, c_factor)
        pil_img = TF.adjust_saturation(pil_img, s_factor)
        pil_img = TF.adjust_hue(pil_img, h_factor)

        return np.array(pil_img, dtype=np.uint8)

    def _augment_observation_images(self, observation: Dict[str, Any], seed: int) -> None:
        """Apply image augmentations in-place to all images."""
        observation["image_primary"] = self._augment_image_np(observation["image_primary"], seed)
        if self.num_images_in_input == 3:
            if "image_secondary" in observation:
                observation["image_secondary"] = self._augment_image_np(observation["image_secondary"], seed + 1)
            if "image_wrist" in observation:
                observation["image_wrist"] = self._augment_image_np(observation["image_wrist"], seed + 2)
        
    def _build_episode_index(self):
        """Build an index of all episodes across all HDF5 files."""
        self.episode_index = []  # List of (hdf5_path, demo_name, num_frames, language_instruction)
        import re

        for hdf5_path in self.hdf5_files:
            # Extract task name from path structure
            # e.g., ".../kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams_im256.hdf5"
            path_parts = Path(hdf5_path).parts
            # task_name is the parent of the date folder (e.g., "CoffeePressButton")
            task_name = path_parts[-3] if len(path_parts) >= 3 else Path(hdf5_path).stem

            with h5py.File(hdf5_path, "r") as f:
                data_grp = f["data"]

                # Sort demo names numerically (demo_1, demo_2, ..., demo_10, ...)
                demo_names = sorted(data_grp.keys(), key=lambda x: int(x.split("_")[1]))

                for demo_name in demo_names:
                    demo_grp = data_grp[demo_name]

                    # Get number of frames (actions)
                    num_frames = len(demo_grp["actions"])

                    # Get language instruction from obs/language
                    language_instruction = None
                    if "obs" in demo_grp and "language" in demo_grp["obs"]:
                        lang_data = demo_grp["obs"]["language"][()]
                        if isinstance(lang_data, bytes):
                            language_instruction = lang_data.decode("utf-8")
                        elif isinstance(lang_data, np.ndarray):
                            # Handle np.array of strings (h5py.string_dtype)
                            if lang_data.ndim == 0:
                                lang_str = lang_data.item()
                            else:
                                lang_str = lang_data[0]
                            language_instruction = lang_str.decode("utf-8") if isinstance(lang_str, bytes) else str(lang_str)
                        else:
                            language_instruction = str(lang_data)

                    # Build tracking path from HDF5 path structure
                    # HDF5: .../single_stage_regenerate/kitchen_coffee/CoffeePressButton/2024-04-25/demo.hdf5
                    # Track: .../pointrack/kitchen_coffee/CoffeePressButton/2024-04-25/demo_1/vertex_tracks.npy
                    # Extract: kitchen_coffee/CoffeePressButton/2024-04-25 from HDF5 path
                    tracking_subpath = None
                    hdf5_parts = Path(hdf5_path).parts
                    # Find the index after the data root marker (e.g., "single_stage_regenerate", "regenerate_single")
                    for marker in ["single_stage_regenerate", "regenerate_single"]:
                        if marker in hdf5_parts:
                            marker_idx = hdf5_parts.index(marker)
                            # Get parts after marker, excluding the filename
                            # e.g., kitchen_coffee/CoffeePressButton/2024-04-25
                            tracking_subpath = "/".join(hdf5_parts[marker_idx + 1:-1])
                            break

                    # If no marker found, try using last 3 directories before filename
                    if tracking_subpath is None:
                        tracking_subpath = "/".join(hdf5_parts[-4:-1])

                    self.episode_index.append({
                        "hdf5_path": hdf5_path,
                        "demo_name": demo_name,
                        "num_frames": num_frames,
                        "language_instruction": language_instruction,
                        "task_name": task_name,
                        "tracking_subpath": tracking_subpath,  # e.g., kitchen_coffee/CoffeePressButton/2024-04-25
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
                # RoboCasa has 12-dim actions, but we only use first 7 dimensions
                actions = demo_grp["actions"][()][:, :7]
                actions_list.append(actions)
                
                # Proprioception: gripper_states (2D) + ee_pos (3D) + ee_ori (3D) = 8D
                if "obs" in demo_grp:
                    gripper_states = demo_grp["obs"]["gripper_states"][()]
                    ee_states = demo_grp["obs"]["ee_states"][()]
                    proprio = np.concatenate([gripper_states, ee_states], axis=1)
                    proprio_list.append(proprio)
            
            # Load ALL tracking data from tracking file
            # Path: tracking_tracks_root / tracking_subpath / demo_name / filename
            # e.g., pointrack/kitchen_coffee/CoffeePressButton/2024-04-25/demo_1/vertex_tracks_face_uniform.npy
            if self.tracking_tracks_root and ep_info["tracking_subpath"]:
                track_file = self.tracking_tracks_root / ep_info["tracking_subpath"] / ep_info["demo_name"] / self.filename
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

    def _pad_or_truncate_points(self, points: np.ndarray) -> np.ndarray:
        """
        Pad or truncate points to match ablation_mean_points.

        Args:
            points: np.ndarray of shape (N, D) or (T, N, D)
                where N is number of points, D is dimension (usually 3)

        Returns:
            Padded/truncated array with N = self.ablation_mean_points
        """
        if not self.ablation_dataset or self.ablation_mean_points is None:
            return points

        target_n = self.ablation_mean_points

        if points.ndim == 2:
            # Shape: (N, D)
            N, D = points.shape
            if N > target_n:
                # Truncate: take first target_n points
                return points[:target_n]
            elif N < target_n:
                # Pad with zeros
                pad_shape = (target_n - N, D)
                padding = np.zeros(pad_shape, dtype=points.dtype)
                return np.concatenate([points, padding], axis=0)
            else:
                return points

        elif points.ndim == 3:
            # Shape: (T, N, D)
            T, N, D = points.shape
            if N > target_n:
                # Truncate: take first target_n points
                return points[:, :target_n, :]
            elif N < target_n:
                # Pad with zeros
                pad_shape = (T, target_n - N, D)
                padding = np.zeros(pad_shape, dtype=points.dtype)
                return np.concatenate([points, padding], axis=1)
            else:
                return points

        else:
            # Unexpected shape, return as-is
            return points

    def _load_frame(self, hdf5_file, demo_grp, frame_idx: int, ep_info: Dict) -> Dict[str, Any]:
        """
        Load a single frame from HDF5 and format it to match RoboCasa format.

        Returns a dictionary with keys:
            - observation: dict with image_primary, image_secondary, image_wrist, proprio, timestep
            - action: np.ndarray of shape (action_chunk_size, action_dim)
            - task: dict with language_instruction
            - dataset_name: str
            - tracking: np.ndarray of shape (action_chunk_size, num_points, 3) - tracking deltas
            - pointcloud: np.ndarray of shape (num_points, 3) - pointcloud at current frame

        Camera views:
            - image_primary: agentview_left_rgb
            - image_secondary: agentview_right_rgb (when num_images_in_input=3)
            - image_wrist: eye_in_hand_rgb (when num_images_in_input=3)
        """
        num_frames = ep_info["num_frames"]

        # Load primary image (agentview_left)
        image_primary = demo_grp["obs"]["agentview_left_rgb"][frame_idx]  # (H, W, 3)
        image_primary = image_primary[::-1, ::-1]  # 180 degree rotation
        # Note: RoboCasa images don't need rotation (unlike LIBERO)

        # Load secondary and wrist images if num_images_in_input=3
        image_secondary = None
        image_wrist = None
        if self.num_images_in_input == 3:
            if "agentview_right_rgb" in demo_grp["obs"]:
                image_secondary = demo_grp["obs"]["agentview_right_rgb"][frame_idx]  # (H, W, 3)
                image_secondary = image_secondary[::-1, ::-1]  # 180 degree rotation
            if "eye_in_hand_rgb" in demo_grp["obs"]:
                image_wrist = demo_grp["obs"]["eye_in_hand_rgb"][frame_idx]  # (H, W, 3)
                image_wrist = image_wrist[::-1, ::-1]  # 180 degree rotation

        # Load proprioception: [gripper_qpos(2), ee_pos(3), ee_ori(3)] = 8D
        gripper_states = demo_grp["obs"]["gripper_states"][frame_idx]  # (2,)
        ee_states = demo_grp["obs"]["ee_states"][frame_idx]  # (6,)
        proprio = np.concatenate([gripper_states, ee_states], axis=0)  # (8,)

        # Load actions with chunking
        # Note: frame_idx is guaranteed to be valid for action chunking by _build_frame_index()
        # RoboCasa has 12-dim actions, but we only use first 7 dimensions
        actions = []
        for offset in range(self.action_chunk_size):
            action_idx = frame_idx + offset
            action = demo_grp["actions"][action_idx][:7]  # Use only first 7 dims
            actions.append(action)
        actions = np.stack(actions, axis=0)  # (action_chunk_size, action_dim=7)

        # Build observation dict
        observation = {
            "image_primary": image_primary,
            "proprio": proprio,
            "timestep": np.array([frame_idx], dtype=np.int32),
        }

        if self.num_images_in_input == 3:
            if image_secondary is not None:
                observation["image_secondary"] = image_secondary
            if image_wrist is not None:
                observation["image_wrist"] = image_wrist

        # Add episode_name for external data loading (point clouds, tracks)
        observation["episode_name"] = f"{ep_info['tracking_subpath']}/{ep_info['demo_name']}"

        # Load tracking data from tracking file if available
        # Path: tracking_tracks_root / tracking_subpath / demo_name / filename
        # e.g., pointrack/kitchen_coffee/CoffeePressButton/2024-04-25/demo_1/vertex_tracks_face_uniform.npy
        pointcloud = None
        tracking_deltas = None
        load_pointcloud = bool(getattr(self.batch_transform, "use_pointcloud_input", False)) or bool(
            getattr(self.batch_transform, "tracking_use_pointcloud_input", False)
        )
        load_tracking = bool(getattr(self.batch_transform, "use_tracking_head", False))

        if (load_pointcloud or load_tracking) and self.tracking_tracks_root and ep_info["tracking_subpath"]:
            track_file = self.tracking_tracks_root / ep_info["tracking_subpath"] / ep_info["demo_name"] / self.filename
            if track_file.exists():
                tracks = np.load(track_file)  # (T, num_points, 3)

                # Apply ablation padding/truncation if enabled
                if self.ablation_dataset:
                    tracks = self._pad_or_truncate_points(tracks)

                if load_pointcloud:
                    # Initial pointcloud (for both VLA input and tracking head)
                    # Use tracks[frame_idx] as the current state
                    # Note: frame_idx is guaranteed to be valid by _build_frame_index()
                    pointcloud = tracks[frame_idx]  # (num_points, 3)

                if load_tracking:
                    if self.use_last_pointcloud_target:
                        # Return final pointcloud instead of tracking deltas
                        # Final pointcloud is at frame_idx + action_chunk_size
                        final_frame_idx = frame_idx + self.action_chunk_size
                        final_pointcloud = tracks[final_frame_idx]  # (num_points, 3)
                        # Add time dimension to match tracking head interface: (N, 3) -> (1, N, 3)
                        tracking_deltas = final_pointcloud[np.newaxis, :, :]  # (1, num_points, 3)
                    else:
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

        # Apply image augmentations (train only)
        if self.train and self.image_aug:
            # Use deterministic seed per-sample for reproducibility
            aug_seed = self.seed + idx
            self._augment_observation_images(frame_data["observation"], aug_seed)
        
        # Apply normalization BEFORE batch transform
        # This way we normalize the raw numpy data before tensorization
        
        # Normalize actions using BOUNDS_Q99 (same as RLDS)
        frame_data["action"] = self.normalize_action(frame_data["action"])
        
        # Normalize proprioceptive state using BOUNDS_Q99
        proprio = frame_data["observation"]["proprio"]
        frame_data["observation"]["proprio"] = self.normalize_proprio(proprio)

        if self.tracking_tracks_root:
            pc = frame_data["pointcloud"]
            if pc is not None:
                frame_data["pointcloud"] = self.normalize_pointcloud(pc)
            
            tracking = frame_data["tracking"]
            if tracking is not None:
                if self.use_last_pointcloud_target:
                    # For last pointcloud target, use pointcloud statistics (not tracking delta statistics)
                    # tracking shape: (1, num_points, 3) - final pointcloud position
                    frame_data["tracking"] = self.normalize_pointcloud(tracking.reshape(-1, 3)).reshape(tracking.shape)
                else:
                    # For tracking deltas, use tracking statistics
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


def make_robocasa_hdf5_datasets(
    data_dir: Path,
    task_suite: str,
    batch_transform: Callable,
    resize_resolution: Tuple[int, int] = (256, 256),
    shuffle_buffer_size: int = 1000,
    image_aug: bool = False,
    num_images_in_input: int = 1,  # 1=left only, 3=left+right+wrist
    tracking_tracks_root: Optional[Path] = None,
    action_chunk_size: int = 1,
    window_stride: int = 1,
    use_val_set: bool = False,
    val_ratio: float = 0.1,
    normalize_pointcloud: bool = True,
    normalize_tracking: bool = True,
    precomputed_statistics_path: Optional[Path] = None,
    filename: str = "vertex_tracks_face_uniform.npy",
    use_last_pointcloud_target: bool = False,
    ablation_dataset: bool = False,
    ablation_mean_points: Optional[int] = None,
) -> Tuple[RoboCasaHdf5Dataset, Optional[RoboCasaHdf5Dataset]]:
    """
    Create train and (optionally) validation datasets from RoboCasa HDF5 files.

    Args:
        data_dir: Root directory containing RoboCasa HDF5 files
        task_suite: Name of the task suite
        batch_transform: Transform to apply to batches
        resize_resolution: Target image resolution
        shuffle_buffer_size: Size of shuffle buffer
        image_aug: Whether to apply image augmentations (train only)
        num_images_in_input: Number of camera views (1=left only, 3=left+right+wrist)
        tracking_tracks_root: Root directory for tracking data (pointcloud + tracking deltas)
        action_chunk_size: Number of future actions to include
        window_stride: Stride for sampling frames (1 = all frames, >1 = skip frames)
        use_val_set: Whether to create validation set
        val_ratio: Ratio of data to use for validation
        normalize_pointcloud: Whether to normalize pointcloud input (x, y, z separately)
        normalize_tracking: Whether to normalize tracking data (x, y, z separately)
        precomputed_statistics_path: Path to precomputed statistics JSON (recommended for large datasets)
        use_last_pointcloud_target: If True, returns final pointcloud as target instead of tracking deltas
        ablation_dataset: If True, pad/truncate points to ablation_mean_points
        ablation_mean_points: Target number of points for ablation mode

    Returns:
        (train_dataset, val_dataset) where val_dataset is None if use_val_set=False
    """
    train_dataset = RoboCasaHdf5Dataset(
        data_dir=data_dir,
        task_suite=task_suite,
        batch_transform=batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
        num_images_in_input=num_images_in_input,
        tracking_tracks_root=tracking_tracks_root,
        action_chunk_size=action_chunk_size,
        window_stride=window_stride,
        seed=42,
        normalize_pointcloud=normalize_pointcloud,
        normalize_tracking=normalize_tracking,
        precomputed_statistics_path=precomputed_statistics_path,
        filename=filename,
        use_last_pointcloud_target=use_last_pointcloud_target,
        ablation_dataset=ablation_dataset,
        ablation_mean_points=ablation_mean_points,
    )

    val_dataset = None
    if use_val_set:
        val_dataset = RoboCasaHdf5Dataset(
            data_dir=data_dir,
            task_suite=task_suite,
            batch_transform=batch_transform,
            resize_resolution=resize_resolution,
            shuffle_buffer_size=shuffle_buffer_size // 10,
            train=False,
            image_aug=False,  # No augmentation for validation
            num_images_in_input=num_images_in_input,
            tracking_tracks_root=tracking_tracks_root,
            action_chunk_size=action_chunk_size,
            window_stride=window_stride,
            seed=123,
            normalize_pointcloud=normalize_pointcloud,
            normalize_tracking=normalize_tracking,
            precomputed_statistics_path=precomputed_statistics_path,
            filename=filename,
            use_last_pointcloud_target=use_last_pointcloud_target,
            ablation_dataset=ablation_dataset,
            ablation_mean_points=ablation_mean_points,
        )

    return train_dataset, val_dataset


# Example usage
if __name__ == "__main__":
    # This is just a placeholder - in actual usage, you'd get these from finetune.py
    class DummyBatchTransform:
        num_images_in_input = 3
        use_pointcloud_input = False
        use_tracking_head = False
        tracking_use_pointcloud_input = False

        def __call__(self, batch):
            return batch

    data_dir = Path("/weka/jisookim/dataset/robocasa/datasets/single_stage_regenerate")

    dataset = RoboCasaHdf5Dataset(
        data_dir=data_dir,
        task_suite="robocasa",
        batch_transform=DummyBatchTransform(),
        resize_resolution=(256, 256),
        shuffle_buffer_size=100,
        train=True,
        image_aug=False,
        num_images_in_input=3,
    )

    print("\nTesting dataset iteration...")
    for i, batch in enumerate(dataset):
        if i == 0:
            print(f"Sample batch keys: {batch.keys()}")
            print(f"  observation keys: {batch['observation'].keys()}")
            print(f"  image_primary shape: {batch['observation']['image_primary'].shape}")
            if 'image_secondary' in batch['observation']:
                print(f"  image_secondary shape: {batch['observation']['image_secondary'].shape}")
            if 'image_wrist' in batch['observation']:
                print(f"  image_wrist shape: {batch['observation']['image_wrist'].shape}")
            print(f"  action shape: {batch['action'].shape}")
            print(f"  language_instruction: {batch['task']['language_instruction']}")
        if i >= 5:
            break

    print("\nDataset test complete!")
