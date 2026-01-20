"""
OMY F3M HDF5 Dataset Loader for OpenVLA Fine-tuning

This module provides a PyTorch dataset that loads OMY F3M robot demonstrations
from HDF5 files (converted from LeRobot v3.0 format) and formats them for OpenVLA training.

HDF5 Structure (converted from LeRobot v3.0):
    data/
        demo_1/  (Note: starts from demo_1, not demo_0)
            obs/
                cam_third: (T, H, W, 3)      # primary camera (480x640)
                cam_top: (T, H, W, 3)        # secondary camera (480x640)
                cam_wrist: (T, H, W, 3)      # wrist camera (480x640)
                gripper_states: (T, 2)       # gripper position (duplicated)
                ee_states: (T, 6)            # end-effector states (placeholder)
                joint_states: (T, 7)         # 6 arm joints + 1 gripper
                language: str                # task description
            actions: (T-1, 7)                # 6 arm joints + 1 gripper
            rewards: (T-1,)
            dones: (T-1,)
        demo_2/
            ...

Data directory structure:
    data_root/
        task_name/
            dataset_name/
                date/
                    demo.hdf5
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
from prismatic.vla.constants import (
    IGNORE_INDEX,
    OMY_F3M_CONSTANTS,
    NormalizationType,
)
from dataclasses import dataclass

# OMY F3M specific constants
ACTION_DIM = OMY_F3M_CONSTANTS["ACTION_DIM"]  # 7 (6 joint + 1 gripper)
PROPRIO_DIM = OMY_F3M_CONSTANTS["PROPRIO_DIM"]  # 7 (6 joint + 1 gripper)
NUM_ACTIONS_CHUNK = OMY_F3M_CONSTANTS["NUM_ACTIONS_CHUNK"]  # 10
ACTION_PROPRIO_NORMALIZATION_TYPE = OMY_F3M_CONSTANTS["ACTION_PROPRIO_NORMALIZATION_TYPE"]  # BOUNDS


@dataclass
class OmyF3mBatchTransform:
    """
    Batch transform for OMY F3M HDF5 dataset.

    Adapted for OMY F3M data format (converted from LeRobot v3.0):
    - Supports multiple camera views:
      - num_images_in_input=1: cam_third only (primary)
      - num_images_in_input=3: cam_third + cam_top + cam_wrist
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
    num_images_in_input: int = 1  # 1=cam_third only, 3=cam_third+cam_top+cam_wrist
    use_proprio: bool = False
    use_pointcloud_input: bool = False
    use_tracking_head: bool = False
    tracking_use_pointcloud_input: bool = False

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts an OMY F3M batch to the format expected by the OpenVLA collator/models.

        Batch format:
            observation:
                image_primary: (H, W, 3) - cam_third (single frame)
                image_secondary: (H, W, 3) - cam_top (single frame, optional)
                image_wrist: (H, W, 3) - cam_wrist (single frame, optional)
                proprio: (7,) - single timestep (joint states)
                episode_name: str
            action: (action_chunk_size, 7) - already chunked
            task:
                language_instruction: str - already string
            dataset_name: str
        """
        dataset_name = batch["dataset_name"]
        current_action = batch["action"][0]  # First action in chunk
        actions = batch["action"]  # Full action chunk

        # Primary image is cam_third (H, W, 3)
        img = Image.fromarray(batch["observation"]["image_primary"])

        # Language instruction is already a string
        lang = batch["task"]["language_instruction"].lower()

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = batch["action"][1:]
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

        # Add secondary (top) and wrist images if num_images_in_input=3
        if self.num_images_in_input == 3:
            # Secondary image: cam_top
            if "image_secondary" in batch["observation"]:
                img_secondary = Image.fromarray(batch["observation"]["image_secondary"])
                pixel_values_secondary = self.image_transform(img_secondary)
                return_dict["pixel_values_secondary"] = pixel_values_secondary

            # Wrist image: cam_wrist
            if "image_wrist" in batch["observation"]:
                img_wrist = Image.fromarray(batch["observation"]["image_wrist"])
                pixel_values_wrist = self.image_transform(img_wrist)
                return_dict["pixel_values_wrist"] = pixel_values_wrist

        # Add proprioception if needed
        if self.use_proprio and "proprio" in batch["observation"]:
            proprio = batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        # Add pointcloud if available and requested
        if "pointcloud" in batch:
            pointcloud = batch["pointcloud"]
            if pointcloud is not None:
                if isinstance(pointcloud, torch.Tensor):
                    pointcloud_tensor = pointcloud.float()
                else:
                    pointcloud_tensor = torch.as_tensor(np.array(pointcloud), dtype=torch.float32)
                return_dict["pointcloud"] = pointcloud_tensor

        # Add tracking deltas if available and requested
        if "tracking" in batch:
            tracking = batch["tracking"]
            if tracking is not None:
                if isinstance(tracking, torch.Tensor):
                    tracking_tensor = tracking.float()
                else:
                    tracking_tensor = torch.as_tensor(np.array(tracking), dtype=torch.float32)
                return_dict["tracking"] = tracking_tensor

        return return_dict


class OmyF3mHdf5Dataset(Dataset):
    """
    PyTorch Dataset that loads OMY F3M demonstrations from HDF5 files
    and yields them in a format compatible with OpenVLA training.

    This is a map-style dataset that supports random access via __getitem__,
    enabling proper shuffling across epochs via DataLoader.

    Data directory structure:
        data_root/
            task_name/
                dataset_name/
                    date/
                        demo.hdf5
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
        num_images_in_input: int = 1,  # 1=cam_third only, 3=cam_third+cam_top+cam_wrist
        tracking_tracks_root: Optional[Path] = None,
        action_chunk_size: int = 1,
        window_stride: int = 1,
        seed: int = 42,
        normalize_pointcloud: bool = True,
        normalize_tracking: bool = True,
        precomputed_statistics_path: Optional[Path] = None,
        filename: str = "vertex_tracks_face_uniform.npy",
        use_last_pointcloud_target: bool = False,
    ) -> None:
        """
        Args:
            data_dir: Root directory containing HDF5 files
            task_suite: Name of the task suite (e.g., "omy_f3m")
            batch_transform: Transform to apply to batches (OmyF3mBatchTransform)
            resize_resolution: Target image resolution (H, W)
            shuffle_buffer_size: Size of shuffle buffer for randomizing samples
            train: Whether this is training dataset (affects shuffling)
            image_aug: Whether to apply image augmentations
            num_images_in_input: Number of camera views (1=cam_third only, 3=all cameras)
            tracking_tracks_root: Root directory for tracking data
            action_chunk_size: Number of future actions to include
            window_stride: Stride for sampling frames (1 = all frames, >1 = skip frames)
            seed: Random seed for shuffling
            normalize_pointcloud: If True, normalize pointcloud input
            normalize_tracking: If True, normalize tracking data
            precomputed_statistics_path: Path to precomputed statistics JSON file
            use_last_pointcloud_target: If True, returns final pointcloud as target
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

        # Find all HDF5 files in the directory
        # Pattern: data_dir/*/*/*/*.hdf5 (e.g., task_name/dataset_name/date/*.hdf5)
        self.hdf5_files = sorted(glob.glob(str(self.data_dir / "*" / "*" / "*" / "*.hdf5")))
        if len(self.hdf5_files) == 0:
            # Also try direct path: data_dir/*.hdf5
            self.hdf5_files = sorted(glob.glob(str(self.data_dir / "*.hdf5")))
        if len(self.hdf5_files) == 0:
            # Try one level: data_dir/*/*.hdf5
            self.hdf5_files = sorted(glob.glob(str(self.data_dir / "*" / "*.hdf5")))
        assert len(self.hdf5_files) > 0, f"No HDF5 files found in {self.data_dir}"

        print(f"\n[OMY F3M HDF5 Dataset]")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Found {len(self.hdf5_files)} HDF5 files")
        print(f"  Task suite: {self.task_suite}")
        print(f"  Image resolution: {resize_resolution}")
        print(f"  Num images in input: {num_images_in_input}")
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

            if self.task_suite in loaded_stats:
                self.dataset_statistics = loaded_stats
                self._statistics_data = loaded_stats[self.task_suite]
            elif "action" in loaded_stats:
                self._statistics_data = loaded_stats
                self.dataset_statistics = {self.task_suite: loaded_stats}
            else:
                raise ValueError(f"Invalid precomputed statistics format")

            print(f"  Statistics loaded successfully")
        else:
            print("  Computing dataset statistics...")
            self._compute_statistics()

    def _augment_image_np(self, img: np.ndarray, seed: int) -> np.ndarray:
        """Apply augmentations with deterministic RNG."""
        rng = np.random.RandomState(seed)
        pil_img = Image.fromarray(img)

        # Resize to target resolution
        pil_img = pil_img.resize(self.resize_resolution[::-1], resample=Image.BILINEAR)

        # RandomResizedCrop with scale=0.9
        w, h = pil_img.size
        scale = 0.9
        crop_size = int(round(scale * min(w, h)))
        crop_h = crop_w = max(1, crop_size)
        j = 0 if w == crop_w else rng.randint(0, w - crop_w + 1)
        i = 0 if h == crop_h else rng.randint(0, h - crop_h + 1)
        pil_img = TF.resized_crop(pil_img, top=i, left=j, height=crop_h, width=crop_w, size=(h, w), antialias=True)

        # Color jitter
        b_factor = 1.0 + rng.uniform(-0.2, 0.2)
        c_factor = rng.uniform(0.8, 1.2)
        s_factor = rng.uniform(0.8, 1.2)
        h_factor = rng.uniform(-0.05, 0.05)
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
        self.episode_index = []

        for hdf5_path in self.hdf5_files:
            path_parts = Path(hdf5_path).parts
            task_name = path_parts[-3] if len(path_parts) >= 3 else Path(hdf5_path).stem

            with h5py.File(hdf5_path, "r") as f:
                data_grp = f["data"]
                demo_names = sorted(data_grp.keys(), key=lambda x: int(x.split("_")[1]))

                for demo_name in demo_names:
                    demo_grp = data_grp[demo_name]

                    # Get number of frames from actions
                    num_actions = len(demo_grp["actions"])
                    num_frames = num_actions + 1  # T frames for T-1 actions

                    # Get language instruction
                    language_instruction = None
                    if "obs" in demo_grp and "language" in demo_grp["obs"]:
                        lang_data = demo_grp["obs"]["language"][()]
                        if isinstance(lang_data, bytes):
                            language_instruction = lang_data.decode("utf-8")
                        elif isinstance(lang_data, np.ndarray):
                            if lang_data.ndim == 0:
                                lang_str = lang_data.item()
                            else:
                                lang_str = lang_data[0]
                            language_instruction = lang_str.decode("utf-8") if isinstance(lang_str, bytes) else str(lang_str)
                        else:
                            language_instruction = str(lang_data)

                    # Build tracking subpath
                    tracking_subpath = None
                    hdf5_parts = Path(hdf5_path).parts
                    for marker in ["real_world_hdf5", "robot_sample_data_hdf5"]:
                        if marker in hdf5_parts:
                            marker_idx = hdf5_parts.index(marker)
                            tracking_subpath = "/".join(hdf5_parts[marker_idx + 1:-1])
                            break
                    if tracking_subpath is None:
                        tracking_subpath = "/".join(hdf5_parts[-4:-1])

                    self.episode_index.append({
                        "hdf5_path": hdf5_path,
                        "demo_name": demo_name,
                        "num_frames": num_frames,
                        "language_instruction": language_instruction,
                        "task_name": task_name,
                        "tracking_subpath": tracking_subpath,
                    })

        print(f"  Total episodes: {len(self.episode_index)}")
        total_frames = sum(ep["num_frames"] for ep in self.episode_index)
        print(f"  Total frames: {total_frames}")

        self.dataset_length = total_frames

    def _build_frame_index(self):
        """Build a flat index mapping global frame index to (episode_idx, frame_idx)."""
        self.frame_index = []

        for ep_idx, ep_info in enumerate(self.episode_index):
            # Valid frame_idx: 0 <= frame_idx <= num_frames - action_chunk_size - 1
            # (need action_chunk_size actions starting from frame_idx)
            max_valid_frame_idx = max(0, ep_info["num_frames"] - self.action_chunk_size - 1)

            for frame_idx in range(0, max_valid_frame_idx + 1, self.window_stride):
                self.frame_index.append((ep_idx, frame_idx))

        self.dataset_length = len(self.frame_index)

        print(f"  Built frame index: {len(self.frame_index)} frames (stride={self.window_stride})")
        print(f"  (Excluded last {self.action_chunk_size} frames per episode for action chunking)")

    def _compute_statistics(self):
        """Compute dataset statistics for normalization."""
        print("  Computing dataset statistics from all episodes...")

        actions_list = []
        proprio_list = []
        pointcloud_list = []
        tracking_list = []

        for ep_info in tqdm.tqdm(self.episode_index, desc="Computing statistics"):
            with h5py.File(ep_info["hdf5_path"], "r") as f:
                demo_grp = f["data"][ep_info["demo_name"]]
                actions = demo_grp["actions"][()]
                actions_list.append(actions)

                # Proprioception: joint_states (7D)
                if "obs" in demo_grp and "joint_states" in demo_grp["obs"]:
                    joint_states = demo_grp["obs"]["joint_states"][()]
                    proprio_list.append(joint_states)

            # Load tracking data if available
            if self.tracking_tracks_root and ep_info["tracking_subpath"]:
                track_file = self.tracking_tracks_root / ep_info["tracking_subpath"] / ep_info["demo_name"] / self.filename
                if track_file.exists():
                    try:
                        tracks = np.load(track_file)
                        if len(tracks) > 0:
                            for t in range(len(tracks)):
                                pointcloud_list.append(tracks[t])
                            for t in range(1, len(tracks)):
                                delta = tracks[t] - tracks[t-1]
                                tracking_list.append(delta)
                    except Exception as e:
                        print(f"  Warning: Error loading {track_file}: {e}")

        all_actions = np.concatenate(actions_list, axis=0)

        # Compute action statistics
        action_mean = np.mean(all_actions, axis=0)
        action_std = np.std(all_actions, axis=0)
        action_min = np.min(all_actions, axis=0)
        action_max = np.max(all_actions, axis=0)

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

        self._statistics_data = stats_data.copy()

        # Add proprio statistics if available
        if proprio_list:
            all_proprio = np.concatenate(proprio_list, axis=0)
            stats_data["proprio"] = {
                "mean": np.mean(all_proprio, axis=0).tolist(),
                "std": np.std(all_proprio, axis=0).tolist(),
                "min": np.min(all_proprio, axis=0).tolist(),
                "max": np.max(all_proprio, axis=0).tolist(),
                "q01": np.percentile(all_proprio, 1, axis=0).tolist(),
                "q99": np.percentile(all_proprio, 99, axis=0).tolist(),
            }
            self._statistics_data["proprio"] = stats_data["proprio"].copy()

        # Add pointcloud statistics if available
        if pointcloud_list:
            print("  Computing pointcloud statistics...")
            all_pointclouds = np.concatenate(pointcloud_list, axis=0)
            pc_mean = np.mean(all_pointclouds, axis=0)
            pc_std = np.std(all_pointclouds, axis=0)

            stats_data["pointcloud"] = {
                "mean": pc_mean.tolist(),
                "std": pc_std.tolist(),
                "min": np.min(all_pointclouds, axis=0).tolist(),
                "max": np.max(all_pointclouds, axis=0).tolist(),
                "q01": np.percentile(all_pointclouds, 1, axis=0).tolist(),
                "q99": np.percentile(all_pointclouds, 99, axis=0).tolist(),
            }
            self._statistics_data["pointcloud"] = stats_data["pointcloud"].copy()
            print(f"    Pointcloud mean: {pc_mean}, std: {pc_std}")

        # Add tracking statistics if available
        if tracking_list:
            print("  Computing tracking statistics...")
            all_tracking = np.concatenate(tracking_list, axis=0)
            track_mean = np.mean(all_tracking, axis=0)
            track_std = np.std(all_tracking, axis=0)

            stats_data["tracking"] = {
                "mean": track_mean.tolist(),
                "std": track_std.tolist(),
                "min": np.min(all_tracking, axis=0).tolist(),
                "max": np.max(all_tracking, axis=0).tolist(),
                "q01": np.percentile(all_tracking, 1, axis=0).tolist(),
                "q99": np.percentile(all_tracking, 99, axis=0).tolist(),
            }
            self._statistics_data["tracking"] = stats_data["tracking"].copy()
            print(f"    Tracking mean: {track_mean}, std: {track_std}")

        print(f"  Action statistics:")
        print(f"    Mean: {action_mean}")
        print(f"    Std: {action_std}")

        self.dataset_statistics = {self.task_suite: stats_data}

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action based on ACTION_PROPRIO_NORMALIZATION_TYPE.
        - BOUNDS: Maps [min, max] -> [-1, 1]
        - BOUNDS_Q99: Maps [q01, q99] -> [-1, 1]
        """
        if "action" not in self._statistics_data:
            return action

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            # BOUNDS: [min, max] -> [-1, 1]
            a_min = np.array(self._statistics_data["action"]["min"])
            a_max = np.array(self._statistics_data["action"]["max"])
            normalized = 2.0 * (action - a_min) / (a_max - a_min + 1e-8) - 1.0
        else:
            # BOUNDS_Q99: [q01, q99] -> [-1, 1]
            q01 = np.array(self._statistics_data["action"]["q01"])
            q99 = np.array(self._statistics_data["action"]["q99"])
            normalized = 2.0 * (action - q01) / (q99 - q01 + 1e-8) - 1.0

        normalized = np.clip(normalized, -1.0, 1.0)
        return normalized

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] back to original scale."""
        if "action" not in self._statistics_data:
            return action

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            # BOUNDS: [-1, 1] -> [min, max]
            a_min = np.array(self._statistics_data["action"]["min"])
            a_max = np.array(self._statistics_data["action"]["max"])
            denormalized = (action + 1.0) * (a_max - a_min) / 2.0 + a_min
        else:
            # BOUNDS_Q99: [-1, 1] -> [q01, q99]
            q01 = np.array(self._statistics_data["action"]["q01"])
            q99 = np.array(self._statistics_data["action"]["q99"])
            denormalized = (action + 1.0) * (q99 - q01) / 2.0 + q01

        return denormalized

    def normalize_pointcloud(self, pointcloud: np.ndarray) -> np.ndarray:
        """Normalize pointcloud using dataset statistics."""
        if "pointcloud" not in self._statistics_data:
            return pointcloud

        pc_mean = np.array(self._statistics_data["pointcloud"]["mean"])
        pc_std = np.array(self._statistics_data["pointcloud"]["std"])
        pc_std = np.where(pc_std < 1e-6, 1.0, pc_std)

        normalized = pointcloud.copy()
        normalized[:, :3] = (pointcloud[:, :3] - pc_mean) / pc_std

        return normalized

    def normalize_tracking(self, tracking: np.ndarray) -> np.ndarray:
        """Normalize tracking data using dataset statistics."""
        if "tracking" not in self._statistics_data:
            return tracking

        track_mean = np.array(self._statistics_data["tracking"]["mean"])
        track_std = np.array(self._statistics_data["tracking"]["std"])
        track_std = np.where(track_std < 1e-6, 1.0, track_std)

        if len(tracking.shape) == 2:
            normalized = (tracking - track_mean) / track_std
        elif len(tracking.shape) == 3:
            normalized = (tracking - track_mean[None, None, :]) / track_std[None, None, :]
        else:
            normalized = tracking

        return normalized

    def denormalize_pointcloud(self, pointcloud: np.ndarray) -> np.ndarray:
        """Denormalize pointcloud back to original scale."""
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
        Normalize proprioceptive state based on ACTION_PROPRIO_NORMALIZATION_TYPE.
        - BOUNDS: Maps [min, max] -> [-1, 1]
        - BOUNDS_Q99: Maps [q01, q99] -> [-1, 1]
        """
        if "proprio" not in self._statistics_data:
            return proprio

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            # BOUNDS: [min, max] -> [-1, 1]
            p_min = np.array(self._statistics_data["proprio"]["min"])
            p_max = np.array(self._statistics_data["proprio"]["max"])
            normalized = 2.0 * (proprio - p_min) / (p_max - p_min + 1e-8) - 1.0
        else:
            # BOUNDS_Q99: [q01, q99] -> [-1, 1]
            q01 = np.array(self._statistics_data["proprio"]["q01"])
            q99 = np.array(self._statistics_data["proprio"]["q99"])
            normalized = 2.0 * (proprio - q01) / (q99 - q01 + 1e-8) - 1.0

        normalized = np.clip(normalized, -1.0, 1.0)
        return normalized

    def denormalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """Denormalize proprioceptive state from [-1, 1] back to original scale."""
        if "proprio" not in self._statistics_data:
            return proprio

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            # BOUNDS: [-1, 1] -> [min, max]
            p_min = np.array(self._statistics_data["proprio"]["min"])
            p_max = np.array(self._statistics_data["proprio"]["max"])
            denormalized = (proprio + 1.0) * (p_max - p_min) / 2.0 + p_min
        else:
            # BOUNDS_Q99: [-1, 1] -> [q01, q99]
            q01 = np.array(self._statistics_data["proprio"]["q01"])
            q99 = np.array(self._statistics_data["proprio"]["q99"])
            denormalized = (proprio + 1.0) * (q99 - q01) / 2.0 + q01

        return denormalized

    def denormalize_tracking(self, tracking: np.ndarray) -> np.ndarray:
        """Denormalize tracking data back to original scale."""
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
        Load a single frame from HDF5 and format it.

        Returns a dictionary with keys:
            - observation: dict with image_primary, image_secondary, image_wrist, proprio, timestep
            - action: np.ndarray of shape (action_chunk_size, 7)
            - task: dict with language_instruction
            - dataset_name: str
            - tracking: np.ndarray of shape (action_chunk_size, num_points, 3)
            - pointcloud: np.ndarray of shape (num_points, 3)

        Camera views:
            - image_primary: cam_third
            - image_secondary: cam_top (when num_images_in_input=3)
            - image_wrist: cam_wrist (when num_images_in_input=3)
        """
        num_frames = ep_info["num_frames"]

        # Load primary image (cam_third)
        image_primary = demo_grp["obs"]["cam_third"][frame_idx]  # (H, W, 3)

        # Load secondary and wrist images if num_images_in_input=3
        image_secondary = None
        image_wrist = None
        if self.num_images_in_input == 3:
            if "cam_top" in demo_grp["obs"]:
                image_secondary = demo_grp["obs"]["cam_top"][frame_idx]  # (H, W, 3)
            if "cam_wrist" in demo_grp["obs"]:
                image_wrist = demo_grp["obs"]["cam_wrist"][frame_idx]  # (H, W, 3)

        # Load proprioception: joint_states (7D)
        proprio = demo_grp["obs"]["joint_states"][frame_idx]  # (7,)

        # Load actions with chunking
        actions = []
        for offset in range(self.action_chunk_size):
            action_idx = frame_idx + offset
            action = demo_grp["actions"][action_idx]  # (7,)
            actions.append(action)
        actions = np.stack(actions, axis=0)  # (action_chunk_size, 7)

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

        observation["episode_name"] = f"{ep_info['tracking_subpath']}/{ep_info['demo_name']}"

        # Load tracking data if available
        pointcloud = None
        tracking_deltas = None
        load_pointcloud = bool(getattr(self.batch_transform, "use_pointcloud_input", False)) or bool(
            getattr(self.batch_transform, "tracking_use_pointcloud_input", False)
        )
        load_tracking = bool(getattr(self.batch_transform, "use_tracking_head", False))

        if (load_pointcloud or load_tracking) and self.tracking_tracks_root and ep_info["tracking_subpath"]:
            track_file = self.tracking_tracks_root / ep_info["tracking_subpath"] / ep_info["demo_name"] / self.filename
            if track_file.exists():
                tracks = np.load(track_file)

                if load_pointcloud:
                    pointcloud = tracks[frame_idx]

                if load_tracking:
                    if self.use_last_pointcloud_target:
                        final_frame_idx = frame_idx + self.action_chunk_size
                        final_pointcloud = tracks[final_frame_idx]
                        tracking_deltas = final_pointcloud[np.newaxis, :, :]
                    else:
                        tracking_deltas = []
                        for offset in range(self.action_chunk_size):
                            t_curr = frame_idx + offset
                            t_next = t_curr + 1
                            delta = tracks[t_next] - tracks[t_curr]
                            tracking_deltas.append(delta)
                        tracking_deltas = np.stack(tracking_deltas, axis=0)

        # Build task dict
        task = {
            "language_instruction": ep_info["language_instruction"],
        }

        return {
            "observation": observation,
            "action": actions,
            "task": task,
            "dataset_name": self.task_suite,
            "pointcloud": pointcloud,
            "tracking": tracking_deltas,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single frame by index."""
        ep_idx, frame_idx = self.frame_index[idx]
        ep_info = self.episode_index[ep_idx]

        with h5py.File(ep_info["hdf5_path"], "r") as f:
            demo_grp = f["data"][ep_info["demo_name"]]
            frame_data = self._load_frame(f, demo_grp, frame_idx, ep_info)

        # Apply image augmentations (train only)
        if self.train and self.image_aug:
            aug_seed = self.seed + idx
            self._augment_observation_images(frame_data["observation"], aug_seed)

        # Normalize actions
        frame_data["action"] = self.normalize_action(frame_data["action"])

        # Normalize proprioceptive state
        proprio = frame_data["observation"]["proprio"]
        frame_data["observation"]["proprio"] = self.normalize_proprio(proprio)

        if self.tracking_tracks_root:
            pc = frame_data["pointcloud"]
            if pc is not None:
                frame_data["pointcloud"] = self.normalize_pointcloud(pc)

            tracking = frame_data["tracking"]
            if tracking is not None:
                if self.use_last_pointcloud_target:
                    frame_data["tracking"] = self.normalize_pointcloud(tracking.reshape(-1, 3)).reshape(tracking.shape)
                else:
                    frame_data["tracking"] = self.normalize_tracking(tracking)

        # Apply batch transform
        transformed = self.batch_transform(frame_data)

        if transformed is None:
            return {}

        return transformed

    def __len__(self) -> int:
        return len(self.frame_index)


def make_omy_f3m_hdf5_datasets(
    data_dir: Path,
    task_suite: str,
    batch_transform: Callable,
    resize_resolution: Tuple[int, int] = (256, 256),
    shuffle_buffer_size: int = 1000,
    image_aug: bool = False,
    num_images_in_input: int = 1,
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
) -> Tuple[OmyF3mHdf5Dataset, Optional[OmyF3mHdf5Dataset]]:
    """
    Create train and (optionally) validation datasets from OMY F3M HDF5 files.

    Args:
        data_dir: Root directory containing HDF5 files
        task_suite: Name of the task suite
        batch_transform: Transform to apply to batches
        resize_resolution: Target image resolution
        shuffle_buffer_size: Size of shuffle buffer
        image_aug: Whether to apply image augmentations (train only)
        num_images_in_input: Number of camera views (1=cam_third only, 3=all cameras)
        tracking_tracks_root: Root directory for tracking data
        action_chunk_size: Number of future actions to include
        window_stride: Stride for sampling frames
        use_val_set: Whether to create validation set
        val_ratio: Ratio of data to use for validation
        normalize_pointcloud: Whether to normalize pointcloud input
        normalize_tracking: Whether to normalize tracking data
        precomputed_statistics_path: Path to precomputed statistics JSON
        use_last_pointcloud_target: If True, returns final pointcloud as target

    Returns:
        (train_dataset, val_dataset) where val_dataset is None if use_val_set=False
    """
    train_dataset = OmyF3mHdf5Dataset(
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
    )

    val_dataset = None
    if use_val_set:
        val_dataset = OmyF3mHdf5Dataset(
            data_dir=data_dir,
            task_suite=task_suite,
            batch_transform=batch_transform,
            resize_resolution=resize_resolution,
            shuffle_buffer_size=shuffle_buffer_size // 10,
            train=False,
            image_aug=False,
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
        )

    return train_dataset, val_dataset


# Example usage
if __name__ == "__main__":
    class DummyBatchTransform:
        num_images_in_input = 3
        use_pointcloud_input = False
        use_tracking_head = False
        tracking_use_pointcloud_input = False

        def __call__(self, batch):
            return batch

    data_dir = Path("/workspace/human_demon_action/robot_sample_data_hdf5/real_world/omy_f3m_simple_santa_pri/2026-01-20")

    dataset = OmyF3mHdf5Dataset(
        data_dir=data_dir,
        task_suite="omy_f3m",
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
