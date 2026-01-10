"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values_stacked = torch.stack(pixel_values)

            # Add secondary image if present (for multi-camera setups like RoboCasa)
            if "pixel_values_secondary" in instances[0]:
                pixel_values_secondary = [instance["pixel_values_secondary"] for instance in instances]
                pixel_values_stacked = torch.cat((pixel_values_stacked, torch.stack(pixel_values_secondary)), dim=1)

            # Add wrist image if present
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values_stacked = torch.cat((pixel_values_stacked, torch.stack(pixel_values_wrist)), dim=1)

            pixel_values = pixel_values_stacked
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        # Stack pointcloud tokens (initial frame)
        if "pointcloud" in instances[0]:
            pointcloud_list = []
            for instance in instances:
                pc_tensor = instance["pointcloud"]
                pc_tensor = pc_tensor if isinstance(pc_tensor, torch.Tensor) else torch.from_numpy(np.array(pc_tensor))
                pointcloud_list.append(pc_tensor)
            pointcloud = torch.stack(pointcloud_list).to(torch.float32)
        else:
            pointcloud = None

        # Stack tracking targets
        if "tracking" in instances[0]:
            tracking_list = []
            for instance in instances:
                tracking_tensor = instance["tracking"]
                tracking_tensor = tracking_tensor if isinstance(tracking_tensor, torch.Tensor) else torch.from_numpy(
                    np.array(tracking_tensor)
                )
                tracking_list.append(tracking_tensor)
            tracking = torch.stack(tracking_list).to(torch.float32)
        else:
            tracking = None

        tracking_pointcloud = None
        if "tracking_pointcloud" in instances[0]:
            tracking_pc_list = []
            for instance in instances:
                pc_tensor = instance["tracking_pointcloud"]
                pc_tensor = pc_tensor if isinstance(pc_tensor, torch.Tensor) else torch.from_numpy(
                    np.array(pc_tensor)
                )
                tracking_pc_list.append(pc_tensor)
            tracking_pointcloud = torch.stack(tracking_pc_list).to(torch.float32)

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if tracking is not None:
            output["tracking"] = tracking
        if pointcloud is not None:
            output["pointcloud"] = pointcloud
        if tracking_pointcloud is not None:
            output["tracking_pointcloud"] = tracking_pointcloud
        return output
