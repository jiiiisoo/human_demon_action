"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import re
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
import open3d as o3d
try:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    imageio = None
    plt = None

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead, PointTrackingHead, PointTrackingHeadWithPointInput
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    PointcloudProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)
    pointcloud_root: Optional[Path] = None           # If set, load per-step pointclouds from disk using episode file path
    pointcloud_subdir: str = "pointclouds_512"       # Subdirectory under episode dir containing pointcloud files
    pointcloud_ext: str = ".npy"                     # Pointcloud file extension (e.g., .npy or .ply)
    tracking_tracks_root: Optional[Path] = None      # If set, load per-episode track npy (T, num_points, 3) for tracking deltas
    tracking_tracks_filename: str = "vertex_tracks.npy"  # Track filename under each episode dir
    use_pointcloud_from_tracks: bool = False         # If True, initial pointcloud input token comes from track npy instead of ply files

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_pointcloud_input: bool = False               # If True, appends a pointcloud token from the initial frame to the VLA input
    pointcloud_input_num_points: int = 0             # Number of points to keep for the pointcloud input token (pad/truncate). 0 => use raw
    pointcloud_input_dim: int = 3                    # Dimensionality of each point for the pointcloud input token
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    use_tracking_head: bool = False                  # If True, predicts tracking targets from action token hidden states
    tracking_dim: int = 0                            # Dimensionality of tracking target per timestep
    tracking_num_points: int = 1                     # Number of tracked points per timestep
    tracking_hidden_dim: int = 0                     # Hidden dimension for tracking head MLP (0 => use llm_dim)
    tracking_num_blocks: int = 2                     # Number of MLP blocks for tracking head
    tracking_loss_weight: float = 1.0                # Scaling factor for tracking loss term
    tracking_label_key: Optional[str] = None         # Dot-delimited key into RLDS batch for point tracking labels
    tracking_use_point_features: bool = False        # If True, fuse base pointcloud into tracking head
    tracking_point_hidden_dim: int = 0               # Hidden dim for point branch in tracking head (0 => use tracking_hidden_dim/llm_dim)
    tracking_use_pointcloud_input: bool = False      # If True, reuse VLA input pointcloud for tracking head fusion
    save_tracking_viz: bool = False                  # If True, save tracking pred/gt visualizations during training
    tracking_viz_freq: int = 100                     # Save tracking visualizations every N gradient steps
    tracking_viz_max_points: int = 2000              # Max points to render in tracking visualizations (for speed)
    tracking_viz_dir: Optional[Path] = None          # Optional override directory for tracking visualizations

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    use_wandb: bool = True                           # If False, disables all WandB init/logging
    use_tensorboard: bool = False                    # If True, logs metrics to TensorBoard
    tensorboard_log_dir: Path = Path("runs/tensorboard")  # TensorBoard log directory
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    tracking_head,
    proprio_projector,
    pointcloud_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_tracking_head,
    use_proprio,
    use_pointcloud_input,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
    tracking_loss_weight=1.0,
    capture_tracking: bool = False,
    tracking_use_point_features: bool = False,
    tracking_use_pointcloud_input: bool = False,
    tracking_num_points: Optional[int] = None,
    tracking_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        tracking_head (nn.Module): Point tracking prediction head.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        pointcloud_projector (nn.Module): Pointcloud input projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_tracking_head (bool): Whether to predict tracking targets.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_pointcloud_input (bool): Whether to use initial-frame pointcloud token as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).
        tracking_loss_weight (float): Weighting factor for tracking loss term.

    Returns:
        tuple: (loss, metrics_dict, tracking_debug)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
            tracking_debug: Optional dict containing predicted/ground-truth tracking and pointcloud input (CPU).
    """
    metrics = {}
    tracking_debug_data: Optional[Dict[str, torch.Tensor]] = None

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    tracking_labels = batch.get("tracking")
    if tracking_labels is not None:
        tracking_labels = tracking_labels.to(device_id).to(torch.bfloat16)
    pointcloud_input = batch.get("pointcloud")
    if pointcloud_input is not None:
        pointcloud_input = pointcloud_input.to(device_id).to(torch.bfloat16)
    tracking_pointcloud = batch.get("tracking_pointcloud")
    if tracking_pointcloud is not None:
        tracking_pointcloud = tracking_pointcloud.to(device_id).to(torch.bfloat16)
    def _pad_or_trim_tracking_pc(pc: torch.Tensor) -> torch.Tensor:
        # Accepts (N, dim) or (B, N, dim)
        batch_first = pc.dim() == 3
        if not batch_first:
            pc = pc.unsqueeze(0)
        B, N, D = pc.shape
        if tracking_dim is not None and D != tracking_dim:
            if D > tracking_dim:
                pc = pc[:, :, :tracking_dim]
            else:
                pad_dim = torch.zeros(B, N, tracking_dim - D, dtype=pc.dtype, device=pc.device)
                pc = torch.cat([pc, pad_dim], dim=2)
        if tracking_num_points is not None and N != tracking_num_points:
            if N > tracking_num_points:
                pc = pc[:, :tracking_num_points]
            else:
                pad = torch.zeros(B, tracking_num_points - N, pc.shape[2], dtype=pc.dtype, device=pc.device)
                pc = torch.cat([pc, pad], dim=1)
        return pc if batch_first else pc.squeeze(0)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            pointcloud=pointcloud_input if use_pointcloud_input else None,
            pointcloud_projector=pointcloud_projector if use_pointcloud_input else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
        )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    requires_action_states = use_l1_regression or use_diffusion or use_tracking_head

    # Compute metrics for discrete action representation (next-token prediction)
    if not requires_action_states:
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)
        loss = None
        predicted_actions = None
        if use_l1_regression:
            # Predict action
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            action_l1_loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
            loss = action_l1_loss if loss is None else loss + action_l1_loss

        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            diffusion_loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
            loss = diffusion_loss if loss is None else loss + diffusion_loss

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        if use_tracking_head and tracking_labels is not None:
            if tracking_head is None:
                raise ValueError("Tracking head is required but not provided.")
            tracking_pointcloud_for_head = None
            if tracking_use_point_features:
                if tracking_use_pointcloud_input and pointcloud_input is not None:
                    tracking_pointcloud_for_head = _pad_or_trim_tracking_pc(pointcloud_input)
                elif tracking_pointcloud is not None:
                    tracking_pointcloud_for_head = _pad_or_trim_tracking_pc(tracking_pointcloud)
            predicted_tracking = tracking_head.module.predict_tracking(
                actions_hidden_states,
                pointcloud=tracking_pointcloud_for_head,
            )
            tracking_l1_loss = torch.nn.L1Loss()(predicted_tracking, tracking_labels)
            weighted_tracking_loss = tracking_loss_weight * tracking_l1_loss
            loss = weighted_tracking_loss if loss is None else loss + weighted_tracking_loss
            metrics.update(
                {
                    "tracking_l1_loss": tracking_l1_loss.item(),
                }
            )
            if capture_tracking:
                if pointcloud_input is not None:
                    tracking_debug_data = {
                        "predicted_tracking": predicted_tracking[:1].detach().to(torch.float32).cpu(),
                        "tracking_labels": tracking_labels[:1].detach().to(torch.float32).cpu(),
                        "pointcloud_input": pointcloud_input[:1].detach().to(torch.float32).cpu()
                        if pointcloud_input is not None
                        else None,
                    }
                elif tracking_pointcloud is not None:
                    print(f'tracking_pointcloud shape: {tracking_pointcloud.shape}')
                    tracking_debug_data = {
                        "predicted_tracking": predicted_tracking[:1].detach().to(torch.float32).cpu(),
                        "tracking_labels": tracking_labels[:1].detach().to(torch.float32).cpu(),
                        "pointcloud_input": tracking_pointcloud[:1].detach().to(torch.float32).cpu()
                    }
            if tracking_labels.shape[1] >= 1:
                ground_truth_curr_tracking = tracking_labels[:, 0]
                predicted_curr_tracking = predicted_tracking[:, 0]
                curr_tracking_l1 = torch.nn.L1Loss()(ground_truth_curr_tracking, predicted_curr_tracking)
                metrics["curr_tracking_l1_loss"] = curr_tracking_l1.item()
            if tracking_labels.shape[1] > 1:
                ground_truth_next_tracking = tracking_labels[:, 1:]
                predicted_next_tracking = predicted_tracking[:, 1:]
                next_tracking_l1 = torch.nn.L1Loss()(ground_truth_next_tracking, predicted_next_tracking)
                metrics["next_tracking_l1_loss"] = next_tracking_l1.item()
        elif use_tracking_head:
            raise ValueError("Tracking targets missing from batch while tracking head is enabled.")

        if loss is None:
            raise RuntimeError("No loss components were produced; check head configuration.")

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = (not use_diffusion or (use_diffusion and compute_diffusion_l1)) and predicted_actions is not None
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics, tracking_debug_data


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


TRACKING_VIZ_DEPS_AVAILABLE = imageio is not None and plt is not None
_TRACKING_VIZ_DEP_WARNED = False
_TRACKING_PC_FALLBACK_WARNED = False


def _ensure_tracking_viz_deps():
    global _TRACKING_VIZ_DEP_WARNED
    if TRACKING_VIZ_DEPS_AVAILABLE:
        return True
    if not _TRACKING_VIZ_DEP_WARNED:
        print("Tracking visualization skipped: imageio/matplotlib not available.")
        _TRACKING_VIZ_DEP_WARNED = True
    return False


def _pad_or_trim_base_np(base_pc: Optional[np.ndarray], num_points: int, dim: int) -> np.ndarray:
    if base_pc is None:
        base_pc = np.zeros((num_points, dim), dtype=np.float32)
    else:
        base_pc = base_pc.astype(np.float32)
    if base_pc.shape[0] > num_points:
        base_pc = base_pc[:num_points]
    elif base_pc.shape[0] < num_points:
        pad = np.zeros((num_points - base_pc.shape[0], base_pc.shape[1]), dtype=base_pc.dtype)
        base_pc = np.concatenate([base_pc, pad], axis=0)
    if base_pc.shape[1] > dim:
        base_pc = base_pc[:, :dim]
    elif base_pc.shape[1] < dim:
        pad_dim = np.zeros((base_pc.shape[0], dim - base_pc.shape[1]), dtype=base_pc.dtype)
        base_pc = np.concatenate([base_pc, pad_dim], axis=1)
    return base_pc


def _build_tracking_sequence(
    base_pc: Optional[np.ndarray], deltas_or_positions: np.ndarray, treat_as_delta: bool
) -> np.ndarray:
    num_points = deltas_or_positions.shape[1]
    dim = deltas_or_positions.shape[2] if deltas_or_positions.ndim >= 3 else 3
    base_pc = _pad_or_trim_base_np(base_pc, num_points, dim)
    if treat_as_delta:
        cum_deltas = np.cumsum(deltas_or_positions, axis=0)
        return np.concatenate([base_pc[None, ...], base_pc[None, ...] + cum_deltas], axis=0)
    return np.concatenate([base_pc[None, ...], deltas_or_positions], axis=0)


def _downsample_sequence_points(sequence: np.ndarray, max_points: Optional[int]) -> np.ndarray:
    if max_points is None or max_points <= 0 or sequence.shape[1] <= max_points:
        return sequence
    return sequence[:, :max_points]


def _save_tracking_sequence_video(points_seq: np.ndarray, video_path: Path, fps: int = 5) -> None:
    if not _ensure_tracking_viz_deps():
        return
    if points_seq.size == 0:
        return
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    pts_all = points_seq.reshape(-1, 3)
    pts_all = pts_all - pts_all.mean(axis=0, keepdims=True)
    max_range = np.linalg.norm(pts_all, axis=1).max() + 1e-6
    ax.set_xlim3d([-max_range, max_range])
    ax.set_ylim3d([-max_range, max_range])
    ax.set_zlim3d([-max_range, max_range])
    scatter = ax.scatter([], [], [], s=1)
    writer = imageio.get_writer(video_path, fps=fps)
    for pts in points_seq:
        pts = pts - pts.mean(axis=0, keepdims=True)
        ax.view_init(elev=20.0, azim=45.0)
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
    writer.close()
    plt.close(fig)


def save_tracking_visualizations(
    tracking_debug: Dict[str, torch.Tensor],
    output_dir: Path,
    log_step: int,
    tracking_labels_are_deltas: bool,
    max_points: Optional[int],
) -> None:
    if not _ensure_tracking_viz_deps():
        return
    pred = tracking_debug.get("predicted_tracking")
    gt = tracking_debug.get("tracking_labels")
    base_pc = tracking_debug.get("pointcloud_input")
    if pred is None or gt is None:
        return
    pred_np = pred[0].to(torch.float32).cpu().numpy()
    gt_np = gt[0].to(torch.float32).cpu().numpy()
    base_np = base_pc[0].to(torch.float32).cpu().numpy() if base_pc is not None else None
    pred_seq = _build_tracking_sequence(base_np, pred_np, treat_as_delta=tracking_labels_are_deltas)
    gt_seq = _build_tracking_sequence(base_np, gt_np, treat_as_delta=tracking_labels_are_deltas)
    pred_seq = _downsample_sequence_points(pred_seq, max_points)
    gt_seq = _downsample_sequence_points(gt_seq, max_points)
    output_dir = Path(output_dir)
    _save_tracking_sequence_video(pred_seq, output_dir / f"tracking_pred_step_{log_step:06d}.mp4")
    _save_tracking_sequence_video(gt_seq, output_dir / f"tracking_gt_step_{log_step:06d}.mp4")


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def log_metrics_to_tensorboard(metrics, prefix, step, writer: SummaryWriter) -> None:
    """
    Log metrics to TensorBoard.

    Args:
        metrics (dict): metrics to log
        prefix (str): metric prefix
        step (int): global step
        writer (SummaryWriter): tensorboard writer
    """
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    pointcloud_projector,
    noisy_action_projector,
    action_head,
    tracking_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        pointcloud_projector (nn.Module): Pointcloud input projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        tracking_head (nn.Module): Tracking head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        # Save run config for reproducibility
        with open(checkpoint_dir / "args_config.json", "w") as f:
            json.dump(asdict(cfg), f, default=str, indent=2)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_pointcloud_input and pointcloud_projector is not None:
            torch.save(
                pointcloud_projector.state_dict(),
                checkpoint_dir / f"pointcloud_projector--{checkpoint_name_suffix}",
            )

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
        if cfg.use_tracking_head and tracking_head is not None:
            torch.save(tracking_head.state_dict(), checkpoint_dir / f"tracking_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    tracking_head,
    proprio_projector,
    pointcloud_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        tracking_head (nn.Module): Tracking head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        pointcloud_projector (nn.Module): Pointcloud input projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics, _ = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                tracking_head=tracking_head if cfg.use_tracking_head else None,
                proprio_projector=proprio_projector,
                pointcloud_projector=pointcloud_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_tracking_head=cfg.use_tracking_head,
                use_proprio=cfg.use_proprio,
                use_pointcloud_input=cfg.use_pointcloud_input,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                tracking_loss_weight=cfg.tracking_loss_weight,
                capture_tracking=False,
                tracking_use_point_features=cfg.tracking_use_point_features,
                tracking_use_pointcloud_input=cfg.tracking_use_pointcloud_input,
                tracking_num_points=cfg.tracking_num_points if cfg.use_tracking_head else None,
                tracking_dim=cfg.tracking_dim if cfg.use_tracking_head else None,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics
    if distributed_state.is_main_process:
        if cfg.use_wandb:
            log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)
        if cfg.use_tensorboard and tb_writer is not None:
            log_metrics_to_tensorboard(avg_val_metrics, "VLA_Val", log_step, tb_writer)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )
    if cfg.use_tracking_head:
        assert cfg.tracking_dim > 0, "tracking_dim must be > 0 when using the tracking head."
        assert cfg.tracking_num_points > 0, "tracking_num_points must be > 0 when using the tracking head."
        assert (
            cfg.tracking_label_key is not None
            or cfg.pointcloud_root is not None
            or cfg.tracking_tracks_root is not None
        ), "Provide tracking_label_key or pointcloud_root or tracking_tracks_root when using the tracking head."
        if cfg.tracking_use_pointcloud_input:
            assert cfg.use_pointcloud_input, "tracking_use_pointcloud_input=True requires use_pointcloud_input=True."
        if cfg.tracking_use_point_features and cfg.use_pointcloud_from_tracks:
            assert cfg.tracking_tracks_root is not None, "use_pointcloud_from_tracks=True requires tracking_tracks_root."
    if cfg.use_pointcloud_from_tracks:
        assert (
            cfg.tracking_tracks_root is not None
        ), "use_pointcloud_from_tracks=True requires tracking_tracks_root with track npy files."
    if cfg.save_tracking_viz:
        assert cfg.use_tracking_head, "save_tracking_viz requires use_tracking_head=True."
        assert cfg.tracking_viz_freq > 0, "tracking_viz_freq must be > 0 when saving tracking visualizations."
    if cfg.use_pointcloud_input:
        assert (
            cfg.pointcloud_root is not None or cfg.use_pointcloud_from_tracks
        ), "Provide pointcloud_root or set use_pointcloud_from_tracks=True when using pointcloud input."
        assert cfg.pointcloud_input_dim > 0, "pointcloud_input_dim must be > 0 when using pointcloud input."

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    pointcloud_input_num_points = (
        cfg.pointcloud_input_num_points
        if cfg.pointcloud_input_num_points > 0
        else (cfg.tracking_num_points if cfg.tracking_num_points > 0 else None)
    )

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if cfg.use_wandb and distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Initialize TensorBoard logging
    tb_writer = None
    if cfg.use_tensorboard and distributed_state.is_main_process:
        tb_log_dir = cfg.tensorboard_log_dir / run_id
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced (only when distributed is initialized)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)
    proprio_projector = None
    pointcloud_projector = None
    action_head = None
    noisy_action_projector = None
    tracking_head = None

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate pointcloud input projector
    if cfg.use_pointcloud_input:
        assert pointcloud_input_num_points is not None and pointcloud_input_num_points > 0, (
            "pointcloud_input_num_points must be > 0 when using pointcloud input."
        )
        pointcloud_projector = init_module(
            PointcloudProjector,
            "pointcloud_projector",
            cfg,
            device_id,
            {
                "llm_dim": vla.module.llm_dim,
                "num_points": pointcloud_input_num_points,
                "point_dim": cfg.pointcloud_input_dim,
            },
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # If applicable, instantiate tracking head
    if cfg.use_tracking_head:
        tracking_hidden_dim = cfg.tracking_hidden_dim if cfg.tracking_hidden_dim > 0 else vla.module.llm_dim
        tracking_point_hidden_dim = (
            cfg.tracking_point_hidden_dim if cfg.tracking_point_hidden_dim > 0 else tracking_hidden_dim
        )
        tracking_head_cls = PointTrackingHeadWithPointInput if cfg.tracking_use_point_features else PointTrackingHead
        tracking_module_args = {
            "input_dim": vla.module.llm_dim,
            "hidden_dim": tracking_hidden_dim,
            "point_hidden_dim": tracking_point_hidden_dim,
            "num_points": cfg.tracking_num_points,
            "tracking_dim": cfg.tracking_dim,
            "num_blocks": cfg.tracking_num_blocks,
        }
        if not cfg.tracking_use_point_features:
            tracking_module_args.pop("point_hidden_dim")
        tracking_head = init_module(
            tracking_head_cls,
            "tracking_head",
            cfg,
            device_id,
            tracking_module_args,
            to_bf16=True,
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # If we have pointcloud inputs, a single pointcloud embedding is appended
    if cfg.use_pointcloud_input:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    if cfg.use_pointcloud_input:
        trainable_params += [param for param in pointcloud_projector.parameters() if param.requires_grad]
    if cfg.use_tracking_head:
        trainable_params += [param for param in tracking_head.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1
    tracks_root = Path(cfg.tracking_tracks_root) if cfg.tracking_tracks_root is not None else None

    # Optional pointcloud loaders from disk using episode_file_path + timestep
    pointcloud_loader = None
    pointcloud_input_loader = None
    tracking_tracks_loader = None
    tracking_pointcloud_loader = None

    def pad_or_trim_pointcloud(pc: torch.Tensor, max_points: Optional[int], dim: Optional[int]) -> torch.Tensor:
        """Utility to pad/trim pointclouds to fixed shape."""
        if dim is not None:
            if pc.shape[1] > dim:
                pc = pc[:, :dim]
            elif pc.shape[1] < dim:
                pad_dim = torch.zeros(pc.shape[0], dim - pc.shape[1], dtype=pc.dtype, device=pc.device)
                pc = torch.cat([pc, pad_dim], dim=1)
        if max_points is not None:
            if pc.shape[0] > max_points:
                pc = pc[:max_points]
            elif pc.shape[0] < max_points:
                pad = torch.zeros(max_points - pc.shape[0], pc.shape[1], dtype=pc.dtype, device=pc.device)
                pc = torch.cat([pc, pad], dim=0)
        return pc

    if cfg.pointcloud_root is not None:
        pc_root = Path(cfg.pointcloud_root)
        pc_len_cache = {}

        def pointcloud_loader_fn(rlds_batch):
            obs = rlds_batch.get("observation", {})

            frame_indices = obs.get("frame_index")
            timesteps = obs.get("timestep")
            traj_lens = obs.get("traj_len")
            episode_name = rlds_batch.get("episode_name").decode("utf-8")
            

            chunk_len = rlds_batch["action"].shape[0]
            pcs = []

            step_idx_anchor = (
                int(frame_indices[0])
                if frame_indices is not None and frame_indices.shape[0] > 0
                else (int(timesteps[0]) if timesteps is not None and timesteps.shape[0] > 0 else None)
            )
            traj_len_anchor = (
                int(traj_lens[0]) if traj_lens is not None and traj_lens.shape[0] > 0 else None
            )

            for i in range(chunk_len):
                # pointcloud is not saved from initial frame
                step_idx = step_idx_anchor + i -1 if step_idx_anchor is not None else None
                if step_idx < 0:
                    step_idx = 0
                # ep_id = f"episode_{ep_idx:05d}"

                pc_path = pc_root / episode_name / cfg.pointcloud_subdir / f"step_{step_idx:04d}{cfg.pointcloud_ext}"
                pc_len_cache[episode_name] = len(list((pc_root / episode_name / cfg.pointcloud_subdir).glob(f"step_*{cfg.pointcloud_ext}")))
                if traj_lens != pc_len_cache[episode_name] :
                    print(f'episode {episode_name} has {traj_lens} actions, but {pc_len_cache[episode_name]} pointclouds')
                    return None
                
                if pc_path.exists():
                    # print(f"Pointcloud file {pc_path} exists")
                    pc_o3d = o3d.io.read_point_cloud(str(pc_path))
                    pc = torch.from_numpy(np.asarray(pc_o3d.points)).float()
                    max_points = cfg.tracking_num_points if cfg.tracking_num_points else None
                    dim = cfg.tracking_dim if cfg.tracking_dim else None
                    pc = pad_or_trim_pointcloud(pc, max_points, dim)
                    pcs.append(pc)
                else :
                    print(f'timestep is later than the last pointcloud')
                    return None

            if any(p is None for p in pcs) or all(p is None for p in pcs):
                return None

            return torch.stack(pcs, dim=0)

        if cfg.use_tracking_head:
            pointcloud_loader = pointcloud_loader_fn

        if cfg.use_pointcloud_input:
            def pointcloud_input_loader_fn(rlds_batch):
                obs = rlds_batch.get("observation", {})
                traj_indices = obs.get("traj_index")
                frame_indices = obs.get("frame_index")
                timesteps = obs.get("timestep")
                episode_name = rlds_batch.get("episode_name").decode("utf-8")

                step_idx = (
                    int(frame_indices[0])
                    if frame_indices is not None and frame_indices.shape[0] > 0
                    else (int(timesteps[0]) if timesteps is not None and timesteps.shape[0] > 0 else None)
                )
                if step_idx is None:
                    return None

                # Option 1: load from tracks npy if requested
                if cfg.use_pointcloud_from_tracks and tracks_root is not None:
                    track_path = tracks_root / episode_name / cfg.tracking_tracks_filename
                    if not track_path.exists():
                        print(f'track path {track_path} does not exist')
                        return None
                    tracks = torch.from_numpy(np.load(track_path)).float()
                    if step_idx >= tracks.shape[0]:
                        print(f'step index {step_idx} is greater than the number of tracks')
                        return None
                    pc = tracks[step_idx]
                else:
                    # Option 2: load from on-disk pointcloud files
                    pc_path = pc_root / episode_name / cfg.pointcloud_subdir / f"step_{step_idx:04d}{cfg.pointcloud_ext}"
                    if not pc_path.exists():
                        print(f'pc path {pc_path} does not exist')
                        return None
                    pc_o3d = o3d.io.read_point_cloud(str(pc_path))
                    pc = torch.from_numpy(np.asarray(pc_o3d.points)).float()

                max_points = pointcloud_input_num_points
                dim = cfg.pointcloud_input_dim if cfg.pointcloud_input_dim > 0 else None
                return pad_or_trim_pointcloud(pc, max_points, dim)

            pointcloud_input_loader = pointcloud_input_loader_fn
    elif cfg.use_pointcloud_input and cfg.use_pointcloud_from_tracks and tracks_root is not None:
        # Allow pointcloud input purely from tracks even if pointcloud_root is not set
        def pointcloud_input_loader_fn(rlds_batch):
            obs = rlds_batch.get("observation", {})
            traj_indices = obs.get("traj_index")
            frame_indices = obs.get("frame_index")
            timesteps = obs.get("timestep")
            traj_lens = obs.get("traj_len")
            episode_name = rlds_batch.get("episode_name").decode("utf-8")

            step_idx = (
                int(frame_indices[0])
                if frame_indices is not None and frame_indices.shape[0] > 0
                else (int(timesteps[0]) if timesteps is not None and timesteps.shape[0] > 0 else None)
            )
            if step_idx is None:
                return None

            track_path = tracks_root / episode_name / cfg.tracking_tracks_filename
            if not track_path.exists():
                print(f'track path {track_path} does not exist')
                return None
            tracks = torch.from_numpy(np.load(track_path)).float()
            if traj_lens != tracks.shape[0]:
                print(f'episode {episode_name} has {traj_lens} actions, but {tracks.shape[0]} tracks')
                return None
            if step_idx - 1 < 0:
                print(f'step index {step_idx} is less than 0, using first track')
                pc = tracks[0]
            else :
                pc = tracks[step_idx - 1]
            max_points = pointcloud_input_num_points
            dim = cfg.pointcloud_input_dim if cfg.pointcloud_input_dim > 0 else None
            return pad_or_trim_pointcloud(pc, max_points, dim)

        pointcloud_input_loader = pointcloud_input_loader_fn

    # Optional tracking loader from per-episode track npy (T, num_points, 3) computing deltas
    if cfg.tracking_tracks_root is not None:
        # Reuse the same tracks_root (already initialized above if present)
        tracks_root = tracks_root or Path(cfg.tracking_tracks_root)

        def pad_or_trim_tracking(pc: torch.Tensor, max_points: Optional[int], dim: Optional[int]) -> torch.Tensor:
            if dim is not None:
                if pc.shape[1] > dim:
                    pc = pc[:, :dim]
                elif pc.shape[1] < dim:
                    pad_dim = torch.zeros(pc.shape[0], dim - pc.shape[1], dtype=pc.dtype, device=pc.device)
                    pc = torch.cat([pc, pad_dim], dim=1)
            if max_points is not None:
                if pc.shape[0] > max_points:
                    pc = pc[:max_points]
                elif pc.shape[0] < max_points:
                    pad = torch.zeros(max_points - pc.shape[0], pc.shape[1], dtype=pc.dtype, device=pc.device)
                    pc = torch.cat([pc, pad], dim=0)
            return pc

        def tracking_tracks_loader_fn(rlds_batch):
            obs = rlds_batch.get("observation", {})
            traj_indices = obs.get("traj_index")
            frame_indices = obs.get("frame_index")
            timesteps = obs.get("timestep")
            traj_lens = obs.get("traj_len")
            action_chunk = rlds_batch.get("action_chunk_indices_raw")
            action_chunk_clip = rlds_batch.get("action_chunk_indices")
            episode_name = rlds_batch.get("episode_name").decode("utf-8")

            chunk_len = rlds_batch["action"].shape[0]

            ep_idx_anchor = int(traj_indices[0]) if traj_indices is not None and traj_indices.shape[0] > 0 else None
            step_idx_anchor = (
                int(frame_indices[0])
                if frame_indices is not None and frame_indices.shape[0] > 0
                else (int(timesteps[0]) if timesteps is not None and timesteps.shape[0] > 0 else None)
            )
            traj_len_anchor = int(traj_lens[0]) if traj_lens is not None and traj_lens.shape[0] > 0 else None

            if step_idx_anchor is None:
                return None

            track_path = tracks_root / episode_name / cfg.tracking_tracks_filename

            if not track_path.exists():
                print(f'track path {track_path} does not exist')
                return None
            tracks = torch.from_numpy(np.load(track_path)).float()  # (T, num_points, dim)
            if traj_lens != tracks.shape[0]:
                print(f'episode {episode_name} has {traj_lens} actions, but {tracks.shape[0]} tracks')
                return None

            deltas = []
            tracks_len = tracks.shape[0]
            for i in range(chunk_len):
                t = step_idx_anchor + i
                if t < tracks_len:
                    if t - 1 < 0:
                        # print(f'step index {t} is less than 0, using first track')
                        # delta = tracks[0]
                        print(f'step index {t-1} is less than 0')
                        return None
                    else :
                        delta = tracks[t] - tracks[t-1]
                    delta = pad_or_trim_tracking(delta, cfg.tracking_num_points, cfg.tracking_dim)
                    deltas.append(delta)
                elif t < tracks_len + chunk_len // 2:
                    print(f'padding with zeros for delta {t}')
                    delta = torch.zeros(cfg.tracking_num_points, cfg.tracking_dim, dtype=tracks.dtype)
                    deltas.append(delta)
                else:
                    print(f"Episode {episode_name} has only {tracks_len} track frames, but expected {t + 1}")
                    return None

            if len(deltas) == 0:
                return None

            return torch.stack(deltas, dim=0)

        tracking_tracks_loader = tracking_tracks_loader_fn
        if cfg.tracking_use_point_features and cfg.use_pointcloud_from_tracks:
            def tracking_pointcloud_loader_fn(rlds_batch):
                obs = rlds_batch.get("observation", {})
                traj_indices = obs.get("traj_index")
                frame_indices = obs.get("frame_index")
                timesteps = obs.get("timestep")
                traj_lens = obs.get("traj_len")
                episode_name = rlds_batch.get("episode_name").decode("utf-8")

                step_idx_anchor = (
                    int(frame_indices[0])
                    if frame_indices is not None and frame_indices.shape[0] > 0
                    else (int(timesteps[0]) if timesteps is not None and timesteps.shape[0] > 0 else None)
                )
                if step_idx_anchor is None:
                    return None

                track_path = tracks_root / episode_name / cfg.tracking_tracks_filename
                
                if not track_path.exists():
                    print(f'track path {track_path} does not exist')
                    return None
                tracks = torch.from_numpy(np.load(track_path)).float()  # (T, num_points, dim)
                if traj_lens != tracks.shape[0]:
                    print(f'episode {episode_name} has {traj_lens} actions, but {tracks.shape[0]} tracks')
                    return None
                base_idx = step_idx_anchor - 1 if step_idx_anchor - 1 >= 0 else 0
                base_pc = tracks[base_idx]
                base_pc = pad_or_trim_tracking(base_pc, cfg.tracking_num_points, cfg.tracking_dim)
                return base_pc

            tracking_pointcloud_loader = tracking_pointcloud_loader_fn

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        pointcloud_from_disk_fn=pointcloud_input_loader if cfg.use_pointcloud_input else None,
        pointcloud_num_points=pointcloud_input_num_points if cfg.use_pointcloud_input else None,
        pointcloud_dim=cfg.pointcloud_input_dim if cfg.use_pointcloud_input else None,
        tracking_key=cfg.tracking_label_key if cfg.use_tracking_head else None,
        tracking_from_disk_fn=tracking_tracks_loader if cfg.tracking_tracks_root is not None else pointcloud_loader,
        tracking_num_points=cfg.tracking_num_points if cfg.use_tracking_head else None,
        tracking_dim=cfg.tracking_dim if cfg.use_tracking_head else None,
        tracking_pointcloud_fn=tracking_pointcloud_loader if (cfg.tracking_use_point_features and cfg.use_pointcloud_from_tracks) else None,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }
    if cfg.use_tracking_head:
        recent_metrics.update(
            {
                "tracking_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
                "curr_tracking_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
                "next_tracking_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
            }
        )

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            capture_tracking = (
                cfg.use_tracking_head
                and cfg.save_tracking_viz
                and distributed_state.is_main_process
                and ((batch_idx + 1) % cfg.grad_accumulation_steps == 0)
                and (log_step % cfg.tracking_viz_freq == 0)
            )
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics, tracking_debug = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                tracking_head=tracking_head if cfg.use_tracking_head else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                pointcloud_projector=pointcloud_projector if cfg.use_pointcloud_input else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_tracking_head=cfg.use_tracking_head,
                use_proprio=cfg.use_proprio,
                use_pointcloud_input=cfg.use_pointcloud_input,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                tracking_loss_weight=cfg.tracking_loss_weight,
                capture_tracking=capture_tracking,
                tracking_use_point_features=cfg.tracking_use_point_features,
                tracking_use_pointcloud_input=cfg.tracking_use_pointcloud_input,
                tracking_num_points=cfg.tracking_num_points if cfg.use_tracking_head else None,
                tracking_dim=cfg.tracking_dim if cfg.use_tracking_head else None,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            if tracking_debug is not None:
                viz_dir = cfg.tracking_viz_dir if cfg.tracking_viz_dir is not None else run_dir / "tracking_viz"
                save_tracking_visualizations(
                    tracking_debug,
                    viz_dir,
                    log_step,
                    tracking_labels_are_deltas=cfg.tracking_tracks_root is not None,
                    max_points=cfg.tracking_viz_max_points,
                )

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                if cfg.use_wandb:
                    log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                if cfg.use_tensorboard and tb_writer is not None:
                    log_metrics_to_tensorboard(smoothened_metrics, "VLA_Train", log_step, tb_writer)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                lr_val = scheduler.get_last_lr()[0]
                if cfg.use_wandb:
                    wandb.log({"VLA Train/Learning Rate": lr_val}, step=log_step)
                if cfg.use_tensorboard and tb_writer is not None:
                    tb_writer.add_scalar("VLA_Train/Learning_Rate", lr_val, log_step)

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    pointcloud_projector=pointcloud_projector if cfg.use_pointcloud_input else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    tracking_head=tracking_head if cfg.use_tracking_head else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    tracking_head=tracking_head if cfg.use_tracking_head else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    pointcloud_projector=pointcloud_projector if cfg.use_pointcloud_input else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

    if cfg.use_tensorboard and tb_writer is not None and distributed_state.is_main_process:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    finetune()
