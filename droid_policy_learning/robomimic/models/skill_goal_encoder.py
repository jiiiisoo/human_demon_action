"""
Utility module to encode goal images into latent skill representations using a
pre-trained VAE from the `vae_latent` project.
"""

from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn
import yaml


class GoalVAEEncoder(nn.Module):
    """
    Thin wrapper that loads a pre-trained VanillaVAE checkpoint and exposes the
    encoder mean as a latent skill representation for goal images.
    """

    def __init__(
        self,
        checkpoint_path: str,
        latent_dim: Optional[int] = None,
        config_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Goal VAE checkpoint not found: {checkpoint_path}")

        # Resolve path to vae_latent project
        repo_root = Path(__file__).resolve().parents[3]
        vae_root = repo_root / "vae_latent"
        if not vae_root.exists():
            raise FileNotFoundError(
                f"Expected vae_latent project at {vae_root}. "
                "Please ensure the repository layout matches the training setup."
            )
        if str(vae_root) not in sys.path:
            sys.path.insert(0, str(vae_root))

        from vanilla_vae_model import VanillaVAE  # type: ignore

        # Load VAE config to match architecture
        if config_path is None:
            config_path = vae_root / "config_vae_droid.yaml"
        config_path = Path(config_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Goal VAE config not found: {config_path}")

        with open(config_path, "r") as f:
            vae_config = yaml.safe_load(f)

        model_params = dict(vae_config.get("model_params", {}))
        if latent_dim is not None and model_params.get("latent_dim", latent_dim) != latent_dim:
            # Override latent dimension to match downstream expectations
            model_params["latent_dim"] = latent_dim
        if latent_dim is None:
            latent_dim = model_params.get("latent_dim")
        if latent_dim is None:
            raise ValueError("Latent dimension must be specified either in config or as an argument.")

        self.latent_dim = latent_dim
        self.vae = VanillaVAE(**model_params)

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint.get("state_dict", checkpoint)
        )
        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[GoalVAEEncoder] Warning: missing keys {missing}, unexpected keys {unexpected}")

        if freeze:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
        else:
            self.vae.train()

        self.freeze = freeze

    def forward(self, goal_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            goal_images: Tensor of shape (B, C, H, W) expected to be in [-1, 1].
        Returns:
            Latent skill tensor of shape (B, latent_dim).
        """
        if self.freeze:
            with torch.no_grad():
                mu, _ = self.vae.encode(goal_images)
        else:
            mu, _ = self.vae.encode(goal_images)
        return mu
