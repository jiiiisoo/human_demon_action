"""
VAE Encoder wrapper for goal-conditioned diffusion policy
Loads pretrained VAE and uses encoder to extract goal latent representations
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import yaml
# Add vae_uniskill to path
vae_latent_path = Path(__file__).parent.parent.parent / 'vae_latent'
sys.path.insert(0, str(vae_latent_path))

from vanilla_vae_model import VanillaVAE


class VAEGoalEncoder(nn.Module):
    """
    Wrapper for VAE encoder to extract goal latent representations
    """
    def __init__(
        self,
        vae_checkpoint_path,
        latent_dim=64,
        freeze_encoder=True,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.freeze_encoder = freeze_encoder
        
        # Load pretrained VAE
        print(f"Loading VAE from {vae_checkpoint_path}")
        checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')

        config = yaml.safe_load(open(vae_latent_path / 'config_vae_droid.yaml'))
        
        # Create VAE model
        self.vae = VanillaVAE(
            **config['model_params'],
        )
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        
        print("VAE loaded successfully")
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
            print("VAE encoder frozen")
    
    def forward(self, goal_images):
        """
        Encode goal images to latent representations
        
        Args:
            goal_images: (B, 3, H, W) or (B, T, 3, H, W)
        
        Returns:
            goal_latents: (B, latent_dim) or (B, T, latent_dim)
        """
        original_shape = goal_images.shape
        
        # Handle both (B, 3, H, W) and (B, T, 3, H, W)
        if len(original_shape) == 5:
            B, T, C, H, W = original_shape
            goal_images = goal_images.reshape(B * T, C, H, W)
        
        # Encode to latent (using mean of distribution)
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            mu, logvar = self.vae.encode(goal_images)
            goal_latents = mu  # Use mean as latent representation
        
        # Reshape back if needed
        if len(original_shape) == 5:
            goal_latents = goal_latents.reshape(B, T, self.latent_dim)
        
        return goal_latents
    
    def output_dim(self):
        """Return output dimension"""
        return self.latent_dim


if __name__ == '__main__':
    # Test encoder
    encoder = VAEGoalEncoder(
        vae_checkpoint_path='/home/jisookim/human_demon_action/vae_uniskill/checkpoints/vae_epoch_10.pt',
        latent_dim=64,
        freeze_encoder=True,
    )
    
    # Test with dummy input
    dummy_goal = torch.randn(4, 3, 256, 256)
    latent = encoder(dummy_goal)
    print(f"Input shape: {dummy_goal.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Test with temporal dimension
    dummy_goal_seq = torch.randn(4, 2, 3, 256, 256)
    latent_seq = encoder(dummy_goal_seq)
    print(f"\nInput shape (with time): {dummy_goal_seq.shape}")
    print(f"Latent shape (with time): {latent_seq.shape}")

