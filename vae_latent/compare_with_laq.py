"""
Script to compare VAE and LAQ encoders for image encoding.
This helps evaluate which encoder is better for your use case.
"""
import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from vanilla_vae_model import VanillaVAE
from laq.laq_model import LatentActionQuantization


def load_vae_model(checkpoint_path, device='cuda'):
    """Load trained VAE model."""
    model = VanillaVAE(
        in_channels=3,
        latent_dim=256,
        hidden_dims=[64, 128, 256, 512, 1024],
        image_size=256
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def load_laq_model(checkpoint_path, device='cuda'):
    """Load trained LAQ model."""
    laq = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=8,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        temporal_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
    ).to(device)
    
    laq.load(checkpoint_path)
    laq.eval()
    return laq


def load_image(image_path, image_size=256):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(img).unsqueeze(0)


def denormalize(tensor):
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


@torch.no_grad()
def compare_encoders(vae_path, laq_path, image_path, output_dir='comparison_results'):
    """Compare VAE and LAQ encoders on sample images."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models
    print("Loading VAE model...")
    vae = load_vae_model(vae_path, device)
    
    print("Loading LAQ model...")
    laq = load_laq_model(laq_path, device)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = load_image(image_path).to(device)
    
    # VAE reconstruction
    print("Running VAE encoding and reconstruction...")
    vae_recon, _, vae_mu, vae_logvar = vae(img)
    
    # LAQ reconstruction
    print("Running LAQ encoding and reconstruction...")
    laq_recon = laq.inference(img, return_only_codebook_ids=False)
    
    # Compute reconstruction errors
    vae_mse = torch.nn.functional.mse_loss(vae_recon, img).item()
    laq_mse = torch.nn.functional.mse_loss(laq_recon, img).item()
    
    print(f"\nReconstruction MSE:")
    print(f"  VAE: {vae_mse:.6f}")
    print(f"  LAQ: {laq_mse:.6f}")
    
    # Save visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Denormalize images
    img_vis = denormalize(img.cpu()).squeeze().permute(1, 2, 0).numpy()
    vae_recon_vis = denormalize(vae_recon.cpu()).squeeze().permute(1, 2, 0).numpy()
    laq_recon_vis = denormalize(laq_recon.cpu()).squeeze().permute(1, 2, 0).numpy()
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_vis)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(vae_recon_vis)
    axes[1].set_title(f'VAE Reconstruction (MSE: {vae_mse:.6f})')
    axes[1].axis('off')
    
    axes[2].imshow(laq_recon_vis)
    axes[2].set_title(f'LAQ Reconstruction (MSE: {laq_mse:.6f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to {output_dir / 'comparison.png'}")
    
    # Analyze latent space
    print(f"\nLatent Space Analysis:")
    print(f"  VAE latent dim: {vae_mu.shape[1]}")
    print(f"  VAE latent mean: {vae_mu.mean().item():.4f} ± {vae_mu.std().item():.4f}")
    print(f"  VAE latent range: [{vae_mu.min().item():.4f}, {vae_mu.max().item():.4f}]")
    
    return {
        'vae_mse': vae_mse,
        'laq_mse': laq_mse,
        'vae_latent_mean': vae_mu.mean().item(),
        'vae_latent_std': vae_mu.std().item()
    }


def main():
    parser = argparse.ArgumentParser(description='Compare VAE and LAQ encoders')
    parser.add_argument('--vae', type=str, required=True, help='Path to VAE checkpoint')
    parser.add_argument('--laq', type=str, required=True, help='Path to LAQ checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory')
    args = parser.parse_args()
    
    compare_encoders(args.vae, args.laq, args.image, args.output)


if __name__ == '__main__':
    main()

