"""
Compare VAE (trained by us) with pretrained LAPA and UniSkill encoders
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import matplotlib.pyplot as plt
import json

# Add paths for LAPA and UniSkill
sys.path.append('/home/jisookim/human_demon_action/LAPA/laq')
sys.path.append('/home/jisookim/human_demon_action/UniSkill')


class EncoderWrapper:
    """Wrapper to provide unified interface for different encoders"""
    
    def __init__(self, model, model_type, device='cuda'):
        self.model = model
        self.model_type = model_type
        self.device = device
        
    def encode(self, images):
        """Encode images to latent representation"""
        with torch.no_grad():
            if self.model_type == 'vae':
                # VAE: return mu (ignore log_var for comparison)
                mu, log_var = self.model.encode(images)
                return mu
                
            elif self.model_type == 'lapa':
                # LAPA LAQ: encode both frames and get latent
                # Note: LAPA expects video input (B, C, T, H, W)
                # We'll treat single image as video with 1 frame
                if images.dim() == 4:  # B, C, H, W
                    images = images.unsqueeze(2)  # B, C, 1, H, W
                
                # Get latent codes (simplified for comparison)
                latent = self.model.encode(images)
                return latent
                
            elif self.model_type == 'uniskill':
                # UniSkill IDM: encode visual features
                # IDM expects (B, T, C, H, W) format
                if images.dim() == 4:  # B, C, H, W
                    images = images.unsqueeze(1)  # B, 1, C, H, W
                
                # Get skill representation
                latent = self.model.forward_encoder(images)
                # Average pool spatial dimensions
                latent = latent.mean(dim=2)  # Average over spatial tokens
                return latent
    
    def decode(self, latent):
        """Decode latent to images"""
        with torch.no_grad():
            if self.model_type == 'vae':
                return self.model.decode(latent)
                
            elif self.model_type == 'lapa':
                # LAPA decoding requires action tokens
                # For reconstruction test, we can use the latent itself
                # This is simplified - full LAPA needs action conditioning
                print("⚠️  LAPA decoder needs action tokens, using simplified version")
                return None
                
            elif self.model_type == 'uniskill':
                # UniSkill doesn't have a decoder (it's for action prediction)
                print("⚠️  UniSkill IDM doesn't have image decoder")
                return None
    
    def get_latent_dim(self):
        """Get latent dimension"""
        # Test with dummy input
        dummy = torch.randn(1, 3, 256, 256).to(self.device)
        latent = self.encode(dummy)
        return latent.shape[-1]


def load_vae_model(checkpoint_path, config_path, device='cuda'):
    """Load trained VAE model"""
    print("Loading VAE model...")
    
    from vanilla_vae_model import VanillaVAE
    
    config = yaml.safe_load(open(config_path))
    model = VanillaVAE(**config['model_params'])
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    print(f"✅ VAE loaded from {checkpoint_path}")
    print(f"   Latent dim: {config['model_params']['latent_dim']}")
    
    return EncoderWrapper(model, 'vae', device)


def load_lapa_model(checkpoint_path, device='cuda'):
    """Load pretrained LAPA model"""
    print("Loading LAPA model...")
    
    try:
        from laq_model import LatentActionQuantization
        
        # LAPA config (from train_sthv2.py)
        model = LatentActionQuantization(
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
        
        # Load checkpoint
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Remove 'module.' prefix if present
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
            print(f"✅ LAPA loaded from {checkpoint_path}")
        else:
            print(f"⚠️  LAPA checkpoint not found: {checkpoint_path}")
            print("   Using randomly initialized model for structure comparison only")
        
        model.eval()
        print(f"   Latent dim (quant_dim): 32")
        
        return EncoderWrapper(model, 'lapa', device)
        
    except Exception as e:
        print(f"❌ Failed to load LAPA: {e}")
        return None


def load_uniskill_model(checkpoint_path, device='cuda'):
    """Load pretrained UniSkill IDM model"""
    print("Loading UniSkill model...")
    
    try:
        from dynamics.idm import IDM
        
        # UniSkill IDM config (from train_uniskill.py)
        model = IDM(
            num_layers=8,
            num_heads=4,
            hidden_dim=256,
            skill_dim=64,
            out_dim=768,  # For diffusion conditioning
            idm_resolution=224,
        ).to(device)
        
        # Load checkpoint
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            print(f"✅ UniSkill loaded from {checkpoint_path}")
        else:
            print(f"⚠️  UniSkill checkpoint not found: {checkpoint_path}")
            print("   Using randomly initialized model for structure comparison only")
        
        model.eval()
        print(f"   Latent dim (skill_dim): 64")
        
        return EncoderWrapper(model, 'uniskill', device)
        
    except Exception as e:
        print(f"❌ Failed to load UniSkill: {e}")
        return None


def compare_latent_representations(encoders, dataloader, device='cuda', save_dir='comparison_results'):
    """Compare latent representations from different encoders"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("COMPARING LATENT REPRESENTATIONS")
    print("="*70)
    
    results = {}
    
    for name, encoder in encoders.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {name.upper()}")
        print(f"{'='*50}")
        
        latents_list = []
        
        # Encode images
        for images, _ in tqdm(dataloader, desc=f"Encoding with {name}"):
            images = images.to(device)
            try:
                latent = encoder.encode(images)
                if latent is not None:
                    latents_list.append(latent.cpu().numpy())
            except Exception as e:
                print(f"⚠️  Error encoding with {name}: {e}")
                break
        
        if len(latents_list) == 0:
            print(f"⚠️  No latents extracted for {name}")
            continue
        
        latents = np.concatenate(latents_list, axis=0)
        
        # Flatten if needed
        if latents.ndim > 2:
            original_shape = latents.shape
            latents = latents.reshape(latents.shape[0], -1)
            print(f"   Reshaped from {original_shape} to {latents.shape}")
        
        # Compute statistics
        results[name] = {
            'latent_dim': latents.shape[-1],
            'mean': float(np.mean(latents)),
            'std': float(np.std(latents)),
            'min': float(np.min(latents)),
            'max': float(np.max(latents)),
            'l2_norm_mean': float(np.mean(np.linalg.norm(latents, axis=1))),
            'l2_norm_std': float(np.std(np.linalg.norm(latents, axis=1))),
        }
        
        # Print results
        print(f"\nLatent Statistics:")
        print(f"  Dimension: {results[name]['latent_dim']}")
        print(f"  Mean: {results[name]['mean']:.4f}")
        print(f"  Std:  {results[name]['std']:.4f}")
        print(f"  Range: [{results[name]['min']:.4f}, {results[name]['max']:.4f}]")
        print(f"  L2 Norm: {results[name]['l2_norm_mean']:.4f} ± {results[name]['l2_norm_std']:.4f}")
    
    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    table = "\n| Encoder | Latent Dim | Mean | Std | L2 Norm | Range |\n"
    table += "|---------|-----------|------|-----|---------|-------|\n"
    
    for name, res in results.items():
        table += f"| {name.upper()} | {res['latent_dim']} | "
        table += f"{res['mean']:.4f} | {res['std']:.4f} | "
        table += f"{res['l2_norm_mean']:.4f} | "
        table += f"[{res['min']:.2f}, {res['max']:.2f}] |\n"
    
    print(table)
    
    # Save results
    with open(save_dir / 'latent_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(save_dir / 'latent_comparison.md', 'w') as f:
        f.write("# Latent Representation Comparison\n\n")
        f.write(table)
    
    print(f"\n✅ Results saved to {save_dir}")
    
    return results


def visualize_latent_distributions(encoders, dataloader, device='cuda', save_dir='comparison_results'):
    """Visualize latent space distributions"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(encoders), figsize=(6*len(encoders), 4))
    if len(encoders) == 1:
        axes = [axes]
    
    # Get one batch for visualization
    images, _ = next(iter(dataloader))
    images = images.to(device)
    
    for idx, (name, encoder) in enumerate(encoders.items()):
        try:
            latent = encoder.encode(images[:64])  # First 64 images
            if latent is not None:
                latent = latent.cpu().numpy()
                
                # Flatten
                if latent.ndim > 2:
                    latent = latent.reshape(latent.shape[0], -1)
                
                # Plot histogram of first dimension
                axes[idx].hist(latent[:, 0], bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{name.upper()}\nLatent Dim[0] Distribution')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(alpha=0.3)
        except Exception as e:
            print(f"⚠️  Error visualizing {name}: {e}")
    
    plt.tight_layout()
    plt.savefig(save_dir / 'latent_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved latent distributions to {save_dir / 'latent_distributions.png'}")
    plt.close()


def main():
    """Main comparison script"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("ENCODER COMPARISON: VAE vs LAPA vs UniSkill")
    print("="*70)
    
    # Paths (modify these!)
    vae_checkpoint = 'logs_vae/VanillaVAE_droid/checkpoints/vae_final.pt'
    vae_config = 'config_vae_droid.yaml'
    lapa_checkpoint = '/home/jisookim/human_demon_action/LAPA/laq/results/model.pt'  # Adjust path
    uniskill_checkpoint = '/home/jisookim/human_demon_action/idm.pth'  # Your idm.pth file
    
    # Load models
    encoders = {}
    
    # VAE (our trained model)
    if Path(vae_checkpoint).exists():
        encoders['vae'] = load_vae_model(vae_checkpoint, vae_config, device)
    else:
        print(f"⚠️  VAE checkpoint not found: {vae_checkpoint}")
        print("   Train VAE first!")
    
    # LAPA (pretrained)
    lapa_encoder = load_lapa_model(lapa_checkpoint, device)
    if lapa_encoder:
        encoders['lapa'] = lapa_encoder
    
    # UniSkill (pretrained)
    uniskill_encoder = load_uniskill_model(uniskill_checkpoint, device)
    if uniskill_encoder:
        encoders['uniskill'] = uniskill_encoder
    
    if len(encoders) == 0:
        print("❌ No encoders loaded! Check your checkpoint paths.")
        return
    
    print(f"\n✅ Loaded {len(encoders)} encoder(s): {list(encoders.keys())}")
    
    # Load validation data
    print("\nLoading validation dataset...")
    from data.dataset_droid import get_dataloaders
    
    config = yaml.safe_load(open(vae_config))
    _, val_loader = get_dataloaders(
        data_path=config['data_params']['data_path'],
        image_size=config['data_params']['image_size'],
        train_batch_size=64,
        val_batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    
    # Run comparisons
    results = compare_latent_representations(encoders, val_loader, device)
    visualize_latent_distributions(encoders, val_loader, device)
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check comparison_results/ for detailed analysis")
    print("2. Use these latent representations for action prediction")
    print("3. Compare which encoder works best for your task")


if __name__ == '__main__':
    main()



















