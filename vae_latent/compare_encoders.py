"""
Encoder Comparison Framework
Compare VAE, VQ-VAE, and IDM-Transformer encoders for action representation learning
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import matplotlib.pyplot as plt
import seaborn as sns


class EncoderComparator:
    """Compare different encoder architectures"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {
            'vae': {},
            'vqvae': {},
            'idm': {}
        }
        
        # Initialize metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        
    def load_model(self, model_type, checkpoint_path, config):
        """Load a trained encoder model"""
        if model_type == 'vae':
            from vanilla_vae_model import VanillaVAE
            model = VanillaVAE(**config['model_params'])
        elif model_type == 'vqvae':
            from vq_vae_model import VQVAE
            model = VQVAE(**config['model_params'])
        elif model_type == 'idm':
            from idm_model import IDMEncoder
            model = IDMEncoder(**config['model_params'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device).eval()
        
        return model
    
    def encode(self, model, model_type, image):
        """Encode image to latent representation"""
        with torch.no_grad():
            if model_type == 'vae':
                mu, log_var = model.encode(image)
                return mu  # Use mean as latent
            elif model_type == 'vqvae':
                latent, _ = model.encode(image)
                return latent
            elif model_type == 'idm':
                latent = model.encode(image)
                return latent
    
    def decode(self, model, model_type, latent):
        """Decode latent to image"""
        with torch.no_grad():
            if model_type in ['vae', 'vqvae']:
                return model.decode(latent)
            elif model_type == 'idm':
                return model.decode(latent)
    
    def test_reconstruction(self, model, model_type, dataloader):
        """Test reconstruction quality"""
        print(f"\n{'='*50}")
        print(f"Testing {model_type.upper()} Reconstruction")
        print(f"{'='*50}")
        
        mse_list = []
        psnr_list = []
        ssim_list = []
        latent_dims = []
        
        model.eval()
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc=f"Evaluating {model_type}"):
                images = images.to(self.device)
                
                # Encode
                latent = self.encode(model, model_type, images)
                latent_dims.append(latent.shape[-1])  # Latent dimension
                
                # Decode
                recon = self.decode(model, model_type, latent)
                
                # Normalize to [0, 1] for metrics
                images_norm = (images + 1) / 2  # [-1,1] -> [0,1]
                recon_norm = (recon + 1) / 2
                
                # Compute metrics
                mse = torch.mean((images - recon) ** 2).item()
                psnr = self.psnr(recon_norm, images_norm).item()
                ssim = self.ssim(recon_norm, images_norm).item()
                
                mse_list.append(mse)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
        
        # Store results
        results = {
            'mse_mean': np.mean(mse_list),
            'mse_std': np.std(mse_list),
            'psnr_mean': np.mean(psnr_list),
            'psnr_std': np.std(psnr_list),
            'ssim_mean': np.mean(ssim_list),
            'ssim_std': np.std(ssim_list),
            'latent_dim': latent_dims[0],
        }
        
        self.results[model_type]['reconstruction'] = results
        
        # Print results
        print(f"\nResults:")
        print(f"  MSE:  {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
        print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
        print(f"  Latent Dim: {results['latent_dim']}")
        
        return results
    
    def test_latent_space(self, model, model_type, dataloader):
        """Analyze latent space properties"""
        print(f"\n{'='*50}")
        print(f"Analyzing {model_type.upper()} Latent Space")
        print(f"{'='*50}")
        
        latents = []
        
        model.eval()
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc=f"Encoding {model_type}"):
                images = images.to(self.device)
                latent = self.encode(model, model_type, images)
                latents.append(latent.cpu().numpy())
        
        latents = np.concatenate(latents, axis=0)
        
        # Compute statistics
        results = {
            'mean': float(np.mean(latents)),
            'std': float(np.std(latents)),
            'min': float(np.min(latents)),
            'max': float(np.max(latents)),
            'l2_norm_mean': float(np.mean(np.linalg.norm(latents, axis=1))),
            'effective_rank': self._effective_rank(latents),
        }
        
        self.results[model_type]['latent_space'] = results
        
        # Print results
        print(f"\nLatent Space Statistics:")
        print(f"  Mean: {results['mean']:.4f}")
        print(f"  Std:  {results['std']:.4f}")
        print(f"  Range: [{results['min']:.4f}, {results['max']:.4f}]")
        print(f"  L2 Norm (mean): {results['l2_norm_mean']:.4f}")
        print(f"  Effective Rank: {results['effective_rank']:.2f}")
        
        return results
    
    def test_frame_prediction(self, model, model_type, dataloader, action_predictor=None):
        """Test frame prediction capability (key metric!)"""
        print(f"\n{'='*50}")
        print(f"Testing {model_type.upper()} Frame Prediction")
        print(f"{'='*50}")
        
        if action_predictor is None:
            print("⚠️  No action predictor provided, skipping frame prediction test")
            return None
        
        prediction_mse = []
        prediction_psnr = []
        prediction_ssim = []
        
        model.eval()
        action_predictor.eval()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Predicting frames")):
                if batch_idx >= len(dataloader) // 2:  # Use first half for curr, second half for next
                    break
                    
                # Get current and next frame pairs
                curr_images = images.to(self.device)
                # Note: In real scenario, you'd have actual (curr, next) pairs from dataset
                # For now, we'll use consecutive batches as proxy
                
                # Encode current frame
                z_curr = self.encode(model, model_type, curr_images)
                
                # TODO: Predict action and next latent
                # z_next_pred = action_predictor(z_curr)
                # frame_next_pred = self.decode(model, model_type, z_next_pred)
                
                # For now, just measure latent space quality
                pass
        
        print("⚠️  Frame prediction test not fully implemented yet")
        print("    Needs action predictor training first")
        
        return None
    
    def _effective_rank(self, X):
        """Compute effective rank of latent representations"""
        # Normalize
        X = X - X.mean(axis=0)
        
        # SVD
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        
        # Normalize singular values
        s = s / s.sum()
        
        # Effective rank (entropy-based)
        s = s[s > 1e-10]  # Remove near-zero values
        eff_rank = np.exp(-np.sum(s * np.log(s)))
        
        return eff_rank
    
    def visualize_reconstructions(self, models, dataloader, save_dir='comparison_results'):
        """Visualize reconstructions from all models side by side"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Get one batch
        images, _ = next(iter(dataloader))
        images = images[:8].to(self.device)  # First 8 images
        
        fig, axes = plt.subplots(len(models) + 1, 8, figsize=(20, 3 * (len(models) + 1)))
        
        # Original images
        for i in range(8):
            img = (images[i].cpu() + 1) / 2  # [-1,1] -> [0,1]
            axes[0, i].imshow(img.permute(1, 2, 0))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', rotation=0, labelpad=40, fontsize=12)
        
        # Reconstructions
        for model_idx, (model_type, model) in enumerate(models.items(), 1):
            model.eval()
            with torch.no_grad():
                latent = self.encode(model, model_type, images)
                recon = self.decode(model, model_type, latent)
            
            for i in range(8):
                img = (recon[i].cpu() + 1) / 2
                axes[model_idx, i].imshow(img.permute(1, 2, 0))
                axes[model_idx, i].axis('off')
                if i == 0:
                    axes[model_idx, i].set_ylabel(model_type.upper(), rotation=0, labelpad=40, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved reconstruction visualization to {save_dir / 'reconstruction_comparison.png'}")
        plt.close()
    
    def create_comparison_table(self, save_dir='comparison_results'):
        """Create a comparison table of all results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Create markdown table
        table = "# Encoder Comparison Results\n\n"
        table += "## Reconstruction Quality\n\n"
        table += "| Model | MSE ↓ | PSNR ↑ | SSIM ↑ | Latent Dim |\n"
        table += "|-------|-------|--------|--------|------------|\n"
        
        for model_type in ['vae', 'vqvae', 'idm']:
            if 'reconstruction' in self.results[model_type]:
                r = self.results[model_type]['reconstruction']
                table += f"| {model_type.upper()} | "
                table += f"{r['mse_mean']:.6f} | "
                table += f"{r['psnr_mean']:.2f} | "
                table += f"{r['ssim_mean']:.4f} | "
                table += f"{r['latent_dim']} |\n"
        
        table += "\n## Latent Space Properties\n\n"
        table += "| Model | Mean | Std | L2 Norm | Effective Rank |\n"
        table += "|-------|------|-----|---------|----------------|\n"
        
        for model_type in ['vae', 'vqvae', 'idm']:
            if 'latent_space' in self.results[model_type]:
                r = self.results[model_type]['latent_space']
                table += f"| {model_type.upper()} | "
                table += f"{r['mean']:.4f} | "
                table += f"{r['std']:.4f} | "
                table += f"{r['l2_norm_mean']:.4f} | "
                table += f"{r['effective_rank']:.2f} |\n"
        
        # Save table
        with open(save_dir / 'comparison_table.md', 'w') as f:
            f.write(table)
        
        print(f"\n✅ Saved comparison table to {save_dir / 'comparison_table.md'}")
        print("\n" + table)
        
        # Save JSON
        with open(save_dir / 'comparison_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Saved detailed results to {save_dir / 'comparison_results.json'}")
    
    def plot_metrics_comparison(self, save_dir='comparison_results'):
        """Plot bar charts comparing all metrics"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        models = []
        mse_vals = []
        psnr_vals = []
        ssim_vals = []
        
        for model_type in ['vae', 'vqvae', 'idm']:
            if 'reconstruction' in self.results[model_type]:
                models.append(model_type.upper())
                r = self.results[model_type]['reconstruction']
                mse_vals.append(r['mse_mean'])
                psnr_vals.append(r['psnr_mean'])
                ssim_vals.append(r['ssim_mean'])
        
        # MSE (lower is better)
        axes[0].bar(models, mse_vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0].set_ylabel('MSE')
        axes[0].set_title('Reconstruction MSE (↓ better)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # PSNR (higher is better)
        axes[1].bar(models, psnr_vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('PSNR (↑ better)')
        axes[1].grid(axis='y', alpha=0.3)
        
        # SSIM (higher is better)
        axes[2].bar(models, ssim_vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[2].set_ylabel('SSIM')
        axes[2].set_title('SSIM (↑ better)')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved metrics comparison to {save_dir / 'metrics_comparison.png'}")
        plt.close()


def main():
    """Example usage"""
    import yaml
    from data.dataset_droid import get_dataloaders
    
    # Load configs
    configs = {
        'vae': yaml.safe_load(open('config_vae_droid.yaml')),
        # 'vqvae': yaml.safe_load(open('config_vqvae_droid.yaml')),
        # 'idm': yaml.safe_load(open('config_idm_droid.yaml')),
    }
    
    # Load data
    print("Loading validation dataset...")
    _, val_loader = get_dataloaders(
        data_path=configs['vae']['data_params']['data_path'],
        image_size=configs['vae']['data_params']['image_size'],
        train_batch_size=64,
        val_batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    
    # Initialize comparator
    comparator = EncoderComparator(device='cuda')
    
    # Load models
    models = {}
    checkpoints = {
        'vae': 'logs_vae/VanillaVAE_droid/checkpoints/vae_final.pt',
        # 'vqvae': 'logs_vqvae/checkpoints/vqvae_final.pt',
        # 'idm': 'logs_idm/checkpoints/idm_final.pt',
    }
    
    for model_type, ckpt_path in checkpoints.items():
        if Path(ckpt_path).exists():
            print(f"\nLoading {model_type.upper()} from {ckpt_path}")
            models[model_type] = comparator.load_model(model_type, ckpt_path, configs[model_type])
        else:
            print(f"⚠️  Checkpoint not found: {ckpt_path}")
    
    # Run comparisons
    for model_type, model in models.items():
        comparator.test_reconstruction(model, model_type, val_loader)
        comparator.test_latent_space(model, model_type, val_loader)
    
    # Create visualizations
    if len(models) > 0:
        comparator.visualize_reconstructions(models, val_loader)
        comparator.create_comparison_table()
        comparator.plot_metrics_comparison()
    
    print("\n" + "="*50)
    print("✅ Comparison complete!")
    print("="*50)


if __name__ == '__main__':
    main()








