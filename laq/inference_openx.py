import torch
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
from PIL import Image

from laq_model import LatentActionQuantization

# Argument parsing
parser = argparse.ArgumentParser(description='LAQ OpenX Inference')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input data file (jsonl format)')
parser.add_argument('--laq_checkpoint', type=str, default='/workspace/LAPA/lapa_checkpoints/laq_openx.pt', help='Path to the laq checkpoint')
parser.add_argument('--output_file', type=str, default='openx_latent_actions.jsonl', help='Output file for latent actions')
parser.add_argument('--output_dir', type=str, default='reconstructed_images', help='Directory to save reconstructed images')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
parser.add_argument('--image_size', type=int, default=256, help='Image size')
parser.add_argument('--window_size', type=int, default=2, help='Window size (number of frames)')
parser.add_argument('--save_images', action='store_true', help='Save reconstructed images as files')

args = parser.parse_args()

# Model parameters inferred from checkpoint
dist_number = 32  # From codebook shape (8, 32)
codebook_size = 8  # From codebook shape (8, 32)
spatial_depth = 8  # From checkpoint structure
temporal_depth = 8  # From checkpoint structure
code_seq_len = 4  # From VQ inference output - returns 4 tokens
dim = 1024  # From checkpoint - confirmed by layer sizes
quant_dim = 32  # From checkpoint - VQ codebook dimension is 32

print(f"Loading LAQ model with parameters:")
print(f"  - codebook_size: {codebook_size}")
print(f"  - dist_number: {dist_number}")
print(f"  - spatial_depth: {spatial_depth}")
print(f"  - temporal_depth: {temporal_depth}")

# Initialize LAQ model
laq = LatentActionQuantization(
    dim=dim,
    quant_dim=quant_dim,
    codebook_size=codebook_size,
    image_size=args.image_size,
    patch_size=32,  # From checkpoint - to_patch_emb layers confirm 32x32 patches
    spatial_depth=spatial_depth,
    temporal_depth=temporal_depth,
    dim_head=64,
    heads=16,
    code_seq_len=code_seq_len,
).cuda()

# Load checkpoint
print(f"Loading checkpoint from: {args.laq_checkpoint}")
checkpoint = torch.load(args.laq_checkpoint, map_location='cuda')
laq.load_state_dict(checkpoint)
laq.eval()

# Create output directory for images
if args.save_images:
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Images will be saved to: {args.output_dir}")

# Load input data
print(f"Loading input data from: {args.input_file}")
input_data = []
with open(args.input_file, 'r') as f:
    for line in f:
        input_data.append(json.loads(line.strip()))

print(f"Loaded {len(input_data)} samples")

# Image transforms
transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor()
])

# Helper function to save tensor as image
def save_tensor_as_image(tensor, filepath):
    """
    Save a tensor as an image file.
    tensor: [C, H, W] or [H, W, C] tensor with values in [0, 1] or [-1, 1]
    """
    import torchvision.transforms as T
    from torchvision.utils import save_image
    
    # Ensure tensor is in [C, H, W] format
    if tensor.dim() == 3:
        if tensor.shape[0] not in [1, 3]:  # If not channel-first
            tensor = tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    # Normalize to [0, 1] if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Save image
    save_image(tensor, filepath)

# Dataset class for OpenX data
class OpenXDataset(Dataset):
    def __init__(self, data, transform, window_size=2):
        self.data = data
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # For OpenX, we assume each sample has 'image' field
        # If it's a single image, we duplicate it for temporal dimension
        if isinstance(sample['image'], str):
            # Single image path
            img = Image.open(sample['image'])
            if self.transform:
                img = self.transform(img)
            # Duplicate for temporal dimension
            images = torch.stack([img] * self.window_size, dim=1)  # [C, T, H, W]
        elif isinstance(sample['image'], list):
            # Multiple images
            images = []
            for img_path in sample['image'][:self.window_size]:
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            
            # Pad if not enough images
            while len(images) < self.window_size:
                images.append(images[-1])  # Repeat last image
            
            images = torch.stack(images, dim=1)  # [C, T, H, W]
        
        return {
            'images': images,
            'id': sample.get('id', f'sample_{idx}'),
            'instruction': sample.get('instruction', '')
        }

# Create dataset and dataloader
dataset = OpenXDataset(input_data, transform, args.window_size)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Inference
print("Starting inference...")
results = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        images = batch['images'].cuda()  # [B, C, T, H, W]
        
        try:
            # Forward pass through LAQ - get both latent codes and reconstructed images
            with torch.no_grad():
                # Get discrete latent codes from images
                latent_codes = laq(images, return_only_codebook_ids=True)
                print(f"Generated latent codes: {latent_codes.shape}")
                
                # Get reconstructed images using return_recons_only=True
                reconstructed_images = laq(images, return_recons_only=True)
                print(f"Reconstructed images shape: {reconstructed_images.shape}")
            
            # Convert to numpy - handle different return types
            if torch.is_tensor(latent_codes):
                latent_codes = latent_codes.cpu().numpy()
            elif isinstance(latent_codes, (list, tuple)):
                # Handle tuple/list of tensors
                if isinstance(latent_codes, tuple) and len(latent_codes) > 0:
                    if torch.is_tensor(latent_codes[0]):
                        latent_codes = [code.cpu().numpy() if torch.is_tensor(code) else code for code in latent_codes]
                latent_codes = np.array(latent_codes)
            else:
                latent_codes = np.array(latent_codes)
            
            # Handle reconstructed_images - LAQ model does have image reconstruction capability
            if isinstance(reconstructed_images, tuple):
                reconstructed_images = reconstructed_images[0]
            
            if torch.is_tensor(reconstructed_images):
                reconstructed_images = reconstructed_images.cpu().numpy()
            else:
                reconstructed_images = np.array(reconstructed_images)
            
            
            for i in range(len(batch['id'])):
                sample_id = batch['id'][i]
                
                # Save reconstructed images if requested
                saved_image_paths = []
                if args.save_images and reconstructed_images.size > 0:
                    try:
                        # Handle temporal dimension - save each frame
                        recon_img = torch.tensor(reconstructed_images[i])  # [C, T, H, W] or [C, H, W]
                        
                        if recon_img.dim() == 4:  # [C, T, H, W]
                            for t in range(recon_img.shape[1]):
                                frame = recon_img[:, t, :, :]  # [C, H, W]
                                img_path = os.path.join(args.output_dir, f"{sample_id}_frame_{t}.png")
                                save_tensor_as_image(frame, img_path)
                                saved_image_paths.append(img_path)
                        else:  # [C, H, W]
                            img_path = os.path.join(args.output_dir, f"{sample_id}_reconstructed.png")
                            save_tensor_as_image(recon_img, img_path)
                            saved_image_paths.append(img_path)
                    except Exception as e:
                        print(f"Could not save images: {e}")
                
                result = {
                    'id': sample_id,
                    'instruction': batch['instruction'][i],
                    'latent_codes': latent_codes[i].tolist() if hasattr(latent_codes[i], 'tolist') else latent_codes[i],
                    'reconstructed_image_shape': list(reconstructed_images[i].shape),
                    'saved_images': saved_image_paths if args.save_images else []
                }
                results.append(result)
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Add empty results for failed samples
            for i in range(len(batch['id'])):
                result = {
                    'id': batch['id'][i],
                    'instruction': batch['instruction'][i],
                    'latent_action': None,
                    'error': str(e)
                }
                results.append(result)

# Save results
print(f"Saving results to: {args.output_file}")
with open(args.output_file, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

print(f"Inference complete! Processed {len(results)} samples.")
print(f"Results saved to: {args.output_file}")
if args.save_images:
    print(f"Reconstructed images saved to: {args.output_dir}")
    total_images = sum(len(r['saved_images']) for r in results if 'saved_images' in r)
    print(f"Total images saved: {total_images}")

