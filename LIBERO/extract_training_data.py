#!/usr/bin/env python
"""
Script to extract agentview images and actions from LIBERO HDF5 datasets for training.

Usage:
    python extract_training_data.py --dataset /path/to/dataset.hdf5 --output ./training_data

Options:
    --dataset: Path to HDF5 dataset file (required)
    --output: Output directory for extracted data (default: ./extracted_data)
    --png: Save images as individual PNG files (default: False, saves as numpy arrays)
    --filter-key: Optional filter key to select specific demonstrations
"""

import argparse
import sys
import os

# Add libero to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libero'))

from libero.libero.utils.dataset_utils import extract_training_data


def main():
    parser = argparse.ArgumentParser(description='Extract training data from LIBERO datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to HDF5 dataset file')
    parser.add_argument('--output', type=str, default='./extracted_data',
                        help='Output directory for extracted data')
    parser.add_argument('--png', action='store_true',
                        help='Save images as individual PNG files (default: save as numpy arrays)')
    parser.add_argument('--filter-key', type=str, default=None,
                        help='Optional filter key to select specific demonstrations')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    print("="*80)
    print("LIBERO Training Data Extraction")
    print("="*80)
    
    # Extract the data
    metadata = extract_training_data(
        dataset_path=args.dataset,
        output_dir=args.output,
        filter_key=args.filter_key,
        save_images_as_png=args.png
    )
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Task: {metadata['language_instruction']}")
    print(f"Total demonstrations: {metadata['num_demonstrations']}")
    print(f"Total frames: {metadata['total_frames']}")
    print(f"Image shape: {metadata['image_shape']}")
    print(f"Action dimension: {metadata['action_dim']}")
    print(f"Save format: {metadata['save_format']}")
    print(f"\nData saved to: {args.output}")
    print("="*80)
    
    if not args.png:
        print("\nTo load the data in your training script:")
        print(f"  import numpy as np")
        print(f"  images = np.load('{os.path.join(args.output, 'all_images.npy')}')")
        print(f"  actions = np.load('{os.path.join(args.output, 'all_actions.npy')}')")
    

if __name__ == "__main__":
    main()


