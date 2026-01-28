"""
Compute and save mean point count statistics for ablation dataset scene_points.npy files.

This script computes:
- Mean, median, min, max, std of the number of points (N dimension) across all demos
- The computed mean is used for padding/truncation when --ablation_dataset is enabled

Usage:
    python scripts/compute_ablation_mean_points.py \
        --tracking_root /weka/jisookim/dataset/robocasa/datasets/scene_pointrack \
        --filename scene_points.npy \
        --output_path ./ablation_mean_points.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def compute_mean_points(tracking_root: Path, filename: str) -> dict:
    """
    Compute mean number of points across all scene_points.npy files.

    Args:
        tracking_root: Root directory for tracking data
        filename: Name of the npy file (e.g., scene_points.npy)

    Returns:
        Dictionary with mean_points, min_points, max_points, std_points, count
    """
    # Find all matching npy files
    # Pattern: tracking_root/*/*/*/demo_*/filename
    # e.g., scene_pointrack/kitchen_coffee/CoffeePressButton/2024-04-25/demo_1/scene_points.npy
    pattern = str(tracking_root / "*" / "*" / "*" / "demo_*" / filename)
    npy_files = sorted(glob.glob(pattern))

    if len(npy_files) == 0:
        raise ValueError(f"No files found matching pattern: {pattern}")

    print(f"Found {len(npy_files)} files")

    point_counts = []

    for npy_file in tqdm(npy_files, desc="Computing point counts"):
        try:
            data = np.load(npy_file)  # Shape: (T, N, 3)
            # N is the number of points (axis 1)
            num_points = data.shape[1]
            point_counts.append(num_points)
        except Exception as e:
            print(f"Warning: Error loading {npy_file}: {e}")
            continue

    if len(point_counts) == 0:
        raise ValueError("No valid files found")

    point_counts = np.array(point_counts)

    return {
        "mean_points": int(np.round(np.mean(point_counts))),
        "median_points": int(np.median(point_counts)),
        "min_points": int(np.min(point_counts)),
        "max_points": int(np.max(point_counts)),
        "std_points": float(np.std(point_counts)),
        "count": len(point_counts),
        "tracking_root": str(tracking_root),
        "filename": filename,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute mean point count for ablation dataset")
    parser.add_argument(
        "--tracking_root",
        type=str,
        required=True,
        help="Root directory for tracking data (e.g., scene_pointrack/)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="scene_points.npy",
        help="Filename for tracking data (default: scene_points.npy)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ablation_mean_points.json",
        help="Output path for JSON file (default: ablation_mean_points.json)",
    )

    args = parser.parse_args()

    tracking_root = Path(args.tracking_root)
    output_path = Path(args.output_path)

    if not tracking_root.exists():
        print(f"Error: Tracking root not found: {tracking_root}")
        return 1

    # Compute statistics
    stats = compute_mean_points(tracking_root, args.filename)

    print("\n" + "=" * 60)
    print("Ablation Dataset Point Statistics")
    print("=" * 60)
    print(f"  Files processed: {stats['count']}")
    print(f"  Mean points: {stats['mean_points']}")
    print(f"  Median points: {stats['median_points']}")
    print(f"  Min points: {stats['min_points']}")
    print(f"  Max points: {stats['max_points']}")
    print(f"  Std points: {stats['std_points']:.2f}")
    print()

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics to {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
