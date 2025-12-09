#!/usr/bin/env python3
"""
Convert tracked points saved in vertex_tracks.npy (T, N, 3)
into per-frame PLY or OBJ point clouds for visualization in Blender.

Example:
  python export_tracks_to_ply.py \
    --npy /path/to/vertex_tracks.npy \
    --out-dir /path/to/out_ply \
    --format ply \
    --frame-index -1   # -1 = export all frames
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def save_points_as_ply(points: np.ndarray, path: Path):
    """
    Save Nx3 points as an ASCII PLY file (no color, just xyz).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.reshape(-1, 3)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    with path.open("w") as f:
        f.write(header)
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_points_as_obj(points: np.ndarray, path: Path):
    """
    Save Nx3 points as a simple OBJ file (vertices only).
    Blender will import them as a point cloud.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.reshape(-1, 3)

    with path.open("w") as f:
        f.write("# OBJ point cloud exported from vertex_tracks.npy\n")
        for x, y, z in points:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Export vertex_tracks.npy (T, N, 3) to per-frame PLY/OBJ for Blender."
    )
    p.add_argument("--npy", required=True, help="Path to vertex_tracks.npy")
    p.add_argument("--out-dir", required=True, help="Output directory for PLY/OBJ files")
    p.add_argument(
        "--format",
        choices=["ply", "obj"],
        default="ply",
        help="Output format (default: ply)",
    )
    p.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help=(
            "Which frame to export. "
            "-1 means export ALL frames as frame_0000.xxx, frame_0001.xxx, ..."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    npy_path = Path(args.npy)
    out_dir = Path(args.out_dir)

    tracks = np.load(npy_path)  # (T, N, 3)
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        raise ValueError(
            f"Expected vertex_tracks shape (T, N, 3), got {tracks.shape}"
        )

    T, N, _ = tracks.shape
    print(f"[INFO] Loaded {npy_path}, T={T}, N={N}")

    if args.frame_index >= 0:
        # export single frame
        if args.frame_index >= T:
            raise IndexError(
                f"frame_index {args.frame_index} out of range [0, {T-1}]"
            )
        points = tracks[args.frame_index]  # (N, 3)
        if args.format == "ply":
            out_path = out_dir / f"frame_{args.frame_index:04d}.ply"
            save_points_as_ply(points, out_path)
        else:
            out_path = out_dir / f"frame_{args.frame_index:04d}.obj"
            save_points_as_obj(points, out_path)
        print(f"[DONE] Saved frame {args.frame_index} -> {out_path}")
    else:
        # export all frames
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in range(T):
            points = tracks[t]
            if args.format == "ply":
                out_path = out_dir / f"frame_{t:04d}.ply"
                save_points_as_ply(points, out_path)
            else:
                out_path = out_dir / f"frame_{t:04d}.obj"
                save_points_as_obj(points, out_path)
            if (t + 1) % 10 == 0 or t == T - 1:
                print(f"[INFO] Exported {t+1}/{T} frames")

        print(f"[DONE] Exported all {T} frames to {out_dir}")


if __name__ == "__main__":
    main()
