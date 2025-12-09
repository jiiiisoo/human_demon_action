#!/usr/bin/env python3
"""
Render tracked 3D points alone (no RGB background) as a video.

Assumes per-episode directory contains:
  - vertex_tracks.npy  (T, N, 3)  # world coordinates of tracked points
  - metadata.json      with keys: "_traj_index", "_frame_index", "_len", "actions_file"
  - actions.npy        (T, A)

Example usage:
  python render_points_only_video.py \
    --episode-dir /mnt/data/libero/modified_libero_whole_mesh/libero_goal_no_noops/1.0.0/episode_00000 \
    --task-suite libero_goal \
    --task-id 0 \
    --camera-name agentview \
    --output /mnt/data/libero/vis/ep00000_agentview_points_only.mp4 \
    --max-points 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl
from tqdm import tqdm

from extract_frames import (
    DatasetInput,
    collect_dataset_inputs,
)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import itertools
import math

import cv2

# ---------------------------------------------------------------------
# Episode data I/O
# ---------------------------------------------------------------------


def load_episode_data(episode_dir: Path):
    """Load vertex tracks, metadata, and actions from an episode directory."""
    meta_path = episode_dir / "metadata.json"
    tracks_path = episode_dir / "vertex_tracks.npy"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.json in {episode_dir}")
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Missing vertex_tracks.npy in {episode_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    actions_file = meta.get("actions_file", "actions.npy")
    actions_path = episode_dir / actions_file
    if not actions_path.is_file():
        raise FileNotFoundError(f"Missing actions file {actions_file} in {episode_dir}")

    vertex_tracks = np.load(tracks_path)  # (T, N, 3)
    actions = np.load(actions_path)       # (T, A)

    traj_index = int(meta.get("_traj_index", 0))
    frame_indices = meta.get("_frame_index", list(range(len(actions))))
    traj_len = int(meta.get("_len", len(frame_indices)))

    return vertex_tracks, actions, traj_index, frame_indices, traj_len


# ---------------------------------------------------------------------
# Projection: world -> image (auto-calibrate axes / signs)
# ---------------------------------------------------------------------

_CAMERA_CALIB_CACHE: Dict[str, Dict[str, Any]] = {}


def project_points_world_to_image(
    points_world: np.ndarray,
    env,
    camera_name: str,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Project world-coordinate points into image pixel coordinates.

    - Uses MuJoCo camera pose, but automatically searches over axis permutations
      and sign flips to maximize number of points that land in the image.
    - Calibration is done once per camera and cached.
    """

    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    sim = env.sim
    model, data = sim.model, sim.data
    cam_id = model.camera_name2id(camera_name)

    # camera -> world rotation; transpose for world -> camera
    R_c2w = data.cam_xmat[cam_id].reshape(3, 3)
    t_c2w = data.cam_xpos[cam_id]
    R_w2c = R_c2w.T

    pts_cam_base = (points_world - t_c2w) @ R_w2c  # (N, 3)

    near = float(model.vis.map.znear)
    far = float(model.vis.map.zfar)

    # intrinsics
    fovy = float(model.cam_fovy[cam_id])  # degrees
    fy = 0.5 * img_height / math.tan(math.radians(fovy) / 2.0)
    fx = fy * img_width / img_height
    cx = (img_width - 1) / 2.0
    cy = (img_height - 1) / 2.0

    cache_key = camera_name
    if cache_key not in _CAMERA_CALIB_CACHE:
        best_score = -1
        best_cfg = None

        pts_for_calib = pts_cam_base
        if pts_for_calib.shape[0] > 5000:
            idx = np.random.choice(pts_for_calib.shape[0], size=5000, replace=False)
            pts_for_calib = pts_for_calib[idx]

        for axes in itertools.permutations([0, 1, 2], 3):
            for signs in itertools.product([-1.0, 1.0], repeat=3):
                x_cam = signs[0] * pts_for_calib[:, axes[0]]
                y_cam = signs[1] * pts_for_calib[:, axes[1]]
                z_cam = signs[2] * pts_for_calib[:, axes[2]]

                # assume camera looks along -Z → depth = -z_cam
                z_forward = -z_cam

                valid_depth = (z_forward > near) & (z_forward < far)
                if not np.any(valid_depth):
                    continue

                x_cam_v = x_cam[valid_depth]
                y_cam_v = y_cam[valid_depth]
                z_f_v = z_forward[valid_depth]

                u = fx * (x_cam_v / z_f_v) + cx
                v = fy * (y_cam_v / z_f_v) + cy

                in_img = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
                score = int(in_img.sum())

                if score > best_score:
                    best_score = score
                    best_cfg = (axes, signs)

        if best_cfg is None:
            best_cfg = ((0, 1, 2), (1.0, 1.0, 1.0))
            print(
                f"[project] WARNING: could not find good axis mapping for camera '{camera_name}', "
                f"using identity."
            )

        axes, signs = best_cfg
        print(
            f"[project] calibrated camera '{camera_name}': axes={axes}, "
            f"signs={signs}, best_in_img={best_score}"
        )
        _CAMERA_CALIB_CACHE[cache_key] = {
            "axes": axes,
            "signs": signs,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "near": near,
            "far": far,
        }

    cfg = _CAMERA_CALIB_CACHE[cache_key]
    axes = cfg["axes"]
    signs = cfg["signs"]
    fx = cfg["fx"]
    fy = cfg["fy"]
    cx = cfg["cx"]
    cy = cfg["cy"]
    near = cfg["near"]
    far = cfg["far"]

    x_cam = signs[0] * pts_cam_base[:, axes[0]]
    y_cam = signs[1] * pts_cam_base[:, axes[1]]
    z_cam = signs[2] * pts_cam_base[:, axes[2]]

    z_forward = -z_cam

    valid_depth = (z_forward > near) & (z_forward < far)
    if not np.any(valid_depth):
        return np.zeros((0, 2), dtype=np.float32)

    x_cam_v = x_cam[valid_depth]
    y_cam_v = y_cam[valid_depth]
    z_f_v = z_forward[valid_depth]

    u = fx * (x_cam_v / z_f_v) + cx
    v = fy * (y_cam_v / z_f_v) + cy

    in_img = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    if not np.any(in_img):
        return np.zeros((0, 2), dtype=np.float32)

    u = u[in_img]
    v = v[in_img]

    return np.stack([u, v], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------
# Main rendering logic (points only)
# ---------------------------------------------------------------------


def render_points_only_video(
    episode_dir: Path,
    task_suite_name: str,
    task_id: int,
    camera_name: str,
    max_points: Optional[int],
    output_path: Path,
    fps: int = 10,
    point_radius: int = 2,
    thickness: int = -1,
    bg_color: Tuple[int, int, int] = (0, 0, 0),  # black background
):
    # Load episode data
    vertex_tracks, actions, traj_index, frame_indices, traj_len = load_episode_data(
        episode_dir
    )
    T, N, _ = vertex_tracks.shape
    assert T == len(actions), f"T mismatch: tracks {T} vs actions {len(actions)}"
    print(f"[INFO] Episode {traj_index}: T={T}, N={N} points")

    # Subsample points for visualization if desired
    if max_points is not None and max_points > 0 and N > max_points:
        sel = np.random.choice(N, size=max_points, replace=False)
        vertex_tracks = vertex_tracks[:, sel, :]
        N = max_points
        print(f"[INFO] Subsampling to {N} points for visualization")

    # Build LIBERO env to replay actions (for camera pose)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = Path(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    # 여기서 camera_heights / widths 는 그냥 프레임 사이즈를 결정하는 용도
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_names": [camera_name],
        "camera_depths": False,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    # 첫 스텝에서 실제 image size를 얻어서 고정
    obs, _, _, _ = env.step(actions[0].tolist())
    img_key = f"{camera_name}_image"
    if img_key not in obs:
        raise KeyError(f"{img_key} not found in observation")
    H, W, _ = obs[img_key].shape
    print(f"[INFO] Using frame size H={H}, W={W}")

    # 다시 초기화 후 제대로 롤아웃
    env.reset()
    env.set_init_state(init_states[0])

    frames_bgr = []

    for t, act in enumerate(actions):
        # 롤아웃해서 카메라 pose 업데이트
        obs, _, done, _ = env.step(act.tolist())

        # 빈 배경 프레임 (BGR)
        frame_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        frame_bgr[:, :] = bg_color  # (B, G, R)

        pts_world = vertex_tracks[t]  # (N, 3)
        pts_px = project_points_world_to_image(
            pts_world, env, camera_name, img_width=W, img_height=H
        )

        print(f"[frame {t}] total points = {N}, projected inside image = {len(pts_px)}")

        for (u, v) in pts_px:
            cv2.circle(
                frame_bgr,
                (int(u), int(v)),
                point_radius,
                (0, 0, 255),  # red
                thickness,
            )

        frames_bgr.append(frame_bgr)

    env.close()

    if not frames_bgr:
        raise RuntimeError("No frames collected; nothing to write to video")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    for f in frames_bgr:
        writer.write(f)
    writer.release()

    print(f"[DONE] Wrote point-only video to {output_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Render point tracks (no RGB background).")
    p.add_argument(
        "--episode-dir",
        required=True,
        help="Path to episode directory containing vertex_tracks.npy and metadata.json",
    )
    p.add_argument(
        "--task-suite",
        default="libero_goal",
        help="LIBERO task suite name (e.g., libero_goal, libero_10, libero_spatial)",
    )
    p.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task id within the suite to replay.",
    )
    p.add_argument(
        "--camera-name",
        default="agentview",
        help="Camera name to render from (e.g., agentview, robot0_eye_in_hand)",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Max number of points to visualize (randomly subsampled). Use -1 or 0 for all.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output video path (e.g., /path/to/episode_00042_points_only.mp4)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output video.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    output_path = Path(args.output)

    max_points = None if args.max_points is None or args.max_points <= 0 else args.max_points

    render_points_only_video(
        episode_dir=episode_dir,
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        camera_name=args.camera_name,
        max_points=max_points,
        output_path=output_path,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
