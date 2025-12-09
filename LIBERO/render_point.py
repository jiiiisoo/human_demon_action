#!/usr/bin/env python3
"""
Render tracked 3D points onto LIBERO RGB frames to visually inspect tracking.

Assumes per-episode directory contains:
  - vertex_tracks.npy  (T, N, 3)  # world coordinates of tracked points
  - metadata.json      with keys: "_traj_index", "_frame_index", "_len", "actions_file"
  - actions.npy        (T, action_dim)
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
from extract_frames import ( DatasetInput, collect_dataset_inputs, ) 
from libero.libero  import benchmark, get_libero_path 
from libero.libero.envs import OffScreenRenderEnv 
import math 
import cv2
import itertools


from export_gt_pointcloud import (  # type: ignore
    collect_world_meshes,
    get_reference_center,
)
_CAMERA_CALIB_CACHE={}
# Lock in the working camera projection (from calibration logs).
_FIXED_CAMERA_PROJ = {
    "agentview": {"rot": "direct", "forward_neg_z": True, "v_flip": False},
    "robot0_eye_in_hand": {"rot": "direct", "forward_neg_z": True, "v_flip": False},
}

def load_episode_data(episode_dir: Path):
    """Load vertex tracks, metadata, and actions from an episode directory."""
    meta_path = episode_dir / "metadata_face_uniform.json"
    # tracks_path = episode_dir / "vertex_tracks_face_uniform.npy"
    tracks_path = episode_dir / f"vertex_tracks_resampled_1024.npy"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.json in {episode_dir}")
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Missing vertex_tracks.npy in {episode_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    actions_file = meta.get("actions_file", "actions_face_uniform.npy")
    actions_path = episode_dir / actions_file
    if not actions_path.is_file():
        raise FileNotFoundError(f"Missing actions file {actions_file} in {episode_dir}")

    vertex_tracks = np.load(tracks_path)  # (T, N, 3) in table-centered local frame
    actions = np.load(actions_path)       # (T, A)

    traj_index = int(meta.get("_traj_index", 0))
    frame_indices = meta.get("_frame_index", list(range(len(actions))))
    traj_len = int(meta.get("_len", len(frame_indices)))

    return vertex_tracks, actions, traj_index, frame_indices, traj_len


def project_points_world_to_image(
    points_world: np.ndarray,
    env,
    camera_name: str,
    img_width: int,
    img_height: int,
    debug: bool = False,
    return_depth: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Project world-coordinate points into image pixel coordinates using MuJoCo camera
    extrinsics/intrinsics. We auto-select z-forward sign and vertical flip that put
    the most points in the image (cached per camera) to handle convention drift.
    """

    if points_world.size == 0:
        return (np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)) if return_depth else (np.zeros((0, 2), dtype=np.float32), None)

    sim = env.sim
    model, data = sim.model, sim.data
    cam_id = model.camera_name2id(camera_name)

    # Extrinsics: world -> camera
    R_c2w = data.cam_xmat[cam_id].reshape(3, 3)
    t_c2w = data.cam_xpos[cam_id]

    near = float(model.vis.map.znear)
    far = float(model.vis.map.zfar)

    # Intrinsics from vertical fov (MuJoCo: fovy is vertical)
    fovy = float(model.cam_fovy[cam_id])
    fy = 0.5 * img_height / math.tan(math.radians(fovy) / 2.0)
    fx = fy * img_width / img_height
    cx = (img_width - 1) / 2.0
    cy = (img_height - 1) / 2.0

    cache_key = camera_name
    if cache_key not in _CAMERA_CALIB_CACHE:
        if camera_name in _FIXED_CAMERA_PROJ:
            _CAMERA_CALIB_CACHE[cache_key] = _FIXED_CAMERA_PROJ[camera_name]
        else:
            candidates = []
            samples = points_world
            if samples.shape[0] > 5000:
                idx = np.random.choice(samples.shape[0], size=5000, replace=False)
                samples = samples[idx]
            # Two rotation hypotheses: R_w2c = R_c2w.T (standard), and R_w2c_alt = R_c2w
            rot_options = [("transpose", R_c2w.T), ("direct", R_c2w)]
            for rot_name, R_w2c in rot_options:
                pts_cam_opt = (samples - t_c2w) @ R_w2c
                for forward_neg_z in (True, False):
                    z_forward = -pts_cam_opt[:, 2] if forward_neg_z else pts_cam_opt[:, 2]
                    valid = (z_forward > near) & (z_forward < far) & np.isfinite(z_forward)
                    if not np.any(valid):
                        continue
                    x_v = pts_cam_opt[valid, 0]
                    y_v = pts_cam_opt[valid, 1]
                    z_v = z_forward[valid]
                    for v_flip in (True, False):
                        u = fx * (x_v / z_v) + cx
                        v = cy - fy * (y_v / z_v) if v_flip else fy * (y_v / z_v) + cy
                        in_img = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
                        score = int(in_img.sum())
                        candidates.append((score, rot_name, forward_neg_z, v_flip))
            if not candidates:
                _CAMERA_CALIB_CACHE[cache_key] = {
                    "rot": "transpose",
                    "forward_neg_z": True,
                    "v_flip": True,
                }
            else:
                best = max(candidates, key=lambda x: x[0])
                _, rot_name, fneg, vflip = best
                print(
                    f"[project] calibrated {camera_name}: rot={rot_name}, forward_neg_z={fneg}, "
                    f"v_flip={vflip}, best_in_img={best[0]}"
                )
                _CAMERA_CALIB_CACHE[cache_key] = {
                    "rot": rot_name,
                    "forward_neg_z": fneg,
                    "v_flip": vflip,
                }

    cfg = _CAMERA_CALIB_CACHE[cache_key]
    rot_name = cfg.get("rot", "transpose")
    R_w2c_use = R_c2w.T if rot_name == "transpose" else R_c2w
    forward_neg_z = cfg.get("forward_neg_z", True)
    v_flip = cfg.get("v_flip", True)

    pts_cam = (points_world - t_c2w) @ R_w2c_use
    z_forward = -pts_cam[:, 2] if forward_neg_z else pts_cam[:, 2]
    valid_depth = (z_forward > near) & (z_forward < far) & np.isfinite(z_forward)
    if not np.any(valid_depth):
        if debug:
            print(f"[project-debug {camera_name}] depth_valid=0 with rot={rot_name}")
        return np.zeros((0, 2), dtype=np.float32)

    x_cam_v = pts_cam[valid_depth, 0]
    y_cam_v = pts_cam[valid_depth, 1]
    z_f_v = z_forward[valid_depth]

    u = fx * (x_cam_v / z_f_v) + cx
    v = cy - fy * (y_cam_v / z_f_v) if v_flip else fy * (y_cam_v / z_f_v) + cy

    in_img = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    if debug:
        total = len(points_world)
        valid_count = int(valid_depth.sum())
        in_count = int(in_img.sum())
        print(
            f"[project-debug {camera_name}] total={total}, depth_valid={valid_count}, "
            f"in_img={in_count}, rot={rot_name}, forward_neg_z={forward_neg_z}, v_flip={v_flip}, "
            f"z_forward_min={z_forward[valid_depth].min():.3f} z_forward_max={z_forward[valid_depth].max():.3f}"
        )
    if not np.any(in_img):
        return (np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)) if return_depth else (np.zeros((0, 2), dtype=np.float32), None)

    u = u[in_img]
    v = v[in_img]
    z_ret = z_forward[valid_depth][in_img]

    pts = np.stack([u, v], axis=-1).astype(np.float32)
    return (pts, z_ret.astype(np.float32)) if return_depth else (pts, None)


def render_tracks_video(
    episode_dir: Path,
    task_suite_name: str,
    task_id: int,
    camera_name: str,
    max_points: Optional[int],
    output_path: Path,
    fps: int = 10,
    point_radius: int = 2,
    thickness: int = -1,
    depth_colormap: bool = True,
):
    # Load per-episode data
    vertex_tracks, actions, traj_index, frame_indices, traj_len = load_episode_data(episode_dir)
    T, N, _ = vertex_tracks.shape
    assert T == len(actions), f"T mismatch: tracks {T} vs actions {len(actions)}"
    print(f"[INFO] Episode {traj_index}: T={T}, N={N} points")

    # Optionally subsample points
    if max_points is not None and N > max_points:
        sel = np.random.choice(N, size=max_points, replace=False)
        vertex_tracks = vertex_tracks[:, sel, :]
        N = max_points
        print(f"[INFO] Subsampling to {N} points for visualization")

    # Build LIBERO env
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = Path(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

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

    frames_bgr = []

    for t, act in enumerate(actions):
        # 동일한 액션 시퀀스로 env 롤아웃
        obs, _, done, _ = env.step(act.tolist())

        img_key = f"{camera_name}_image"
        if img_key not in obs:
            raise KeyError(f"{img_key} not found in observation")
        img = obs[img_key]  # H, W, 3 (uint8, RGB)

        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H, W, _ = frame_bgr.shape

        # vertex_tracks are stored in world coordinates from export_gt_track
        pts_world = vertex_tracks[t]              # (N, 3) world frame

        # 첫 프레임만 디버그
        debug = (t == 0)
        pts_px, z_vals = project_points_world_to_image(
            pts_world,
            env,
            camera_name,
            img_width=W,
            img_height=H,
            debug=(t == 0),
            return_depth=True,
        )

        if debug:
            print(f"[frame 0] total points = {len(pts_world)}, projected inside image = {len(pts_px)}")

        # Draw projected points with depth-based color
        if depth_colormap and len(pts_px) > 0 and z_vals is not None and len(z_vals) > 0:
            z_forward = z_vals
            z_norm = (z_forward - z_forward.min()) / (z_forward.max() - z_forward.min() + 1e-6)
            colors = np.stack(
                [
                    (z_norm * 255).astype(np.uint8),           # B
                    np.zeros_like(z_norm, dtype=np.uint8),     # G
                    ((1 - z_norm) * 255).astype(np.uint8),     # R
                ],
                axis=1,
            )
            for (u, v), color in zip(pts_px, colors):
                cv2.circle(
                    frame_bgr,
                    (int(u), int(v)),
                    point_radius,
                    (int(color[0]), int(color[1]), int(color[2])),
                    thickness,
                )
        else:
            for (u, v) in pts_px:
                cv2.circle(
                    frame_bgr,
                    (int(u), int(v)),
                    point_radius,
                    (0, 0, 255),  # red
                    thickness,
                )

        # Rotate output frame 180 degrees if desired
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
        frames_bgr.append(frame_bgr)

    env.close()

    if not frames_bgr:
        raise RuntimeError("No frames collected; nothing to write to video")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    H, W, _ = frames_bgr[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for f in frames_bgr:
        writer.write(f)
    writer.release()

    print(f"[DONE] Wrote video to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Render point tracks onto LIBERO RGB video.")
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
        help="Max number of points to visualize (randomly subsampled). Use -1 for all.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output video path (e.g., /path/to/episode_00042_agentview.mp4)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output video.",
    )
    p.add_argument(
        "--depth-colormap",
        action="store_true",
        help="Use depth-based color mapping for points.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    output_path = Path(args.output)

    max_points = None if args.max_points is None or args.max_points < 0 else args.max_points

    render_tracks_video(
        episode_dir=episode_dir,
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        camera_name=args.camera_name,
        max_points=max_points,
        output_path=output_path,
        fps=args.fps,
        depth_colormap=args.depth_colormap,
    )


if __name__ == "__main__":
    main()
