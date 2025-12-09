#!/usr/bin/env python3
"""
Render depth from a LIBERO env and back-project to point clouds (camera frame).

Saves per-step PLYs under ./depth_pointclouds/<camera>_step_<n>.ply
"""

import argparse
import glob
import math
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

from extract_frames import iter_tfrecord_records
from export_rlds_cropped_meshes import actions_from_record_bytes, ensure_action_shape
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def depth_buffer_to_meters(depth_buffer, near, far):
    depth_buffer = np.squeeze(depth_buffer)
    if depth_buffer.max() <= 1.0 + 1e-6:
        return (near * far) / (far - (far - near) * depth_buffer)
    return depth_buffer


def get_camera_intrinsics(sim, camera_name, width, height):
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    fy = 0.5 * height / math.tan(math.radians(fovy) / 2.0)
    fx = fy * width / height
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def depth_to_point_cloud(depth_m, fx, fy, cx, cy):
    h, w = depth_m.shape
    i, j = np.indices((h, w))
    z = depth_m
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]


def save_point_cloud_ply(points, path):
    header = "ply\nformat ascii 1.0\n"
    header += f"element vertex {len(points)}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "end_header\n"
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f")


def actions_from_tfrecord(pattern: str) -> np.ndarray:
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matched pattern: {pattern}")
    first = Path(shards[0])
    for record in iter_tfrecord_records(first):
        acts = actions_from_record_bytes(record)
        acts = acts.reshape(-1, 7)
        if acts.size == 0:
            continue
        acts = ensure_action_shape(acts, expected_dim=7)
        return acts
    raise ValueError(f"No actions found in TFRecord {first}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-suite", default="libero_goal")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--camera-names", nargs="+", default=["agentview", "robot0_eye_in_hand"])
    p.add_argument("--steps", type=int, default=10, help="Used if no tfrecord provided")
    p.add_argument("--tfrecord", default=None, help="Glob for TFRecord to pull actions from")
    p.add_argument("--output-dir", default="depth_pointclouds")
    return p.parse_args()


def main():
    args = parse_args()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
        "camera_names": args.camera_names,
        "camera_depths": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(args.task_id)
    env.set_init_state(init_states[0])

    os.makedirs(args.output_dir, exist_ok=True)
    if args.tfrecord:
        actions = actions_from_tfrecord(args.tfrecord)

    for step, act in enumerate(actions):
        obs, _, _, _ = env.step(act.tolist())
        near, far = env.sim.model.vis.map.znear, env.sim.model.vis.map.zfar
        for cam in args.camera_names:
            depth_key = f"{cam}_depth"
            if depth_key not in obs:
                continue
            depth_m = depth_buffer_to_meters(obs[depth_key], near, far)
            h, w = depth_m.shape
            fx, fy, cx, cy = get_camera_intrinsics(env.sim, cam, w, h)
            pts = depth_to_point_cloud(depth_m, fx, fy, cx, cy)
            ply_path = os.path.join(args.output_dir, f"{cam}_step_{step}.ply")
            save_point_cloud_ply(pts, ply_path)
            print(f"[step {step}] saved {len(pts)} pts -> {ply_path}")

    env.close()


if __name__ == "__main__":
    main()
