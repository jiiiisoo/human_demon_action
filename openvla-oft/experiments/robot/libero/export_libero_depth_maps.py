"""
Export per-frame depth maps by replaying actions from LIBERO HDF5 demos.

This mirrors the environment setup and action replay flow from
regenerate_libero_dataset_with_tracks.py, and pulls depth observations
in the same way as LIBERO/scripts/create_dataset.py.
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional

import cv2
import h5py
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.libero.libero_utils import get_libero_dummy_action


DEPTH_KEY_BY_CAMERA = {
    "agentview": "agentview_depth",
    "robot0_eye_in_hand": "robot0_eye_in_hand_depth",
}

RGB_KEY_BY_CAMERA = {
    "agentview": "agentview_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
}


def _build_env(task, resolution: int, camera_names: List[str]):
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
        "camera_depths": True,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "ignore_done": True,
        "reward_shaping": True,
        "control_freq": 20,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env




def _get_camera_intrinsics(sim, camera_name: str, width: int, height: int):
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    fy = 0.5 * height / math.tan(math.radians(fovy) / 2.0)
    fx = fy * width / height
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "fovy_deg": float(fovy),
    }


def _get_camera_extrinsics(sim, camera_name: str):
    cam_id = sim.model.camera_name2id(camera_name)
    r_c2w = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    t_w = sim.data.cam_xpos[cam_id].copy()
    return r_c2w, t_w


def _save_depth_png(depth_m: np.ndarray, out_path: Path):
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), depth_mm)


def _save_depth_npy(depth_m: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, depth_m.astype(np.float32))


def _save_depth_viz_png(depth_m: np.ndarray, out_path: Path):
    depth = depth_m.astype(np.float32)
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    denom = max(depth_max - depth_min, 1e-6)
    depth_norm = ((depth - depth_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), depth_norm)


def _save_rgb_png(rgb: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def _save_intrinsics_json(intrinsics: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(intrinsics, f, indent=2)


def _save_extrinsics_npz(r_c2w: np.ndarray, t_w: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, r_c2w=r_c2w, t_w=t_w)


def _mujoco_to_opencv_camera(r_c2w_mujoco: np.ndarray, t_w: np.ndarray):
    flip = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        dtype=np.float32,
    )
    r_c2w_cv = r_c2w_mujoco @ flip
    r_w2c = r_c2w_cv.T
    t_c = -r_w2c @ t_w
    return r_w2c, t_c


def _save_camera_npz(
    sim,
    camera_name: str,
    width: int,
    height: int,
    out_path: Path,
    convert_to_opencv: bool,
):
    intrinsics = _get_camera_intrinsics(sim, camera_name, width, height)
    k = np.array(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    r_c2w, t_w = _get_camera_extrinsics(sim, camera_name)
    if convert_to_opencv:
        r_w2c, t_c = _mujoco_to_opencv_camera(r_c2w, t_w)
    else:
        r_w2c = r_c2w.T
        t_c = -r_w2c @ t_w
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = r_w2c
    extrinsic[:3, 3] = t_c
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        intrinsic=k,
        extrinsic=extrinsic,
        width=np.int32(width),
        height=np.int32(height),
    )


def _replay_demo(
    env,
    demo_data,
    depth_key: str,
    rgb_key: str,
    out_dir: Path,
    stop_on_done: bool,
    save_npy: bool,
    save_rgb: bool,
    save_depth_viz: bool,
    save_camera_params: bool,
    camera_params_format: str,
):
    actions = demo_data["actions"][()]
    states = demo_data["states"][()]

    env.reset()
    env.set_init_state(states[0])
    obs = None
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("llava"))

    if save_camera_params:
        depth = obs[depth_key]
        height, width = depth.shape
        camera_name = depth_key.replace("_depth", "")
        if camera_params_format == "mujoco":
            intrinsics = _get_camera_intrinsics(env.sim, camera_name, width, height)
            intrinsics["width"] = int(width)
            intrinsics["height"] = int(height)
            intrinsics["camera_name"] = camera_name
            intrinsics["near"] = float(env.sim.model.vis.map.znear)
            intrinsics["far"] = float(env.sim.model.vis.map.zfar)
            _save_intrinsics_json(intrinsics, out_dir / "camera_params" / "intrinsics.json")
        elif camera_params_format == "spatracker_v2":
            _save_camera_npz(
                env.sim,
                camera_name,
                width,
                height,
                out_dir / "camera_params" / "spatracker_v2.npz",
                convert_to_opencv=False,
            )
        elif camera_params_format == "opencv":
            _save_camera_npz(
                env.sim,
                camera_name,
                width,
                height,
                out_dir / "camera_params" / "opencv.npz",
                convert_to_opencv=True,
            )

    for step_idx, action in enumerate(actions):
        if obs is None or depth_key not in obs:
            raise KeyError(f"Depth key '{depth_key}' not found in observation.")
        if save_rgb and rgb_key not in obs:
            raise KeyError(f"RGB key '{rgb_key}' not found in observation.")

        depth = obs[depth_key]
        os.makedirs(out_dir / "frame_depth", exist_ok=True)
        frame_out_path = out_dir / "frame_depth" / f"{step_idx:04d}.png"
        _save_depth_png(depth, frame_out_path)
        if save_npy:
            os.makedirs(out_dir / "depth_npy", exist_ok=True)
            npy_path = out_dir / "depth_npy" / f"{step_idx:04d}.npy"
            _save_depth_npy(depth, npy_path)
        if save_depth_viz:
            os.makedirs(out_dir / "frame_depth_viz", exist_ok=True)
            viz_path = out_dir / "frame_depth_viz" / f"{step_idx:04d}.png"
            _save_depth_viz_png(depth, viz_path)
        if save_rgb:
            os.makedirs(out_dir / "frame_rgb", exist_ok=True)
            rgb_path = out_dir / "frame_rgb" / f"{step_idx:04d}.png"
            _save_rgb_png(obs[rgb_key], rgb_path)
        if save_camera_params and camera_params_format == "mujoco":
            r_c2w, t_w = _get_camera_extrinsics(env.sim, depth_key.replace("_depth", ""))
            extr_path = out_dir / "camera_params" / "extrinsics" / f"{step_idx:04d}.npz"
            _save_extrinsics_npz(r_c2w, t_w, extr_path)

        obs, _, done, _ = env.step(action.tolist())
        if stop_on_done and done:
            break


def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    if args.task_index is None:
        task_indices = list(range(task_suite.n_tasks))
    else:
        task_indices = [args.task_index]

    depth_key = DEPTH_KEY_BY_CAMERA[args.camera_name]
    rgb_key = RGB_KEY_BY_CAMERA[args.camera_name]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for task_id in tqdm.tqdm(task_indices):
        task = task_suite.get_task(task_id)
        env = _build_env(task, args.resolution, [args.camera_name])

        data_path = os.path.join(args.libero_data_dir, f"{task.name}_demo.hdf5")
        if not os.path.exists(data_path):
            print(f"Skipping task {task.name}: missing {data_path}")
            env.close()
            continue

        with h5py.File(data_path, "r") as data_file:
            demos = data_file["data"]
            demo_keys = sorted(demos.keys(), key=lambda k: int(k.split("_")[1]))
            if args.max_episodes is not None:
                demo_keys = demo_keys[: args.max_episodes]

            for demo_key in demo_keys:
                demo_idx = int(demo_key.split("_")[1])
                demo_out_dir = output_root / task.name / f"demo_{demo_idx}" / args.camera_name
                if args.skip_existing and demo_out_dir.exists():
                    continue
                _replay_demo(
                    env=env,
                    demo_data=demos[demo_key],
                    depth_key=depth_key,
                    rgb_key=rgb_key,
                    out_dir=demo_out_dir,
                    stop_on_done=args.stop_on_done,
                    save_npy=args.save_npy,
                    save_rgb=args.save_rgb,
                    save_depth_viz=args.save_depth_viz,
                    save_camera_params=args.save_camera_params,
                    camera_params_format=args.camera_params_format,
                )

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        required=True,
        help="LIBERO task suite (e.g., libero_goal).",
    )
    parser.add_argument(
        "--libero_data_dir",
        type=str,
        required=True,
        help="Directory with libero_no_oops HDF5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for depth images.",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        choices=["agentview", "robot0_eye_in_hand"],
        default="agentview",
        help="Which camera to export depth from.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Camera resolution (square).",
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=None,
        help="Optional task index to export.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional limit on episodes per task.",
    )
    parser.add_argument(
        "--stop_on_done",
        action="store_true",
        help="Stop exporting frames once done=True is encountered.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip demos if output directory already exists.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Also save float32 depth maps as .npy in the same folder.",
    )
    parser.add_argument(
        "--save_rgb",
        action="store_true",
        help="Also save RGB frames as PNG with _rgb suffix.",
    )
    parser.add_argument(
        "--save_depth_viz",
        action="store_true",
        help="Also save 8-bit normalized depth PNGs for visualization.",
    )
    parser.add_argument(
        "--save_camera_params",
        action="store_true",
        help="Save camera parameters in the requested format.",
    )
    parser.add_argument(
        "--camera_params_format",
        choices=["mujoco", "spatracker_v2", "opencv"],
        default="mujoco",
        help="Camera params format to save when --save_camera_params is set.",
    )
    main(parser.parse_args())
