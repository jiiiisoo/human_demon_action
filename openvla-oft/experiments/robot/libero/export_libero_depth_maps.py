"""
Export per-frame depth maps by replaying actions from LIBERO HDF5 demos.

This mirrors the environment setup and action replay flow from
regenerate_libero_dataset_with_tracks.py, and pulls depth observations
in the same way as LIBERO/scripts/create_dataset.py.
"""

import argparse
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


def _save_depth_png(depth_m: np.ndarray, out_path: Path):
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), depth_mm)


def _save_depth_npy(depth_m: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, depth_m.astype(np.float32))


def _replay_demo(
    env,
    demo_data,
    depth_key: str,
    out_dir: Path,
    stop_on_done: bool,
    save_npy: bool,
):
    actions = demo_data["actions"][()]
    states = demo_data["states"][()]

    env.reset()
    env.set_init_state(states[0])
    obs = None
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("llava"))

    for step_idx, action in enumerate(actions):
        if obs is None or depth_key not in obs:
            raise KeyError(f"Depth key '{depth_key}' not found in observation.")

        depth = obs[depth_key]
        out_path = out_dir / f"frame_{step_idx:04d}.png"
        _save_depth_png(depth, out_path)
        if save_npy:
            npy_path = out_dir / f"frame_{step_idx:04d}.npy"
            _save_depth_npy(depth, npy_path)

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
                    out_dir=demo_out_dir,
                    stop_on_done=args.stop_on_done,
                    save_npy=args.save_npy,
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
    main(parser.parse_args())
