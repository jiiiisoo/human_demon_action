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
from export_gt_pointcloud import (  # type: ignore
    collect_world_meshes,
    get_reference_center,
    center_and_crop_meshes,
    save_meshes_as_obj,
)
import glob
import hashlib

# def _dataset_builder_from_shards(shards: List[Path]):
#     """Infer TFDS builder from shard paths."""
#     if not shards:
#         raise ValueError("No shards provided to build dataset")
#     version_dir = shards[0].parent
#     dataset_dir = version_dir.parent
#     data_root = dataset_dir.parent
#     dataset_name = dataset_dir.name
#     if not dataset_name:
#         raise ValueError(f"Could not infer dataset name from shard path: {shards[0]}")
#     return tfds.builder(dataset_name, data_dir=str(data_root))

def _patch_dlimp_options():
    """Avoid TF options not available on older TF builds."""
    try:
        import dlimp.dataset as ds
    except Exception:
        return

    def _safe_apply_options(self):
        options = tf.data.Options()
        options.autotune.enabled = True
        options.deterministic = True
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.map_and_filter_fusion = True
        options.experimental_optimization.inject_prefetch = False
        return self.with_options(options)

    ds.DLataset._apply_options = _safe_apply_options

def load_rlds_dataset(
    shards: List[Path],
    split: str = "train",
    num_parallel_reads: int = tf.data.AUTOTUNE,
):
    """
    Wrap TFDS RLDS with dlimp.DLataset. Trajectories will contain:
      - "action"
      - "observation"
      - "_traj_index" (scalar)
      - "_frame_index" (length-T int vector)
      - "_len" (scalar, trajectory length)
    """
    # _patch_dlimp_options()
    # builder = _dataset_builder_from_shards(shards)
    builder = tfds.builder("libero_goal_no_noops", data_dir="/mnt/data/modified_libero_rlds")
    return dl.DLataset.from_rlds(
        builder,
        split=split,
        shuffle=False,
        num_parallel_reads=num_parallel_reads,
    )


def process_dataset(
    dataset: DatasetInput,
    output_root: Path,
    task_suite_name: str,
    task_id: int,
    max_episodes: Optional[int],
    max_steps: Optional[int],
    cube_half: float,
    include_table: bool,
    split: str,
    max_track_points: int,
    episode_offset: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> int:
    """
    Given one DatasetInput (e.g., libero_goal_no_noops/1.0.0), iterate over
    RLDS trajectories and for each one:

      - Extract 'action', '_traj_index', '_frame_index', '_len'.
      - Replay in LIBERO env.
      - Save cropped meshes and vertex tracks.
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = Path(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    camera_names = ["agentview", "robot0_eye_in_hand"]
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": 128,
        "camera_widths": 128,
        "camera_names": camera_names,
        "camera_depths": False,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    # dataset_output = output_root / dataset.display_path
    episode_idx = 0

    rlds_dataset = load_rlds_dataset(dataset, split=split)

    return rlds_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "/mnt/data/modified_libero_rlds/libero_goal_no_noops/1.0.0/libero_goal-train.tfrecord-*"
        ],
        help="Glob patterns or dataset directories containing TFRecord shards.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/data/libero/modified_libero_mesh_with_tracks",
        help="Base directory to write meshes and tracks into.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load via dlimp/tfds (e.g., train, val, all).",
    )
    parser.add_argument(
        "--task-suite",
        default="libero_goal",
        help="LIBERO task suite name (e.g., libero_10, libero_spatial).",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task id within the suite to replay.",
    )
    parser.add_argument(
        "--cube-half",
        type=float,
        default=0.5,
        help="Half-edge length (meters) of the cube crop around the table center.",
    )
    parser.add_argument(
        "--max-track-points",
        type=int,
        default=5000,
        help="Maximum number of cropped vertices to track per episode.",
    )
    parser.add_argument(
        "--exclude-table",
        action="store_true",
        help="If set, remove meshes whose name contains 'table' from the saved outputs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Worker index [0, num_shards-1] for sharding episodes.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of workers to shard episodes across.",
    )
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Offset to add to episode numbering (useful when sharding across multiple workers).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on number of episodes to export.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    datasets = collect_dataset_inputs(args.inputs)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total = 0
    hash_list = []
    # for dataset in tqdm(datasets):
    exported = process_dataset(
        dataset=datasets,
        output_root=output_root,
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        cube_half=args.cube_half,
        include_table=not args.exclude_table,
        split=args.split,
        max_track_points=args.max_track_points,
        episode_offset=args.episode_offset,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    lang_dict = {}
    # b = tfds.builder("libero_goal_no_noops", data_dir="/mnt/data/modified_libero_rlds")
    # d = dl.DLataset.from_rlds(b, split="train", shuffle=False, num_parallel_reads=tf.data.AUTOTUNE)
    # for t in d :
    #     act = t['action'].numpy()
    #     print(t['language_instruction'][0])
    #     print(t['_len'][0])
    #     print(act)
    #     1/0
    for traj in exported:
        # print(tf.cast(traj['_traj_index'], tf.int64))
        # print(traj['_frame_index'])
        # print(traj['_len'])
        # print(traj['traj_metadata'].keys())
        # print(traj['observation'].keys())
        # print(traj.keys())
        # 1/0
        traj_len = tf.cast(traj['_len'][0], tf.int64)
        traj_index = tf.cast(traj['_traj_index'][0], tf.int64)
        # print(traj.keys())
        # print(traj['is_terminal'])
        # print(traj['is_last'])
        # print(traj['is_first'])
        # print(traj['traj_metadata']['episode_metadata']['file_path'][0])
        # print(traj['observation'].keys())
        # 1/0
        # action = np.round(traj['action'].numpy(), 3)
        language = traj['language_instruction'][0].numpy().decode("utf-8")
        # act = tf.cast(traj['action'], tf.float32)
        # print(act)
        # if language not in lang_dict.keys():
        #     lang_dict[language] = []
        # if traj_len in lang_dict[language]:
        #     print(f"Duplicate traj_len found: {traj_len} for language: {language}")
        #     1/0
        # else :
        #     lang_dict[language].append(traj_len)
        # action = np.round(traj['action'].numpy(), 3)
        action = np.asarray(traj['action'])
            # 1/0
        # language = traj['language_instruction'][0]
        frame_idx = np.asarray(traj['_frame_index'])
        h = hashlib.sha1()
        h.update(action.tobytes())
        h.update(frame_idx.tobytes().tobytes())
        hash_value = h.hexdigest()[:16]
        print(hash_value)
        1/0
        # print(action)
        # print(np.array2string(action, formatter={"float_kind": lambda x: f"{x:.3f}"}))
        # print(np.array2string(frame_idx, formatter={"int_kind": lambda x: f"{x:05d}"}))
        # print(hash_value)
        # print(action.dtype)
        # print(frame_idx.dtype)
        if h.hexdigest()[:16] in hash_list:
            print(f"Duplicate hash found: {h.hexdigest()[:16]}")
        if h.hexdigest()[:16].startswith('d6c243'):
            print(language)
            print(f"Hash starts with d6c243: {h.hexdigest()[:16]}")
            1/0
        hash_list.append(h.hexdigest()[:16])
        print(h.hexdigest()[:16])
        # print(f'traj_index : {traj_index} / action : {action} / traj_len : {traj_len}')
        # point_len = len(glob.glob(f'/mnt/data/libero/modified_libero_meshes_face_tracks_final/1.0.0/episode_{traj_index:05d}/cropped_scene/*'))
        # if action != point_len:
        #     print(f"action {action} != point_len {point_len} for traj_index {traj_index}")
        # if traj_len != point_len:
        #     print(f"traj_len {traj_len} != point_len {point_len} for traj_index {traj_index}")

        # total += exported
    print(f"Done. Total episodes exported: {total}")

if __name__ == "__main__":
    main()
