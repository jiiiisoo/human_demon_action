#!/usr/bin/env python3
"""
Replay RLDS episodes by applying stored actions in a LIBERO environment and
export per-step meshes (cropped around the table) AND 4D point tracks.

Point tracks are initialized from the *cropped, table-removed region* at the
first step of the episode:

    - At the first step, we compute the crop center (table center) and cube crop.
    - We convert all MuJoCo GEOM_MESH vertices to world coordinates.
    - We discard vertices belonging to table (if exclude-table) and those
      outside the cube crop.
    - From the remaining vertices, we randomly sample up to `max_track_points`
      and remember (geom_id, local_idx).
    - For every step, we recompute those vertices' world positions and store
      a [T, N, 3] track array.

Output layout (per dataset):

  <output_root>/<dataset>/episode_00000/cropped_scene/step_0000.obj
  <output_root>/<dataset>/episode_00000/actions.npy
  <output_root>/<dataset>/episode_00000/metadata.json
  <output_root>/<dataset>/episode_00000/vertex_tracks.npy
  <output_root>/<dataset>/episode_00000/vertex_ids.json
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
from export_gt_pointcloud import (  # type: ignore
    collect_world_meshes,
    get_reference_center,
    center_and_crop_meshes,
    save_meshes_as_obj,
)


# ===============================
#  Utility: action + dlimp helpers
# ===============================

def ensure_action_shape(actions: np.ndarray, expected_dim: int) -> np.ndarray:
    """
    Ensure actions is [T, D]. If flat, reshape; check last dim.
    """
    if actions.size == 0:
        raise ValueError("No actions found in trajectory")
    if actions.ndim == 1:
        if actions.size % expected_dim != 0:
            raise ValueError(
                f"Flat action array length {actions.size} not divisible by {expected_dim}"
            )
        actions = actions.reshape(-1, expected_dim)
    if actions.shape[1] != expected_dim:
        raise ValueError(
            f"Action dim mismatch: expected {expected_dim}, got {actions.shape[1]}"
        )
    return actions


def _patch_dlimp_options():
    """
    Avoid setting TF options that may not exist on older TF builds.
    Overwrite DLataset._apply_options with a safe variant.
    """
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


def _dataset_builder_from_shards(shards: List[Path]):
    """
    Infer TFDS builder from a list of TFRecord shards.
    Assumes standard TFDS layout: <data_root>/<dataset>/<version>/<split>.tfrecord-*
    """
    if not shards:
        raise ValueError("No shards provided to build dataset")
    version_dir = shards[0].parent
    dataset_dir = version_dir.parent
    data_root = dataset_dir.parent
    dataset_name = dataset_dir.name
    if not dataset_name:
        raise ValueError(f"Could not infer dataset name from shard path: {shards[0]}")
    return tfds.builder(dataset_name, data_dir=str(data_root))


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
    _patch_dlimp_options()
    builder = _dataset_builder_from_shards(shards)
    return dl.DLataset.from_rlds(
        builder,
        split=split,
        shuffle=False,
        num_parallel_reads=num_parallel_reads,
    )


# ===============================
#  Vertex-track helpers (MuJoCo)
# ===============================

def _compute_geom_world_verts(sim, geom_local_verts: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    For current sim state, compute world-frame vertices for each geom_id.
    """
    model = sim.model
    data = sim.data
    geom_world: Dict[int, np.ndarray] = {}
    for geom_id, local_verts in geom_local_verts.items():
        R = data.geom_xmat[geom_id].reshape(3, 3)
        t = data.geom_xpos[geom_id]
        world = local_verts @ R.T + t  # (V_i, 3)
        geom_world[geom_id] = world
    return geom_world


def _build_tracking_points_from_cropped_region(
    sim,
    cube_center: np.ndarray,
    cube_half: float,
    include_table: bool,
    max_points: int,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]]]:
    """
    Initialize tracking points from the cropped, table-removed region at t=0.

    Steps:
      1) For each GEOM_MESH:
         - Get local vertices (mesh_vert).
         - Transform to world using geom_xmat, geom_xpos.
      2) Filter vertices:
         - Drop table geoms if include_table is False.
         - Keep only vertices inside cube [center - cube_half, center + cube_half].
      3) Randomly sample up to max_points candidates.
      4) Return:
         - geom_local_verts: dict[geom_id -> (V_i, 3) local vertex positions]
         - point_meta: list of {geom_id, local_idx, body_name} length N (<= max_points).
    """
    model = sim.model
    data = sim.data

    geom_local_verts: Dict[int, np.ndarray] = {}
    candidate_meta: List[Dict[str, Any]] = []
    candidate_world: List[np.ndarray] = []

    cube_center = np.asarray(cube_center, dtype=np.float32)

    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = model.body_id2name(body_id) or f"body_{body_id}"
        if not include_table and "table" in body_name.lower():
            continue

        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0:
            # Skip primitive geoms (box/sphere/etc.) for now
            continue

        v_adr = model.mesh_vertadr[mesh_id]
        v_num = model.mesh_vertnum[mesh_id]
        if v_num == 0:
            continue

        local_verts = model.mesh_vert[v_adr : v_adr + v_num]  # (V_i, 3)
        geom_local_verts[geom_id] = local_verts

        # world transform
        R = data.geom_xmat[geom_id].reshape(3, 3)
        t = data.geom_xpos[geom_id]
        world_verts = local_verts @ R.T + t  # (V_i, 3)

        # crop filter
        rel = world_verts - cube_center[None, :]
        inside = (
            (np.abs(rel[:, 0]) <= cube_half)
            & (np.abs(rel[:, 1]) <= cube_half)
            & (np.abs(rel[:, 2]) <= cube_half)
        )
        idxs = np.where(inside)[0]

        for li in idxs:
            candidate_world.append(world_verts[li])
            candidate_meta.append(
                {
                    "geom_id": int(geom_id),
                    "local_idx": int(li),
                    "body_name": body_name,
                }
            )

    if not candidate_meta:
        raise RuntimeError(
            "No vertices inside cropped region to initialize tracks "
            "(maybe cube_half too small or include_table=False removed everything?)."
        )

    # Random subset ≈ Poisson-ish. If you really want Poisson disk, you can
    # replace this with FPS or actual Poisson-disk sampling on candidate_world.
    if len(candidate_meta) > max_points:
        idxs = np.random.choice(len(candidate_meta), size=max_points, replace=False)
        point_meta = [candidate_meta[i] for i in idxs]
    else:
        point_meta = candidate_meta

    return geom_local_verts, point_meta


def replay_episode_with_mesh_and_tracks(
    env: OffScreenRenderEnv,
    actions: np.ndarray,
    out_cropped: Path,
    cube_half: float,
    camera_names: List[str],
    traj_index: int,
    frame_indices: List[int],
    traj_len: int,
    max_track_points: int = 5000,
    include_table: bool = True,
) -> None:
    """
    One episode:
      - For each frame_index, apply action, render mesh, crop, and save OBJ.
      - At the first step, initialize mesh-vertex-based tracking points from the
        cropped, table-removed region.
      - For every step, track those same vertices via (geom_id, local_idx) and
        save a [T, N, 3] world-coordinate track array.

    Saves into:
      out_cropped/step_XXXX.obj
      out_cropped.parent/actions.npy
      out_cropped.parent/metadata.json
      out_cropped.parent/vertex_tracks.npy
      out_cropped.parent/vertex_ids.json
    """
    out_cropped.mkdir(parents=True, exist_ok=True)

    if len(actions) != len(frame_indices):
        raise ValueError(
            f"Action length {len(actions)} does not match frame_indices length {len(frame_indices)} "
            f"for traj {traj_index}"
        )

    sim = env.sim

    geom_local_verts: Optional[Dict[int, np.ndarray]] = None
    point_meta: Optional[List[Dict[str, Any]]] = None
    tracks_per_step: List[np.ndarray] = []

    for t, (frame_idx, act) in enumerate(zip(frame_indices, actions)):
        # Step environment
        obs, _, done, _ = env.step(act.tolist())

        # Collect meshes and crop (same as before)
        meshes = collect_world_meshes(
            env,
            include_robot=True,
            include_statics=True,
            exclude_body_substrings=(),
        )
        ref_center = get_reference_center(meshes, keyword="table")
        filtered = [m for m in meshes if include_table or "table" not in m["name"].lower()]
        cropped = center_and_crop_meshes(filtered, ref_center, cube_half)

        mesh_path = out_cropped / f"step_{frame_idx:04d}.obj"
        if not mesh_path.exists():
            save_meshes_as_obj(cropped, mesh_path)

        # Initialize tracking points at the first step
        if point_meta is None:
            geom_local_verts, point_meta = _build_tracking_points_from_cropped_region(
                sim=sim,
                cube_center=ref_center,
                cube_half=cube_half,
                include_table=include_table,
                max_points=max_track_points,
            )
            num_points = len(point_meta)
        else:
            # should already be initialized
            num_points = len(point_meta)

        # Compute world positions of tracked vertices at this step
        geom_world = _compute_geom_world_verts(sim, geom_local_verts)
        step_pts = np.zeros((num_points, 3), dtype=np.float32)
        for i, meta in enumerate(point_meta):
            g = meta["geom_id"]
            li = meta["local_idx"]
            step_pts[i] = geom_world[g][li]
        tracks_per_step.append(step_pts)

        # 보통 RLDS 데모는 마지막까지 step이 정의되어 있어서
        # done 을 무시하고 끝까지 돌리는 게 안정적이라 break 안 걸어둠.
        # if done:
        #     break

    vertex_tracks = np.stack(tracks_per_step, axis=0)  # [T, N, 3]

    # Save actions
    actions_path = out_cropped.parent / "actions.npy"
    if not actions_path.exists():
        np.save(actions_path, actions[: len(frame_indices)])

    # Save metadata
    meta_path = out_cropped.parent / "metadata.json"
    metadata = {
        "_traj_index": int(traj_index),
        "_frame_index": [int(i) for i in frame_indices],
        "_len": int(traj_len),
        "actions_file": actions_path.name,
        "num_track_points": int(vertex_tracks.shape[1]),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save vertex tracks + point meta
    tracks_path = out_cropped.parent / "vertex_tracks.npy"
    ids_path = out_cropped.parent / "vertex_ids.json"

    np.save(tracks_path, vertex_tracks)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(point_meta, f, indent=2)


# ===============================
#  Main dataset processing
# ===============================

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

    dataset_output = output_root / dataset.display_path
    episode_idx = 0

    rlds_dataset = load_rlds_dataset(dataset.shards, split=split)

    for traj_i, traj in enumerate(rlds_dataset):
        if not traj_i == 147 :
            continue
        # Shard across workers
        if traj_i % num_shards != shard_index:
            continue

        if max_episodes is not None and episode_idx >= max_episodes:
            break

        # --- actions ---
        actions = np.asarray(traj["action"])
        actions = ensure_action_shape(actions, expected_dim=actions.shape[-1])

        # --- frame indices ---
        frame_indices = np.asarray(traj["_frame_index"]).astype(int).tolist()
        traj_len_arr = np.asarray(traj["_len"])
        traj_len = int(traj_len_arr[0]) if traj_len_arr.ndim > 0 else int(traj_len_arr)

        if traj_len != len(frame_indices):
            raise ValueError(
                f"traj_len {traj_len} does not match frame_indices length {len(frame_indices)} "
                f"for traj index {traj_i}"
            )

        traj_index_arr = np.asarray(traj["_traj_index"])
        traj_index_raw = int(traj_index_arr[0]) if traj_index_arr.ndim > 0 else int(traj_index_arr)

        # --- truncate by max_steps (if any) ---
        if max_steps is not None:
            actions = actions[:max_steps]
            frame_indices = frame_indices[: len(actions)]
            traj_len = min(traj_len, len(frame_indices))

        if len(actions) != len(frame_indices):
            raise ValueError(
                f"Action length {len(actions)} does not match frame_indices length {len(frame_indices)} "
                f"after truncation for traj_index_raw {traj_index_raw}"
            )

        # Reset env to initial state
        env.reset()
        env.set_init_state(init_states[0])

        traj_index = episode_offset + traj_index_raw

        episode_dir_name = f"episode_{traj_index:05d}"
        ep_dir = dataset_output / episode_dir_name
        cropped_dir = ep_dir / "cropped_scene"

        replay_episode_with_mesh_and_tracks(
            env=env,
            actions=actions,
            out_cropped=cropped_dir,
            cube_half=cube_half,
            camera_names=camera_names,
            traj_index=traj_index,
            frame_indices=frame_indices,
            traj_len=traj_len,
            max_track_points=max_track_points,
            include_table=include_table,
        )

        episode_idx += 1

    env.close()
    return episode_idx


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
    for dataset in tqdm(datasets):
        exported = process_dataset(
            dataset=dataset,
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
        print(
            f"{dataset.display_path}: exported {exported} episodes to "
            f"{output_root / dataset.display_path}"
        )
        total += exported
    print(f"Done. Total episodes exported: {total}")


if __name__ == "__main__":
    main()
