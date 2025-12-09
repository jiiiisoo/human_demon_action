#!/usr/bin/env python3
"""
Replay RLDS episodes and export cropped meshes plus face-area-sampled point tracks.

Difference vs export_gt_track.py:
  - Tracking points are sampled uniformly over triangle surface area (within the
    cropped region) instead of uniformly over vertices.
  - Each tracked point stores (geom_id, vert_indices[3], barycentric[3]) so its
    world position is recomputed every step.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl
from tqdm import tqdm

from extract_frames import DatasetInput, collect_dataset_inputs
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from export_gt_pointcloud import (  # type: ignore
    collect_world_meshes,
    get_reference_center,
    center_and_crop_meshes,
    save_meshes_as_obj,
    load_meshes_from_obj,
    box_geom_in_world,
)
import os
import glob
import time
import hashlib

# ===============================
#  Utility: action + dlimp helpers
# ===============================

def ensure_action_shape(actions: np.ndarray, expected_dim: int) -> np.ndarray:
    """Ensure actions is [T, D]."""
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


def _dataset_builder_from_shards(shards: List[Path]):
    """Infer TFDS builder from shard paths."""
    if not shards:
        raise ValueError("No shards provided to build dataset")
    version_dir = shards[0].parent
    dataset_dir = version_dir.parent
    data_root = dataset_dir.parent
    dataset_name = dataset_dir.name
    if not dataset_name:
        raise ValueError(f"Could not infer dataset name from shard path: {shards[0]}")
    return tfds.builder(dataset_name, data_dir=str(data_root))

# ===============================
#  Vertex-track helpers (MuJoCo)
# ===============================

def _compute_geom_world_verts(sim, geom_local_verts: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """For current sim state, compute world-frame vertices for each geom_id in dict."""
    data = sim.data
    geom_world: Dict[int, np.ndarray] = {}
    for geom_id, local_verts in geom_local_verts.items():
        R = data.geom_xmat[geom_id].reshape(3, 3)
        t = data.geom_xpos[geom_id]
        geom_world[geom_id] = local_verts @ R.T + t
    return geom_world


def _triangle_areas(tri: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _build_tracking_points_from_faces(
    sim,
    cube_center: np.ndarray,
    cube_half: float,
    include_table: bool,
    max_points: int,
    include_wall: bool = False,
    table_weight: float = 1.0,
    vertex_ratio_default: float = 0.6,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
    """
    Initialize tracking points by sampling triangles proportional to area inside the crop cube.
    Returns:
      geom_local_verts: dict[geom_id] -> (V_i, 3) local verts
      point_meta: list of dicts with geom_id, vert_indices[3], barycentric[3], body_name
    """
    model = sim.model
    data = sim.data

    geom_local_verts: Dict[int, np.ndarray] = {}
    face_meta: List[Dict[str, Any]] = []
    face_areas: List[float] = []
    table_face_indices: List[int] = []
    vertex_candidates: List[Dict[str, Any]] = []
    vertex_weights: List[float] = []

    cube_center = np.asarray(cube_center, dtype=np.float32)

    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = model.body_id2name(body_id) or f"body_{body_id}"

        if not include_table and "table" in body_name.lower():
            continue
        if (not include_wall) and ("world" in body_name.lower() or "mount0" in body_name.lower()):
            continue
        # if include_table and "table" in body_name.lower():
        #     print("geom_id", geom_id, "body_name", body_name, "geom_type", model.geom_type[geom_id], "geom_size", model.geom_size[geom_id])


        mesh_id = model.geom_dataid[geom_id]
        if mesh_id >= 0:
            v_adr = model.mesh_vertadr[mesh_id]
            v_num = model.mesh_vertnum[mesh_id]
            f_adr = model.mesh_faceadr[mesh_id]
            f_num = model.mesh_facenum[mesh_id]
            if v_num == 0 or f_num == 0:
                # print(f"geom_id {body_name} {geom_id} has no mesh")
                continue
            local_verts = model.mesh_vert[v_adr : v_adr + v_num]
            faces = model.mesh_face[f_adr : f_adr + f_num]
        elif include_table and "table" in body_name.lower():
            hx, hy, hz = model.geom_size[geom_id]
            # plane 같은 경우 size[2]가 0일 수 있음 → 두께 없는 테이블
            if hz < 1e-6:
                # 윗면만 4개 vertex + 2개 triangle로 정의 (z=0 기준 local frame)
                local_verts = np.array(
                    [
                        [-hx, -hy, 0.0],
                        [-hx,  hy, 0.0],
                        [ hx, -hy, 0.0],
                        [ hx,  hy, 0.0],
                    ],
                    dtype=np.float32,
                )
                faces = np.array(
                    [
                        [0, 1, 3],
                        [0, 3, 2],
                    ],
                    dtype=np.int32,
                )
            else:
                # 진짜 box인 경우: 기존 8-vertex 박스 그대로 사용
                local_verts = np.array(
                    [
                        [-hx, -hy, -hz],
                        [-hx, -hy,  hz],
                        [-hx,  hy, -hz],
                        [-hx,  hy,  hz],
                        [ hx, -hy, -hz],
                        [ hx, -hy,  hz],
                        [ hx,  hy, -hz],
                        [ hx,  hy,  hz],
                    ],
                    dtype=np.float32,
                )
                faces = np.array(
                    [
                        [0, 1, 3],
                        [0, 3, 2],
                        [4, 6, 7],
                        [4, 7, 5],
                        [0, 4, 5],
                        [0, 5, 1],
                        [2, 3, 7],
                        [2, 7, 6],
                        [0, 2, 6],
                        [0, 6, 4],
                        [1, 5, 7],
                        [1, 7, 3],
                    ],
                    dtype=np.int32,
                )
        else :
            # print(f"geom_id {body_name} {geom_id} is not a mesh")
            continue

        geom_local_verts[geom_id] = local_verts

        R = data.geom_xmat[geom_id].reshape(3, 3)
        t = data.geom_xpos[geom_id]
        world_verts = local_verts @ R.T + t  # (V_i, 3)
        tri_world = world_verts[faces]  # (F, 3, 3)
        centroids = tri_world.mean(axis=1)
        inside = np.all(np.abs(centroids - cube_center[None]) <= cube_half, axis=1)
        # if include_table and "table" in body_name.lower():
        #     # Always keep table faces for sampling when tables are included
        #     inside = np.ones_like(inside, dtype=bool)
        if not inside.any():
            continue
        tri_inside = tri_world[inside]
        faces_inside = faces[inside]
        areas = _triangle_areas(tri_inside)
        valid = areas > 1e-9
        if not valid.any():
            continue
        tri_inside = tri_inside[valid]
        faces_inside = faces_inside[valid]
        areas = areas[valid]
        # if include_table and "table" in body_name.lower():
        #     print(f'number of faces : {faces[inside]}')
        #     print(f'area : {areas}')

        for face_indices, area in zip(faces_inside, areas):
            face_meta.append(
                {
                    "geom_id": geom_id,
                    "vert_indices": face_indices.tolist(),
                    "body_name": body_name,
                }
            )
            weight = table_weight if "table" in body_name.lower() else 1.0
            face_areas.append(float(area * weight))
            if "table" in body_name.lower():
                table_face_indices.append(len(face_meta) - 1)

        # vertex candidates (uniform over vertices inside cube) with per-body weighting
        vert_inside = np.all(np.abs(world_verts - cube_center[None]) <= cube_half, axis=1)
        for li in np.where(vert_inside)[0]:
            vertex_candidates.append(
                {
                    "geom_id": geom_id,
                    "vert_indices": [int(li), int(li), int(li)],
                    "body_name": body_name,
                }
            )
            bname = body_name.lower()
            if "gripper0" in bname:
                vertex_weights.append(3.0)
            elif "robot0" in bname:
                vertex_weights.append(2.0)
            else:
                vertex_weights.append(1.0)

    if not face_meta and not vertex_candidates:
        return geom_local_verts, []

    face_areas_np = np.asarray(face_areas, dtype=np.float64) if face_areas else np.zeros((0,), dtype=np.float64)
    probs = face_areas_np / face_areas_np.sum() if face_areas_np.size > 0 else None
    vertex_count = int(max_points * vertex_ratio_default)
    vertex_count = max(0, min(vertex_count, max_points))
    face_count = max_points - vertex_count

    point_meta: List[Dict[str, Any]] = []

    if vertex_candidates and vertex_count > 0:
        v_count = min(vertex_count, len(vertex_candidates))
        v_probs = None
        if vertex_weights:
            w = np.asarray(vertex_weights, dtype=np.float64)
            v_probs = w / w.sum()
        v_idx = np.random.choice(len(vertex_candidates), size=v_count, replace=False, p=v_probs)
        for idx in v_idx:
            meta = dict(vertex_candidates[idx])
            meta["barycentric"] = [1.0, 0.0, 0.0]
            point_meta.append(meta)

    if face_count > 0 and face_meta:
        chosen = np.random.choice(len(face_meta), size=face_count, replace=True, p=probs)
        if include_table and table_face_indices and not any(idx in table_face_indices for idx in chosen):
            chosen[0] = np.random.choice(table_face_indices)
        for idx in chosen:
            meta = face_meta[idx]
            r1 = np.sqrt(np.random.rand())
            r2 = np.random.rand()
            w0 = 1 - r1
            w1 = r1 * (1 - r2)
            w2 = r1 * r2
            meta = dict(meta)
            meta["barycentric"] = [float(w0), float(w1), float(w2)]
            point_meta.append(meta)

    return geom_local_verts, point_meta, face_meta, face_areas


# ===============================
#  Replay episode with face-area tracks
# ===============================

def replay_episode_with_face_tracks(
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
    include_wall: bool = True,
    table_weight: float = 1.0,
) -> None:
    out_cropped.mkdir(parents=True, exist_ok=True)

    if len(actions) != len(frame_indices):
        raise ValueError(
            f"Action length {len(actions)} does not match frame_indices length {len(frame_indices)} "
            f"for traj {traj_index}"
        )

    sim = env.sim
    geom_local_verts: Optional[Dict[int, np.ndarray]] = None
    point_meta: Optional[List[Dict[str, Any]]] = None
    cached_face_meta: Optional[List[Dict[str, Any]]] = None
    cached_face_areas: Optional[List[float]] = None
    cached_pose_seq: List[Dict[str, Any]] = []
    tracks_per_step: List[np.ndarray] = []

    ref_center: Optional[np.ndarray] = None
    step_done = 0
    for frame_idx, act in zip(frame_indices, actions):
        mesh_path = out_cropped / f"step_{frame_idx:04d}.obj"
        # if mesh_path.exists():
        #     print(f"Loading mesh from {mesh_path}")
        #     cropped = load_meshes_from_obj(mesh_path)
        #     if ref_center is None:
        #         ref_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # else:
        obs, _, done, _ = env.step(act.tolist())

        meshes = collect_world_meshes(
            env,
            include_robot=True,
            include_statics=True,
            exclude_body_substrings=(),
        )
        ref_center = get_reference_center(meshes, keyword="table")
        # print(f'keys : {[m["name"] for m in meshes]}')
        # 1/0
        if not include_wall :
            filtered = [m for m in meshes if "world_geom" not in m["name"].lower() and "mount0" not in m["name"].lower()]
        elif not include_table :
            filtered = [m for m in meshes if "table" not in m["name"].lower()]
        else :
            filtered = meshes
        cropped = center_and_crop_meshes(filtered, ref_center, cube_half)
        # cropped = meshes
        # if not mesh_path.exists():
        # print(f"Saving mesh to {mesh_path}")
        save_meshes_as_obj(cropped, mesh_path)

        # record geom poses for resampling (per step)
        pose_step: Dict[str, Any] = {
            "geom_id": [],
            "xmat": [],
            "xpos": [],
        }
        for gid in range(sim.model.ngeom):
            pose_step["geom_id"].append(int(gid))
            pose_step["xmat"].append(sim.data.geom_xmat[gid].reshape(3, 3).tolist())
            pose_step["xpos"].append(sim.data.geom_xpos[gid].tolist())
        cached_pose_seq.append(pose_step)

        if point_meta is None:
            geom_local_verts, point_meta, cached_face_meta, cached_face_areas = _build_tracking_points_from_faces(
                sim=sim,
                cube_center=ref_center,
                cube_half=cube_half,
                include_table=include_table,
                max_points=max_track_points,
                include_wall=include_wall,
                table_weight=table_weight,
            )
            num_points = len(point_meta or [])

        geom_world = _compute_geom_world_verts(sim, geom_local_verts or {})
        step_pts = np.zeros((num_points, 3), dtype=np.float32)
        for i, meta in enumerate(point_meta or []):
            g = meta["geom_id"]
            idxs = meta["vert_indices"]
            w = meta["barycentric"]
            tri = geom_world[g][idxs]
            step_pts[i] = w[0] * tri[0] + w[1] * tri[1] + w[2] * tri[2]
        tracks_per_step.append(step_pts)
        step_done += 1
    print(f'step_done : {step_done} / actions : {len(actions)}')
    if step_done < len(actions):
        print(f"step_done {step_done} < len(actions) {len(actions)}")

    vertex_tracks = np.stack(tracks_per_step, axis=0) if tracks_per_step else np.zeros((0, 0, 3))

    actions_path = out_cropped.parent / "actions.npy"
    if not actions_path.exists():
        np.save(actions_path, actions[: len(frame_indices)])

    # save face pool, geom poses, and local verts for offline resampling
    if cached_face_meta is not None and cached_face_areas is not None and geom_local_verts is not None:
        pool_meta_path = out_cropped.parent / "face_pool_meta.json"
        pool_area_path = out_cropped.parent / "face_pool_areas.npy"
        pose_path = out_cropped.parent / "geom_pose_seq.json"
        verts_path = out_cropped.parent / "geom_local_verts.npz"
        with pool_meta_path.open("w", encoding="utf-8") as f:
            json.dump(cached_face_meta, f, indent=2)
        np.save(pool_area_path, np.asarray(cached_face_areas, dtype=np.float32))
        with pose_path.open("w", encoding="utf-8") as f:
            json.dump(cached_pose_seq, f)
        np.savez(verts_path, **{str(k): v for k, v in geom_local_verts.items()})

    meta_path = out_cropped.parent / "metadata_face_uniform.json"
    metadata = {
        "_traj_index": int(traj_index),
        "_frame_index": [int(i) for i in frame_indices],
        "_len": int(traj_len),
        "actions_file": actions_path.name,
        "num_track_points": int(vertex_tracks.shape[1]),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    tracks_path = out_cropped.parent / "vertex_tracks_face_uniform.npy"
    ids_path = out_cropped.parent / "vertex_ids_face_uniform.json"

    np.save(tracks_path, vertex_tracks)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(point_meta or [], f, indent=2)
    # 1/0


# ===============================
#  Main dataset processing
# ===============================

def process_dataset(
    dataset: DatasetInput,
    output_root: Path,
    task_suite_name: str,
    task_id: int,
    max_episodes: Optional[int],
    cube_half: float,
    include_table: bool,
    include_wall: bool,
    table_weight: float,
    split: str,
    max_track_points: int,
    shard_index: int = 0,
    num_shards: int = 1,
) -> int:
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

    builder = tfds.builder("libero_goal_no_noops", data_dir="/mnt/data/modified_libero_rlds")
    rlds_dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=False, num_parallel_reads=tf.data.AUTOTUNE)

    for traj in rlds_dataset:
        traj_i = tf.cast(traj['_traj_index'][0], tf.int64)
        actions = np.asarray(traj["action"])
        frame_idx = np.asarray(traj["_frame_index"])
        h = hashlib.sha1()
        h.update(actions.tobytes())
        h.update(frame_idx.tobytes())
        hash_value = h.hexdigest()[:16]
        print(hash_value)
        folder_path = dataset_output / f"episode_{hash_value}"
        print(folder_path)
        # print(hash_value)
        traj_len = tf.cast(traj['_len'][0], tf.int64)
        if traj_i % num_shards != shard_index:
            continue
        if max_episodes is not None and episode_idx >= max_episodes:
            break

        if os.path.exists(folder_path) and traj_len == len(glob.glob(f'{folder_path}/cropped_scene/*')):
            print(f'Traj_index {traj_i} already processed (traj_len {traj_len} == {len(glob.glob(f"{folder_path}/cropped_scene/*"))})')
            continue
        # while os.path.exists(folder_path) or traj_len != len(glob.glob(f'{folder_path}/cropped_scene/*')):
        else :
            print(f'Start processing traj_index {traj_i}')
            # time.sleep(1)
            actions = ensure_action_shape(actions, expected_dim=actions.shape[-1])

            frame_indices = np.asarray(traj["_frame_index"]).astype(int).tolist()
            traj_len_arr = np.asarray(traj["_len"])
            traj_len = int(traj_len_arr[0]) if traj_len_arr.ndim > 0 else int(traj_len_arr)

            if traj_len != len(frame_indices):
                raise ValueError(
                    f"traj_len {traj_len} does not match frame_indices length {len(frame_indices)} "
                    f"for traj index {traj_i}"
                )

            if len(actions) != len(frame_indices):
                raise ValueError(
                    f"Action length {len(actions)} does not match frame_indices length {len(frame_indices)} "
                    f"after truncation for traj_index_raw {traj_index_raw}"
                )

            env.reset()
            env.set_init_state(init_states[0])

            ep_dir = folder_path
            cropped_dir = ep_dir / "cropped_scene"
            replay_episode_with_face_tracks(
                env=env,
                actions=actions,
                out_cropped=cropped_dir,
                cube_half=cube_half,
                camera_names=camera_names,
                traj_index=traj_i,
                frame_indices=frame_indices,
                traj_len=traj_len,
                max_track_points=max_track_points,
                include_table=include_table,
                include_wall=include_wall,
                table_weight=table_weight,
            )
            episode_idx += 1

    env.close()
    return episode_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cropped meshes and face-area point tracks from RLDS shards."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["/mnt/data/modified_libero_rlds/libero_goal_no_noops/1.0.0/libero_goal-train.tfrecord-*"],
        help="Glob patterns or dataset directories containing TFRecord shards.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/data/libero/modified_libero_meshes_face_tracks",
        help="Base directory to write outputs into.",
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
        "--exclude-table",
        action="store_true",
        help="If set, remove meshes whose name contains 'table' from the saved outputs.",
    )
    parser.add_argument(
        "--exclude-wall",
        action="store_true",
        help="If set, include meshes whose name contains 'table' in the saved outputs.",
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
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load via dlimp/tfds (e.g., train, val, all).",
    )
    parser.add_argument(
        "--max-track-points",
        type=int,
        default=5000,
        help="Max number of points to track (sampled uniformly over triangle area).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of CPU workers to parallelize episodes (uses shard-index/num-shards internally).",
    )
    parser.add_argument(
        "--table-weight",
        type=float,
        default=1.0,
        help="Multiplier for table face areas during sampling (<1 reduces table samples).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = collect_dataset_inputs(args.inputs)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for dataset in tqdm(datasets):
        if args.num_workers <= 1:
            exported = process_dataset(
                dataset,
                output_root,
                args.task_suite,
                args.task_id,
                args.max_episodes,
                args.cube_half,
                include_table=not args.exclude_table,
                include_wall=not args.exclude_wall,
                table_weight=args.table_weight,
                split=args.split,
                max_track_points=args.max_track_points,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            total += exported
            print(f"{dataset.display_path}: exported {exported} episodes to {output_root/dataset.display_path}")
        else:
            num_workers = args.num_workers
            with mp.Pool(processes=num_workers) as pool:
                tasks = []
                for worker_idx in range(num_workers):
                    tasks.append(
                        pool.apply_async(
                            process_dataset,
                            (
                                dataset,
                                output_root,
                                args.task_suite,
                                args.task_id,
                                args.max_episodes,
                                args.cube_half,
                                not args.exclude_table,
                                not args.exclude_wall,
                                args.table_weight,
                                args.split,
                                args.max_track_points,
                                worker_idx,
                                num_workers,
                            ),
                        )
                    )
                pool.close()
                pool.join()
                exported_list = [t.get() for t in tasks]
                exported = sum(exported_list)
                total += exported
                print(f"{dataset.display_path}: exported {exported} episodes across {num_workers} workers to {output_root/dataset.display_path}")
    print(f"Done. Total episodes exported: {total}")


if __name__ == "__main__":
    main()
