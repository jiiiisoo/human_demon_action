#!/usr/bin/env python3
"""
Resample face-area-based tracking points from a saved face pool and regenerate tracks.

Expects an episode directory containing:
  - actions.npy
  - metadata_face_uniform.json (with _frame_index, _len)
  - face_pool_meta.json (list of dicts: geom_id, vert_indices, body_name)
  - face_pool_areas.npy (areas aligned with face_pool_meta)

Usage:
  python resample_tracks_from_pool.py --episode-dir /path/to/episode_xxxxx \
      --task-suite libero_goal --task-id 0 --max-track-points 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def ensure_action_shape(actions: np.ndarray, expected_dim: int) -> np.ndarray:
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


def load_episode_pool(ep_dir: Path) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
    actions = np.load(ep_dir / "actions.npy")
    meta_path = ep_dir / "metadata_face_uniform.json"
    pool_meta_path = ep_dir / "face_pool_meta.json"
    pool_area_path = ep_dir / "face_pool_areas.npy"

    if not meta_path.exists() or not pool_meta_path.exists() or not pool_area_path.exists():
        raise FileNotFoundError("Expected metadata_face_uniform.json, face_pool_meta.json, face_pool_areas.npy in episode dir")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with pool_meta_path.open("r", encoding="utf-8") as f:
        face_meta = json.load(f)
    face_areas = np.load(pool_area_path)
    return actions, meta, face_meta, face_areas


def resample_face_indices(face_meta: List[Dict[str, Any]], face_areas: np.ndarray, max_points: int, include_table: bool) -> List[Dict[str, Any]]:
    if max_points <= 0 or not face_meta:
        return []
    areas = face_areas.astype(np.float64)
    probs = areas / areas.sum()
    table_indices = [i for i, m in enumerate(face_meta) if "table" in m.get("body_name", "").lower()]
    chosen = np.random.choice(len(face_meta), size=max_points, replace=True, p=probs)
    if include_table and table_indices and not any(idx in table_indices for idx in chosen):
        chosen[0] = np.random.choice(table_indices)
    out: List[Dict[str, Any]] = []
    for idx in chosen:
        meta = face_meta[idx]
        r1 = np.sqrt(np.random.rand())
        r2 = np.random.rand()
        w0 = 1 - r1
        w1 = r1 * (1 - r2)
        w2 = r1 * r2
        m = dict(meta)
        m["barycentric"] = [float(w0), float(w1), float(w2)]
        out.append(m)
    return out


def compute_tracks(env, actions: np.ndarray, point_meta: List[Dict[str, Any]], frame_indices: List[int]) -> np.ndarray:
    if len(actions) != len(frame_indices):
        raise ValueError("actions and frame_indices length mismatch")
    sim = env.sim
    geom_local_cache: Dict[int, np.ndarray] = {}
    tracks: List[np.ndarray] = []
    for act in actions:
        env.step(act.tolist())
        step_pts = np.zeros((len(point_meta), 3), dtype=np.float32)
        for i, meta in enumerate(point_meta):
            g = int(meta["geom_id"])
            idxs = meta["vert_indices"]
            w = meta["barycentric"]
            if g not in geom_local_cache:
                # lazily fetch local verts for this geom
                model = sim.model
                mesh_id = model.geom_dataid[g]
                if mesh_id >= 0:
                    v_adr = model.mesh_vertadr[mesh_id]
                    v_num = model.mesh_vertnum[mesh_id]
                    geom_local_cache[g] = model.mesh_vert[v_adr : v_adr + v_num]
                else:
                    # fallback primitive -> box using geom_size
                    hx, hy, hz = sim.model.geom_size[g]
                    geom_local_cache[g] = np.array(
                        [
                            [-hx, -hy, -hz],
                            [-hx, -hy, hz],
                            [-hx, hy, -hz],
                            [-hx, hy, hz],
                            [hx, -hy, -hz],
                            [hx, -hy, hz],
                            [hx, hy, -hz],
                            [hx, hy, hz],
                        ],
                        dtype=np.float32,
                    )
            local_verts = geom_local_cache[g]
            R = sim.data.geom_xmat[g].reshape(3, 3)
            t = sim.data.geom_xpos[g]
            world_verts = local_verts @ R.T + t
            tri = world_verts[idxs]
            step_pts[i] = w[0] * tri[0] + w[1] * tri[1] + w[2] * tri[2]
        tracks.append(step_pts)
    return np.stack(tracks, axis=0)


def replay_with_resampled_tracks(
    episode_dir: Path,
    task_suite_name: str,
    task_id: int,
    max_track_points: int,
    include_table: bool,
) -> None:
    actions, meta, face_meta, face_areas = load_episode_pool(episode_dir)
    frame_indices = meta.get("_frame_index", list(range(len(actions))))
    actions = ensure_action_shape(actions, expected_dim=actions.shape[-1])

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = Path(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": 128,
        "camera_widths": 128,
        "camera_names": ["agentview"],
        "camera_depths": False,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    point_meta = resample_face_indices(face_meta, face_areas, max_track_points, include_table=include_table)
    tracks = compute_tracks(env, actions, point_meta, frame_indices)
    env.close()

    # Save outputs
    out_tracks = episode_dir / "vertex_tracks_resampled.npy"
    out_ids = episode_dir / "vertex_ids_resampled.json"
    np.save(out_tracks, tracks)
    with out_ids.open("w", encoding="utf-8") as f:
        json.dump(point_meta, f, indent=2)
    print(f"[resample] Saved {len(point_meta)} points to {out_tracks}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resample tracking points from saved face pool.")
    p.add_argument("--episode-dir", required=True, help="Episode directory with face_pool_meta.json/face_pool_areas.npy/actions.npy")
    p.add_argument("--task-suite", default="libero_goal", help="LIBERO task suite name")
    p.add_argument("--task-id", type=int, default=0, help="Task id within suite")
    p.add_argument("--max-track-points", type=int, default=5000, help="Number of points to resample")
    p.add_argument("--include-table", action="store_true", help="Ensure table faces can be sampled")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    replay_with_resampled_tracks(
        episode_dir=Path(args.episode_dir),
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        max_track_points=args.max_track_points,
        include_table=args.include_table,
    )


if __name__ == "__main__":
    main()
