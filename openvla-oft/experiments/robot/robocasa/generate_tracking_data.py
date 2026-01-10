"""Generate mesh crops and point tracks for RoboCasa HDF5 demonstrations.

This mirrors the LIBERO `regenerate_libero_dataset_with_tracks.py` utility but
operates directly on RoboCasa datasets. Each demonstration is replayed in the
simulator, a cropped mesh is exported for every step, and triangle-area-sampled
tracking points are recorded to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

import robocasa  # noqa: F401  # ensure envs are registered
from robocasa.scripts.playback_dataset import (
    get_env_metadata_from_dataset,
    reset_to,
)
import json
import robosuite


# ===============================
#  Mesh helpers (adapted from LIBERO export_gt_pointcloud.py)
# ===============================

GEOM_MESH = 7
GEOM_BOX = 6


def _geom_mesh_in_world(sim, geom_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    model, data = sim.model, sim.data
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        return None
    v_adr = model.mesh_vertadr[mesh_id]
    v_num = model.mesh_vertnum[mesh_id]
    f_adr = model.mesh_faceadr[mesh_id]
    f_num = model.mesh_facenum[mesh_id]
    verts_local = model.mesh_vert[v_adr : v_adr + v_num]
    faces = model.mesh_face[f_adr : f_adr + f_num]
    R = data.geom_xmat[geom_id].reshape(3, 3)
    t = data.geom_xpos[geom_id]
    verts_world = verts_local @ R.T + t
    return verts_world, faces


def _box_geom_in_world(sim, geom_id: int) -> Tuple[np.ndarray, np.ndarray]:
    model, data = sim.model, sim.data
    hx, hy, hz = model.geom_size[geom_id]
    corners = np.array(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [hx, hy, -hz],
            [hx, hy, hz],
        ]
    )
    R = data.geom_xmat[geom_id].reshape(3, 3)
    t = data.geom_xpos[geom_id]
    verts_world = corners @ R.T + t
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
    return verts_world, faces


def collect_world_meshes(
    env,
    *,
    include_robot: bool = True,
    include_statics: bool = True,
    exclude_body_substrings: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    sim, model = env.sim, env.sim.model
    meshes: List[Dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = model.body_id2name(body_id) or f"body_{body_id}"
        lname = body_name.lower()
        if not include_robot and (
            "panda" in lname or "robot" in lname or "gripper" in lname
        ):
            continue
        is_static = body_id == 0 or any(word in lname for word in ("floor", "table", "ground"))
        if not include_statics and is_static:
            continue
        if any(substr.lower() in lname for substr in exclude_body_substrings):
            continue

        geom_type = model.geom_type[geom_id]
        mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if geom_type == GEOM_MESH:
            mesh = _geom_mesh_in_world(sim, geom_id)
        elif geom_type == GEOM_BOX:
            mesh = _box_geom_in_world(sim, geom_id)
        if mesh is None:
            continue
        verts_world, faces = mesh
        meshes.append(
            {
                "name": f"{body_name}_geom{geom_id}",
                "verts": verts_world,
                "faces": faces,
            }
        )
    return meshes


def get_reference_center(meshes: Sequence[Dict[str, Any]], keyword: str = "table") -> np.ndarray:
    for mesh in meshes:
        if keyword in mesh["name"].lower() and len(mesh["verts"]) > 0:
            return mesh["verts"].mean(axis=0)
    return np.zeros(3)


def get_body_center(env, body_id: Optional[int]) -> Optional[np.ndarray]:
    if body_id is None:
        return None
    return env.sim.data.body_xpos[body_id].copy()


def get_anchor_center(env, meshes, anchor_body_id: Optional[int], keyword: str = "table") -> np.ndarray:
    center = get_body_center(env, anchor_body_id)
    if center is None:
        center = get_reference_center(meshes, keyword=keyword)
    return center


def compute_bounds(cube_half: Union[float, Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(cube_half, (list, tuple)):
        if len(cube_half) != 4:
            raise ValueError("--cube_half tuple must be [front back lateral vertical]")
        front, back, lateral, vertical = map(float, cube_half)
        bounds_min = np.array([-back, -lateral, -vertical], dtype=np.float32)
        bounds_max = np.array([front, lateral, vertical], dtype=np.float32)
    else:
        bounds_min = np.array([-cube_half, -cube_half, -cube_half], dtype=np.float32)
        bounds_max = np.array([cube_half, cube_half, cube_half], dtype=np.float32)
    return bounds_min, bounds_max


def compute_adjusted_center(
    ref_center: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    cube_offset: Union[Sequence[float], float],
    cube_offset_m: Union[Sequence[float], float],
) -> np.ndarray:
    if isinstance(cube_offset, (list, tuple, np.ndarray)):
        if len(cube_offset) != 3:
            raise ValueError("--cube_offset expects 3 fractions (x y z) in [0,1]")
        frac = np.array(cube_offset, dtype=np.float32)
    else:
        frac = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    if isinstance(cube_offset_m, (list, tuple, np.ndarray)):
        if len(cube_offset_m) != 3:
            raise ValueError("--cube_offset_m expects 3 values (x y z) in meters")
        extra = np.array(cube_offset_m, dtype=np.float32)
    else:
        extra = np.zeros(3, dtype=np.float32)

    span = bounds_max - bounds_min
    shift = bounds_min + frac * span
    return ref_center + shift + extra


def _allocate_samples(probs: np.ndarray, total: int) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=int)
    expected = probs * total
    counts = np.floor(expected).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        fractional = expected - counts
        order = np.argsort(-fractional)
        counts[order[:remainder]] += 1
    elif remainder < 0:
        # shouldn't happen, but clip in case of numerical issues
        for idx in np.argsort(expected):
            if remainder == 0:
                break
            if counts[idx] > 0:
                counts[idx] -= 1
                remainder += 1
    if counts.sum() == 0:
        counts[np.argmax(probs)] = total
    idxs = np.repeat(np.arange(len(probs)), counts)
    if len(idxs) < total:
        extra = np.random.choice(len(probs), size=total - len(idxs), p=probs)
        idxs = np.concatenate([idxs, extra])
    return idxs


def center_and_crop_meshes(
    meshes: Sequence[Dict[str, Any]],
    ref_center: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> List[Dict[str, Any]]:
    centered: List[Dict[str, Any]] = []
    for mesh in meshes:
        verts = mesh["verts"] - ref_center
        faces = mesh["faces"]
        if len(verts) == 0 or len(faces) == 0:
            continue
        tri = verts[faces]
        tri_min = tri.min(axis=1)
        tri_max = tri.max(axis=1)
        overlap = (
            (tri_min[:, 0] <= bounds_max[0])
            & (tri_max[:, 0] >= bounds_min[0])
            & (tri_min[:, 1] <= bounds_max[1])
            & (tri_max[:, 1] >= bounds_min[1])
            & (tri_min[:, 2] <= bounds_max[2])
            & (tri_max[:, 2] >= bounds_min[2])
        )
        faces_filtered = faces[overlap]
        if len(faces_filtered) == 0:
            continue
        centered.append(
            {
                "name": mesh["name"],
                "verts": verts,
                "faces": faces_filtered,
            }
        )
    return centered


def save_meshes_as_obj(meshes: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Generated by generate_tracking_data.py\n")
        vert_offset = 0
        for mesh in meshes:
            name = mesh["name"].replace(" ", "_")
            verts = mesh["verts"]
            faces = mesh["faces"]
            f.write(f"o {name}\n")
            np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
            faces_shifted = faces + vert_offset + 1
            np.savetxt(f, faces_shifted, fmt="f %d %d %d")
            vert_offset += verts.shape[0]


def _save_points_as_ply(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")


def _align_points_to_neg_x(
    points: np.ndarray, direction_vec: np.ndarray, *, center: np.ndarray
) -> np.ndarray:
    dx = float(direction_vec[0])
    dy = float(direction_vec[1])
    if abs(dx) >= abs(dy):
        if dx > 0:
            rot = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        else:
            rot = np.eye(3, dtype=np.float32)
    else:
        if dy > 0:
            rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        else:
            rot = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (points - center) @ rot.T + center


# ===============================
#  Vertex-track helpers (from export_gt_track_area.py)
# ===============================


def _compute_geom_world_verts(sim, geom_local_verts: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    data = sim.data
    geom_world: Dict[int, np.ndarray] = {}
    for geom_id, local_verts in geom_local_verts.items():
        R = data.geom_xmat[geom_id].reshape(3, 3)
        t = data.geom_xpos[geom_id]
        geom_world[geom_id] = local_verts @ R.T + t
    return geom_world


def _build_tracking_points_from_faces(
    sim,
    cube_center: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    include_table: bool,
    max_points: int,
    include_wall: bool = True,
    table_weight: float = 1.0,
    robot_weight: float = 1.0,
    gripper_weight: float = 1.0,
    keyword_weight: float = 1.0,
    keyword: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
    model = sim.model
    geom_local_verts: Dict[int, np.ndarray] = {}
    face_meta: List[Dict[str, Any]] = []
    face_areas: List[float] = []
    face_tris: List[np.ndarray] = []
    point_meta: List[Dict[str, Any]] = []

    cube_center = np.asarray(cube_center, dtype=np.float32)
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    cube_min = cube_center + bounds_min
    cube_max = cube_center + bounds_max

    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = model.body_id2name(body_id) or f"body_{body_id}"
        lname = body_name.lower()

        if not include_table and "table" in lname:
            continue
        if (not include_wall) and (
            "world" in lname or "mount0" in lname or lname.startswith("wall")
        ):
            continue

        mesh_id = model.geom_dataid[geom_id]
        if mesh_id >= 0:
            v_adr = model.mesh_vertadr[mesh_id]
            v_num = model.mesh_vertnum[mesh_id]
            f_adr = model.mesh_faceadr[mesh_id]
            f_num = model.mesh_facenum[mesh_id]
            if v_num == 0 or f_num == 0:
                continue
            local_verts = model.mesh_vert[v_adr : v_adr + v_num]
            faces = model.mesh_face[f_adr : f_adr + f_num]
        else:
            hx, hy, hz = model.geom_size[geom_id]
            if hz < 1e-6:
                local_verts = np.array(
                    [
                        [-hx, -hy, 0.0],
                        [-hx, hy, 0.0],
                        [hx, -hy, 0.0],
                        [hx, hy, 0.0],
                    ],
                    dtype=np.float32,
                )
                faces = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
            else:
                local_verts = np.array(
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

        geom_local_verts[geom_id] = local_verts

        pose_triangles = _compute_geom_world_verts(sim, {geom_id: local_verts})[geom_id][faces]
        tri_min = pose_triangles.min(axis=1)
        tri_max = pose_triangles.max(axis=1)
        overlap = np.all(tri_min <= cube_max, axis=1) & np.all(tri_max >= cube_min, axis=1)
        valid_faces = faces[overlap]
        if len(valid_faces) == 0:
            continue

        tri = pose_triangles[overlap]
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        if "table" in lname:
            areas = areas * table_weight
        if lname.startswith("robot0"):
            areas = areas * robot_weight
        if lname.startswith("gripper0"):
            areas = areas * gripper_weight
        if keyword:
            if any(kw in lname for kw in keyword):
                areas = areas * keyword_weight

        for face, area in zip(valid_faces, areas):
            face_meta.append(
                {
                    "geom_id": int(geom_id),
                    "vert_indices": face.tolist(),
                    "body_name": body_name,
                }
            )
            face_areas.append(float(area))
        face_tris.extend(list(tri))

    if not face_meta:
        return geom_local_verts, [], [], []

    face_areas_np = np.asarray(face_areas, dtype=np.float64)
    probs = face_areas_np / np.maximum(face_areas_np.sum(), 1e-6)
    sample_count = max_points if max_points > 0 else 0
    if sample_count == 0:
        return geom_local_verts, [], [], face_areas
    max_attempts = max(sample_count * 200, 5000)
    attempts = 0
    while len(point_meta) < sample_count and attempts < max_attempts:
        attempts += 1
        idx = int(np.random.choice(len(face_meta), p=probs))
        tri_world = face_tris[idx]
        barycentric = np.random.dirichlet(alpha=np.ones(3)).astype(np.float32)
        point = barycentric[0] * tri_world[0] + barycentric[1] * tri_world[1] + barycentric[2] * tri_world[2]
        if np.all(point >= cube_min) and np.all(point <= cube_max):
            meta = face_meta[idx]
            vert_indices = np.asarray(meta["vert_indices"], dtype=np.int32)
            point_meta.append(
                {
                    "geom_id": meta["geom_id"],
                    "vert_indices": vert_indices.tolist(),
                    "barycentric": barycentric.tolist(),
                    "body_name": meta["body_name"],
                }
            )

    return geom_local_verts, point_meta, face_meta, face_areas


def _sample_cube_points(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    num_points: int,
) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    samples = np.random.uniform(bounds_min, bounds_max, size=(num_points, 3)).astype(np.float32)
    return samples


# ===============================
#  Tracking generation logic
# ===============================


def is_noop(action: np.ndarray, prev_action: Optional[np.ndarray], threshold: float) -> bool:
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def _capture_scene(
    env,
    out_dir: Path,
    step_idx: int,
    geom_local_verts: Optional[Dict[int, np.ndarray]],
    point_meta: Optional[List[Dict[str, Any]]],
    cached_pose_seq: List[Dict[str, Any]],
    mesh_cache: Dict[str, Dict[str, Any]],
    mesh_order: List[str],
    cube_half: Union[float, Sequence[float]],
    cube_offset: Union[float, Sequence[float]],
    cube_offset_m: Union[float, Sequence[float]],
    anchor_body_id: Optional[int],
    direction_anchor_body_id: Optional[int],
    direction_target_body_id: Optional[int],
    *,
    cube_points_only: bool,
    skip_mesh_save: bool,
    save_first_ply: bool,
    recenter_points: bool,
    align_forward_to_neg_x: bool,
    include_table: bool,
    include_wall: bool,
    max_track_points: int,
    table_weight: float,
    robot_weight: float,
    gripper_weight: float,
    keyword_weight: float,
    keyword: Optional[Sequence[str]],
    direction_offset: float,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]], List[float], np.ndarray]:
    sim = env.sim
    bounds_min, bounds_max = compute_bounds(cube_half)
    meshes: List[Dict[str, Any]] = []
    ref_center = get_body_center(env, anchor_body_id)
    if ref_center is None:
        meshes = collect_world_meshes(env, include_robot=True, include_statics=True)
        ref_center = get_anchor_center(env, meshes, anchor_body_id, keyword="table")
    if ref_center is None:
        ref_center = np.zeros(3, dtype=np.float32)
    if not skip_mesh_save and not meshes:
        meshes = collect_world_meshes(env, include_robot=True, include_statics=True)
    adjusted_center = compute_adjusted_center(ref_center, bounds_min, bounds_max, cube_offset, cube_offset_m)
    direction_vec = mesh_cache.get("__direction_vec__")
    if (direction_offset != 0.0 or align_forward_to_neg_x) and direction_anchor_body_id is not None and direction_target_body_id is not None:
        if direction_vec is None:
            anchor_pos = get_body_center(env, direction_anchor_body_id)
            target_pos = get_body_center(env, direction_target_body_id)
            if anchor_pos is not None and target_pos is not None:
                direction = target_pos - anchor_pos
                direction[2] = 0
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction_vec = (direction / norm).astype(np.float32)
                    di_x = direction_vec[0]
                    di_y = direction_vec[1]
                    if abs(di_x) >= abs(di_y) :
                        direction_vec[1] = 0
                        if di_x > 0 :
                            direction_vec[0] = 1.0
                        else :
                            direction_vec[0] = -1.0
                    else :
                        direction_vec[0] = 0
                        if di_y > 0:
                            direction_vec[1] = 1.0
                        else :
                            direction_vec[1] = -1.0
                    mesh_cache["__direction_vec__"] = direction_vec
        if direction_vec is not None and direction_offset != 0.0:
            adjusted_center = adjusted_center + direction_offset * direction_vec

    if recenter_points and "__recenter_origin__" not in mesh_cache:
        mesh_cache["__recenter_origin__"] = adjusted_center.copy()

    if cube_points_only:
        if point_meta is None:
            local_offsets = _sample_cube_points(bounds_min, bounds_max, max_track_points)
            point_meta = [{"local_offset": offset.tolist()} for offset in local_offsets]
        local_offsets = (
            np.asarray([meta["local_offset"] for meta in point_meta], dtype=np.float32)
            if point_meta
            else np.zeros((0, 3), dtype=np.float32)
        )
        step_pts = adjusted_center + local_offsets
        if align_forward_to_neg_x and direction_vec is not None and not recenter_points:
            step_pts = _align_points_to_neg_x(step_pts, direction_vec, center=adjusted_center)
        if save_first_ply and (not recenter_points) and step_idx == 0 and step_pts.size > 0:
            _save_points_as_ply(step_pts, out_dir / "pointcloud_step_0000.ply")
        return {}, point_meta or [], [], [], step_pts

    filtered = meshes
    if not include_wall:
        filtered = [
            m
            for m in filtered
            if (
                "world" not in m["name"].lower()
                and "mount0" not in m["name"].lower()
                and not m["name"].lower().startswith("wall")
            )
        ]
    if not include_table:
        filtered = [m for m in filtered if "table" not in m["name"].lower()]

    if not skip_mesh_save:
        cropped = center_and_crop_meshes(filtered, adjusted_center, bounds_min, bounds_max)

        # maintain persistent mesh order so previously seen meshes remain saved
        for mesh in cropped:
            name = mesh["name"]
            if name not in mesh_order:
                mesh_order.append(name)
            mesh_cache[name] = mesh
        ordered_meshes = [mesh_cache[name] for name in mesh_order if name in mesh_cache]

        mesh_path = out_dir / "cropped_scene" / f"step_{step_idx:04d}.obj"
        save_meshes_as_obj(ordered_meshes, mesh_path)

        pose_step = {"geom_id": [], "xmat": [], "xpos": []}
        for gid in range(sim.model.ngeom):
            pose_step["geom_id"].append(int(gid))
            pose_step["xmat"].append(sim.data.geom_xmat[gid].reshape(3, 3).tolist())
            pose_step["xpos"].append(sim.data.geom_xpos[gid].tolist())
        cached_pose_seq.append(pose_step)

    cached_face_meta: List[Dict[str, Any]] = []
    cached_face_areas: List[float] = []
    if point_meta is None or geom_local_verts is None:
        geom_local_verts, point_meta, cached_face_meta, cached_face_areas = _build_tracking_points_from_faces(
            sim=sim,
            cube_center=adjusted_center,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            include_table=include_table,
            max_points=max_track_points,
            include_wall=include_wall,
            table_weight=table_weight,
            robot_weight=robot_weight,
            gripper_weight=gripper_weight,
            keyword_weight=keyword_weight,
            keyword=keyword,
        )
    else:
        cached_face_meta = []
        cached_face_areas = []

    geom_world = _compute_geom_world_verts(sim, geom_local_verts or {})
    num_points = len(point_meta or [])
    step_pts = np.zeros((num_points, 3), dtype=np.float32)
    for idx, meta in enumerate(point_meta or []):
        g = meta["geom_id"]
        vert_indices = meta["vert_indices"]
        bary = meta["barycentric"]
        tri = geom_world[g][vert_indices]
        step_pts[idx] = bary[0] * tri[0] + bary[1] * tri[1] + bary[2] * tri[2]

    if align_forward_to_neg_x and direction_vec is not None and not recenter_points:
        step_pts = _align_points_to_neg_x(step_pts, direction_vec, center=adjusted_center)

    if save_first_ply and (not recenter_points) and step_idx == 0 and step_pts.size > 0:
        _save_points_as_ply(step_pts, out_dir / "pointcloud_step_0000.ply")

    return geom_local_verts or {}, point_meta or [], cached_face_meta, cached_face_areas, step_pts


def generate_tracking(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).expanduser().resolve()
    point_root = Path(args.point_cloud_dir).expanduser().resolve()
    point_root.mkdir(parents=True, exist_ok=True)
    tracking_index: Dict[str, Any] = {}

    env_meta = get_env_metadata_from_dataset(dataset_path.as_posix())
    env_kwargs = env_meta["env_kwargs"]
    env_name = env_meta.get("env_name")
    env_kwargs["env_name"] = env_name

    with h5py.File(dataset_path, "r") as f:
        anchor_body = args.anchor_body
        demos = sorted(f["data"].keys())
        if args.demo_keys:
            demos = [k for k in demos if k in set(args.demo_keys)]
        if args.limit is not None:
            demos = demos[: args.limit]

        for demo_key in tqdm(demos, desc="Generating tracks"):
            demo = f["data"][demo_key]
            states = demo["states"][:]
            actions = demo["actions"][:]
            initial_state = dict(states=states[0])
            initial_state['model'] = demo.attrs['model_file']
            initial_state['ep_meta'] = demo.attrs['ep_meta']
            # initial = {
            #     "model": demo.attrs.get("model_file"),
            #     "states": states[0],
            #     "ep_meta": demo.attrs.get("ep_meta"),
            # }
            ep_meta = json.loads(initial_state['ep_meta'])
            env_kwargs["layout_ids"] = ep_meta["layout_id"]
            env_kwargs["style_ids"] = ep_meta["style_id"]

            env = robosuite.make(**env_kwargs)

            anchor_body_id = None
            if anchor_body is not None:
                anchor_body_id = env.sim.model.body_name2id(anchor_body)
            direction_anchor_body_id = None
            if args.direction_anchor_body:
                try:
                    direction_anchor_body_id = env.sim.model.body_name2id(
                        args.direction_anchor_body
                    )
                except Exception:
                    direction_anchor_body_id = None
            direction_target_body_id = None
            if args.direction_target_body:
                try:
                    direction_target_body_id = env.sim.model.body_name2id(
                        args.direction_target_body
                    )
                except Exception:
                    direction_target_body_id = None
            reset_to(env, initial_state)

            demo_dir = point_root / demo_key
            (demo_dir / "cropped_scene").mkdir(parents=True, exist_ok=True)

            cached_pose_seq: List[Dict[str, Any]] = []
            tracks_per_step: List[np.ndarray] = []
            geom_local_verts: Optional[Dict[int, np.ndarray]] = None
            point_meta: Optional[List[Dict[str, Any]]] = None
            cached_face_meta: Optional[List[Dict[str, Any]]] = None
            cached_face_areas: Optional[List[float]] = None
            mesh_cache: Dict[str, Dict[str, Any]] = {}
            mesh_order: List[str] = []

            # capture the initial scene before playing actions
            (
                geom_local_verts,
                point_meta,
                cached_face_meta,
                cached_face_areas,
                step_pts,
            ) = _capture_scene(
                env,
                demo_dir,
                0,
                geom_local_verts,
                point_meta,
                cached_pose_seq,
                mesh_cache,
                mesh_order,
                args.cube_half,
                args.cube_offset,
                args.cube_offset_m,
                anchor_body_id,
                direction_anchor_body_id,
                direction_target_body_id,
                cube_points_only=args.cube_points_only,
                skip_mesh_save=args.skip_mesh_save,
                save_first_ply=args.save_first_ply,
                recenter_points=args.recenter_points,
                align_forward_to_neg_x=args.align_forward_to_neg_x,
                include_table=not args.exclude_table,
                include_wall=not args.exclude_wall,
                max_track_points=args.max_track_points,
                table_weight=args.table_weight,
                robot_weight=args.robot_weight,
                gripper_weight=args.gripper_weight,
                keyword_weight=args.keyword_weight,
                keyword=args.keyword,
                direction_offset=args.direction_offset,
            )
            if step_pts.size > 0:
                tracks_per_step.append(step_pts)

            filtered_actions: List[np.ndarray] = []
            skipped_action_indices: List[int] = []

            for idx, action in enumerate(actions):
                prev_action = filtered_actions[-1] if filtered_actions else None
                if args.skip_noops and is_noop(action, prev_action, args.noop_threshold):
                    skipped_action_indices.append(idx)
                    continue

                filtered_actions.append(action)
                obs, reward, done, info = env.step(action)
                (
                    geom_local_verts,
                    point_meta,
                    cached_face_meta,
                    cached_face_areas,
                    step_pts,
                ) = _capture_scene(
                    env,
                    demo_dir,
                    idx + 1,
                    geom_local_verts,
                    point_meta,
                    cached_pose_seq,
                    mesh_cache,
                    mesh_order,
                    args.cube_half,
                    args.cube_offset,
                    args.cube_offset_m,
                    anchor_body_id,
                    direction_anchor_body_id,
                    direction_target_body_id,
                    cube_points_only=args.cube_points_only,
                    skip_mesh_save=args.skip_mesh_save,
                    save_first_ply=args.save_first_ply,
                    recenter_points=args.recenter_points,
                    align_forward_to_neg_x=args.align_forward_to_neg_x,
                    include_table=not args.exclude_table,
                    include_wall=not args.exclude_wall,
                    max_track_points=args.max_track_points,
                    table_weight=args.table_weight,
                    robot_weight=args.robot_weight,
                    gripper_weight=args.gripper_weight,
                    keyword_weight=args.keyword_weight,
                    keyword=args.keyword,
                    direction_offset=args.direction_offset,
                )
                if step_pts.size > 0:
                    tracks_per_step.append(step_pts)

            vertex_tracks = (
                np.stack(tracks_per_step, axis=0)
                if tracks_per_step
                else np.zeros((0, 0, 3), dtype=np.float32)
            )
            if args.recenter_points and vertex_tracks.size > 0:
                origin = mesh_cache.get("__recenter_origin__")
                if origin is not None:
                    vertex_tracks = vertex_tracks - origin
                if args.align_forward_to_neg_x:
                    direction_vec = mesh_cache.get("__direction_vec__")
                    if direction_vec is not None:
                        vertex_tracks = _align_points_to_neg_x(
                            vertex_tracks.reshape(-1, 3),
                            direction_vec,
                            center=np.zeros(3, dtype=np.float32),
                        ).reshape(vertex_tracks.shape)
                if args.save_first_ply and vertex_tracks.shape[0] > 0:
                    _save_points_as_ply(vertex_tracks[0], demo_dir / "pointcloud_step_0000.ply")
            np.save(demo_dir / "vertex_tracks_face_uniform.npy", vertex_tracks)
            np.save(demo_dir / "actions.npy", np.asarray(filtered_actions, dtype=np.float32))
            with (demo_dir / "vertex_ids_face_uniform.json").open("w", encoding="utf-8") as f_json:
                json.dump(point_meta or [], f_json, indent=2)

            if cached_face_meta is not None and cached_face_areas is not None and geom_local_verts is not None:
                with (demo_dir / "face_pool_meta.json").open("w", encoding="utf-8") as f_json:
                    json.dump(cached_face_meta, f_json, indent=2)
                np.save(demo_dir / "face_pool_areas.npy", np.asarray(cached_face_areas, dtype=np.float32))
                with (demo_dir / "geom_pose_seq.json").open("w", encoding="utf-8") as f_json:
                    json.dump(cached_pose_seq, f_json)
                np.savez(demo_dir / "geom_local_verts.npz", **{str(k): v for k, v in geom_local_verts.items()})

            metadata = {
                "dataset": dataset_path.as_posix(),
                "demo_key": demo_key,
                "num_actions": len(filtered_actions),
                "num_track_points": int(vertex_tracks.shape[1]) if vertex_tracks.size else 0,
                "skipped_noop_indices": skipped_action_indices,
                "tracking_mode": "cube" if args.cube_points_only else "mesh",
            }
            with (demo_dir / "metadata_face_uniform.json").open("w", encoding="utf-8") as f_json:
                json.dump(metadata, f_json, indent=2)

            if args.skip_noops:
                with (demo_dir / "skipped_noop_indices.json").open("w", encoding="utf-8") as f_json:
                    json.dump(skipped_action_indices, f_json, indent=2)

            tracking_index[demo_key] = {
                "path": demo_dir.as_posix(),
                "num_frames": int(vertex_tracks.shape[0]),
                "num_track_points": metadata["num_track_points"],
            }

    index_path = point_root / (args.index_name or "tracking_index.json")
    with index_path.open("w", encoding="utf-8") as f_idx:
        json.dump(tracking_index, f_idx, indent=2)
    print(f"Saved tracking assets for {len(tracking_index)} demos to {point_root}")
    print(f"Index written to {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RoboCasa tracking data")
    parser.add_argument("--dataset", required=True, help="Path to RoboCasa HDF5 dataset")
    parser.add_argument("--point_cloud_dir", required=True, help="Output directory for tracking assets")
    parser.add_argument(
        "--cube_half",
        type=float,
        nargs="+",
        default=(0.5,),
        help="Half edge length (m) for the crop cube. Provide four floats as `front back lateral vertical` to use asymmetric bounds.",
    )
    parser.add_argument(
        "--anchor_body",
        type=str,
        default="table",
        help="Body name used as crop center. Defaults to 'table'; use e.g. 'robot0_link0'",
    )
    parser.add_argument(
        "--cube_offset",
        type=float,
        nargs="+",
        default=(0.5, 0.5, 0.5),
        help="Offset of anchor within cube as fractions (x y z) in [0,1]; e.g., 0.5 0.2 0.8 shifts cube so anchor sits lower/forward.",
    )
    parser.add_argument(
        "--cube_offset_m",
        type=float,
        nargs="+",
        default=(0.0, 0.0, 0.0),
        help="Additional cube-center offset in meters (x y z) applied after --cube_offset.",
    )
    parser.add_argument("--max_track_points", type=int, default=5000, help="Maximum number of tracked points")
    parser.add_argument(
        "--cube_points_only",
        action="store_true",
        help="Sample tracking points uniformly inside the cube instead of mesh faces",
    )
    parser.add_argument(
        "--skip_mesh_save",
        action="store_true",
        help="Skip writing cropped mesh .obj files",
    )
    parser.add_argument(
        "--save_first_ply",
        action="store_true",
        help="Save the first-frame tracking point cloud as a PLY file",
    )
    parser.add_argument(
        "--recenter_points",
        action="store_true",
        help="Shift sampled points so the first-frame cube center becomes the origin",
    )
    parser.add_argument(
        "--align_forward_to_neg_x",
        action="store_true",
        help="Rotate points so the robot forward direction aligns with -x",
    )
    parser.add_argument("--exclude_table", action="store_true", help="Drop meshes whose name contains 'table'")
    parser.add_argument("--exclude_wall", action="store_true", help="Drop world / wall meshes")
    parser.add_argument("--table_weight", type=float, default=1.0, help="Triangle area multiplier for table faces")
    parser.add_argument(
        "--robot_weight",
        type=float,
        default=6.0,
        help="Triangle area multiplier for bodies starting with 'robot0'",
    )
    parser.add_argument(
        "--gripper_weight",
        type=float,
        default=20.0,
        help="Triangle area multiplier for bodies starting with 'gripper0'",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        nargs="+",
        default=None,
        help="If set, upweight faces whose body name contains any of these substrings",
    )
    parser.add_argument(
        "--keyword_weight",
        type=float,
        default=10.0,
        help="Triangle area multiplier for bodies matching --keyword",
    )
    parser.add_argument(
        "--direction_anchor_body",
        type=str,
        default="robot0_link0",
        help="Anchor body used to define the forward direction vector",
    )
    parser.add_argument(
        "--direction_target_body",
        type=str,
        default="gripper0",
        help="Target body used to define the forward direction vector",
    )
    parser.add_argument(
        "--direction_offset",
        type=float,
        default=0.0,
        help="Meters to shift the cube center along (target - anchor) direction",
    )
    parser.add_argument("--demo_keys", nargs="*", default=None, help="Subset of demo_<id> keys to process")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N demos")
    parser.add_argument("--skip_noops", action="store_true", help="Filter out no-op actions before stepping")
    parser.add_argument("--noop_threshold", type=float, default=1e-4, help="L2 threshold for no-op detection")
    parser.add_argument("--index_name", type=str, default=None, help="Optional custom filename for the index JSON")
    parsed = parser.parse_args()
    if isinstance(parsed.cube_half, list):
        if len(parsed.cube_half) == 1:
            parsed.cube_half = parsed.cube_half[0]
        elif len(parsed.cube_half) == 4:
            parsed.cube_half = tuple(parsed.cube_half)
        else:
            raise ValueError("--cube_half expects 1 value (symmetric) or 4 values (front back lateral vertical)")
    if isinstance(parsed.cube_offset, list):
        if len(parsed.cube_offset) == 3:
            parsed.cube_offset = tuple(parsed.cube_offset)
        else:
            raise ValueError("--cube_offset expects 3 values (x y z fractions)")
    if isinstance(parsed.cube_offset_m, list):
        if len(parsed.cube_offset_m) == 3:
            parsed.cube_offset_m = tuple(parsed.cube_offset_m)
        else:
            raise ValueError("--cube_offset_m expects 3 values (x y z meters)")
    if parsed.keyword:
        parsed.keyword = [kw.lower() for kw in parsed.keyword]
    return parsed


if __name__ == "__main__":
    args = parse_args()
    generate_tracking(args)
