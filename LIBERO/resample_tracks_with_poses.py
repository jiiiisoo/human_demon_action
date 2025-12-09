#!/usr/bin/env python3
"""
Resample tracking points from saved face pools *without* rerunning the env rollout.

Requires per-episode files:
  - actions.npy (used only for length/frame indices)
  - metadata_face_uniform.json (provides _frame_index/_len)
  - face_pool_meta.json (list of {geom_id, vert_indices, body_name})
  - face_pool_areas.npy (float area per face)
  - geom_pose_seq.json (list per step: {geom_id[], xmat[], xpos[]})

Output:
  - vertex_tracks_resampled.npy
  - vertex_ids_resampled.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import glob


def load_episode_pool(ep_dir: Path):
    actions = np.load(ep_dir / "actions.npy")
    with (ep_dir / "metadata_face_uniform.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (ep_dir / "face_pool_meta.json").open("r", encoding="utf-8") as f:
        face_meta = json.load(f)
    face_areas = np.load(ep_dir / "face_pool_areas.npy")
    with (ep_dir / "geom_pose_seq.json").open("r", encoding="utf-8") as f:
        pose_seq = json.load(f)
    verts_npz = np.load(ep_dir / "geom_local_verts.npz")
    geom_local = {int(k): verts_npz[k] for k in verts_npz.files}
    return actions, meta, face_meta, face_areas, pose_seq, geom_local


def _vertex_candidates_from_faces(face_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build unique vertex candidates from faces that were already inside the crop cube."""
    verts = []
    seen = set()
    for fm in face_meta:
        gid = int(fm["geom_id"])
        bname = fm.get("body_name", "")
        for v in fm["vert_indices"]:
            key = (gid, int(v))
            if key in seen:
                continue
            seen.add(key)
            verts.append(
                {
                    "geom_id": gid,
                    "vert_indices": [int(v), int(v), int(v)],
                    "body_name": bname,
                }
            )
    return verts


def resample_point_meta(face_meta, face_areas, max_points, include_table, vertex_ratio: float = 0.6):
    """
    Hybrid sampling: a share of points from vertices (uniform over cached vertices),
    the rest from faces proportional to area (already cropped + weighted).
    """
    if max_points <= 0 or not face_meta:
        return []

    vertex_candidates = _vertex_candidates_from_faces(face_meta)
    vertex_weights = []
    for vc in vertex_candidates:
        bname = vc.get("body_name", "").lower()
        if "gripper0" in bname:
            vertex_weights.append(3.0)
        elif "robot0" in bname:
            vertex_weights.append(2.0)
        else:
            vertex_weights.append(0.6)

    areas = face_areas.astype(np.float64)
    probs = areas / areas.sum()
    table_indices = [i for i, m in enumerate(face_meta) if "table" in m.get("body_name", "").lower()]

    vertex_count = int(max_points * vertex_ratio)
    vertex_count = max(0, min(vertex_count, len(vertex_candidates), max_points))
    face_count = max_points - vertex_count

    out: List[Dict[str, Any]] = []

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
            out.append(meta)

    if face_count > 0:
        chosen = np.random.choice(len(face_meta), size=face_count, replace=True, p=probs)
        if include_table and table_indices and not any(idx in table_indices for idx in chosen):
            chosen[0] = np.random.choice(table_indices)
        for idx in chosen:
            meta = dict(face_meta[idx])
            r1 = np.sqrt(np.random.rand())
            r2 = np.random.rand()
            w0 = 1 - r1
            w1 = r1 * (1 - r2)
            w2 = r1 * r2
            meta["barycentric"] = [float(w0), float(w1), float(w2)]
            out.append(meta)
    return out


def compute_tracks_from_poses(pose_seq, point_meta, frame_indices, geom_local):
    """Project sampled points for every stored pose."""
    tracks = []
    for step_idx, geom_data in enumerate(pose_seq):
        if step_idx >= len(frame_indices):
            break
        gid_list = geom_data["geom_id"]
        xmat_list = geom_data["xmat"]
        xpos_list = geom_data["xpos"]
        geom_cache = {}
        step_pts = np.zeros((len(point_meta), 3), dtype=np.float32)
        for i, meta in enumerate(point_meta):
            g = int(meta["geom_id"])
            if g not in geom_cache:
                try:
                    idx = gid_list.index(g)
                except ValueError:
                    continue
                R = np.array(xmat_list[idx], dtype=np.float32)
                t = np.array(xpos_list[idx], dtype=np.float32)
                geom_cache[g] = (R, t)
            R, t = geom_cache[g]
            local_verts = geom_local.get(g)
            if local_verts is None:
                continue
            idxs = meta["vert_indices"]
            w = meta["barycentric"]
            tri = local_verts[idxs]
            tri_world = tri @ R.T + t
            step_pts[i] = w[0] * tri_world[0] + w[1] * tri_world[1] + w[2] * tri_world[2]
        tracks.append(step_pts)
    return np.stack(tracks, axis=0)


def main():
    p = argparse.ArgumentParser(description="Resample tracks using saved face pools and geom poses.")
    p.add_argument("--episode-dir", required=True)
    p.add_argument("--max-track-points", type=int, default=5000)
    p.add_argument("--include-table", action="store_true")
    args = p.parse_args()

    episodes = glob.glob(args.episode_dir + "/*")
    for episode in episodes:

        ep_dir = Path(episode)
        actions, meta, face_meta, face_areas, pose_seq, geom_local = load_episode_pool(ep_dir)
        frame_indices = meta.get("_frame_index", list(range(len(actions))))
        if not pose_seq:
            raise ValueError(f"No poses found in {ep_dir/'geom_pose_seq.json'}")

        expected_steps = len(frame_indices)
        if len(pose_seq) < expected_steps:
            # Pad by repeating the last pose so that we still emit a full-length track sequence.
            last_pose = pose_seq[-1]
            pose_seq = pose_seq + [last_pose] * (expected_steps - len(pose_seq))
        elif len(pose_seq) > expected_steps:
            pose_seq = pose_seq[:expected_steps]

        point_meta = resample_point_meta(
            face_meta=face_meta,
            face_areas=face_areas,
            max_points=args.max_track_points,
            include_table=args.include_table,
            vertex_ratio=0.5,
        )
        tracks = compute_tracks_from_poses(pose_seq, point_meta, frame_indices, geom_local)

        out_tracks = ep_dir / f"vertex_tracks_resampled_{args.max_track_points}.npy"
        out_ids = ep_dir / f"vertex_ids_resampled_{args.max_track_points}.json"
        np.save(out_tracks, tracks)
        with out_ids.open("w", encoding="utf-8") as f:
            json.dump(point_meta, f, indent=2)
        print(
            f"[resample] saved tracks shape={tracks.shape} (steps={tracks.shape[0]}, points={tracks.shape[1] if tracks.size else 0}) to {out_tracks}"
        )


if __name__ == "__main__":
    main()
