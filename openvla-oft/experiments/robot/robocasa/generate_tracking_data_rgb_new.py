"""Generate point tracks + per-point RGB colors for RoboCasa HDF5 demonstrations.

This is a lightweight wrapper around `generate_tracking_data.py` that produces the
same tracked 3D points (mesh-face sampling) and additionally saves a stable RGB
color for each tracked point.

The RGB values here are *not* camera-rendered colors. Instead, they are derived
from MuJoCo geom/material colors when available, with a deterministic fallback
palette keyed by body name / geom id.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm

import robocasa  # noqa: F401  # ensure envs are registered
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset, reset_to

import robosuite

# Reuse the full tracking implementation (mesh extraction + point sampling + stepping)
_THIS_DIR = Path(__file__).resolve().parent
if _THIS_DIR.as_posix() not in sys.path:
	sys.path.insert(0, _THIS_DIR.as_posix())

import generate_tracking_data as base


def _make_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
	"""Create 4x4 pose matrix (world_from_frame) from position and rotation."""
	T = np.eye(4, dtype=np.float32)
	T[:3, :3] = np.asarray(rot, dtype=np.float32)
	T[:3, 3] = np.asarray(pos, dtype=np.float32)
	return T


def _align_rotation_from_direction(direction_vec: np.ndarray) -> np.ndarray:
	"""Return the 3x3 yaw rotation matrix R used by generate_tracking_data.

	The forward transform in generate_tracking_data uses row-vector form:
	  p_aligned = (p - c) @ R.T + c
	To invert, use:
	  p = (p_aligned - c) @ R + c
	"""
	d = np.asarray(direction_vec, dtype=np.float32).reshape(-1)
	if d.size < 2:
		return np.eye(3, dtype=np.float32)
	# Snap to axis-aligned direction if not already.
	x, y = float(d[0]), float(d[1])
	if abs(x) >= abs(y):
		dir_x = 1.0 if x >= 0 else -1.0
		axis = (dir_x, 0.0)
	else:
		dir_y = 1.0 if y >= 0 else -1.0
		axis = (0.0, dir_y)

	# These match base._align_points_to_neg_x selection.
	if axis == (1.0, 0.0):
		# +x -> -x (flip x)
		R = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
	elif axis == (-1.0, 0.0):
		# already -x
		R = np.eye(3, dtype=np.float32)
	elif axis == (0.0, 1.0):
		# +y -> -x
		R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
	else:
		# -y -> -x
		R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
	return R


def _camera_params_snapshot(
	env,
	*,
	camera_name: str,
	width: int,
	height: int,
) -> Dict[str, Any]:
	"""Capture a single (static) camera snapshot compatible with robosuite conventions."""
	sim = env.sim
	model = sim.model
	data = sim.data

	if not camera_name:
		raise ValueError("camera_name is empty; cannot capture camera params")

	try:
		cam_id = int(model.camera_name2id(camera_name))
	except Exception as e:
		raise ValueError(f"Failed to resolve camera_name={camera_name!r} in MuJoCo model") from e

	pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float32).reshape(3)
	xmat = np.asarray(data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)

	# Robosuite uses an axis correction so camera coords match the projection utilities.
	T_world_from_cam_raw = _make_pose(pos, xmat)
	axis_correction = np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32))
	T_world_from_cam = (T_world_from_cam_raw @ axis_correction).astype(np.float32)
	T_cam_from_world = np.linalg.inv(T_world_from_cam).astype(np.float32)

	# Intrinsics: fovy is vertical field-of-view in degrees.
	try:
		fovy_deg = float(np.asarray(model.cam_fovy[cam_id]).item())
	except Exception as e:
		raise RuntimeError("Failed to read model.cam_fovy for camera") from e

	fovy = math.radians(fovy_deg)
	fy = 0.5 * float(height) / float(math.tan(0.5 * fovy))
	fx = fy
	cx = 0.5 * float(width)
	cy = 0.5 * float(height)
	K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

	return {
		"camera_name": camera_name,
		"camera_id": cam_id,
		"image_width": int(width),
		"image_height": int(height),
		"cam_fovy_deg": fovy_deg,
		"K_3x3": K.tolist(),
		"T_world_from_cam_4x4": T_world_from_cam.tolist(),
		"T_cam_from_world_4x4": T_cam_from_world.tolist(),
		# This script writes frames exactly as returned by sim.render(); MuJoCo/robosuite
		# renders are commonly vertically flipped relative to standard pixel coordinates.
		# The visualizer uses this to flip projected point v (without altering the video).
		"render_vertical_flip": True,
	}


def _write_camera_and_trackspace_params(
	demo_dir: Path,
	*,
	env,
	camera_name: str,
	width: int,
	height: int,
	recenter_points: bool,
	align_forward_to_neg_x: bool,
	mesh_cache: Dict[str, Any],
) -> None:
	params: Dict[str, Any] = {}
	params.update(_camera_params_snapshot(env, camera_name=camera_name, width=width, height=height))

	origin = mesh_cache.get("__recenter_origin__")
	direction_vec = mesh_cache.get("__direction_vec__")
	R_align = None
	if direction_vec is not None and align_forward_to_neg_x:
		try:
			R_align = _align_rotation_from_direction(np.asarray(direction_vec, dtype=np.float32)).tolist()
		except Exception:
			R_align = None

	params["tracks_space"] = {
		"recenter_points": bool(recenter_points),
		"align_forward_to_neg_x": bool(align_forward_to_neg_x),
		"recenter_origin_world": (np.asarray(origin, dtype=np.float32).reshape(3).tolist() if origin is not None else None),
		"direction_vec_world": (
			np.asarray(direction_vec, dtype=np.float32).reshape(-1)[:3].tolist() if direction_vec is not None else None
		),
		"R_align_3x3": R_align,
	}

	path = demo_dir / "camera_params.json"
	with path.open("w", encoding="utf-8") as f:
		json.dump(params, f, indent=2)


def _pick_camera_name(env, preferred: Optional[str]) -> str:
	# Try preferred first.
	if preferred:
		try:
			if hasattr(env, "camera_names") and preferred in getattr(env, "camera_names"):
				return preferred
		except Exception:
			pass
		try:
			if hasattr(env.sim.model, "camera_name2id"):
				env.sim.model.camera_name2id(preferred)
				return preferred
		except Exception:
			# Fall back to auto-pick if preferred isn't available.
			pass

	# Prefer RoboCasa / Robosuite-style camera names that typically show robot + scene.
	# We try more specific names first.
	candidates = (
		"robot0_agentview_left",
		"robot0_agentview_right",
		"robot0_eye_in_hand",
  "robot0_agentview_center",
  "robot0_frontview",
		"agentview",
		"frontview",
		"sideview",
	)
	for name in candidates:
		try:
			# If env exposes camera list.
			if hasattr(env, "camera_names") and name in getattr(env, "camera_names"):
				return name
		except Exception:
			pass
		try:
			# If mujoco model can resolve camera name.
			if hasattr(env.sim.model, "camera_name2id"):
				env.sim.model.camera_name2id(name)
				return name
		except Exception:
			pass

	# Fallback: first camera in the model.
	try:
		model = env.sim.model
		if hasattr(model, "ncam") and int(model.ncam) > 0 and hasattr(model, "cam_id2name"):
			return str(model.cam_id2name(0))
	except Exception:
		pass

	# Last resort: empty string (some render APIs default to a camera).
	return ""


def _render_rgb_frame(env, *, camera_name: str, width: int, height: int) -> np.ndarray:
	"""Render an RGB frame from the simulator.

	Returns uint8 image in shape (H, W, 3).
	"""
	sim = env.sim

	# Try MuJoCo/Robosuite render signatures.
	img = None
	# Some sim.render() signatures error if camera_name is an empty string.
	kwargs_list = []
	if camera_name:
		kwargs_list.extend(
			[
				{"width": width, "height": height, "camera_name": camera_name},
				{"width": width, "height": height, "camera_name": camera_name, "depth": False},
			]
		)
	kwargs_list.append({"width": width, "height": height})

	for kwargs in tuple(kwargs_list):
		try:
			img = sim.render(**kwargs)
			break
		except Exception:
			img = None

	if img is None:
		# Fallback to env.render (some robosuite envs expose this).
		try:
			img = env.render(mode="rgb_array", width=width, height=height, camera_name=camera_name)
		except Exception as e:
			raise RuntimeError(
				"Failed to render RGB frames. If you're running headless, ensure offscreen rendering is enabled "
				"and the environment supports sim.render()."
			) from e

	img = np.asarray(img)
	if img.ndim == 3 and img.shape[-1] == 3:
		pass
	elif img.ndim == 3 and img.shape[-1] == 4:
		img = img[..., :3]
	else:
		raise ValueError(f"Unexpected render output shape: {img.shape}")

	# Convert float images in [0,1] to uint8.
	if img.dtype != np.uint8:
		img = np.clip(img, 0.0, 1.0)
		img = (img * 255.0 + 0.5).astype(np.uint8)
	return img


def _geom_base_rgb_uint8(sim, geom_id: int, *, fallback_body_name: str = "") -> np.ndarray:
	model = sim.model
	# MuJoCo renderer conceptually applies both geom and material properties.
	# For "base color" matching, we combine them when both are available.
	geom_rgb = None
	mat_rgb = None

	try:
		if hasattr(model, "geom_rgba"):
			geom_rgba = np.asarray(model.geom_rgba)
			if geom_rgba.ndim == 2 and geom_rgba.shape[1] >= 3 and 0 <= geom_id < geom_rgba.shape[0]:
				geom_rgb = np.asarray(geom_rgba[geom_id, :3], dtype=np.float32)
	except Exception:
		geom_rgb = None

	try:
		if hasattr(model, "geom_matid") and hasattr(model, "mat_rgba"):
			geom_matid = np.asarray(model.geom_matid)
			mat_rgba = np.asarray(model.mat_rgba)
			if mat_rgba.ndim == 2 and mat_rgba.shape[1] >= 3 and 0 <= geom_id < len(geom_matid):
				mat_id = int(geom_matid[geom_id])
				if 0 <= mat_id < mat_rgba.shape[0]:
					mat_rgb = np.asarray(mat_rgba[mat_id, :3], dtype=np.float32)
	except Exception:
		mat_rgb = None

	if geom_rgb is None and mat_rgb is None:
		return _fallback_palette_rgb(fallback_body_name, geom_id)
	if geom_rgb is None:
		geom_rgb = np.ones(3, dtype=np.float32)
	if mat_rgb is None:
		mat_rgb = np.ones(3, dtype=np.float32)

	eff = np.clip(geom_rgb * mat_rgb, 0.0, 1.0)
	return _as_uint8_rgb(eff)


def _render_geomcolor_frame(
	env,
	*,
	camera_name: str,
	width: int,
	height: int,
) -> Optional[np.ndarray]:
	"""Render a frame where each visible geom is colored by its base MuJoCo color.

	This is meant for *debug/verification*: it will match `point_rgb_face_uniform.npy`
	when that file is derived from geom/material colors.

	Returns uint8 (H,W,3) or None if segmentation rendering isn't available.
	"""
	sim = env.sim
	seg = None
	# Try common segmentation signatures.
	try_kwargs = []
	if camera_name:
		try_kwargs.append({"width": width, "height": height, "camera_name": camera_name, "segmentation": True})
		try_kwargs.append({"width": width, "height": height, "camera_name": camera_name, "segmentation": True, "depth": False})
	try_kwargs.append({"width": width, "height": height, "segmentation": True})

	for kwargs in try_kwargs:
		try:
			seg = sim.render(**kwargs)
			break
		except Exception:
			seg = None

	if seg is None:
		return None

	seg = np.asarray(seg)
	# MuJoCo commonly returns (H, W, 2): [objtype, objid]
	if seg.ndim == 3 and seg.shape[-1] == 2:
		objtype = seg[..., 0].astype(np.int32)
		objid = seg[..., 1].astype(np.int32)
		geom_type_id = None
		try:
			import mujoco

			geom_type_id = int(mujoco.mjtObj.mjOBJ_GEOM)
		except Exception:
			# Common mjtObj enum value for GEOM in MuJoCo.
			geom_type_id = 5
		mask = objtype == geom_type_id
		geom_ids = np.where(mask, objid, -1)
	elif seg.ndim == 2:
		# Some wrappers may return raw geom ids directly.
		geom_ids = seg.astype(np.int32)
		mask = geom_ids >= 0
	else:
		return None

	# Heuristic: some wrappers may return 1-based geom ids.
	ngeom = int(getattr(sim.model, "ngeom", 0))
	if ngeom > 0:
		try:
			max_id = int(np.max(geom_ids[mask])) if np.any(mask) else -1
			min_id = int(np.min(geom_ids[mask])) if np.any(mask) else -1
			if min_id >= 1 and max_id == ngeom:
				geom_ids = geom_ids - 1
		except Exception:
			pass

	# Build lookup table of geom base colors.
	ngeom = int(getattr(sim.model, "ngeom", 0))
	lookup = np.zeros((max(ngeom, 1), 3), dtype=np.uint8)
	for gid in range(ngeom):
		lookup[gid] = _geom_base_rgb_uint8(sim, gid)

	out = np.zeros((height, width, 3), dtype=np.uint8)
	valid = mask & (geom_ids >= 0) & (geom_ids < ngeom)
	if np.any(valid):
		out[valid] = lookup[geom_ids[valid]]
	return out


def _as_uint8_rgb(rgb01: np.ndarray) -> np.ndarray:
	rgb01 = np.asarray(rgb01, dtype=np.float32)
	rgb01 = np.clip(rgb01, 0.0, 1.0)
	return (rgb01 * 255.0 + 0.5).astype(np.uint8)


def _fallback_palette_rgb(body_name: str, geom_id: int) -> np.ndarray:
	# If MuJoCo geom/material colors are unavailable, use a single neutral color.
	# This avoids misleading “randomly colorful” tracks.
	return np.array([128, 128, 128], dtype=np.uint8)


def _derive_point_rgb(sim, point_meta: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, str]:
	"""Return (N,3) uint8 RGB for points and a string describing the source.

	We prefer MuJoCo's base geom/material colors (which are closer to what the
	simulator is configured with) and only use a muted fallback palette when the
	model doesn't expose usable RGBA.
	"""

	model = sim.model
	n = len(point_meta)
	if n == 0:
		return np.zeros((0, 3), dtype=np.uint8), "empty"

	out = np.zeros((n, 3), dtype=np.uint8)
	counts = {"mat_rgba": 0, "geom_rgba": 0, "fallback_palette": 0}

	# Per point: prefer material base color (what geoms typically inherit), then geom_rgba.
	for i, meta in enumerate(point_meta):
		geom_id = int(meta.get("geom_id", -1))
		body_name = str(meta.get("body_name", ""))
		# Use the same logic as the geom-color video.
		c = _geom_base_rgb_uint8(sim, geom_id, fallback_body_name=body_name)
		out[i] = c
		# Best-effort source accounting.
		if hasattr(model, "geom_matid") and hasattr(model, "mat_rgba"):
			try:
				geom_matid = np.asarray(model.geom_matid)
				mat_rgba = np.asarray(model.mat_rgba)
				mat_id = int(geom_matid[geom_id]) if 0 <= geom_id < len(geom_matid) else -1
				if mat_rgba.ndim == 2 and mat_rgba.shape[1] >= 3 and 0 <= mat_id < mat_rgba.shape[0]:
					counts["mat_rgba"] += 1
					continue
			except Exception:
				pass
		if hasattr(model, "geom_rgba"):
			try:
				geom_rgba = np.asarray(model.geom_rgba)
				if geom_rgba.ndim == 2 and geom_rgba.shape[1] >= 3 and 0 <= geom_id < geom_rgba.shape[0]:
					counts["geom_rgba"] += 1
					continue
			except Exception:
				pass
		counts["fallback_palette"] += 1

	# Describe source.
	if counts["fallback_palette"] == 0 and (counts["mat_rgba"] > 0 or counts["geom_rgba"] > 0):
		# We compute effective color from both when available.
		return out, "geom_rgba_x_mat_rgba"
	if counts["fallback_palette"] > 0 and (counts["geom_rgba"] > 0 or counts["mat_rgba"] > 0):
		return out, "mixed_with_fallback"
	return out, "fallback_palette"


def generate_tracking(args: argparse.Namespace) -> None:
	if getattr(args, "cube_points_only", False):
		raise ValueError("This RGB script assumes mesh-based points; do not use --cube_points_only")

	dataset_path = Path(args.dataset).expanduser().resolve()
	point_root = Path(args.point_cloud_dir).expanduser().resolve()
	point_root.mkdir(parents=True, exist_ok=True)
	tracking_index: Dict[str, Any] = {}

	env_meta = get_env_metadata_from_dataset(dataset_path.as_posix())
	env_kwargs = dict(env_meta["env_kwargs"])
	env_name = env_meta.get("env_name")
	env_kwargs["env_name"] = env_name

	# If saving videos, force offscreen rendering.
	if getattr(args, "save_video", False):
		env_kwargs["has_offscreen_renderer"] = True
		# Prefer headless mode.
		if "has_renderer" not in env_kwargs:
			env_kwargs["has_renderer"] = False

	with h5py.File(dataset_path, "r") as f:
		anchor_body = args.anchor_body
		demos = sorted(f["data"].keys())
		if args.demo_keys:
			demos = [k for k in demos if k in set(args.demo_keys)]
		if args.limit is not None:
			demos = demos[: args.limit]

		for demo_key in tqdm(demos, desc="Generating tracks + RGB"):
			demo = f["data"][demo_key]
			states = demo["states"][:]
			actions = demo["actions"][:]

			initial_state = dict(states=states[0])
			initial_state["model"] = demo.attrs["model_file"]
			initial_state["ep_meta"] = demo.attrs["ep_meta"]

			ep_meta = json.loads(initial_state["ep_meta"])
			env_kwargs["layout_ids"] = ep_meta["layout_id"]
			env_kwargs["style_ids"] = ep_meta["style_id"]
			env = robosuite.make(**env_kwargs)

			camera_name = _pick_camera_name(env, getattr(args, "camera_name", None))
			if getattr(args, "save_video", False) and not camera_name:
				raise RuntimeError(
					"--save_video requires a valid camera_name, but no camera could be selected. "
					"Pass --camera_name explicitly."
				)
			video_writer = None
			video_path = None
			if getattr(args, "save_video", False):
				# Stream frames directly to disk to avoid large RAM usage.
				try:
					import imageio.v2 as imageio  # type: ignore
				except Exception as e:
					raise RuntimeError(
						"Saving MP4 requires imageio. Install it in your env (pip/conda install imageio)."
					) from e
				video_path = (point_root / demo_key / (getattr(args, "video_name", None) or "render.mp4"))
				geomcolor_video_path = video_path.with_name(f"geomcolor_{video_path.name}")
				video_path.parent.mkdir(parents=True, exist_ok=True)
				geomcolor_writer = None
				try:
					video_writer = imageio.get_writer(
						video_path.as_posix(),
						fps=int(getattr(args, "video_fps", 30)),
						codec="libx264",
						quality=8,
					)
				except Exception:
					# Fallback: let imageio pick defaults (useful if codec isn't available).
					video_writer = imageio.get_writer(
						video_path.as_posix(),
						fps=int(getattr(args, "video_fps", 30)),
					)
				# Geom-color video: may fail if segmentation rendering isn't supported.
				try:
					geomcolor_writer = imageio.get_writer(
						geomcolor_video_path.as_posix(),
						fps=int(getattr(args, "video_fps", 30)),
						codec="libx264",
						quality=8,
					)
				except Exception:
					try:
						geomcolor_writer = imageio.get_writer(
							geomcolor_video_path.as_posix(),
							fps=int(getattr(args, "video_fps", 30)),
						)
					except Exception:
						geomcolor_writer = None

			anchor_body_id = None
			if anchor_body is not None:
				anchor_body_id = env.sim.model.body_name2id(anchor_body)

			direction_anchor_body_id = None
			if args.direction_anchor_body:
				try:
					direction_anchor_body_id = env.sim.model.body_name2id(args.direction_anchor_body)
				except Exception:
					direction_anchor_body_id = None

			direction_target_body_id = None
			if args.direction_target_body:
				try:
					direction_target_body_id = env.sim.model.body_name2id(args.direction_target_body)
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

			# Capture initial scene and sample points.
			(
				geom_local_verts,
				point_meta,
				cached_face_meta,
				cached_face_areas,
				step_pts,
			) = base._capture_scene(
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
				cube_points_only=False,
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

			# Save a single camera snapshot + track-space transform metadata.
			# Camera is static (user confirmed), so one file is enough.
			_write_camera_and_trackspace_params(
				demo_dir,
				env=env,
				camera_name=camera_name,
				width=int(getattr(args, "video_width", 640)),
				height=int(getattr(args, "video_height", 480)),
				recenter_points=bool(args.recenter_points),
				align_forward_to_neg_x=bool(args.align_forward_to_neg_x),
				mesh_cache=mesh_cache,
			)

			# Write initial rendered frame.
			if video_writer is not None:
				frame = _render_rgb_frame(
					env,
					camera_name=camera_name,
					width=int(getattr(args, "video_width", 640)),
					height=int(getattr(args, "video_height", 480)),
				)
				video_writer.append_data(frame)
			if getattr(args, "save_video", False) and "geomcolor_writer" in locals() and geomcolor_writer is not None:
				gc = _render_geomcolor_frame(
					env,
					camera_name=camera_name,
					width=int(getattr(args, "video_width", 640)),
					height=int(getattr(args, "video_height", 480)),
				)
				if gc is not None:
					geomcolor_writer.append_data(gc)
				else:
					# Disable writing if segmentation isn't available.
					geomcolor_writer.close()
					geomcolor_writer = None

			# Derive and persist per-point RGB once (stable across frames).
			point_rgb, color_source = _derive_point_rgb(env.sim, point_meta or [])
			np.save(demo_dir / "point_rgb_face_uniform.npy", point_rgb)

			filtered_actions: List[np.ndarray] = []
			skipped_action_indices: List[int] = []

			for idx, action in enumerate(actions):
				prev_action = filtered_actions[-1] if filtered_actions else None
				if args.skip_noops and base.is_noop(action, prev_action, args.noop_threshold):
					skipped_action_indices.append(idx)
					continue

				filtered_actions.append(action)
				env.step(action)
				(
					geom_local_verts,
					point_meta,
					cached_face_meta,
					cached_face_areas,
					step_pts,
				) = base._capture_scene(
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
					cube_points_only=False,
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
				if video_writer is not None:
					frame = _render_rgb_frame(
						env,
						camera_name=camera_name,
						width=int(getattr(args, "video_width", 640)),
						height=int(getattr(args, "video_height", 480)),
					)
					video_writer.append_data(frame)
				if getattr(args, "save_video", False) and "geomcolor_writer" in locals() and geomcolor_writer is not None:
					gc = _render_geomcolor_frame(
						env,
						camera_name=camera_name,
						width=int(getattr(args, "video_width", 640)),
						height=int(getattr(args, "video_height", 480)),
					)
					if gc is not None:
						geomcolor_writer.append_data(gc)
					else:
						geomcolor_writer.close()
						geomcolor_writer = None

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
						vertex_tracks = base._align_points_to_neg_x(
							vertex_tracks.reshape(-1, 3),
							direction_vec,
							center=np.zeros(3, dtype=np.float32),
						).reshape(vertex_tracks.shape)
				if args.save_first_ply and vertex_tracks.shape[0] > 0:
					base._save_points_as_ply(vertex_tracks[0], demo_dir / "pointcloud_step_0000.ply")

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
				"tracking_mode": "mesh",
				"point_rgb_file": "point_rgb_face_uniform.npy",
				"point_rgb_dtype": "uint8",
				"point_rgb_range": "[0,255]",
				"point_rgb_source": color_source,
			}
			if video_path is not None:
				metadata.update(
					{
						"render_video_file": Path(video_path).name,
						"render_geomcolor_video_file": (Path(geomcolor_video_path).name if "geomcolor_video_path" in locals() else None),
						"render_camera_name": camera_name,
						"render_width": int(getattr(args, "video_width", 640)),
						"render_height": int(getattr(args, "video_height", 480)),
						"render_fps": int(getattr(args, "video_fps", 30)),
					}
				)
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

			if video_writer is not None:
				video_writer.close()
			if getattr(args, "save_video", False) and "geomcolor_writer" in locals() and geomcolor_writer is not None:
				geomcolor_writer.close()

	index_path = point_root / (args.index_name or "tracking_index.json")
	with index_path.open("w", encoding="utf-8") as f_idx:
		json.dump(tracking_index, f_idx, indent=2)
	print(f"Saved tracking assets for {len(tracking_index)} demos to {point_root}")
	print(f"Index written to {index_path}")


def parse_args() -> argparse.Namespace:
	# base.parse_args() errors on unknown args, so we peel off our extra flags first.
	extra = argparse.ArgumentParser(add_help=False)
	extra.add_argument("--save_video", action="store_true", help="Save an MP4 rendering per demo")
	extra.add_argument(
		"--camera_name",
		type=str,
		default=None,
		help="Camera name for rendering (e.g., agentview). If unset, picks a sensible default.",
	)
	extra.add_argument("--video_width", type=int, default=640, help="Rendered video width")
	extra.add_argument("--video_height", type=int, default=480, help="Rendered video height")
	extra.add_argument("--video_fps", type=int, default=30, help="Rendered video FPS")
	extra.add_argument(
		"--video_name",
		type=str,
		default="render.mp4",
		help="Filename to use for the per-demo MP4 inside each demo output directory",
	)

	extra_args, remaining = extra.parse_known_args(sys.argv[1:])
	old_argv = sys.argv
	try:
		sys.argv = [old_argv[0]] + remaining
		base_args = base.parse_args()
	finally:
		sys.argv = old_argv

	for k, v in vars(extra_args).items():
		setattr(base_args, k, v)
	return base_args


if __name__ == "__main__":
	args = parse_args()
	generate_tracking(args)

