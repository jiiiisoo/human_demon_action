"""Visualize RoboCasa point tracks (optionally with per-point RGB colors).

This script expects a demo output directory that contains at least:
  - vertex_tracks_face_uniform.npy     (T, N, 3) float32

If present, it will also use:
  - point_rgb_face_uniform.npy         (N, 3) uint8

The RGB colors are assumed to be *stable per point* (not per frame).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_ANIM_REF = None  # prevent matplotlib animation GC warnings


def _load_camera_params(demo_dir: Path, camera_params_path: Optional[Path]) -> Dict[str, Any]:
	path = camera_params_path or (demo_dir / "camera_params.json")
	if not path.exists():
		raise FileNotFoundError(
			f"Missing camera params: {path}. Run generate_tracking_data_rgb_new.py first (it writes camera_params.json)."
		)
	with path.open("r", encoding="utf-8") as f:
		params = json.load(f)
	if not isinstance(params, dict):
		raise ValueError(f"Invalid camera params JSON (expected object): {path}")
	return params


def _align_rotation_from_direction(direction_vec: np.ndarray) -> np.ndarray:
	"""Same discrete yaw rotation used by generate_tracking_data(_rgb_new).

	Forward (generator) uses: p_aligned = (p - c) @ R.T + c
	Inverse (this visualizer) uses: p = (p_aligned - c) @ R + c
	"""
	d = np.asarray(direction_vec, dtype=np.float32).reshape(-1)
	if d.size < 2:
		return np.eye(3, dtype=np.float32)
	x, y = float(d[0]), float(d[1])
	if abs(x) >= abs(y):
		dir_x = 1.0 if x >= 0 else -1.0
		axis = (dir_x, 0.0)
	else:
		dir_y = 1.0 if y >= 0 else -1.0
		axis = (0.0, dir_y)

	if axis == (1.0, 0.0):
		R = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
	elif axis == (-1.0, 0.0):
		R = np.eye(3, dtype=np.float32)
	elif axis == (0.0, 1.0):
		R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
	else:
		R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
	return R


def _tracks_to_world(tracks: np.ndarray, camera_params: Dict[str, Any]) -> np.ndarray:
	"""Convert saved track coordinates back to MuJoCo world coordinates."""
	space = camera_params.get("tracks_space") or {}
	recenter = bool(space.get("recenter_points", False))
	align = bool(space.get("align_forward_to_neg_x", False))

	if align and not recenter:
		raise ValueError(
			"camera2d requires recenter_points=True when align_forward_to_neg_x=True, "
			"because inverting alignment without per-frame centers is ambiguous."
		)

	out = tracks.astype(np.float32)
	if align:
		R_list = space.get("R_align_3x3")
		if R_list is not None:
			R = np.asarray(R_list, dtype=np.float32).reshape(3, 3)
		else:
			d = space.get("direction_vec_world")
			if d is None:
				raise ValueError("Missing direction_vec_world / R_align_3x3 in camera_params.json")
			R = _align_rotation_from_direction(np.asarray(d, dtype=np.float32))
		# Invert p_aligned = p @ R.T (center=0) => p = p_aligned @ R
		out = out @ R

	if recenter:
		origin = space.get("recenter_origin_world")
		if origin is None:
			raise ValueError("Missing recenter_origin_world in camera_params.json")
		out = out + np.asarray(origin, dtype=np.float32).reshape(1, 1, 3)
	return out


def _project_world_to_pixels(points_world: np.ndarray, camera_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
	"""Project (N,3) world points to pixels.

	Returns:
	  uv: (N,2) float32 in (u,v) pixel coords
	  valid: (N,) bool for points with positive depth and finite projection
	"""
	K = np.asarray(camera_params["K_3x3"], dtype=np.float32).reshape(3, 3)
	T = np.asarray(camera_params["T_cam_from_world_4x4"], dtype=np.float32).reshape(4, 4)

	pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
	pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
	cam = (T @ pts_h.T).T
	x = cam[:, 0]
	y = cam[:, 1]
	z = cam[:, 2]

	valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 1e-6)
	uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
	if np.any(valid):
		fx = float(K[0, 0])
		fy = float(K[1, 1])
		cx = float(K[0, 2])
		cy = float(K[1, 2])
		u = fx * (x[valid] / z[valid]) + cx
		v = fy * (y[valid] / z[valid]) + cy
		uv[valid, 0] = u
		uv[valid, 1] = v
	return uv, valid


def _draw_points_on_image(
	img: np.ndarray,
	uv: np.ndarray,
	colors_u8: np.ndarray,
	*,
	radius: int,
	alpha: float,
) -> np.ndarray:
	"""Draw colored points on an RGB uint8 image."""
	if img.dtype != np.uint8:
		img = np.clip(img, 0.0, 255.0).astype(np.uint8)
	out = img.copy()
	H, W = out.shape[:2]

	r = int(max(radius, 0))
	if r == 0:
		offsets = [(0, 0)]
	else:
		offsets = []
		for dy in range(-r, r + 1):
			for dx in range(-r, r + 1):
				if dx * dx + dy * dy <= r * r:
					offsets.append((dy, dx))

	a = float(np.clip(alpha, 0.0, 1.0))
	for (u, v), c in zip(uv, colors_u8):
		if not np.isfinite(u) or not np.isfinite(v):
			continue
		x = int(round(float(u)))
		y = int(round(float(v)))
		if x < 0 or x >= W or y < 0 or y >= H:
			continue
		cr, cg, cb = int(c[0]), int(c[1]), int(c[2])
		for dy, dx in offsets:
			xi = x + dx
			yi = y + dy
			if xi < 0 or xi >= W or yi < 0 or yi >= H:
				continue
			if a >= 1.0:
				out[yi, xi, 0] = cr
				out[yi, xi, 1] = cg
				out[yi, xi, 2] = cb
			else:
				out[yi, xi, 0] = int(a * cr + (1.0 - a) * int(out[yi, xi, 0]))
				out[yi, xi, 1] = int(a * cg + (1.0 - a) * int(out[yi, xi, 1]))
				out[yi, xi, 2] = int(a * cb + (1.0 - a) * int(out[yi, xi, 2]))
	return out


def visualize_camera2d(
	demo_dir: Path,
	*,
	stride: int,
	max_frames: Optional[int],
	fps: int,
	save_path: Path,
	dpi: int,
	color_mode: str,
	overlay_video: bool,
	video_file: str,
	camera_params_path: Optional[Path],
	radius: int,
	alpha: float,
	background_alpha: float,
	show_trails: bool,
	trail_len: int,
	trail_alpha: float,
) -> None:
	try:
		import imageio.v2 as imageio  # type: ignore
	except Exception as e:
		raise RuntimeError("camera2d mode requires imageio (pip/conda install imageio)") from e

	tracks, point_rgb = _load_tracks_and_colors(demo_dir)
	point_meta = _load_point_meta(demo_dir, expected_n=int(tracks.shape[1]))

	# Colors: fixed per point.
	colors = None
	if color_mode == "rgb":
		if point_rgb is not None:
			colors = (point_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
		else:
			colors = None
	elif color_mode == "body":
		if point_meta is not None:
			colors = _colors_from_body_names(point_meta)
		elif point_rgb is not None:
			colors = (point_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
		else:
			colors = None
	else:
		raise ValueError(f"Unknown color_mode={color_mode!r}. Use 'body' or 'rgb'.")

	if colors is None:
		colors = np.full((tracks.shape[1], 3), 0.5, dtype=np.float32)
	colors_u8 = (np.clip(colors, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

	camera_params = _load_camera_params(demo_dir, camera_params_path)
	world_tracks = _tracks_to_world(tracks, camera_params)
	W = int(camera_params["image_width"])
	H = int(camera_params["image_height"])
	# MuJoCo / robosuite renders are commonly vertically flipped (upside down).
	# The user's requirement is to keep the original render orientation (upside down),
	# and align points to it. So we do NOT flip video frames; instead we flip point v.
	flip_point_v = bool(camera_params.get("render_vertical_flip", True))

	bg_a = float(np.clip(background_alpha, 0.0, 1.0))

	frame_idxs = np.arange(world_tracks.shape[0], dtype=int)
	if stride > 1:
		frame_idxs = frame_idxs[::stride]
	if max_frames is not None:
		frame_idxs = frame_idxs[:max_frames]
	if frame_idxs.size == 0:
		raise ValueError("No frames to visualize (check --stride / --max_frames)")

	# Source frames.
	reader = None
	if overlay_video:
		video_path = demo_dir / video_file
		if not video_path.exists():
			raise FileNotFoundError(f"Missing overlay video: {video_path}")
		reader = imageio.get_reader(video_path.as_posix())
		try:
			meta = reader.get_meta_data()
			fps = int(meta.get("fps", fps))
		except Exception:
			pass

	save_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		writer = imageio.get_writer(save_path.as_posix(), fps=int(fps), codec="libx264", quality=8)
	except Exception:
		writer = imageio.get_writer(save_path.as_posix(), fps=int(fps))

	# Precompute pixel projections for trails / per-frame drawing.
	uv_all = np.full((world_tracks.shape[0], world_tracks.shape[1], 2), np.nan, dtype=np.float32)
	valid_all = np.zeros((world_tracks.shape[0], world_tracks.shape[1]), dtype=bool)
	for t in range(world_tracks.shape[0]):
		uv_t, valid_t = _project_world_to_pixels(world_tracks[t], camera_params)
		if flip_point_v:
			uv_t = uv_t.copy()
			uv_t[:, 1] = (float(H) - 1.0) - uv_t[:, 1]
		uv_all[t] = uv_t.reshape(-1, 2)
		valid_all[t] = valid_t.reshape(-1)

	try:
		for out_i, t in enumerate(frame_idxs.tolist()):
			if reader is not None:
				try:
					frame = reader.get_data(int(t))
				except Exception:
					# If random access fails, stop.
					break
				frame = np.asarray(frame)
				if frame.ndim == 3 and frame.shape[-1] == 4:
					frame = frame[..., :3]
				if frame.dtype != np.uint8:
					frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
			else:
				frame = np.full((H, W, 3), 255, dtype=np.uint8)

			# Camera-ready: keep original render orientation; fade background so points dominate.
			if bg_a < 1.0:
				frame = (bg_a * frame.astype(np.float32) + (1.0 - bg_a) * 255.0).astype(np.uint8)

			# Trails: draw older points first with fading alpha.
			frame_out = frame
			if show_trails:
				k = int(t)
				if trail_len <= 0:
					start = 0
				else:
					start = max(0, k - int(trail_len) + 1)
				m = max(1, k - start + 1)
				for j in range(start, k):
					age = (k - 1) - j  # 0 for newest trail frame
					denom = max(m - 2, 1)
					a_j = float(trail_alpha) * (1.0 - float(age) / float(denom))
					a_j = float(np.clip(a_j, 0.0, 1.0))
					valid = valid_all[j]
					if not np.any(valid):
						continue
					uv = uv_all[j][valid]
					c = colors_u8[valid]
					frame_out = _draw_points_on_image(frame_out, uv, c, radius=max(1, radius - 1), alpha=a_j)

			# Current frame points on top.
			valid = valid_all[int(t)]
			if np.any(valid):
				uv = uv_all[int(t)][valid]
				c = colors_u8[valid]
				frame_out = _draw_points_on_image(frame_out, uv, c, radius=radius, alpha=alpha)

			writer.append_data(frame_out)
	finally:
		try:
			writer.close()
		except Exception:
			pass
		if reader is not None:
			try:
				reader.close()
			except Exception:
				pass


def _load_tracks_and_colors(demo_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	tracks_path = demo_dir / "vertex_tracks_face_uniform.npy"
	if not tracks_path.exists():
		raise FileNotFoundError(f"Missing tracks file: {tracks_path}")
	tracks = np.load(tracks_path)
	if tracks.ndim != 3 or tracks.shape[-1] != 3:
		raise ValueError(f"Expected tracks with shape (T, N, 3); got {tracks.shape}")

	rgb_path = demo_dir / "point_rgb_face_uniform.npy"
	point_rgb: Optional[np.ndarray] = None
	if rgb_path.exists():
		point_rgb = np.load(rgb_path)
		if point_rgb.ndim != 2 or point_rgb.shape[1] != 3:
			raise ValueError(f"Expected point RGB with shape (N, 3); got {point_rgb.shape}")
		if point_rgb.shape[0] != tracks.shape[1]:
			raise ValueError(
				f"RGB point count mismatch: rgb has N={point_rgb.shape[0]} but tracks has N={tracks.shape[1]}"
			)
	return tracks, point_rgb


def _load_point_meta(demo_dir: Path, *, expected_n: int) -> Optional[List[Dict[str, Any]]]:
	meta_path = demo_dir / "vertex_ids_face_uniform.json"
	if not meta_path.exists():
		return None
	with meta_path.open("r", encoding="utf-8") as f:
		meta = json.load(f)
	if not isinstance(meta, list):
		raise ValueError(f"Expected a list in {meta_path}, got {type(meta)}")
	if expected_n is not None and len(meta) != int(expected_n):
		raise ValueError(
			f"Meta point count mismatch: meta has N={len(meta)} but tracks has N={expected_n}"
		)
	return meta


def _category_from_body_name(body_name: str) -> str:
	name = (body_name or "").lower()

	# Priority matters. E.g. "microwave_door" should be microwave, not door.
	# Robot: keep all links same except gripper.
	if any(k in name for k in ("gripper", "finger", "eef", "hand")):
		return "gripper"
	if any(k in name for k in ("robot0", "robot", "panda", "iiwa", "ur5", "arm")):
		return "robot"

	if any(k in name for k in ("microwave",)):
		return "microwave"
	if any(k in name for k in ("coffee_machine", "coffee", "espresso")):
		return "coffee_machine"
	if any(k in name for k in ("fridge", "refrigerator")):
		return "fridge"
	if any(k in name for k in ("dishwasher",)):
		return "dishwasher"
	if any(k in name for k in ("stove", "oven", "burner", "cooktop")):
		return "stove"
	if any(k in name for k in ("sink",)):
		return "sink"
	if any(k in name for k in ("faucet",)):
		return "faucet"
	if any(k in name for k in ("cabinet",)):
		return "cabinet"
	if any(k in name for k in ("drawer",)):
		return "drawer"
	if any(k in name for k in ("door", "handle", "knob", "hinge")):
		return "door"
	if any(k in name for k in ("table", "counter", "countertop", "island")):
		return "table"
	if any(k in name for k in ("wall", "floor", "world", "mount0")):
		return "world"
	return "default"


def _category_color_rgb01(category: str) -> np.ndarray:
	# Muted, readable colors. (RGB in [0,1])
	colors = {
		"robot": np.array([60, 110, 160], dtype=np.float32),
		"gripper": np.array([180, 120, 60], dtype=np.float32),
		"microwave": np.array([95, 115, 105], dtype=np.float32),
		"door": np.array([140, 95, 120], dtype=np.float32),
		"coffee_machine": np.array([125, 95, 95], dtype=np.float32),
		"fridge": np.array([90, 120, 140], dtype=np.float32),
		"dishwasher": np.array([105, 105, 125], dtype=np.float32),
		"stove": np.array([120, 95, 80], dtype=np.float32),
		"sink": np.array([85, 115, 130], dtype=np.float32),
		"faucet": np.array([95, 110, 120], dtype=np.float32),
		"cabinet": np.array([135, 110, 85], dtype=np.float32),
		"drawer": np.array([135, 110, 85], dtype=np.float32),
		"table": np.array([125, 110, 95], dtype=np.float32),
		"world": np.array([110, 110, 110], dtype=np.float32),
		"default": np.array([128, 128, 128], dtype=np.float32),
	}
	rgb = colors.get(category, colors["default"])
	return (rgb / 255.0).clip(0.0, 1.0)


def _colors_from_body_names(point_meta: List[Dict[str, Any]]) -> np.ndarray:
	n = len(point_meta)
	colors = np.zeros((n, 3), dtype=np.float32)
	for i, meta in enumerate(point_meta):
		body_name = str(meta.get("body_name", ""))
		cat = _category_from_body_name(body_name)
		colors[i] = _category_color_rgb01(cat)
	return colors


def _compute_axis_limits(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
	# points: (..., 3)
	finite = np.isfinite(points).all(axis=-1)
	if not np.any(finite):
		return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)
	pts = points[finite]
	mins = pts.min(axis=0)
	maxs = pts.max(axis=0)
	center = 0.5 * (mins + maxs)
	span = np.maximum(maxs - mins, 1e-6)
	# make cube-ish bounds for nicer 3D viewing
	half = 0.55 * float(np.max(span))
	return (
		(float(center[0] - half), float(center[0] + half)),
		(float(center[1] - half), float(center[1] + half)),
		(float(center[2] - half), float(center[2] + half)),
	)


def visualize(
	demo_dir: Path,
	*,
	stride: int,
	max_frames: Optional[int],
	point_size: float,
	fps: int,
	save_path: Optional[Path],
	dpi: int,
	color_mode: str,
	show_trails: bool,
	trail_len: int,
	trail_alpha: float,
	trail_linewidth: float,
	style: str,
	elev: float,
	azim: float,
) -> None:
	# Lazy import so just loading doesn't require a display backend.
	import matplotlib
	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation
	from mpl_toolkits.mplot3d.art3d import Line3DCollection

	# If running headless (non-interactive backend), plt.show() won't display anything.
	# Bail out early before creating the animation to avoid GC warnings.
	backend = matplotlib.get_backend().lower()
	if save_path is None and "agg" in backend:
		raise SystemExit(
			f"Matplotlib backend '{matplotlib.get_backend()}' is non-interactive. "
			"Use --save /path/out.mp4 (needs ffmpeg) or --save /path/out.gif (needs pillow)."
		)

	tracks, point_rgb = _load_tracks_and_colors(demo_dir)
	point_meta = _load_point_meta(demo_dir, expected_n=int(tracks.shape[1]))

	# Subsample frames for speed.
	frame_idxs = np.arange(tracks.shape[0], dtype=int)
	if stride > 1:
		frame_idxs = frame_idxs[::stride]
	if max_frames is not None:
		frame_idxs = frame_idxs[:max_frames]
	if frame_idxs.size == 0:
		raise ValueError("No frames to visualize (check --stride / --max_frames)")

	# Colors: fixed per point.
	colors = None
	if color_mode == "rgb":
		if point_rgb is not None:
			colors = (point_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
		else:
			colors = None
	elif color_mode == "body":
		if point_meta is not None:
			colors = _colors_from_body_names(point_meta)
		else:
			# If meta is missing, fall back to RGB if available; otherwise default (matplotlib will pick).
			if point_rgb is not None:
				colors = (point_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
			else:
				colors = None
	else:
		raise ValueError(f"Unknown color_mode={color_mode!r}. Use 'body' or 'rgb'.")

	base_colors = colors
	if base_colors is None:
		# Matplotlib default will be used for points; for trails we still need explicit RGBA.
		base_colors = np.full((tracks.shape[1], 3), 0.2, dtype=np.float32)

	# Compute stable axis limits based on the frames we show.
	sample_points = tracks[frame_idxs]
	(xlim, ylim, zlim) = _compute_axis_limits(sample_points.reshape(-1, 3))

	fig = plt.figure(figsize=(8, 6), dpi=dpi)
	ax = fig.add_subplot(111, projection="3d")

	# Paper-friendly styling.
	if style == "paper":
		ax.set_title("")
		ax.set_axis_off()
		try:
			ax.grid(False)
		except Exception:
			pass
		# Remove panes (best-effort across mpl versions).
		for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
			try:
				axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
				axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
			except Exception:
				pass
		try:
			ax.set_proj_type("ortho")
		except Exception:
			pass
	else:
		ax.set_title(demo_dir.name)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

	try:
		ax.set_box_aspect((1, 1, 1))
	except Exception:
		pass
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)
	ax.set_zlim(*zlim)
	ax.view_init(elev=float(elev), azim=float(azim))

	first = tracks[frame_idxs[0]]
	scat = ax.scatter(first[:, 0], first[:, 1], first[:, 2], s=point_size, c=colors, linewidths=0)

	trail = None
	if show_trails:
		# Some Matplotlib versions error if a 3D collection is added with no segments.
		# Initialize with a single invisible degenerate segment; we replace it on first update.
		p = first[0].astype(np.float32)
		dummy_seg = np.stack([p, p], axis=0)[None, :, :]  # (1, 2, 3)
		dummy_rgba = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
		trail = Line3DCollection(dummy_seg, linewidths=float(trail_linewidth))
		trail.set_color(dummy_rgba)
		ax.add_collection3d(trail)

	def _build_trail_segments(frame_i: int) -> Tuple[np.ndarray, np.ndarray]:
		"""Return (segments, rgba_colors) for history trails up to this frame."""
		k = int(frame_idxs[frame_i])
		if trail_len <= 0:
			start = 0
		else:
			start = max(0, k - int(trail_len) + 1)
		sub = tracks[start : k + 1]  # (m, N, 3)
		m = int(sub.shape[0])
		if m <= 1:
			return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
		# Build line segments between consecutive timesteps.
		segs = np.stack([sub[:-1], sub[1:]], axis=2)  # (m-1, N, 2, 3)
		segs = segs.reshape(-1, 2, 3).astype(np.float32)

		# Fade older segments.
		ages = np.arange(m - 1, dtype=np.float32)
		ages = (m - 2) - ages  # 0 for newest, larger for older
		denom = max(float(m - 2), 1.0)
		alpha_by_step = float(trail_alpha) * (1.0 - (ages / denom))
		alpha_by_step = np.clip(alpha_by_step, 0.0, 1.0)

		step_colors = np.repeat(base_colors[None, :, :], m - 1, axis=0)  # (m-1, N, 3)
		step_alpha = np.repeat(alpha_by_step[:, None], step_colors.shape[1], axis=1)  # (m-1, N)
		rgba = np.concatenate([step_colors.reshape(-1, 3), step_alpha.reshape(-1, 1)], axis=1).astype(np.float32)
		return segs, rgba

	def _update(frame_i: int):
		pts = tracks[frame_idxs[frame_i]]
		scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
		if style != "paper":
			ax.set_title(f"{demo_dir.name}  frame={int(frame_idxs[frame_i])}/{tracks.shape[0]-1}")
		artists = [scat]
		if trail is not None:
			segs, rgba = _build_trail_segments(frame_i)
			trail.set_segments(segs)
			trail.set_color(rgba)
			artists.append(trail)
		return tuple(artists)

	interval_ms = int(1000 / max(fps, 1))
	anim = FuncAnimation(fig, _update, frames=len(frame_idxs), interval=interval_ms, blit=False)

	# Keep reference to avoid garbage collection warnings.
	global _ANIM_REF
	_ANIM_REF = anim

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		suffix = save_path.suffix.lower()
		if suffix == ".gif":
			try:
				from matplotlib.animation import PillowWriter

				anim.save(save_path.as_posix(), writer=PillowWriter(fps=fps), dpi=dpi)
			except Exception as e:
				raise RuntimeError(
					"Failed to save GIF. Install pillow (pip install pillow) or save as .mp4 if ffmpeg is available."
				) from e
		elif suffix == ".mp4":
			try:
				from matplotlib.animation import FFMpegWriter

				anim.save(save_path.as_posix(), writer=FFMpegWriter(fps=fps), dpi=dpi)
			except Exception as e:
				raise RuntimeError(
					"Failed to save MP4. Ensure ffmpeg is installed and visible on PATH, or save as .gif (requires pillow)."
				) from e
		else:
			raise ValueError(f"Unsupported --save extension '{save_path.suffix}'. Use .mp4 or .gif")
		return

	plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize RoboCasa point tracks with per-point colors")
	parser.add_argument(
		"--demo_dir",
		type=str,
		required=True,
		help="Path to a demo output directory containing vertex_tracks_face_uniform.npy",
	)
	parser.add_argument("--stride", type=int, default=1, help="Visualize every k-th frame")
	parser.add_argument("--max_frames", type=int, default=None, help="Cap number of frames visualized")
	parser.add_argument("--point_size", type=float, default=2.0, help="Scatter point size")
	parser.add_argument("--fps", type=int, default=30, help="Playback speed")
	parser.add_argument(
		"--view_mode",
		type=str,
		default="3d",
		choices=("3d", "camera2d"),
		help="View mode: '3d' (matplotlib) or 'camera2d' (project into render camera pixels)",
	)
	parser.add_argument(
		"--style",
		type=str,
		default="paper",
		choices=("paper", "debug"),
		help="Rendering style: 'paper' is clean/minimal; 'debug' shows axes and titles",
	)
	parser.add_argument("--elev", type=float, default=20.0, help="3D view elevation angle")
	parser.add_argument("--azim", type=float, default=-60.0, help="3D view azimuth angle")
	parser.add_argument(
		"--show_trails",
		action="store_true",
		help="If set, draw per-point history trails (recommended for papers)",
	)
	parser.add_argument(
		"--trail_len",
		type=int,
		default=75,
		help="Number of history frames to show in trails; <=0 means full history",
	)
	parser.add_argument("--trail_alpha", type=float, default=0.35, help="Max trail alpha for newest segments")
	parser.add_argument("--trail_linewidth", type=float, default=0.8, help="Trail line width")
	parser.add_argument(
		"--save",
		type=str,
		default=None,
		help="If set, save the animation to this path (.mp4 or .gif) instead of showing it",
	)
	parser.add_argument("--dpi", type=int, default=150, help="DPI used when saving")
	parser.add_argument(
		"--color_mode",
		type=str,
		default="body",
		choices=("body", "rgb"),
		help="Coloring mode: 'body' uses vertex_ids_face_uniform.json body_name keywords; 'rgb' uses point_rgb_face_uniform.npy",
	)
	parser.add_argument(
		"--overlay_video",
		action="store_true",
		help="(camera2d) Overlay points on render.mp4 from the demo_dir",
	)
	parser.add_argument(
		"--video_file",
		type=str,
		default="render.mp4",
		help="(camera2d) Video filename inside demo_dir to overlay onto",
	)
	parser.add_argument(
		"--camera_params",
		type=str,
		default=None,
		help="(camera2d) Path to camera_params.json (default: demo_dir/camera_params.json)",
	)
	parser.add_argument(
		"--radius",
		type=int,
		default=2,
		help="(camera2d) Point radius in pixels",
	)
	parser.add_argument(
		"--alpha",
		type=float,
		default=0.9,
		help="(camera2d) Point alpha for overlay blending",
	)
	parser.add_argument(
		"--background_alpha",
		type=float,
		default=0.35,
		help="(camera2d) Background strength. 1.0 = full video, 0.0 = white background",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	demo_dir = Path(args.demo_dir).expanduser().resolve()
	view_mode = str(getattr(args, "view_mode", "3d"))
	if view_mode == "camera2d":
		if not args.save:
			raise SystemExit("camera2d mode requires --save /path/out.mp4")
		visualize_camera2d(
			demo_dir,
			stride=max(int(args.stride), 1),
			max_frames=args.max_frames,
			fps=max(int(args.fps), 1),
			save_path=Path(args.save).expanduser().resolve(),
			dpi=max(int(args.dpi), 50),
			color_mode=str(args.color_mode),
			overlay_video=bool(args.overlay_video),
			video_file=str(args.video_file),
			camera_params_path=(Path(args.camera_params).expanduser().resolve() if args.camera_params else None),
			radius=int(args.radius),
			alpha=float(args.alpha),
			background_alpha=float(args.background_alpha),
			show_trails=bool(args.show_trails),
			trail_len=int(args.trail_len),
			trail_alpha=float(args.trail_alpha),
		)
		return

	visualize(
		demo_dir,
		stride=max(int(args.stride), 1),
		max_frames=args.max_frames,
		point_size=float(args.point_size),
		fps=max(int(args.fps), 1),
		save_path=(Path(args.save).expanduser().resolve() if args.save else None),
		dpi=max(int(args.dpi), 50),
		color_mode=str(args.color_mode),
		show_trails=bool(args.show_trails),
		trail_len=int(args.trail_len),
		trail_alpha=float(args.trail_alpha),
		trail_linewidth=float(args.trail_linewidth),
		style=str(args.style),
		elev=float(args.elev),
		azim=float(args.azim),
	)


if __name__ == "__main__":
	main()
