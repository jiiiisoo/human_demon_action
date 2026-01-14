"""
run_robocasa_eval.py

Evaluate a trained policy on RoboCasa tasks.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import glob
import robocasa

import draccus
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tqdm
from PIL import Image
import torch
import robosuite.utils.transform_utils as T

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.models.projectors import PointcloudProjector
from prismatic.models.action_heads import PointTrackingHead
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS
# from robosuite import load_controller_config
from robosuite.controllers import load_composite_controller_config
import robosuite


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ===============================
#  Mesh helpers (from generate_tracking_data.py)
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


def compute_adjusted_center(
    ref_center: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    cube_offset: Tuple[float, float, float],
    cube_offset_m: Tuple[float, float, float],
) -> np.ndarray:
    """
    Compute adjusted center with cube_offset and cube_offset_m.
    Same logic as generate_tracking_data.py.
    """
    frac = np.array(cube_offset, dtype=np.float32)
    extra = np.array(cube_offset_m, dtype=np.float32)
    span = bounds_max - bounds_min
    shift = bounds_min + frac * span
    return ref_center + shift + extra


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 3
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    action_chunk_size: Optional[int] = None
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Pointcloud input
    use_pointcloud_input: bool = False
    pointcloud_num_points: int = 512
    pointcloud_dim: int = 3
    pointcloud_cube_half: float = 0.5
    include_table: bool = False
    save_pc_debug: bool = False
    point_visualize: bool = False
    tracking_num_points: int = 512
    tracking_dim: int = 3
    normalize_pointcloud: bool = True
    normalize_tracking: bool = True
    precomputed_statistics_path: Optional[Union[str, Path]] = None
    
    # Pointcloud visualization
    visualize_pc_image: bool = False
    save_pc_ply: bool = False
    pc_viz_freq: int = 1
    pc_viz_max_points: int = 2000

    # LIBERO env
    task_suite_name: str = "robocasa"
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    max_episode_steps: int = 720
    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    rollout_dir: str = "./rollouts"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7
    num_episodes : int = 50
    # fmt: on

def create_eval_env(
    env_name,
    # robosuite-related configs
    robots="PandaMobile",
    controllers="OSC_POSE",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=256,
    camera_heights=256,
    seed=None,
    # robocasa-related configs
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    # controller_configs = load_controller_config(default_controller=controllers)
    controller_configs = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
        )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env

def get_task_description(env, fallback_name: str) -> str:
    ep_meta = env.get_ep_meta() if hasattr(env, "get_ep_meta") else {}
    return ep_meta.get("lang", fallback_name)


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    pointcloud_projector = (
        PointcloudProjector(model.llm_dim, num_points=cfg.pointcloud_num_points, point_dim=cfg.pointcloud_dim)
        if cfg.use_pointcloud_input
        else None
    )
    tracking_head = None
    if cfg.point_visualize:
        tracking_head = PointTrackingHead(
            input_dim=model.llm_dim,
            hidden_dim=model.llm_dim,
            num_points=cfg.tracking_num_points,
            tracking_dim=cfg.tracking_dim,
        )
        # Try to load tracking_head weights from checkpoint dir
        if os.path.isdir(cfg.pretrained_checkpoint):
            ckpt_glob = glob.glob(os.path.join(cfg.pretrained_checkpoint, "tracking_head--*.pt"))
            if ckpt_glob:
                sd = torch.load(ckpt_glob[-1], map_location="cpu")
                tracking_head.load_state_dict(remove_ddp_prefix(sd))
        tracking_head = tracking_head.to(model.device if hasattr(model, "device") else 0)

    return model, action_head, proprio_projector, noisy_action_projector, processor, pointcloud_projector, tracking_head

def setup_logging(cfg: GenerateConfig, env_name: str):
    safe_env_name = env_name.replace("/", "_")
    run_id = f"EVAL-{safe_env_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    env_log_dir = os.path.join(cfg.local_log_dir, safe_env_name)
    os.makedirs(env_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(env_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def save_rollout_video(frames, video_path):
    video_writer = imageio.get_writer(video_path, fps=20)
    for frame in frames:
        video_writer.append_data(frame)
    video_writer.close()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    # processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    left_img = obs["robot0_agentview_left_image"][::-1, ::-1]
    right_img = obs["robot0_agentview_right_image"][::-1, ::-1]
    wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    # left_img = obs["robot0_agentview_left_image"]
    # right_img = obs["robot0_agentview_right_image"]
    # wrist_img = obs["robot0_eye_in_hand_image"]
    proprio = np.concatenate([obs["robot0_gripper_qpos"], np.hstack([obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])])])
    observation = {
        "left_image": left_img,
        "right_image": right_img,
        "wrist_image": wrist_img,
        "state": proprio,
    }
    return observation


def _align_points_to_neg_x(
    points: np.ndarray, direction_vec: np.ndarray, *, center: np.ndarray
) -> np.ndarray:
    """
    Rotate points so that direction_vec aligns with -x axis.
    Same implementation as generate_tracking_data.py.
    """
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


def center_crop_and_resize_np(img: np.ndarray, resize_size: Union[int, tuple], crop_scale: float = 0.9) -> np.ndarray:
    """
    Center-crop and resize an image to match training-time distribution.

    Args:
        img: HWC uint8 image
        resize_size: Target size as int (square) or (height, width) tuple
        crop_scale: Area scale for the center crop (0 < scale <= 1)
    """
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(resize_size[::-1], resample=Image.BILINEAR)
    w, h = pil_img.size
    crop_size = int(round(crop_scale * min(w, h)))
    crop_h = crop_w = max(1, crop_size)
    i = max(0, (h - crop_h) // 2)
    j = max(0, (w - crop_w) // 2)
    pil_img = pil_img.crop((j, i, j + crop_w, i + crop_h))
    pil_img = pil_img.resize((w, h), resample=Image.BILINEAR)
    return np.array(pil_img, dtype=np.uint8)


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    # if model_family == "openvla":
    #     action = invert_gripper_action(action)
    return action


def _build_tracking_points_from_faces(
    env,
    cube_center: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    max_points: int,
    include_table: bool = False,
    include_wall: bool = False,
    table_weight: float = 1.0,
    robot_weight: float = 6.0,
    gripper_weight: float = 20.0,
) -> np.ndarray:
    """
    Sample points from mesh faces with weighted sampling (same as generate_tracking_data.py).
    
    This uses the same weighted sampling strategy as training data generation:
    - Robot meshes: 6x weight
    - Gripper meshes: 20x weight
    - Table meshes: 1x weight
    """
    sim = env.sim
    model = sim.model
    
    face_tris: List[np.ndarray] = []
    face_areas: List[float] = []
    
    cube_center = np.asarray(cube_center, dtype=np.float32)
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    cube_min = cube_center + bounds_min
    cube_max = cube_center + bounds_max
    
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = model.body_id2name(body_id) or f"body_{body_id}"
        lname = body_name.lower()
        
        # Skip based on include flags
        if not include_table and "table" in lname:
            continue
        if not include_wall and ("world" in lname or "mount0" in lname or lname.startswith("wall")):
            continue
        
        # Get mesh geometry
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
            # Box geometry
            hx, hy, hz = model.geom_size[geom_id]
            local_verts = np.array([
                [-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, -hz], [-hx, hy, hz],
                [hx, -hy, -hz], [hx, -hy, hz], [hx, hy, -hz], [hx, hy, hz],
            ], dtype=np.float32)
            faces = np.array([
                [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
            ], dtype=np.int32)
        
        # Transform to world coordinates
        R = sim.data.geom_xmat[geom_id].reshape(3, 3)
        t = sim.data.geom_xpos[geom_id]
        world_verts = local_verts @ R.T + t
        
        # Filter faces that overlap with cube
        pose_triangles = world_verts[faces]
        tri_min = pose_triangles.min(axis=1)
        tri_max = pose_triangles.max(axis=1)
        overlap = np.all(tri_min <= cube_max, axis=1) & np.all(tri_max >= cube_min, axis=1)
        valid_tris = pose_triangles[overlap]
        
        if len(valid_tris) == 0:
            continue
        
        # Compute areas with weights
        areas = 0.5 * np.linalg.norm(np.cross(valid_tris[:, 1] - valid_tris[:, 0], valid_tris[:, 2] - valid_tris[:, 0]), axis=1)
        
        # Apply weights (same as generate_tracking_data.py)
        if "table" in lname:
            areas = areas * table_weight
        if lname.startswith("robot0"):
            areas = areas * robot_weight
        if lname.startswith("gripper0"):
            areas = areas * gripper_weight
        
        face_tris.extend(list(valid_tris))
        face_areas.extend(list(areas))
    
    if not face_tris:
        return np.zeros((0, 3), dtype=np.float32)
    
    # Sample points using weighted face areas
    face_areas_np = np.asarray(face_areas, dtype=np.float64)
    probs = face_areas_np / np.maximum(face_areas_np.sum(), 1e-6)
    
    points = []
    max_attempts = max(max_points * 200, 5000)
    attempts = 0
    
    while len(points) < max_points and attempts < max_attempts:
        attempts += 1
        idx = int(np.random.choice(len(face_tris), p=probs))
        tri_world = face_tris[idx]
        barycentric = np.random.dirichlet(alpha=np.ones(3)).astype(np.float32)
        point = barycentric[0] * tri_world[0] + barycentric[1] * tri_world[1] + barycentric[2] * tri_world[2]
        
        # Check if point is inside cube
        if np.all(point >= cube_min) and np.all(point <= cube_max):
            points.append(point)
    
    return np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)


def _allocate_counts_by_area(meshes, total_points: int, min_per_mesh: int):
    areas = []
    for m in meshes:
        tri = m["verts"][m["faces"]] if len(m["faces"]) > 0 else np.zeros((0, 3, 3))
        if len(tri) == 0:
            areas.append(0.0)
        else:
            a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
            areas.append(float(a))
    areas = np.array(areas, dtype=np.float64)
    if areas.sum() <= 1e-9:
        return [min_per_mesh] * len(meshes)
    weights = areas / areas.sum()
    counts = np.floor(weights * total_points).astype(int)
    counts = np.maximum(counts, min_per_mesh)
    diff = total_points - counts.sum()
    if diff > 0:
        order = np.argsort(-weights)
        for i in range(diff):
            counts[order[i % len(order)]] += 1
    elif diff < 0:
        order = np.argsort(weights)
        i = 0
        while diff < 0 and i < len(order):
            idx = order[i]
            if counts[idx] > min_per_mesh:
                counts[idx] -= 1
                diff += 1
            else:
                i += 1
    return counts.tolist()


def sample_points_from_meshes(meshes, total_points: int, min_per_mesh: int = 200) -> np.ndarray:
    """Sample a total number of points across meshes proportional to their surface area."""
    if not meshes or total_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    counts = _allocate_counts_by_area(meshes, total_points, min_per_mesh)
    pts = []
    for m, c in zip(meshes, counts):
        pts.append(sample_points_on_mesh(m["verts"], m["faces"], c))
    return np.concatenate(pts, axis=0).astype(np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


def pointcloud_from_env(
    env,
    cube_half: float,
    num_points: int,
    include_table: bool,
    recenter_origin: Optional[np.ndarray] = None,
    direction_vec: Optional[np.ndarray] = None,
    recenter_points: bool = True,
    align_forward_to_neg_x: bool = True,
    cube_offset: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    cube_offset_m: Tuple[float, float, float] = (-0.5, 0.0, 0.0),
    direction_offset: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract pointcloud from environment matching the training data generation process.
    
    Args:
        env: RoboCasa environment
        cube_half: Half-size of the cube for cropping
        num_points: Number of points to sample
        include_table: Whether to include table meshes
        recenter_origin: Origin for recentering (if None, will be computed on first call)
        direction_vec: Direction vector for alignment (if None, will be computed)
        recenter_points: Whether to recenter points to origin
        align_forward_to_neg_x: Whether to align forward direction to -x
        cube_offset: Offset of anchor within cube as fractions (x, y, z) in [0,1]
        cube_offset_m: Additional cube-center offset in meters (x, y, z)
        direction_offset: Offset along direction vector for center adjustment
    
    Returns:
        Tuple of (points, recenter_origin, direction_vec)
    """
    # Collect meshes
    meshes = collect_world_meshes(env, include_robot=True, include_statics=True, exclude_body_substrings=())
    
    # Use anchor_body (robot0_link0) for center, same as generate_tracking_data.sh
    anchor_body_name = "robot0_link0"
    try:
        anchor_body_id = env.sim.model.body_name2id(anchor_body_name)
    except Exception:
        anchor_body_id = None
    
    ref_center = get_anchor_center(env, meshes, anchor_body_id, keyword="table")
    
    # Compute bounds
    bounds_min = np.array([-cube_half, -cube_half, -cube_half], dtype=np.float32)
    bounds_max = np.array([cube_half, cube_half, cube_half], dtype=np.float32)
    
    # Compute adjusted center with cube_offset and cube_offset_m
    adjusted_center = compute_adjusted_center(ref_center, bounds_min, bounds_max, cube_offset, cube_offset_m)
    
    # Calculate direction vector if needed (robot0_link0 -> gripper0_right_right_gripper)
    if direction_vec is None and (direction_offset != 0.0 or align_forward_to_neg_x):
        # Find body IDs
        anchor_body_name = "robot0_link0"
        target_body_name = "gripper0_right_right_gripper"
        try:
            anchor_body_id = env.sim.model.body_name2id(anchor_body_name)
            target_body_id = env.sim.model.body_name2id(target_body_name)
            
            anchor_pos = env.sim.data.body_xpos[anchor_body_id].copy()
            target_pos = env.sim.data.body_xpos[target_body_id].copy()
            
            direction = target_pos - anchor_pos
            direction[2] = 0  # Project to xy plane
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction_vec = (direction / norm).astype(np.float32)
                
                # Snap to axis-aligned direction (same logic as generate_tracking_data.py)
                di_x = direction_vec[0]
                di_y = direction_vec[1]
                if abs(di_x) >= abs(di_y):
                    if di_x >= 0:
                        direction_vec[0] = 1.0
                        direction_vec[1] = 0.0
                    else:
                        direction_vec[0] = -1.0
                        direction_vec[1] = 0.0
                else:
                    if di_y >= 0:
                        direction_vec[0] = 0.0
                        direction_vec[1] = 1.0
                    else:
                        direction_vec[0] = 0.0
                        direction_vec[1] = -1.0
        except Exception as e:
            logger.warning(f"Could not compute direction vector: {e}")
            direction_vec = None
    
    # Apply direction offset
    if direction_vec is not None and direction_offset != 0.0:
        adjusted_center = adjusted_center + direction_offset * direction_vec
    
    # Store recenter origin on first call
    if recenter_points and recenter_origin is None:
        recenter_origin = adjusted_center.copy()
    
    # Use weighted sampling (same as generate_tracking_data.py)
    # This applies robot_weight=6.0, gripper_weight=20.0 automatically
    pts = _build_tracking_points_from_faces(
        env=env,
        cube_center=adjusted_center,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        max_points=num_points,
        include_table=include_table,
        include_wall=False,  # Same as --exclude_wall in training
        table_weight=1.0,     # Default from generate_tracking_data.py
        robot_weight=6.0,     # Default from generate_tracking_data.py
        gripper_weight=20.0,  # Default from generate_tracking_data.py
    )
    
    if pts.size == 0:
        pts = np.zeros((num_points, 3), dtype=np.float32)
        if recenter_points and recenter_origin is not None:
            pts = pts - recenter_origin
        return pts, recenter_origin, direction_vec
    
    # Apply transformations
    if recenter_points and recenter_origin is not None:
        pts = pts - (adjusted_center - recenter_origin)
        
    if align_forward_to_neg_x and direction_vec is not None and recenter_points:
        # Rotate around origin (after recentering)
        pts = _align_points_to_neg_x(pts, direction_vec, center=np.zeros(3, dtype=np.float32))
    
    return pts, recenter_origin, direction_vec


def save_pc_tensor_as_ply(pc_tensor: torch.Tensor, path: Path, batch_idx: int = 0) -> None:
    pc_np = pc_tensor[batch_idx].to(torch.float32).detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)
    o3d.io.write_point_cloud(str(path), pcd)


def save_pc_np_as_ply(pc_np: np.ndarray, path: Path) -> None:
    """Save numpy pointcloud array as PLY file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)
    logger.info(f"Saved pointcloud to {path}")


def save_pointcloud_image(
    pointcloud: np.ndarray,
    image_path: Path,
    max_points: Optional[int] = None,
    title: str = 'Pointcloud',
) -> None:
    """
    Save pointcloud visualization as image (same as finetune_libero.py).
    
    Args:
        pointcloud: Input pointcloud (N, 3)
        image_path: Path to save the image
        max_points: Maximum points to render
        title: Title for the plot
    """
    if pointcloud.size == 0:
        return
    
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Downsample if needed
    if max_points and pointcloud.shape[0] > max_points:
        indices = np.linspace(0, pointcloud.shape[0] - 1, max_points, dtype=int)
        pointcloud = pointcloud[indices]
    
    fig = None
    try:
        import time
        start_time = time.time()
        
        # Create figure
        fig = plt.figure(figsize=(6.4, 4.8), dpi=80)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_axis_off()
        
        # Center pointcloud
        center = pointcloud.mean(axis=0, keepdims=True)
        pc_centered = pointcloud - center
        max_range = np.linalg.norm(pc_centered, axis=1).max() + 1e-6
        max_range *= 1.2  # Add margin
        
        # Set view limits
        ax.set_xlim3d([-max_range, max_range])
        ax.set_ylim3d([-max_range, max_range])
        ax.set_zlim3d([-max_range, max_range])
        
        # Set view angle
        ax.view_init(elev=20.0, azim=45.0)
        
        # Plot points
        ax.scatter(
            pc_centered[:, 0],
            pc_centered[:, 1],
            pc_centered[:, 2],
            c='blue',
            marker='o',
            s=3,
            alpha=0.6
        )
        
        plt.tight_layout()
        fig.savefig(image_path, dpi=80, bbox_inches='tight')
        
        elapsed = time.time() - start_time
        logger.info(f"Saved pointcloud image to {image_path} (took {elapsed:.2f}s)")
    except Exception as e:
        logger.warning(f"Failed to save pointcloud image to {image_path}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if fig is not None:
            try:
                plt.close(fig)
            except:
                pass


def save_sequence_video(points_seq: np.ndarray, video_path: Path, fps: int = 5, elev: float = 20.0, azim: float = 45.0):
    """Render a sequence of point clouds (T, N, 3) into a fixed-view MP4."""
    if imageio is None or plt is None:
        raise RuntimeError("imageio/matplotlib not available for sequence video export.")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    pts_all = points_seq.reshape(-1, 3)
    pts_all = pts_all - pts_all.mean(axis=0, keepdims=True)
    max_range = np.linalg.norm(pts_all, axis=1).max() + 1e-6
    ax.set_xlim3d([-max_range, max_range])
    ax.set_ylim3d([-max_range, max_range])
    ax.set_zlim3d([-max_range, max_range])
    scatter = ax.scatter([], [], [], s=1)
    writer = imageio.get_writer(video_path, fps=fps)
    for pts in points_seq:
        pts = pts - pts.mean(axis=0, keepdims=True)
        ax.view_init(elev=elev, azim=azim, roll=0)
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
    writer.close()
    plt.close(fig)


def remove_ddp_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    return new_sd


def normalize_pointcloud(pointcloud: np.ndarray, dataset_statistics: dict) -> np.ndarray:
    """
    Normalize pointcloud using dataset statistics (x, y, z separately).
    Same implementation as LIBEROHdf5Dataset.normalize_pointcloud().
    
    Args:
        pointcloud: np.ndarray of shape (N, 3) or (N, D) where D >= 3
        dataset_statistics: dict containing pointcloud statistics
    
    Returns:
        Normalized pointcloud of same shape
    """
    if "pointcloud" not in dataset_statistics:
        return pointcloud
    
    pc_mean = np.array(dataset_statistics["pointcloud"]["mean"])
    pc_std = np.array(dataset_statistics["pointcloud"]["std"])
    
    # Avoid division by zero
    pc_std = np.where(pc_std < 1e-6, 1.0, pc_std)
    
    # Normalize (only first 3 dimensions if D > 3)
    normalized = pointcloud.copy()
    normalized[:, :3] = (pointcloud[:, :3] - pc_mean) / pc_std
    
    return normalized


def denormalize_tracking(tracking: np.ndarray, dataset_statistics: dict) -> np.ndarray:
    """
    Denormalize tracking data back to original scale.
    Same implementation as LIBEROHdf5Dataset.denormalize_tracking().
    
    Args:
        tracking: Normalized tracking of shape (num_points, 3) or (T, num_points, 3)
        dataset_statistics: dict containing tracking statistics
    
    Returns:
        Denormalized tracking of same shape
    """
    if "tracking" not in dataset_statistics:
        return tracking
    
    track_mean = np.array(dataset_statistics["tracking"]["mean"])
    track_std = np.array(dataset_statistics["tracking"]["std"])

    track_std = np.where(track_std < 1e-6, 1.0, track_std)
    
    if len(tracking.shape) == 2:
        denormalized = tracking * track_std + track_mean
    elif len(tracking.shape) == 3:
        denormalized = tracking * track_std[None, None, :] + track_mean[None, None, :]
    else:
        denormalized = tracking
    
    return denormalized


def normalize_proprio(proprio: np.ndarray, dataset_statistics: dict) -> np.ndarray:
    """
    Normalize proprioceptive state using BOUNDS_Q99 method (same as action normalization).
    Same implementation as LIBEROHdf5Dataset.normalize_proprio().
    
    Args:
        proprio: np.ndarray of shape (proprio_dim,)
        dataset_statistics: dict containing proprio statistics
    
    Returns:
        Normalized proprio of same shape
    """
    if "proprio" not in dataset_statistics:
        return proprio
    
    q01 = np.array(dataset_statistics["proprio"]["q01"])
    q99 = np.array(dataset_statistics["proprio"]["q99"])
    
    # BOUNDS_Q99: [q01, q99] -> [-1, 1]
    normalized = 2.0 * (proprio - q01) / (q99 - q01 + 1e-8) - 1.0
    
    # Clip to [-1, 1] for safety
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def denormalize_action(action: np.ndarray, dataset_statistics: dict) -> np.ndarray:
    """
    Denormalize action from [-1, 1] back to original scale using BOUNDS_Q99.
    Same implementation as LIBEROHdf5Dataset.denormalize_action().
    
    Args:
        action: np.ndarray of shape (action_dim,) or (chunk_size, action_dim)
        dataset_statistics: dict containing action statistics
    
    Returns:
        Denormalized action of same shape
    """
    if "action" not in dataset_statistics:
        return action
    
    q01 = np.array(dataset_statistics["action"]["q01"])
    q99 = np.array(dataset_statistics["action"]["q99"])
    
    # Inverse of BOUNDS_Q99: denormalized = (normalized + 1) * (q99 - q01) / 2 + q01
    denormalized = (action + 1.0) * (q99 - q01) / 2.0 + q01
    
    return denormalized


def run_episode(
    cfg: GenerateConfig,
    env,
    env_name: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    pointcloud_projector=None,
    tracking_head=None,
    dataset_statistics=None,
):
    env.reset()
    obs = env._get_observations(force_update=True)
    task_description = get_task_description(env, env_name)
    log_message(f"\nTask: {task_description}", log_file)
    # if initial_state is not None:
    #     obs = env.set_init_state(initial_state)
    # else:
    #     obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
            f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
            "both speed and success rate), we recommend executing the full action chunk."
        )
        raise ValueError("cfg.num_open_loop_steps does not match the NUM_ACTIONS_CHUNK constant defined in prismatic.vla.constants!")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    max_steps = cfg.max_episode_steps

    # Initialize recenter_origin and direction_vec for consistent pointcloud generation
    recenter_origin = None
    direction_vec = None

    # Check if we need to generate pointcloud
    need_pointcloud = cfg.point_visualize or cfg.use_pointcloud_input or cfg.visualize_pc_image or cfg.save_pc_ply

    # initial pointcloud
    if need_pointcloud:
        pc_np, recenter_origin, direction_vec = pointcloud_from_env(
            env,
            cube_half=cfg.pointcloud_cube_half,
            num_points=cfg.pointcloud_num_points,
            include_table=cfg.include_table,
            recenter_origin=recenter_origin,
            direction_vec=direction_vec,
            recenter_points=True,
            align_forward_to_neg_x=True,
            cube_offset=(0.5, 0.5, 0.5),  # Default from generate_tracking_data.py
            cube_offset_m=(-0.5, 0.0, 0.0),  # Actual training data generation parameter
            direction_offset=0.0,  # No direction offset used in training
        )
    
    # Normalize pointcloud if statistics are available and normalization is enabled
    if need_pointcloud and cfg.normalize_pointcloud:
        pc_np = normalize_pointcloud(pc_np, dataset_statistics)
    
    device = model.device if hasattr(model, "device") else 0
    pc_tensor = None
    if cfg.use_pointcloud_input and need_pointcloud:
        pc_tensor = torch.from_numpy(pc_np).to(torch.bfloat16).to(device).unsqueeze(0)
    
    # Setup visualization directories
    pc_viz_dir = None
    if cfg.visualize_pc_image or cfg.save_pc_ply or cfg.save_pc_debug:
        pc_viz_dir = Path(cfg.rollout_dir) / DATE / "pointcloud_viz"
        pc_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial pointcloud (step 0)
        if need_pointcloud:
            # Denormalize for visualization
            pc_denorm = pc_np.copy()
            if cfg.normalize_pointcloud and dataset_statistics and "pointcloud" in dataset_statistics:
                pc_mean = np.array(dataset_statistics["pointcloud"]["mean"])
                pc_std = np.array(dataset_statistics["pointcloud"]["std"])
                pc_std = np.where(pc_std < 1e-6, 1.0, pc_std)
                pc_denorm[:, :3] = pc_np[:, :3] * pc_std + pc_mean
            
            if cfg.visualize_pc_image:
                save_pointcloud_image(
                    pc_denorm,
                    pc_viz_dir / "pointcloud_step_0000.png",
                    max_points=cfg.pc_viz_max_points,
                    title=f'Pointcloud Step 0',
                )
            
            if cfg.save_pc_ply:
                save_pc_np_as_ply(pc_denorm, pc_viz_dir / "pointcloud_step_0000.ply")
    
    # Legacy pc_debug_dir for backward compatibility
    pc_debug_dir = None
    if cfg.point_visualize and cfg.save_pc_debug and pc_tensor is not None:
        pc_debug_dir = Path(cfg.rollout_dir) / DATE / "pc_debug"
        pc_debug_dir.mkdir(parents=True, exist_ok=True)
        save_pc_tensor_as_ply(pc_tensor, pc_debug_dir / "pc_init.ply")

    success = False
    while t < max_steps:
        frame = env.sim.render(
            height=cfg.env_img_res,
            width=cfg.env_img_res,
            camera_name="robot0_agentview_left",
        )[::-1, ::-1]
        replay_images.append(frame)

        observation = prepare_observation(obs, resize_size)

        if len(action_queue) == 0:
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            all_images = [observation["left_image"]]
            if cfg.num_images_in_input > 1:
                all_images.append(observation["right_image"])
            if cfg.num_images_in_input > 2:
                all_images.append(observation["wrist_image"])
            if cfg.center_crop:
                all_images = [center_crop_and_resize_np(img, resize_size) for img in all_images]
            primary_image = all_images.pop(0)
            primary_image = Image.fromarray(primary_image)
            inputs = processor(prompt, primary_image, return_tensors="pt").to(
                model.device if hasattr(model, "device") else 0
            )
            if all_images:
                all_wrist_inputs = [
                    processor(prompt, Image.fromarray(image_wrist), return_tensors="pt").to(
                        model.device if hasattr(model, "device") else 0
                    )
                    for image_wrist in all_images
                ]
                primary_pixel_values = inputs["pixel_values"]
                all_wrist_pixel_values = [wi["pixel_values"] for wi in all_wrist_inputs]
                inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

            proprio = None
            if cfg.use_proprio:
                proprio = observation["state"]
                proprio = normalize_proprio(proprio, dataset_statistics)
                proprio = torch.tensor(proprio, device=inputs["input_ids"].device).unsqueeze(0)
            if cfg.use_pointcloud_input:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    # Get normalized actions from model (unnorm_key=None to prevent model's denormalization)
                    actions_pred, actions_hidden_states = model.predict_action(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                        proprio=proprio,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        pointcloud=pc_tensor,
                        pointcloud_projector=pointcloud_projector if cfg.use_pointcloud_input else None,
                        unnorm_key=None,  # Don't use model's denormalization
                        do_sample=False,
                        use_film=cfg.use_film,
                        action_head=action_head,
                    )
                    
                    # Denormalize actions using dataset statistics (same as training)
                    # Convert to numpy, denormalize, convert back
                    actions_pred_np = actions_pred.cpu().numpy()  # (chunk_size, action_dim)
                    actions_pred_denorm = denormalize_action(actions_pred_np, dataset_statistics)
                    actions = [actions_pred_denorm[i] for i in range(len(actions_pred_denorm))]

                    # Optional: visualize tracking head outputs as sequence video
                    if cfg.point_visualize and tracking_head is not None:
                        pred_tracking = tracking_head.predict_tracking(actions_hidden_states)
                        pred_seq = pred_tracking[0].detach().to(torch.float32).cpu().numpy()  # (chunk_len, num_points, dim)
                        
                        # Denormalize tracking deltas if statistics are available
                        if cfg.normalize_tracking:
                            pred_seq = denormalize_tracking(pred_seq, dataset_statistics)
                        
                        # Build cumulative sequence: initial, initial+delta1, initial+delta1+delta2, ...
                        # Note: init_pc is already normalized, need to denormalize it for visualization
                        init_pc_normalized = pc_tensor[0].detach().to(torch.float32).cpu().numpy()
                        
                        # Denormalize initial pointcloud for visualization
                        if cfg.normalize_pointcloud:
                            # Denormalize using inverse transformation
                            if "pointcloud" in dataset_statistics:
                                pc_mean = np.array(dataset_statistics["pointcloud"]["mean"])
                                pc_std = np.array(dataset_statistics["pointcloud"]["std"])
                                pc_std = np.where(pc_std < 1e-6, 1.0, pc_std)
                                init_pc = init_pc_normalized.copy()
                                init_pc[:, :3] = init_pc_normalized[:, :3] * pc_std + pc_mean
                            else:
                                init_pc = init_pc_normalized
                        else:
                            init_pc = init_pc_normalized
                        
                        # Now both init_pc and pred_seq (deltas) are in original space
                        cum_deltas = np.cumsum(pred_seq, axis=0)
                        pred_seq_with_input = np.concatenate([init_pc[None, ...], init_pc[None, ...] + cum_deltas], axis=0)
                        if pc_debug_dir is not None:
                            seq_video_path = pc_debug_dir / f"track_pred_{t:04d}.mp4"
                            save_sequence_video(pred_seq_with_input, seq_video_path)
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    # Get normalized actions from model (unnorm_key=None to prevent model's denormalization)
                    actions_pred, normalized_actions, actions_hidden_states = model.predict_action(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                        proprio=proprio,
                        proprio_projector= proprio_projector,
                        pointcloud=None,
                        pointcloud_projector= None,
                        unnorm_key=None,  # Don't use model's denormalization
                        do_sample=False,
                        use_film=cfg.use_film,
                        action_head=action_head,
                    )

                actions = denormalize_action(normalized_actions, dataset_statistics)
            
            actions = actions
            action_queue.extend(actions)

        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)
        if action.shape[-1] == 7:
            pad = np.array([0.0, 0.0, 0.0, -1.0], dtype=action.dtype)
            action = np.concatenate([action, pad], axis=-1)
        obs, reward, done, info = env.step(action.tolist())
        if env._check_success():
            success = True
            break
        t += 1

        # refresh pointcloud and visualize
        if need_pointcloud:
            # Regenerate pointcloud
            pc_np, _, _ = pointcloud_from_env(
                env,
                cube_half=cfg.pointcloud_cube_half,
                num_points=cfg.pointcloud_num_points,
                include_table=cfg.include_table,
                recenter_origin=recenter_origin,
                direction_vec=direction_vec,
                recenter_points=True,
                align_forward_to_neg_x=True,
                cube_offset=(0.5, 0.5, 0.5),
                cube_offset_m=(-0.5, 0.0, 0.0),
                direction_offset=0.0,
            )
            
            # Normalize pointcloud if statistics are available and normalization is enabled
            if cfg.normalize_pointcloud:
                pc_np = normalize_pointcloud(pc_np, dataset_statistics)
            
            if cfg.use_pointcloud_input:
                pc_tensor = torch.from_numpy(pc_np).to(torch.bfloat16).to(device).unsqueeze(0)
            
            # Save visualization at specified frequency
            if pc_viz_dir is not None and t % cfg.pc_viz_freq == 0:
                # Denormalize for visualization
                pc_denorm = pc_np.copy()
                if cfg.normalize_pointcloud and dataset_statistics and "pointcloud" in dataset_statistics:
                    pc_mean = np.array(dataset_statistics["pointcloud"]["mean"])
                    pc_std = np.array(dataset_statistics["pointcloud"]["std"])
                    pc_std = np.where(pc_std < 1e-6, 1.0, pc_std)
                    pc_denorm[:, :3] = pc_np[:, :3] * pc_std + pc_mean
                
                if cfg.visualize_pc_image:
                    save_pointcloud_image(
                        pc_denorm,
                        pc_viz_dir / f"pointcloud_step_{t:04d}.png",
                        max_points=cfg.pc_viz_max_points,
                        title=f'Pointcloud Step {t}',
                    )
                
                if cfg.save_pc_ply:
                    save_pc_np_as_ply(pc_denorm, pc_viz_dir / f"pointcloud_step_{t:04d}.ply")
            
            # Legacy pc_debug for backward compatibility
            if pc_debug_dir is not None and cfg.save_pc_debug and pc_tensor is not None:
                save_pc_tensor_as_ply(pc_tensor, pc_debug_dir / f"pc_step_{t:04d}.ply")

    # except Exception as e:
    #     log_message(f"Episode error: {e}", log_file)

    return success, replay_images, task_description


def run_task(
    cfg: GenerateConfig,
    env_name: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    pointcloud_projector=None,
    tracking_head=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    rollout_dir=None,
    dataset_statistics=None,
):
    # task = task_suite.get_task(task_id)
    # initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    # env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    env = create_eval_env(env_name=env_name, camera_widths=cfg.env_img_res, camera_heights=cfg.env_img_res)
    # task_description = get_task_description(env, env_name)
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_episodes)):
        # if cfg.initial_states_path == "DEFAULT":
        #     initial_state = initial_states[episode_idx]
        # else:
        #     initial_states_task_key = task_description.replace(" ", "_")
        #     episode_key = f"demo_{episode_idx}"
        #     if not all_initial_states[initial_states_task_key][episode_key]["success"]:
        #         log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
        #         continue
        #     initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        success, replay_images, task_description = run_episode(
            cfg,
            env,
            env_name,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            None,
            log_file,
            pointcloud_projector,
            tracking_head,
            dataset_statistics,
        )
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1
        safe_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        base_dir = rollout_dir or os.path.join(cfg.rollout_dir, DATE)
        os.makedirs(base_dir, exist_ok=True)
        video_path = (
            f"{base_dir}/{DATE_TIME}--openvla_oft--episode={total_episodes}--success={success}--task={safe_task}.mp4"
        )
        save_rollout_video(replay_images, video_path)
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)
    log_message(f"Final {env_name} success rate: {task_success_rate}", log_file)
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )
    return total_episodes, total_successes, task_success_rate


@draccus.wrap()
def eval_robocasa(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    
    # Load action_chunk_size from checkpoint if available
    ## Initializing mdoel
    global NUM_ACTIONS_CHUNK
    if cfg.pretrained_checkpoint and os.path.isdir(cfg.pretrained_checkpoint):
        args_config_path = Path(cfg.pretrained_checkpoint) / "args_config.json"
        if args_config_path.exists():
            with open(args_config_path, "r") as f:
                saved_args = json.load(f)
                
                # Load action_chunk_size from checkpoint
                if "action_chunk_size" in saved_args and saved_args["action_chunk_size"] is not None:
                    checkpoint_action_chunk_size = saved_args["action_chunk_size"]
                    logger.info(f"Loading action_chunk_size from checkpoint: {checkpoint_action_chunk_size}")
                    NUM_ACTIONS_CHUNK = checkpoint_action_chunk_size
                    
                    # Update the constant in all relevant modules
                    import prismatic.vla.constants as constants_module
                    constants_module.NUM_ACTIONS_CHUNK = checkpoint_action_chunk_size
                    
                    import prismatic.models.action_heads as action_heads_module
                    action_heads_module.NUM_ACTIONS_CHUNK = checkpoint_action_chunk_size
                    
                    # Override cfg.action_chunk_size if user mistakenly provided it
                    if cfg.action_chunk_size is not None and cfg.action_chunk_size != checkpoint_action_chunk_size:
                        logger.warning(
                            f"Ignoring command-line action_chunk_size={cfg.action_chunk_size}, "
                            f"using checkpoint value={checkpoint_action_chunk_size}"
                        )
                    cfg.action_chunk_size = checkpoint_action_chunk_size
        else:
            # No args_config.json found, use command-line value if provided
            if cfg.action_chunk_size is not None:
                logger.info(f"Overriding NUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK} -> {cfg.action_chunk_size}")
                NUM_ACTIONS_CHUNK = cfg.action_chunk_size
                
                # Update the constant in all relevant modules
                import prismatic.vla.constants as constants_module
                constants_module.NUM_ACTIONS_CHUNK = cfg.action_chunk_size
                
                import prismatic.models.action_heads as action_heads_module
                action_heads_module.NUM_ACTIONS_CHUNK = cfg.action_chunk_size
    elif cfg.action_chunk_size is not None:
        # Not a directory checkpoint, use command-line value if provided
        logger.info(f"Overriding NUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK} -> {cfg.action_chunk_size}")
        NUM_ACTIONS_CHUNK = cfg.action_chunk_size
        
        # Update the constant in all relevant modules
        import prismatic.vla.constants as constants_module
        constants_module.NUM_ACTIONS_CHUNK = cfg.action_chunk_size
        
        import prismatic.models.action_heads as action_heads_module
        action_heads_module.NUM_ACTIONS_CHUNK = cfg.action_chunk_size
    
    # Print detected constants
    logger.info(
        f"Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}"
    )
    
    # Set num_open_loop_steps to match NUM_ACTIONS_CHUNK if not explicitly set
    if cfg.num_open_loop_steps == 8 and NUM_ACTIONS_CHUNK != 8:
        logger.info(f"Setting num_open_loop_steps to match NUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}")
        cfg.num_open_loop_steps = NUM_ACTIONS_CHUNK
    
    # Load dataset statistics for normalization/denormalization
    dataset_statistics = None
    if cfg.precomputed_statistics_path:
        stats_path = Path(cfg.precomputed_statistics_path)
        if stats_path.exists():
            logger.info(f"Loading dataset statistics from {stats_path}")
            with open(stats_path, "r") as f:
                loaded_stats = json.load(f)
            
            # Check if it's in RLDS format {dataset_name: {pointcloud: ...}} or flat format {pointcloud: ...}
            if cfg.task_suite_name in loaded_stats:
                # RLDS format
                dataset_statistics = loaded_stats[cfg.task_suite_name]
            elif "pointcloud" in loaded_stats or "tracking" in loaded_stats:
                # Flat format - use directly
                dataset_statistics = loaded_stats
            else:
                logger.warning(f"Could not find pointcloud/tracking statistics in {stats_path}")
            
            if dataset_statistics:
                logger.info("✓ Dataset statistics loaded successfully")
                if "action" in dataset_statistics:
                    action_q01 = dataset_statistics["action"]["q01"]
                    action_q99 = dataset_statistics["action"]["q99"]
                    logger.info(f"  Action q01: {action_q01}")
                    logger.info(f"  Action q99: {action_q99}")
                if "proprio" in dataset_statistics:
                    proprio_q01 = dataset_statistics["proprio"]["q01"]
                    proprio_q99 = dataset_statistics["proprio"]["q99"]
                    logger.info(f"  Proprio q01: {proprio_q01}")
                    logger.info(f"  Proprio q99: {proprio_q99}")
                if "pointcloud" in dataset_statistics:
                    pc_mean = dataset_statistics["pointcloud"]["mean"]
                    pc_std = dataset_statistics["pointcloud"]["std"]
                    logger.info(f"  Pointcloud mean: {pc_mean}")
                    logger.info(f"  Pointcloud std: {pc_std}")
                if "tracking" in dataset_statistics:
                    track_mean = dataset_statistics["tracking"]["mean"]
                    track_std = dataset_statistics["tracking"]["std"]
                    logger.info(f"  Tracking mean: {track_mean}")
                    logger.info(f"  Tracking std: {track_std}")
        else:
            logger.warning(f"Statistics file not found: {stats_path}")
    elif cfg.pretrained_checkpoint and os.path.isdir(cfg.pretrained_checkpoint):
        # Try to load from checkpoint directory
        stats_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
        if stats_path.exists():
            logger.info(f"Loading dataset statistics from checkpoint: {stats_path}")
            with open(stats_path, "r") as f:
                loaded_stats = json.load(f)
            
            if cfg.task_suite_name in loaded_stats:
                dataset_statistics = loaded_stats[cfg.task_suite_name]
            elif "pointcloud" in loaded_stats or "tracking" in loaded_stats:
                dataset_statistics = loaded_stats
            
            if dataset_statistics:
                logger.info("✓ Dataset statistics loaded from checkpoint")
        else:
            logger.info("No dataset statistics found in checkpoint directory")
    
    (
        model,
        action_head,
        proprio_projector,
        noisy_action_projector,
        processor,
        pointcloud_projector,
        tracking_head,
    ) = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    # Initializing envs
    env_names = sorted([name for name in SINGLE_STAGE_TASK_DATASETS if name != "NavigateKitchen"])[18:]
    total_episodes, total_successes = 0, 0
    per_env_summary = []
    for e_name in env_names:
        log_file, local_log_filepath, run_id = setup_logging(cfg, e_name)
        env_rollout_dir = os.path.join(cfg.rollout_dir, e_name.replace("/", "_"), DATE)
        total_episodes, total_successes, env_success_rate = run_task(
            cfg,
            e_name,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            pointcloud_projector,
            tracking_head,
            total_episodes,
            total_successes,
            log_file,
            env_rollout_dir,
            dataset_statistics,
        )
        per_env_summary.append((e_name, env_success_rate))
        if log_file:
            log_file.close()

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    summary_path = os.path.join(cfg.local_log_dir, "overall_summary.txt")
    with open(summary_path, "w") as summary_file:
        summary_file.write("Final results:\n")
        summary_file.write("Per-env success rates:\n")
        for env_name, env_success_rate in per_env_summary:
            summary_file.write(f"- {env_name}: {env_success_rate:.4f}\n")
        summary_file.write(f"Total episodes: {total_episodes}\n")
        summary_file.write(f"Total successes: {total_successes}\n")
        summary_file.write(
            f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)\n"
        )

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(summary_path)
    return final_success_rate


if __name__ == "__main__":
    eval_robocasa()
