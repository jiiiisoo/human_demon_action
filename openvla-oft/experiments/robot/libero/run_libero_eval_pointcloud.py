"""
run_libero_eval_pointcloud.py

Same as run_libero_eval.py, but adds pointcloud input support by sampling a pointcloud
from the LIBERO environment at each step and passing it through a PointcloudProjector.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import glob
from transformers import AutoProcessor

import draccus
try:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    imageio = None
    plt = None
import numpy as np
import open3d as o3d
import tqdm
from libero.libero import benchmark
from PIL import Image

import wandb
import torch
import robosuite.utils.transform_utils as T

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
    prepare_images_for_vla,
    normalize_proprio,
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

# Mesh helpers
sys.path.append("/home/jisookim/LIBERO")
from export_gt_pointcloud import (  # type: ignore  # noqa: E402
    collect_world_meshes,
    get_reference_center,
    center_and_crop_meshes,
)
from pathlib import Path


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
    num_images_in_input: int = 2
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

    # LIBERO env
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    rollout_dir: str = "./rollouts"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)
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

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key

def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
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
    img = get_libero_image(obs) # for 180 degree rotation
    # img = Image.fromarray(img)
    # img = processor.image_processor.apply_transform(img)
    # img = obs["agentview_image"]
    wrist_img = get_libero_wrist_image(obs)
    # wrist_img = Image.fromarray(wrist_img)
    # wrist_img = processor.image_processor.apply_transform(wrist_img)
    # img_resized = resize_image_for_policy(img, resize_size)
    # wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    proprio = np.concatenate([obs["robot0_gripper_qpos"], np.hstack([obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])])])
    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": proprio,
    }
    return observation, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    # if model_family == "openvla":
    #     action = invert_gripper_action(action)
    return action


def sample_points_on_mesh(verts: np.ndarray, faces: np.ndarray, n_samples: int) -> np.ndarray:
    """Sample points uniformly on a mesh surface using face areas."""
    if len(faces) == 0 or len(verts) == 0 or n_samples <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    tri = verts[faces]  # (F, 3, 3)
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = areas.sum()
    if total <= 1e-9:
        return np.zeros((0, 3), dtype=np.float32)
    probs = areas / total
    face_idx = np.random.choice(len(faces), size=n_samples, p=probs)
    tri_sel = tri[face_idx]
    r1 = np.sqrt(np.random.rand(n_samples, 1))
    r2 = np.random.rand(n_samples, 1)
    samples = tri_sel[:, 0] + r1 * (tri_sel[:, 1] - tri_sel[:, 0]) + r2 * (tri_sel[:, 2] - tri_sel[:, 0])
    return samples.astype(np.float32)


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


def pointcloud_from_env(env, cube_half: float, num_points: int, include_table: bool) -> np.ndarray:
    meshes = collect_world_meshes(env, include_robot=True, include_statics=True, exclude_body_substrings=())
    ref_center = get_reference_center(meshes, keyword="table")
    filtered = [m for m in meshes if include_table or "table" not in m["name"].lower()]
    cropped = center_and_crop_meshes(filtered, ref_center, cube_half)
    verts_list, faces_list = [], []
    vert_offset = 0
    for m in cropped:
        if "verts" not in m or "faces" not in m:
            continue
        verts = m["verts"]
        faces = m["faces"] + vert_offset
        verts_list.append(verts)
        faces_list.append(faces)
        vert_offset += verts.shape[0]

    if not verts_list or not faces_list:
        return np.zeros((num_points, 3), dtype=np.float32)

    verts_all = np.concatenate(verts_list, axis=0)
    faces_all = np.concatenate(faces_list, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_all.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces_all.astype(np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    pc_o3d = mesh.sample_points_uniformly(num_points)
    pts = np.asarray(pc_o3d.points, dtype=np.float32)
    return pts


def save_pc_tensor_as_ply(pc_tensor: torch.Tensor, path: Path, batch_idx: int = 0) -> None:
    pc_np = pc_tensor[batch_idx].to(torch.float32).detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)
    o3d.io.write_point_cloud(str(path), pcd)


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
    task_description: str,
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
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

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
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # initial pointcloud
    pc_np = pointcloud_from_env(
        env, cube_half=cfg.pointcloud_cube_half, num_points=cfg.pointcloud_num_points, include_table=cfg.include_table
    )
    
    # Normalize pointcloud if statistics are available and normalization is enabled
    if cfg.normalize_pointcloud and dataset_statistics is not None:
        pc_np = normalize_pointcloud(pc_np, dataset_statistics)
    
    device = model.device if hasattr(model, "device") else 0
    pc_tensor = torch.from_numpy(pc_np).to(torch.bfloat16).to(device).unsqueeze(0)
    pc_debug_dir = None
    if cfg.use_pointcloud_input and (cfg.save_pc_debug or cfg.point_visualize):
        pc_debug_dir = Path(cfg.rollout_dir) / DATE / "pc_debug"
        pc_debug_dir.mkdir(parents=True, exist_ok=True)
        if cfg.save_pc_debug:
            save_pc_tensor_as_ply(pc_tensor, pc_debug_dir / "pc_init.ply")

    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)

        if len(action_queue) == 0:
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            all_images = [observation["full_image"]]
            if cfg.num_images_in_input > 1:
                all_images.append(observation["wrist_image"])
            # all_images = prepare_images_for_vla(all_images, cfg)
            primary_image = all_images.pop(0)
            primary_image = Image.fromarray(primary_image)
            inputs = processor(prompt, primary_image, return_tensors="pt").to(
                model.device if hasattr(model, "device") else 0
            )
            if all_images:
                image_wrist = Image.fromarray(image_wrist)
                all_wrist_inputs = [
                    processor(prompt, image_wrist, return_tensors="pt").to(
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
                # Normalize proprio using dataset statistics (same as training)
                # if dataset_statistics is not None and "proprio" in dataset_statistics:
                #     proprio = normalize_proprio(proprio, dataset_statistics)
                # else:
                #     # Fallback to model's normalization if dataset statistics not available
                #     logger.warning("Dataset statistics not available for proprio, using model's norm_stats")
                #     if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
                #         proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
                #         q01 = np.array(proprio_norm_stats["q01"])
                #         q99 = np.array(proprio_norm_stats["q99"])
                #         proprio = 2.0 * (proprio - q01) / (q99 - q01 + 1e-8) - 1.0
                #         proprio = np.clip(proprio, -1.0, 1.0)
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
                    if dataset_statistics is not None and "action" in dataset_statistics:
                        # Convert to numpy, denormalize, convert back
                        actions_pred_np = actions_pred.cpu().numpy()  # (chunk_size, action_dim)
                        actions_pred_denorm = denormalize_action(actions_pred_np, dataset_statistics)
                        actions = [actions_pred_denorm[i] for i in range(len(actions_pred_denorm))]
                    else:
                        # Fallback to model's denormalization if statistics not available
                        logger.warning("Dataset statistics not available, using model's denormalization (may be inconsistent with training)")
                        if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
                            action_norm_stats = model.norm_stats[cfg.unnorm_key]["action"]
                            q01, q99 = (
                                action_norm_stats["q01"],
                                action_norm_stats["q99"],
                            )
                            actions_pred_np = actions_pred.cpu().numpy()
                            actions_pred_denorm = (actions_pred_np + 1) * (q99 - q01) / 2 + q01
                            actions = [actions_pred_denorm[i] for i in range(len(actions_pred_denorm))]
                        else:
                            actions = [actions_pred[i].cpu().numpy() for i in range(len(actions_pred))]

                    # Optional: visualize tracking head outputs as sequence video
                    if cfg.point_visualize and tracking_head is not None:
                        pred_tracking = tracking_head.predict_tracking(actions_hidden_states)
                        pred_seq = pred_tracking[0].detach().to(torch.float32).cpu().numpy()  # (chunk_len, num_points, dim)
                        
                        # Denormalize tracking deltas if statistics are available
                        if cfg.normalize_tracking and dataset_statistics is not None:
                            pred_seq = denormalize_tracking(pred_seq, dataset_statistics)
                        
                        # Build cumulative sequence: initial, initial+delta1, initial+delta1+delta2, ...
                        # Note: init_pc is already normalized, need to denormalize it for visualization
                        init_pc_normalized = pc_tensor[0].detach().to(torch.float32).cpu().numpy()
                        
                        # Denormalize initial pointcloud for visualization
                        if cfg.normalize_pointcloud and dataset_statistics is not None:
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
                # get_action() uses model's unnorm_key for denormalization
                # We need to re-normalize and then denormalize with dataset statistics
                # actions = get_action(
                #     cfg,
                #     model,
                #     observation,
                #     task_description,
                #     processor=processor,
                #     action_head=action_head,
                #     proprio_projector=proprio_projector,
                #     noisy_action_projector=noisy_action_projector,
                #     use_film=cfg.use_film,
                # )
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    # Get normalized actions from model (unnorm_key=None to prevent model's denormalization)
                    actions_pred, normalized_actions, actions_hidden_states = model.predict_action(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                        proprio=proprio,
                        proprio_projector= None,
                        pointcloud=None,
                        pointcloud_projector= None,
                        unnorm_key=None,  # Don't use model's denormalization
                        do_sample=False,
                        use_film=cfg.use_film,
                        action_head=action_head,
                    )
                # actions_pred_np = normalized_actions.cpu().numpy()
                actions = denormalize_action(normalized_actions, dataset_statistics)
                
                # If dataset statistics available, re-normalize and denormalize with correct statistics
                # if dataset_statistics is not None and "action" in dataset_statistics:
                #     # get_action() returns denormalized actions using model's norm_stats
                #     # We need to use dataset_statistics instead
                #     # First, re-normalize using model's norm_stats
                #     if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
                #         print(f"Re-normalizing actions using model's norm_stats: {cfg.unnorm_key}")
                #         action_norm_stats = model.norm_stats[cfg.unnorm_key]["action"]
                #         q01_model = np.array(action_norm_stats["q01"])
                #         q99_model = np.array(action_norm_stats["q99"])
                        
                #         # Convert actions to numpy if needed
                #         actions_np = np.array([a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in actions])
                        
                #         # Re-normalize using model's statistics: normalized = 2 * (action - q01) / (q99 - q01) - 1
                #         actions_normalized = 2.0 * (actions_np - q01_model) / (q99_model - q01_model + 1e-8) - 1.0
                        
                #         # Denormalize using dataset statistics
                #         actions_denorm = denormalize_action(actions_normalized, dataset_statistics)
                #         actions = [actions_denorm[i] for i in range(len(actions_denorm))]
                #     else:
                #         logger.warning("Model norm_stats not available for re-normalization")
            
            actions = actions[:8]
            action_queue.extend(actions)

        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

        # refresh pointcloud
        if cfg.point_visualize :
            print('Refresh pointcloud')
            pc_np = pointcloud_from_env(
                env, cube_half=cfg.pointcloud_cube_half, num_points=cfg.pointcloud_num_points, include_table=cfg.include_table
            )
            
            # Normalize pointcloud if statistics are available and normalization is enabled
            if cfg.normalize_pointcloud and dataset_statistics is not None:
                pc_np = normalize_pointcloud(pc_np, dataset_statistics)
            
            pc_tensor = torch.from_numpy(pc_np).to(torch.bfloat16).to(device).unsqueeze(0)
            if pc_debug_dir is not None and cfg.save_pc_debug:
                save_pc_tensor_as_ply(pc_tensor, pc_debug_dir / f"pc_step_{t:04d}.ply")
            print('Refresh pointcloud done')

    # except Exception as e:
    #     log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
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
    dataset_statistics=None,
):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)
        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
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
        save_rollout_video(
            rollout_images=replay_images,
            idx=total_episodes,
            success=success,
            task_description=task_description,
            rollout_dir=os.path.join(cfg.rollout_dir, DATE),
            log_file=log_file,
        )
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )
    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    
    # Load action_chunk_size from checkpoint if available
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
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
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
            dataset_statistics,
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)
    if log_file:
        log_file.close()
    return final_success_rate


if __name__ == "__main__":
    eval_libero()
