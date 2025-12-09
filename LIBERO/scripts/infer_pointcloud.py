import argparse
import glob
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    imageio = None
    plt = None

from prismatic.models.projectors import PointcloudProjector, ProprioProjector
from prismatic.models.action_heads import PointTrackingHead
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.datasets.rlds.dataset import make_interleaved_dataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction


def pad_or_trim_pointcloud(pc: torch.Tensor, max_points: int, dim: int) -> torch.Tensor:
    if pc.shape[1] > dim:
        pc = pc[:, :dim]
    elif pc.shape[1] < dim:
        pad_dim = torch.zeros(pc.shape[0], dim - pc.shape[1], dtype=pc.dtype, device=pc.device)
        pc = torch.cat([pc, pad_dim], dim=1)
    if pc.shape[0] > max_points:
        pc = pc[:max_points]
    elif pc.shape[0] < max_points:
        pad = torch.zeros(max_points - pc.shape[0], pc.shape[1], dtype=pc.dtype, device=pc.device)
        pc = torch.cat([pc, pad], dim=0)
    return pc


def build_pointcloud_loader(pc_root: Path, subdir: str, ext: str, max_points: int, dim: int):
    def loader(rlds_batch):
        obs = rlds_batch.get("observation", {})
        traj_indices = obs.get("traj_index")
        frame_indices = obs.get("frame_index")
        timesteps = obs.get("timestep")
        ep_idx = int(traj_indices[0]) if traj_indices is not None and len(traj_indices) > 0 else None
        step_idx = (
            int(frame_indices[0])
            if frame_indices is not None and len(frame_indices) > 0
            else (int(timesteps[0]) if timesteps is not None and len(timesteps) > 0 else None)
        )
        if ep_idx is None or step_idx is None:
            return None
        ep_id = f"episode_{ep_idx:05d}"
        pc_path = pc_root / ep_id / subdir / f"step_{step_idx:04d}{ext}"
        print(pc_path)
        if not pc_path.exists():
            return None
        pc_o3d = o3d.io.read_point_cloud(str(pc_path))
        pc = torch.from_numpy(np.asarray(pc_o3d.points)).float()
        return pad_or_trim_pointcloud(pc, max_points, dim)

    return loader


def build_pointcloud_loader_from_tracks(tracks_root: Path, filename: str, max_points: int, dim: int):
    def loader(rlds_batch):
        obs = rlds_batch.get("observation", {})
        traj_indices = obs.get("traj_index")
        frame_indices = obs.get("frame_index")
        timesteps = obs.get("timestep")
        ep_idx = int(traj_indices[0]) if traj_indices is not None and len(traj_indices) > 0 else None
        step_idx = (
            int(frame_indices[0])
            if frame_indices is not None and len(frame_indices) > 0
            else (int(timesteps[0]) if timesteps is not None and len(timesteps) > 0 else None)
        )
        if ep_idx is None or step_idx is None:
            return None
        ep_id = f"episode_{ep_idx:05d}"
        track_path = tracks_root / ep_id / filename
        if not track_path.exists():
            return None
        tracks = torch.from_numpy(np.load(track_path)).float()
        if step_idx >= tracks.shape[0]:
            return None
        pc = tracks[step_idx]
        return pad_or_trim_pointcloud(pc, max_points, dim)

    return loader


def load_state(path_glob: str):
    matches = sorted(glob.glob(path_glob))
    if not matches:
        return None
    state = torch.load(matches[-1], map_location="cpu")
    # Strip potential DDP "module." prefix
    if isinstance(state, dict) and all(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def save_ply(points: np.ndarray, path: Path):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path), cloud)


def save_rotation_video(
    points: np.ndarray,
    video_path: Path,
    num_frames: int = 120,
    fps: int = 30,
    elev: float = 20.0,
    dist: float = 1.5,
) -> None:
    if imageio is None or plt is None:
        raise ImportError("imageio and matplotlib are required for video export. Please `pip install imageio matplotlib`.")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    pts = points - points.mean(axis=0, keepdims=True)
    max_range = np.linalg.norm(pts, axis=1).max()
    ax.set_xlim3d([-max_range, max_range])
    ax.set_ylim3d([-max_range, max_range])
    ax.set_zlim3d([-max_range, max_range])

    scatter = ax.scatter([], [], [], s=1)
    writer = imageio.get_writer(video_path, fps=fps)

    for i in range(num_frames):
        azim = 360.0 * i / num_frames
        ax.view_init(elev=elev, azim=azim, roll=0)
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
    writer.close()
    plt.close(fig)


def save_static_video(
    points: np.ndarray,
    video_path: Path,
    num_frames: int = 120,
    fps: int = 30,
    elev: float = 20.0,
    azim: float = 45.0,
) -> None:
    if imageio is None or plt is None:
        raise ImportError("imageio and matplotlib are required for video export. Please `pip install imageio matplotlib`.")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    pts = points - points.mean(axis=0, keepdims=True)
    max_range = np.linalg.norm(pts, axis=1).max()
    ax.set_xlim3d([-max_range, max_range])
    ax.set_ylim3d([-max_range, max_range])
    ax.set_zlim3d([-max_range, max_range])

    scatter = ax.scatter([], [], [], s=1)
    writer = imageio.get_writer(video_path, fps=fps)

    # Fixed view; just repeat the same frame
    ax.view_init(elev=elev, azim=azim, roll=0)
    scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    for _ in range(num_frames):
        writer.append_data(frame)
    writer.close()
    plt.close(fig)


def save_sequence_video(
    points_seq: np.ndarray,
    video_path: Path,
    fps: int = 5,
    elev: float = 20.0,
    azim: float = 45.0,
) -> None:
    """
    Render a sequence of point clouds (T, N, 3) from a fixed view into a video.
    """
    if imageio is None or plt is None:
        raise ImportError("imageio and matplotlib are required for video export. Please `pip install imageio matplotlib`.")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    # Normalize across entire sequence for consistent scale
    pts_all = points_seq.reshape(-1, 3) - points_seq.reshape(-1, 3).mean(axis=0, keepdims=True)
    max_range = np.linalg.norm(pts_all, axis=1).max()
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


def infer_and_save(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    adapter_dir = checkpoint_dir / "lora_adapter"

    # Load processor
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load VLA (merged model takes priority)
    try:
        vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_loaded = True
    except Exception:
        base_model = AutoModelForVision2Seq.from_pretrained(
            args.base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        vla = PeftModel.from_pretrained(base_model, adapter_dir)
        merged_loaded = False

    vla = vla.to(device)
    vla.eval()
    vla.vision_backbone.set_num_images_in_input(args.num_images_in_input)

    # Projectors
    proprio_projector = None
    if args.use_proprio:
        proprio_state = load_state(str(checkpoint_dir / "proprio_projector--*.pt"))
        proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
        if proprio_state:
            proprio_projector.load_state_dict(proprio_state)
        proprio_projector = proprio_projector.to(device).to(torch.bfloat16)

    pointcloud_projector = PointcloudProjector(
        llm_dim=vla.llm_dim, num_points=args.pointcloud_input_num_points, point_dim=args.pointcloud_input_dim
    )
    pc_proj_state = load_state(str(checkpoint_dir / "pointcloud_projector--*.pt"))
    if pc_proj_state:
        pointcloud_projector.load_state_dict(pc_proj_state)
    pointcloud_projector = pointcloud_projector.to(device).to(torch.bfloat16)

    tracking_head = PointTrackingHead(
        input_dim=vla.llm_dim,
        hidden_dim=vla.llm_dim,
        num_points=args.tracking_num_points,
        tracking_dim=args.tracking_dim,
    )
    tracking_state = load_state(str(checkpoint_dir / "tracking_head--*.pt"))
    if tracking_state:
        tracking_head.load_state_dict(tracking_state)
    tracking_head = tracking_head.to(device).to(torch.bfloat16)

    # Data
    pointcloud_loader = None
    if args.use_pointcloud_from_tracks:
        assert args.tracking_tracks_root is not None, "tracking_tracks_root is required when use_pointcloud_from_tracks=True"
        pointcloud_loader = build_pointcloud_loader_from_tracks(
            Path(args.tracking_tracks_root),
            args.tracking_tracks_filename,
            args.pointcloud_input_num_points,
            args.pointcloud_input_dim,
        )
    else:
        assert args.pointcloud_root is not None, "pointcloud_root is required when not using pointcloud_from_tracks"
        pointcloud_loader = build_pointcloud_loader(
            Path(args.pointcloud_root),
            args.pointcloud_subdir,
            args.pointcloud_ext,
            args.pointcloud_input_num_points,
            args.pointcloud_input_dim,
        )

    batch_transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=args.num_images_in_input > 1,
        use_proprio=args.use_proprio,
        pointcloud_from_disk_fn=pointcloud_loader,
        pointcloud_num_points=args.pointcloud_input_num_points,
        pointcloud_dim=args.pointcloud_input_dim,
        tracking_from_disk_fn=None,
        tracking_num_points=args.tracking_num_points,
        tracking_dim=args.tracking_dim,
    )

    # Minimal dataset: use RLDSDataset to reuse transforms; single batch
    train_dataset = RLDSDataset(
        Path(args.data_root_dir),
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.vision_backbone.featurizer.patch_embed.img_size),
        shuffle_buffer_size=1_000,
        image_aug=False,
        train=True,
    )
    save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(train_dataset, batch_size=1, sampler=None, collate_fn=collator, num_workers=0)
    batch = next(iter(dataloader))

    # Forward
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=batch["proprio"].to(device) if args.use_proprio else None,
            proprio_projector=proprio_projector if args.use_proprio else None,
            pointcloud=batch["pointcloud"].to(device),
            pointcloud_projector=pointcloud_projector,
            use_film=False,
        )

    # Extract hidden states for actions
    num_patches = (
        vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        + (1 if args.use_proprio else 0)
        + 1  # pointcloud token
    )
    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    mask = current_action_mask | next_actions_mask

    # Align mask length to hidden state length if needed
    if mask.dim() > 2:
        mask = mask.view(mask.shape[0], -1)
    if mask.shape[1] != text_hidden_states.shape[1]:
        min_len = min(mask.shape[1], text_hidden_states.shape[1])
        mask = mask[:, :min_len]
        text_slice = text_hidden_states[:, :min_len]
    else:
        text_slice = text_hidden_states

    if mask.sum() == 0:
        raise ValueError("Empty action mask")

    actions_hidden_states = (
        text_slice[mask]
        .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    # PointTrackingHead defines predict_tracking (forward is not implemented)
    predicted_tracking = tracking_head.predict_tracking(actions_hidden_states)  # (1, chunk_len, num_points, dim)
    predicted_seq = predicted_tracking[0].detach().float().cpu().numpy()  # (chunk_len, num_points, dim)

    # Save first timestep as PLY for quick inspection
    predicted_points = predicted_seq[0]
    save_ply(predicted_points, Path(args.output_ply))
    print(f"Saved predicted pointcloud (t=0) to {args.output_ply}")

    if args.video_path is not None:
        save_rotation_video(
            predicted_points,
            Path(args.video_path),
            num_frames=args.video_frames,
            fps=args.video_fps,
            elev=args.video_elev,
            dist=1.5,
        )
        print(f"Saved rotating pointcloud video to {args.video_path}")

    if args.static_video_path is not None:
        save_static_video(
            predicted_points,
            Path(args.static_video_path),
            num_frames=args.video_frames,
            fps=args.video_fps,
            elev=args.video_elev,
            azim=args.video_azim,
        )
        print(f"Saved static-view pointcloud video to {args.static_video_path}")

    if args.sequence_video_path is not None:
        save_sequence_video(
            predicted_seq,
            Path(args.sequence_video_path),
            fps=args.video_fps,
            elev=args.video_elev,
            azim=args.video_azim,
        )
        print(f"Saved sequence pointcloud video to {args.sequence_video_path}")

    # Optional: reconstruct absolute positions by accumulating predicted deltas on top of input pointcloud
    if args.reconstruct_sequence_path is not None:
        init_pc = batch["pointcloud"][0].detach().cpu().numpy()  # (num_points, dim)
        recon_seq = [init_pc]
        curr = init_pc
        for delta in predicted_seq:
            curr = curr + delta
            recon_seq.append(curr)
        recon_seq = np.stack(recon_seq, axis=0)  # (chunk_len+1, num_points, dim)
        save_sequence_video(
            recon_seq,
            Path(args.reconstruct_sequence_path),
            fps=args.video_fps,
            elev=args.video_elev,
            azim=args.video_azim,
        )
        # Also drop first frame as PLY for debugging
        save_ply(recon_seq[0], Path(args.output_ply).with_name("reconstructed_t0.ply"))
        save_ply(recon_seq[-1], Path(args.output_ply).with_name("reconstructed_last.ply"))
        print(f"Saved reconstructed sequence video to {args.reconstruct_sequence_path}")


def main():
    parser = argparse.ArgumentParser(description="Run pointcloud inference and save predicted pointcloud to PLY.")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory (run or chkpt folder).")
    parser.add_argument("--base_model_path", default="openvla/openvla-7b", help="Base model path if adapter is separate.")
    parser.add_argument("--data_root_dir", required=True, help="RLDS root.")
    parser.add_argument("--dataset_name", required=True, help="Dataset name (e.g., libero_goal_no_noops).")
    parser.add_argument("--pointcloud_root", default=None, help="Pointcloud root (ply/npy) for input token.")
    parser.add_argument("--pointcloud_subdir", default="pointclouds_512")
    parser.add_argument("--pointcloud_ext", default=".ply")
    parser.add_argument("--tracking_num_points", type=int, default=256)
    parser.add_argument("--tracking_dim", type=int, default=3)
    parser.add_argument("--pointcloud_input_num_points", type=int, default=256)
    parser.add_argument("--pointcloud_input_dim", type=int, default=3)
    parser.add_argument("--use_pointcloud_from_tracks", action="store_true")
    parser.add_argument("--tracking_tracks_root", default=None, help="Root containing per-episode track npy (T, N, 3).")
    parser.add_argument("--tracking_tracks_filename", default="vertex_tracks.npy")
    parser.add_argument("--num_images_in_input", type=int, default=1)
    parser.add_argument("--use_proprio", action="store_true")
    parser.add_argument("--output_ply", default="predicted_pointcloud.ply")
    parser.add_argument("--video_path", default=None, help="If set, save a rotating video (mp4) of the predicted pointcloud.")
    parser.add_argument(
        "--static_video_path", default=None, help="If set, save a fixed-view video (mp4) of the predicted pointcloud."
    )
    parser.add_argument(
        "--sequence_video_path",
        default=None,
        help="If set, save a fixed-view video over the predicted pointcloud sequence (t dimension).",
    )
    parser.add_argument(
        "--reconstruct_sequence_path",
        default=None,
        help="If set, accumulate predicted deltas on the input pointcloud and save the reconstructed sequence video.",
    )
    parser.add_argument("--video_frames", type=int, default=120)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_elev", type=float, default=20.0)
    parser.add_argument("--video_azim", type=float, default=45.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    infer_and_save(args)


if __name__ == "__main__":
    main()
