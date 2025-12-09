import argparse
import glob
import multiprocessing as mp
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm


def load_mesh(obj_path):
    # print(f"[INFO] Loading mesh: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    return mesh


def sample_pcd(mesh, n_points=200000, method="poisson"):
    print(f"[INFO] Sampling {n_points} points using {method} method...")

    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=n_points,
            init_factor=5
        )
    elif method == "uniform":
        pcd = mesh.sample_points_uniformly(
            number_of_points=n_points
        )
    else:
        raise ValueError("method must be 'poisson' or 'uniform'")

    print(f"[INFO] Resulting point cloud has {np.asarray(pcd.points).shape[0]} points")
    return pcd


def save_pcd(pcd, save_path):
    print(f"[INFO] Saving PLY: {save_path}")
    o3d.io.write_point_cloud(save_path, pcd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample point clouds from cropped mesh OBJ files."
    )
    parser.add_argument(
        "--mesh-root",
        default="/mnt/data/libero/modified_libero_wotable_mesh/1.0.0",
        help="Root directory that contains scene folders with cropped meshes.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPU workers to shard the work across.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit GPU ids to assign per worker (defaults to [0, ..., num_gpus-1]).",
    )
    parser.add_argument(
        "--method",
        choices=["poisson", "uniform"],
        default="poisson",
        help="Sampling method to use for point cloud generation.",
    )
    return parser.parse_args()


def ensure_output_dirs(obj_path: str, specs: Sequence[Tuple[int, str]]) -> None:
    for _, dirname in specs:
        os.makedirs(os.path.join(obj_path, dirname), exist_ok=True)


def process_obj_path(
    obj_path: str, specs: Sequence[Tuple[int, str]], method: str, progress_queue: Optional[mp.Queue] = None
) -> None:
    steps = sorted(glob.glob(f"{obj_path}/cropped_scene/*"))
    ensure_output_dirs(obj_path, specs)
    for step in steps:
        mesh = load_mesh(step)
        for n_points, dirname in specs:
            output_path = os.path.join(obj_path, dirname, os.path.basename(step).replace(".obj", ".ply"))
            if os.path.exists(output_path):
                continue
            pcd = sample_pcd(mesh, n_points=n_points, method=method)
            save_pcd(
                pcd,output_path,
            )
        if progress_queue is not None:
            progress_queue.put(1)


def worker_main(
    rank: int,
    obj_paths: List[str],
    specs: Sequence[Tuple[int, str]],
    method: str,
    gpu_id: int,
    progress_queue: Optional[mp.Queue] = None,
) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Worker {rank}] Assigned GPU {gpu_id}")
    desc = f"worker-{rank}"
    for obj_path in tqdm(obj_paths, desc=desc):
        process_obj_path(obj_path, specs, method, progress_queue)


def shard_paths(items: List[str], num_shards: int) -> List[List[str]]:
    return [items[i::num_shards] for i in range(num_shards)]


def main():
    args = parse_args()
    obj_paths = sorted(glob.glob(os.path.join(args.mesh_root, "*")))
    obj_paths = obj_paths[len(obj_paths)//5*4:]
    if not obj_paths:
        raise FileNotFoundError(f"No scene folders found under {args.mesh_root}")

    if args.num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")

    gpu_ids: List[int] = (
        list(range(args.num_gpus)) if args.gpu_ids is None else list(args.gpu_ids)
    )
    if len(gpu_ids) < args.num_gpus:
        raise ValueError("gpu_ids length must match num_gpus")

    specs: Sequence[Tuple[int, str]] = (
        (512, "pointclouds_512"),
        # (256, "pointclouds_256"),
        # (128, "pointclouds_128"),
    )

    total_steps = sum(len(glob.glob(f"{p}/cropped_scene/*")) for p in obj_paths)
    if total_steps == 0:
        raise FileNotFoundError("No cropped_scene steps found to process.")
    print(f"[INFO] Total steps to process: {total_steps}")

    shards = shard_paths(obj_paths, args.num_gpus)
    ctx = mp.get_context("spawn")
    progress_queue: mp.Queue = ctx.Queue()
    procs = []
    for rank, shard in enumerate(shards):
        if not shard:
            continue
        p = ctx.Process(
            target=worker_main,
            args=(rank, shard, specs, args.method, gpu_ids[rank], progress_queue),
        )
        p.start()
        procs.append(p)

    with tqdm(total=total_steps, desc="overall") as pbar:
        completed = 0
        while completed < total_steps:
            steps_done = progress_queue.get()
            completed += steps_done
            pbar.update(steps_done)

    for p in procs:
        p.join()

    print("[DONE] Point cloud saved!")


if __name__ == "__main__":
    main()
