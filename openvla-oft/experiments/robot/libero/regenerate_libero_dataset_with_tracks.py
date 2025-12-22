"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.
Additionally saves point cloud meshes and tracking information for each episode.

This script combines:
    - regenerate_libero_dataset.py: HDF5 regeneration with no-op filtering
    - export_gt_track_area.py: Point cloud mesh and tracking information export

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - We save cropped meshes and face-area-sampled point tracks for each episode.
    - Metadata includes {task_id}_{episode_idx}_{task_name} identifier for point cloud folder.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset_with_tracks.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR> \
        --point_cloud_dir <PATH TO POINT CLOUD OUTPUT DIR>

    Example (LIBERO-Goal):
        python experiments/robot/libero/regenerate_libero_dataset_with_tracks.py \
            --libero_task_suite libero_goal \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_goal \
            --libero_target_dir ./LIBERO/libero/datasets/libero_goal_no_noops \
            --point_cloud_dir ./LIBERO/libero/point_clouds/libero_goal

"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark

# Add parent directory to path for imports
# sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

# Point cloud and tracking related imports
# If these are in a separate file, import them
import sys
sys.path.append('/scratch2/jisoo6687/human_demon_action/LIBERO')
from export_gt_pointcloud import (
    collect_world_meshes,
    get_reference_center,
    center_and_crop_meshes,
    save_meshes_as_obj,
)

from export_gt_track_area import _build_tracking_points_from_faces

IMAGE_RESOLUTION = 256


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


# ===============================
#  Point Cloud & Tracking Functions (from export_gt_track_area.py)
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
    """Calculate triangle areas."""
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset with point cloud tracking!")

    os.makedirs(args.libero_target_dir, exist_ok=True)
    
    point_cloud_root = Path(args.point_cloud_dir)
    point_cloud_root.mkdir(parents=True, exist_ok=True)

    # Prepare JSON file to record success/failure and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"./experiments/robot/libero/{args.libero_task_suite}_metainfo_with_tracks.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    # num_tasks_in_suite = task_suite.n_tasks
    if args.task_index is None:
        num_tasks_in_suite = list(range(task_suite.n_tasks))
    else:
        num_tasks_in_suite = [args.task_index]

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0
    print(f"num_tasks_in_suite: {list(range(task_suite.n_tasks))}")

    for task_id in tqdm.tqdm(num_tasks_in_suite):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        if os.path.exists(new_data_path):
            print(f"Skipping task {task.name} because it already exists.")
            continue
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for i in range(len(orig_data.keys())):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            # for point tracking
            sim = env.sim
            geom_local_verts: Optional[Dict[int, np.ndarray]] = None
            point_meta: Optional[List[Dict[str, Any]]] = None
            cached_face_meta: Optional[List[Dict[str, Any]]] = None
            cached_face_areas: Optional[List[float]] = None
            cached_pose_seq: List[Dict[str, Any]] = []
            tracks_per_step: List[np.ndarray] = []

            ref_center: Optional[np.ndarray] = None
            step_done = 0

            # Create identifier: {task_id}_{episode_idx}_{task_name}
            point_cloud_identifier = f"{task_id}_{i}_{task.name}"
            point_cloud_dir = point_cloud_root / point_cloud_identifier
            os.makedirs(Path(point_cloud_dir), exist_ok=True)
            os.makedirs(Path(f'{point_cloud_dir}/cropped_scene'), exist_ok=True)

            # Replay original demo actions in environment and record observations
            for k, action in enumerate(orig_actions):
                if k==0 :
                    mesh_path = f'{point_cloud_dir}/cropped_scene/step_0000.obj'
                    meshes = collect_world_meshes(
                        env,
                        include_robot=True,
                        include_statics=True,
                        exclude_body_substrings=(),
                    )
                    ref_center = get_reference_center(meshes, keyword="table")
                    # print(f'keys : {[m["name"] for m in meshes]}')
                    # 1/0
                    if args.exclude_wall :
                        filtered = [m for m in meshes if "world_geom" not in m["name"].lower() and "mount0" not in m["name"].lower()]
                    elif args.exclude_table :
                        filtered = [m for m in meshes if "table" not in m["name"].lower()]
                    else :
                        filtered = meshes
                    cropped = center_and_crop_meshes(filtered, ref_center, args.cube_half)
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
                            cube_half=args.cube_half,
                            include_table=not args.exclude_table,
                            max_points=args.max_track_points,
                            include_wall=not args.exclude_wall,
                            table_weight=args.table_weight,
                        )
                        num_points = len(point_meta or [])

                    geom_world = _compute_geom_world_verts(sim, geom_local_verts or {})
                    step_pts = np.zeros((num_points, 3), dtype=np.float32)
                    for j, meta in enumerate(point_meta or []):
                        g = meta["geom_id"]
                        idxs = meta["vert_indices"]
                        w = meta["barycentric"]
                        tri = geom_world[g][idxs]
                        step_pts[j] = w[0] * tri[0] + w[1] * tri[1] + w[2] * tri[2]
                    tracks_per_step.append(step_pts)
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])


                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())
                # save mesh and pointrack
                mesh_path = f'{point_cloud_dir}/cropped_scene/step_{k+1:04d}.obj'
                meshes = collect_world_meshes(
                    env,
                    include_robot=True,
                    include_statics=True,
                    exclude_body_substrings=(),
                )
                ref_center = get_reference_center(meshes, keyword="table")
                if args.exclude_wall :
                    filtered = [m for m in meshes if "world_geom" not in m["name"].lower() and "mount0" not in m["name"].lower()]
                elif args.exclude_table :
                    filtered = [m for m in meshes if "table" not in m["name"].lower()]
                else :
                    filtered = meshes
                cropped = center_and_crop_meshes(filtered, ref_center, args.cube_half)
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
                        cube_half=args.cube_half,
                        include_table=not args.exclude_table,
                        max_points=args.max_track_points,
                        include_wall=not args.exclude_wall,
                        table_weight=args.table_weight,
                    )
                    num_points = len(point_meta or [])

                geom_world = _compute_geom_world_verts(sim, geom_local_verts or {})
                step_pts = np.zeros((num_points, 3), dtype=np.float32)
                for j, meta in enumerate(point_meta or []):
                    g = meta["geom_id"]
                    idxs = meta["vert_indices"]
                    w = meta["barycentric"]
                    tri = geom_world[g][idxs]
                    step_pts[j] = w[0] * tri[0] + w[1] * tri[1] + w[2] * tri[2]
                tracks_per_step.append(step_pts)   

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                # Save point cloud and tracking information
                print(f'Done {i} and save')

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                # Save point cloud identifier as string dataset
                obs_grp.create_dataset("point_meta", data=point_cloud_identifier, dtype=h5py.string_dtype(encoding='utf-8'))
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                # save
                vertex_tracks = np.stack(tracks_per_step, axis=0) if tracks_per_step else np.zeros((0, 0, 3))

                actions_path = Path(f'{point_cloud_dir}/actions.npy')
                if not actions_path.exists():
                    np.save(actions_path, actions)

                # save face pool, geom poses, and local verts for offline resampling
                if cached_face_meta is not None and cached_face_areas is not None and geom_local_verts is not None:
                    pool_meta_path = point_cloud_dir / "face_pool_meta.json"
                    pool_area_path = point_cloud_dir / "face_pool_areas.npy"
                    pose_path = point_cloud_dir / "geom_pose_seq.json"
                    verts_path = point_cloud_dir / "geom_local_verts.npz"
                    with pool_meta_path.open("w", encoding="utf-8") as f:
                        json.dump(cached_face_meta, f, indent=2)
                    np.save(pool_area_path, np.asarray(cached_face_areas, dtype=np.float32))
                    with pose_path.open("w", encoding="utf-8") as f:
                        json.dump(cached_pose_seq, f)
                    np.savez(verts_path, **{str(k): v for k, v in geom_local_verts.items()})

                meta_path = point_cloud_dir / "metadata_face_uniform.json"
                metadata = {
                    "_traj_index": point_cloud_identifier,
                    "_len": len(actions),
                    "actions_file": actions_path.name,
                    "num_track_points": int(vertex_tracks.shape[1]),
                }
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                tracks_path = point_cloud_dir / "vertex_tracks_face_uniform.npy"
                ids_path = point_cloud_dir / "vertex_ids_face_uniform.json"

                np.save(tracks_path, vertex_tracks)
                with ids_path.open("w", encoding="utf-8") as f:
                    json.dump(point_meta or [], f, indent=2)

                num_success += 1

            num_replays += 1

            # Record success/failure and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()
            
            # Add point cloud identifier
            if done:
                metainfo_json_dict[task_key][episode_key]["point_cloud_id"] = f"{task_id}_{i}_{task.name}"

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"\nDataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved point cloud data at: {args.point_cloud_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, 
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", 
                        required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", 
                        required=True)
    parser.add_argument("--point_cloud_dir", type=str,
                        help="Path to point cloud output directory. Example: ./LIBERO/libero/point_clouds/libero_spatial",
                        required=True)
    # Point cloud and tracking arguments
    parser.add_argument("--cube_half", type=float, default=0.5,
                        help="Half-edge length (meters) of the cube crop around the table center.")
    parser.add_argument("--max_track_points", type=int, default=5000,
                        help="Max number of points to track (sampled uniformly over triangle area).")
    parser.add_argument("--exclude_table", action="store_true",
                        help="If set, remove meshes whose name contains 'table' from the saved outputs.")
    parser.add_argument("--exclude_wall", action="store_true",
                        help="If set, remove meshes whose name contains 'world' or 'mount0' from the saved outputs.")
    parser.add_argument("--table_weight", type=float, default=1.0,
                        help="Multiplier for table face areas during sampling (<1 reduces table samples).")
    parser.add_argument("--task_index", type=int, default=None,
                        help="Task index to regenerate.")
    
    args = parser.parse_args()

    # Start data regeneration
    main(args)

