"""
Regenerates RoboCasa dataset (HDF5 files) by converting to the target format.

This script converts RoboCasa datasets to match the format used in LIBERO dataset (regenerate_libero_dataset.py).
Unlike the LIBERO version, this does NOT:
    - Filter out no-op actions
    - Filter out unsuccessful demonstrations
    - Replay demos in environment

It simply reformats the data structure to match the expected format (obs_grp, ep_data_grp).

Usage:
    python experiments/robot/robocasa/regenerate_robocasa_dataset.py \
        --robocasa_raw_data_dir /weka/jisookim/dataset/robocasa/datasets/v0.1/single_stage \
        --robocasa_target_dir /weka/jisookim/dataset/robocasa/datasets/v0.1/single_stage_reformatted

"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm


def convert_hdf5_format(input_path, output_path):
    """
    Converts a single HDF5 file from RoboCasa format to LIBERO-style format.

    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
    """
    print(f"Converting {input_path}")

    # Open input file
    with h5py.File(input_path, "r") as in_file:
        # Create output file
        with h5py.File(output_path, "w") as out_file:
            # Create data group
            grp = out_file.create_group("data")

            # Get input data group
            in_data = in_file["data"]

            # Process each demo
            demo_keys = sorted([k for k in in_data.keys() if k.startswith("demo_")],
                             key=lambda x: int(x.split('_')[1]))

            num_demos = len(demo_keys)
            print(f"  Found {num_demos} demos")

            for demo_key in tqdm.tqdm(demo_keys, desc=f"  Processing"):
                in_demo = in_data[demo_key]
                in_obs = in_demo["obs"]

                # Create output demo group with same naming
                ep_data_grp = grp.create_group(demo_key)

                # Create obs group (matching LIBERO format)
                obs_grp = ep_data_grp.create_group("obs")

                # Copy observation data to obs group
                # RoboCasa specific: has 3 RGB cameras
                obs_grp.create_dataset("agentview_left_rgb",
                                      data=in_obs["robot0_agentview_left_image"][()])
                obs_grp.create_dataset("agentview_right_rgb",
                                      data=in_obs["robot0_agentview_right_image"][()])
                obs_grp.create_dataset("eye_in_hand_rgb",
                                      data=in_obs["robot0_eye_in_hand_image"][()])

                # Copy robot state observations
                obs_grp.create_dataset("gripper_states",
                                      data=in_obs["robot0_gripper_qpos"][()])
                obs_grp.create_dataset("joint_states",
                                      data=in_obs["robot0_joint_pos"][()])

                # Create ee_states from eef_pos and eef_quat (convert quat to axis-angle)
                eef_pos = in_obs["robot0_eef_pos"][()]
                eef_quat = in_obs["robot0_eef_quat"][()]

                # Convert quaternion to axis-angle for each timestep
                ee_states = []
                for i in range(len(eef_pos)):
                    ee_state = np.hstack((
                        eef_pos[i],
                        T.quat2axisangle(eef_quat[i])
                    ))
                    ee_states.append(ee_state)
                ee_states = np.stack(ee_states, axis=0)

                obs_grp.create_dataset("ee_states", data=ee_states)
                obs_grp.create_dataset("ee_pos", data=eef_pos)
                obs_grp.create_dataset("ee_ori", data=ee_states[:, 3:])

                # Copy actions, states, rewards, dones to demo group (not obs group)
                ep_data_grp.create_dataset("actions", data=in_demo["actions"][()])
                ep_data_grp.create_dataset("states", data=in_demo["states"][()])

                # Create robot_states from gripper_qpos, eef_pos, eef_quat
                # Matching LIBERO format: [gripper_qpos, eef_pos, eef_quat]
                gripper_qpos = in_obs["robot0_gripper_qpos"][()]
                robot_states = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=1)
                ep_data_grp.create_dataset("robot_states", data=robot_states)

                ep_data_grp.create_dataset("rewards", data=in_demo["rewards"][()])
                ep_data_grp.create_dataset("dones", data=in_demo["dones"][()])

    print(f"  Saved to {output_path}")


def main(args):
    print(f"Converting RoboCasa dataset")
    print(f"Source: {args.robocasa_raw_data_dir}")
    print(f"Target: {args.robocasa_target_dir}")
    print()

    # Create target directory
    if os.path.isdir(args.robocasa_target_dir):
        user_input = input(
            f"Target directory already exists at path: {args.robocasa_target_dir}\n"
            f"Enter 'y' to overwrite, or anything else to exit: "
        )
        if user_input != 'y':
            exit()
    os.makedirs(args.robocasa_target_dir, exist_ok=True)

    # Get all subdirectories (task categories)
    raw_data_path = Path(args.robocasa_raw_data_dir)

    if not raw_data_path.exists():
        print(f"Error: Raw data directory does not exist: {args.robocasa_raw_data_dir}")
        return

    # Get all task directories (kitchen_coffee, kitchen_doors, etc.)
    task_dirs = [d for d in raw_data_path.iterdir() if d.is_dir()]

    if len(task_dirs) == 0:
        print(f"No task directories found in {args.robocasa_raw_data_dir}")
        return

    print(f"Found {len(task_dirs)} task categories:")
    for d in sorted(task_dirs):
        print(f"  - {d.name}")
    print()

    total_files = 0

    # Process each task directory
    for task_dir in sorted(task_dirs):
        print(f"{'='*60}")
        print(f"Processing task category: {task_dir.name}")
        print(f"{'='*60}")

        # Find all HDF5 files recursively
        hdf5_files = list(task_dir.rglob("*.hdf5"))

        if len(hdf5_files) == 0:
            print(f"  No HDF5 files found in {task_dir}")
            continue

        print(f"Found {len(hdf5_files)} HDF5 files in {task_dir.name}")
        print()

        # Convert each HDF5 file
        for hdf5_file in sorted(hdf5_files):
            # Create output path maintaining subdirectory structure
            relative_path = hdf5_file.relative_to(task_dir)
            output_file = Path(args.robocasa_target_dir) / task_dir.name / relative_path

            # Create parent directories if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)

            convert_hdf5_format(str(hdf5_file), str(output_file))
            total_files += 1

        print()

    print(f"{'='*60}")
    print(f"Dataset conversion complete!")
    print(f"Converted {total_files} HDF5 files")
    print(f"Output saved to: {args.robocasa_target_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robocasa_raw_data_dir",
        type=str,
        help="Path to directory containing raw RoboCasa HDF5 dataset",
        required=True
    )
    parser.add_argument(
        "--robocasa_target_dir",
        type=str,
        help="Path to target directory for reformatted dataset",
        required=True
    )
    args = parser.parse_args()

    # Start data conversion
    main(args)
