#!/usr/bin/env python
"""
Convert LIBERO HDF5 datasets: Action rotation only (Euler → 6D)
Keep observations in Euler format (matching DROID)

Original: actions [7] = pos(3) + rot_euler(3) + gripper(1)
Converted: actions [10] = pos(3) + rot_6d(6) + gripper(1)

Observations stay as Euler!
"""

import h5py
import numpy as np
import os
import sys
from tqdm import tqdm

# Add robomimic to path
sys.path.insert(0, '/home/jisookim/human_demon_action/droid_policy_learning')
from robomimic.utils.torch_utils import euler_angles_to_rot_6d
import torch

def convert_euler_to_6d(euler_angles):
    """Convert Euler angles (N, 3) to 6D rotation (N, 6)"""
    euler_tensor = torch.FloatTensor(euler_angles)
    rot_6d_tensor = euler_angles_to_rot_6d(euler_tensor, convention="XYZ")
    return rot_6d_tensor.numpy()

def convert_hdf5_file(input_path, output_path):
    """Convert a single HDF5 file: actions Euler → 6D, keep observations as Euler"""
    
    print(f"\nConverting: {os.path.basename(input_path)}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    
    # Open input file
    with h5py.File(input_path, 'r') as f_in:
        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            # Copy metadata
            f_out.attrs.update(f_in.attrs)
            
            # Create data group
            data_in = f_in['data']
            data_out = f_out.create_group('data')
            data_out.attrs.update(data_in.attrs)
            
            # Process each demonstration
            for demo_name in tqdm(list(data_in.keys()), desc="  Demos"):
                demo_in = data_in[demo_name]
                demo_out = data_out.create_group(demo_name)
                demo_out.attrs.update(demo_in.attrs)
                
                # Copy and process observations
                obs_in = demo_in['obs']
                obs_out = demo_out.create_group('obs')
                for obs_key in obs_in.keys():
                    if obs_key == 'gripper_states':
                        # Convert 2D gripper to 1D (matching DROID)
                        # LIBERO: [left, right] symmetric parallel jaw
                        # DROID: single gripper width/opening value
                        gripper_2d = obs_in[obs_key][()]  # (T, 2)
                        # Use width: left - right (standard for parallel jaw grippers)
                        # Physical meaning: gripper opening distance
                        # 0 = closed (grasping), large = open (releasing)
                        gripper_width = (gripper_2d[:, 0] - gripper_2d[:, 1]).reshape(-1, 1)  # (T, 1)
                        obs_out.create_dataset(obs_key, data=gripper_width)
                    else:
                        obs_out.create_dataset(obs_key, data=obs_in[obs_key][()])
                
                # Copy and process next_obs (keep Euler!)
                if 'next_obs' in demo_in:
                    next_obs_in = demo_in['next_obs']
                    next_obs_out = demo_out.create_group('next_obs')
                    for obs_key in next_obs_in.keys():
                        if obs_key == 'gripper_states':
                            # Convert 2D gripper to 1D using width
                            gripper_2d = next_obs_in[obs_key][()]
                            gripper_width = (gripper_2d[:, 0] - gripper_2d[:, 1]).reshape(-1, 1)
                            next_obs_out.create_dataset(obs_key, data=gripper_width)
                        else:
                            next_obs_out.create_dataset(obs_key, data=next_obs_in[obs_key][()])
                
                # Convert actions ONLY: [7] → [10]
                actions_euler = demo_in['actions'][()]  # Shape: (T, 7)
                T = actions_euler.shape[0]
                
                # Split: pos(3) + rot_euler(3) + gripper(1)
                pos = actions_euler[:, :3]
                rot_euler = actions_euler[:, 3:6]
                gripper = actions_euler[:, 6:7]
                
                # Convert rotation to 6D
                rot_6d = convert_euler_to_6d(rot_euler)  # Shape: (T, 6)
                
                # Concatenate: pos(3) + rot_6d(6) + gripper(1)
                actions_10d = np.concatenate([pos, rot_6d, gripper], axis=1)  # Shape: (T, 10)
                
                # Save converted actions
                demo_out.create_dataset('actions', data=actions_10d)
                
                # Copy other data (rewards, dones, etc.)
                for key in demo_in.keys():
                    if key not in ['obs', 'next_obs', 'actions']:
                        demo_out.create_dataset(key, data=demo_in[key][()])
    
    print(f"  ✓ Converted successfully!")
    
    # Verify
    with h5py.File(output_path, 'r') as f:
        demo_0 = f['data'][list(f['data'].keys())[0]]
        actions_shape = demo_0['actions'].shape
        ee_ori_shape = demo_0['obs']['ee_ori'].shape
        ee_states_shape = demo_0['obs']['ee_states'].shape
        
        print(f"  Verification:")
        gripper_shape = demo_0['obs']['gripper_states'].shape
        
        print(f"    - actions shape:   {actions_shape}, expected (T, 10) {'✓' if actions_shape[1] == 10 else '✗'}")
        print(f"    - ee_ori shape:    {ee_ori_shape}, expected (T, 3) {'✓' if ee_ori_shape[1] == 3 else '✗'}")
        print(f"    - ee_states shape: {ee_states_shape}, expected (T, 6) {'✓' if ee_states_shape[1] == 6 else '✗'}")
        print(f"    - gripper shape:   {gripper_shape}, expected (T, 1) {'✓' if gripper_shape[1] == 1 else '✗'}")
        
        assert actions_shape[1] == 10, f"Expected 10-DOF actions, got {actions_shape[1]}"
        assert ee_ori_shape[1] == 3, f"Expected 3-DOF ee_ori (Euler), got {ee_ori_shape[1]}"
        assert ee_states_shape[1] == 6, f"Expected 6-DOF ee_states (Euler), got {ee_states_shape[1]}"
        assert gripper_shape[1] == 1, f"Expected 1-DOF gripper, got {gripper_shape[1]}"

def main():
    # LIBERO Spatial dataset paths
    input_dir = "/mnt/data/libero/libero_spatial"
    output_dir = "/mnt/data/libero/libero_spatial_actions_6d"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List of LIBERO Spatial task files
    task_files = [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5",
    ]
    
    print("="*70)
    print("LIBERO Dataset Conversion: Actions Only (Euler → 6D)")
    print("Observations stay in Euler format (matching DROID)")
    print("="*70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tasks to convert: {len(task_files)}")
    print("="*70)
    
    for task_file in task_files:
        input_path = os.path.join(input_dir, task_file)
        output_path = os.path.join(output_dir, task_file)
        
        if not os.path.exists(input_path):
            print(f"\n⚠ Skipping {task_file}: file not found")
            continue
        
        try:
            convert_hdf5_file(input_path, output_path)
        except Exception as e:
            print(f"\n❌ Error converting {task_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("Conversion complete!")
    print("="*70)
    print(f"\nConverted datasets saved to: {output_dir}")
    print("\nWhat was converted:")
    print("  ✓ actions:        [7] → [10]  (pos(3) + rot_euler(3) + gripper(1) → pos(3) + rot_6d(6) + gripper(1))")
    print("  ✓ ee_ori:         [3] UNCHANGED (Euler - matching DROID)")
    print("  ✓ ee_states:      [6] UNCHANGED (Euler - matching DROID)")
    print("  ✓ gripper_states: [2] → [1]   ([left, right] → width = left - right)")
    print("\nDimension summary (matching DROID):")
    print("  Low-dim observations: ee_pos(3) + ee_ori(3) + gripper(1) = 7")
    print("  Actions: pos(3) + rot_6d(6) + gripper(1) = 10")
    print("\nNext steps:")
    print("1. Update libero_spatial_finetune.json:")
    print(f"   data paths: {output_dir}/*.hdf5")
    print("2. Update observation config:")
    print("   - ee_ori: [3] (not [6])")
    print("   - gripper_states: [1] (not [2])")
    print("3. Now combine layer dimensions match DROID!")
    print("   - DROID: Visual(1024) + Low-dim(7) = 1031")
    print("   - LIBERO: Visual(1024) + Low-dim(7) = 1031 ✓")

if __name__ == "__main__":
    main()

