#!/usr/bin/env python
"""
Convert LIBERO HDF5 datasets from Euler angles to 6D rotation representation.
This creates new HDF5 files with actions in 10-DOF format matching DROID.

Original LIBERO: actions [7] = pos(3) + rot_euler(3) + gripper(1)
Converted:      actions [10] = pos(3) + rot_6d(6) + gripper(1)
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
    """Convert a single HDF5 file from Euler to 6D rotation"""
    
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
                
                # Copy and convert observations
                obs_in = demo_in['obs']
                obs_out = demo_out.create_group('obs')
                for obs_key in obs_in.keys():
                    if obs_key == 'ee_ori':
                        # Convert ee_ori from Euler (3,) to 6D rotation (6,)
                        ee_ori_euler = obs_in[obs_key][()]  # Shape: (T, 3)
                        ee_ori_6d = convert_euler_to_6d(ee_ori_euler)  # Shape: (T, 6)
                        obs_out.create_dataset(obs_key, data=ee_ori_6d)
                        print(f"    ✓ Converted ee_ori: {ee_ori_euler.shape} → {ee_ori_6d.shape}")
                    elif obs_key == 'ee_states':
                        # ee_states = [ee_pos(3), ee_ori(3)] → [ee_pos(3), ee_ori_6d(6)]
                        ee_states = obs_in[obs_key][()]  # Shape: (T, 6)
                        ee_pos = ee_states[:, :3]
                        ee_ori_euler = ee_states[:, 3:6]
                        ee_ori_6d = convert_euler_to_6d(ee_ori_euler)  # Shape: (T, 6)
                        ee_states_6d = np.concatenate([ee_pos, ee_ori_6d], axis=1)  # Shape: (T, 9)
                        obs_out.create_dataset(obs_key, data=ee_states_6d)
                        print(f"    ✓ Converted ee_states: {ee_states.shape} → {ee_states_6d.shape}")
                    else:
                        # Copy other observations unchanged
                        obs_out.create_dataset(obs_key, data=obs_in[obs_key][()])
                
                # Copy and convert next_obs if exists
                if 'next_obs' in demo_in:
                    next_obs_in = demo_in['next_obs']
                    next_obs_out = demo_out.create_group('next_obs')
                    for obs_key in next_obs_in.keys():
                        if obs_key == 'ee_ori':
                            # Convert next_obs ee_ori from Euler (3,) to 6D rotation (6,)
                            ee_ori_euler = next_obs_in[obs_key][()]
                            ee_ori_6d = convert_euler_to_6d(ee_ori_euler)
                            next_obs_out.create_dataset(obs_key, data=ee_ori_6d)
                        elif obs_key == 'ee_states':
                            # ee_states = [ee_pos(3), ee_ori(3)] → [ee_pos(3), ee_ori_6d(6)]
                            ee_states = next_obs_in[obs_key][()]
                            ee_pos = ee_states[:, :3]
                            ee_ori_euler = ee_states[:, 3:6]
                            ee_ori_6d = convert_euler_to_6d(ee_ori_euler)
                            ee_states_6d = np.concatenate([ee_pos, ee_ori_6d], axis=1)
                            next_obs_out.create_dataset(obs_key, data=ee_states_6d)
                        else:
                            next_obs_out.create_dataset(obs_key, data=next_obs_in[obs_key][()])
                
                # Convert actions: [7] -> [10]
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
        print(f"    - actions shape:   {actions_shape}, expected (T, 10) ✓" if actions_shape[1] == 10 else f"    - actions shape:   {actions_shape}, expected (T, 10) ✗")
        print(f"    - ee_ori shape:    {ee_ori_shape}, expected (T, 6) ✓" if ee_ori_shape[1] == 6 else f"    - ee_ori shape:    {ee_ori_shape}, expected (T, 6) ✗")
        print(f"    - ee_states shape: {ee_states_shape}, expected (T, 9) ✓" if ee_states_shape[1] == 9 else f"    - ee_states shape: {ee_states_shape}, expected (T, 9) ✗")
        
        assert actions_shape[1] == 10, f"Expected 10-DOF actions, got {actions_shape[1]}"
        assert ee_ori_shape[1] == 6, f"Expected 6-DOF ee_ori, got {ee_ori_shape[1]}"
        assert ee_states_shape[1] == 9, f"Expected 9-DOF ee_states, got {ee_states_shape[1]}"

def main():
    # LIBERO Spatial dataset paths
    input_dir = "/mnt/data/libero/libero_spatial"
    output_dir = "/mnt/data/libero/libero_spatial_6d"
    
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
    print("LIBERO Dataset Conversion: Euler Angles → 6D Rotation")
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
    print("  • actions:    [7] → [10]  (pos(3) + rot_euler(3) + gripper(1) → pos(3) + rot_6d(6) + gripper(1))")
    print("  • ee_ori:     [3] → [6]   (rot_euler(3) → rot_6d(6))")
    print("  • ee_states:  [6] → [9]   (pos(3) + rot_euler(3) → pos(3) + rot_6d(6))")
    print("\nNext steps:")
    print("1. Update libero_spatial_finetune.json to use the new dataset paths:")
    print(f"   Change: /mnt/data/libero/libero_spatial/*.hdf5")
    print(f"   To:     /mnt/data/libero/libero_spatial_6d/*.hdf5")
    print("2. Action dimension will now be 10 (matching DROID)")
    print("3. Observation dimensions updated to match 6D rotation")
    print("4. Action prediction head will fully transfer from DROID checkpoint!")
    print("5. All rotation representations now consistent (6D everywhere)!")

if __name__ == "__main__":
    main()

