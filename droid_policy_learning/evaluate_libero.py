#!/usr/bin/env python
"""
Evaluate a robomimic diffusion policy checkpoint on LIBERO environments.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import h5py
from pathlib import Path
from collections import deque

# Add robomimic to path
sys.path.append(os.path.dirname(__file__))

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory

# LIBERO imports
try:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
    LIBERO_AVAILABLE = True
except ImportError:
    print("[Warning] LIBERO not installed. Please install LIBERO to run evaluation.")
    LIBERO_AVAILABLE = False


BENCHMARK_MAP = {
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_10": "LIBERO_10",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}


def rot_6d_to_euler(rot_6d):
    """
    Convert 6D rotation representation to Euler angles (XYZ convention).
    
    Args:
        rot_6d: (N, 6) array of 6D rotation representations
    
    Returns:
        euler: (N, 3) array of Euler angles in radians
    """
    import torch
    import robomimic.utils.torch_utils as TorchUtils
    
    # Convert to torch if numpy
    is_numpy = isinstance(rot_6d, np.ndarray)
    if is_numpy:
        rot_6d = torch.from_numpy(rot_6d).float()
    
    # Convert 6D to Euler using robomimic utilities
    euler = TorchUtils.rot_6d_to_euler_angles(rot_6d, convention="XYZ")
    
    if is_numpy:
        euler = euler.numpy()
    
    return euler


def process_libero_obs(raw_obs, obs_keys, obs_history, observation_horizon, goal_obs, device):
    """
    Process raw LIBERO observations to robomimic format with history.
    
    Args:
        raw_obs: Raw observation dict from LIBERO environment  
        obs_keys: List of observation keys expected by the policy
        obs_history: Deque of past observations
        observation_horizon: Number of observation timesteps to stack
        goal_obs: Goal observation (for skill conditioning)
        device: torch device
    
    Returns:
        obs_dict: Processed observation dict for robomimic policy
    """
    # Add current obs to history
    obs_history.append(raw_obs)
    
    obs_dict = {}
    
    # Stack observations over time
    for key in obs_keys:
        obs_list = []
        
        for hist_obs in obs_history:
            if key in ["agentview_rgb", "eye_in_hand_rgb"]:
                # Image observations
                libero_key = "agentview_image" if key == "agentview_rgb" else "robot0_eye_in_hand_image"
                
                if libero_key in hist_obs:
                    # raw_obs[libero_key] is (H, W, C) numpy array [0, 255]
                    img = hist_obs[libero_key]
                    # Add batch dimension: (1, H, W, C)
                    img = img[np.newaxis, ...]
                    obs_list.append(torch.from_numpy(img).float().to(device))
            
            elif key in ["ee_pos", "ee_ori", "gripper_states"]:
                # Low-dim observations
                if key == "ee_pos":
                    val = torch.from_numpy(hist_obs["ee_pos"]).float().to(device).unsqueeze(0)
                elif key == "ee_ori":
                    val = torch.from_numpy(hist_obs["ee_ori"]).float().to(device).unsqueeze(0)
                elif key == "gripper_states":
                    # LIBERO provides 2D gripper, we converted to 1D width in training
                    gripper_2d = hist_obs["gripper_qpos"]  # (2,)
                    gripper_width = np.array([gripper_2d[0] - gripper_2d[1]])  # (1,)
                    val = torch.from_numpy(gripper_width).float().to(device).unsqueeze(0)
                obs_list.append(val)
        
        # Stack along time dimension
        if obs_list:
            obs_dict[key] = torch.stack(obs_list, dim=1)  # (B, T, ...)
    
    # Add goal observations for skill conditioning
    if goal_obs is not None:
        for key in ["agentview_rgb", "eye_in_hand_rgb"]:
            if key in obs_keys:
                libero_key = "agentview_image" if key == "agentview_rgb" else "robot0_eye_in_hand_image"
                if libero_key in goal_obs:
                    # Goal image (final frame from demo)
                    goal_img = goal_obs[libero_key]
                    goal_img = goal_img[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, C)
                    obs_dict[f"goal_obs/{key}"] = torch.from_numpy(goal_img).float().to(device)
    
    return obs_dict


def evaluate_policy_on_task(
    policy,
    task,
    task_id,
    benchmark_name,
    num_episodes=20,
    max_steps=300,
    obs_keys=None,
    device="cuda",
    save_videos=False,
    video_folder=None,
):
    """
    Evaluate a policy on a single LIBERO task.
    
    Returns:
        success_rate: Fraction of successful episodes
    """
    if not LIBERO_AVAILABLE:
        raise ImportError("LIBERO is not installed")
    
    # Get paths
    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")
    
    # Environment configuration
    env_args = {
        "bddl_file_name": os.path.join(bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": 128,  # LIBERO default
        "camera_widths": 128,
    }
    
    # Create vectorized environment
    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_episodes)]
    )
    env.reset()
    env.seed(0)
    
    # Load initial states
    init_states_path = os.path.join(init_states_folder, task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path)
    indices = np.arange(num_episodes) % init_states.shape[0]
    init_states_ = init_states[indices]
    
    # Reset environment with initial states
    obs = env.set_init_state(init_states_)
    
    # Simulate physics for a few steps
    for _ in range(5):
        env.step(np.zeros((num_episodes, 7)))
    
    # Get goal observation (final frame from first demo)
    # Load demonstration to get goal image
    dataset_path = os.path.join(get_libero_path("datasets"), benchmark_name, 
                                benchmark.get_task_demonstration(task_id))
    goal_obs = None
    try:
        with h5py.File(dataset_path, 'r') as f:
            demo_keys = list(f['data'].keys())
            if demo_keys:
                first_demo = f['data'][demo_keys[0]]
                # Get final observation
                final_idx = len(first_demo['actions'][()])  - 1
                goal_obs = {}
                
                # Load final frame images
                for img_key, h5_key in [('agentview_image', 'agentview_rgb'), 
                                        ('robot0_eye_in_hand_image', 'eye_in_hand_rgb')]:
                    obs_key = f'obs/{h5_key}'
                    if obs_key in first_demo:
                        goal_obs[img_key] = first_demo[obs_key][final_idx]
                
                print(f"  Loaded goal observation with keys: {list(goal_obs.keys())}")
    except Exception as e:
        print(f"  Warning: Could not load goal observation: {e}")
    
    # Reset policy
    policy.reset()
    policy.set_eval()
    
    # Tracking
    dones = [False] * num_episodes
    steps = 0
    
    # Observation history for each environment
    observation_horizon = 2  # Default for diffusion policy
    obs_histories = [deque(maxlen=observation_horizon) for _ in range(num_episodes)]
    
    # Initialize observation histories with first observation
    for i in range(num_episodes):
        env_obs = {k: v[i] for k, v in obs.items()}
        obs_histories[i].append(env_obs)
    
    print(f"[Evaluation] Task {task_id}: {task.language}")
    print(f"  Episodes: {num_episodes}, Max steps: {max_steps}")
    
    with torch.no_grad():
        while steps < max_steps:
            steps += 1
            
            # Process observations for each environment
            batch_obs_list = []
            for i in range(num_episodes):
                env_obs = {k: v[i] for k, v in obs.items()}
                
                # Convert to policy format with history
                policy_obs = process_libero_obs(
                    env_obs, obs_keys, obs_histories[i], 
                    observation_horizon, goal_obs, device
                )
                batch_obs_list.append(policy_obs)
            
            # Concatenate batch
            batch_obs = {}
            for key in batch_obs_list[0].keys():
                batch_obs[key] = torch.cat([obs[key] for obs in batch_obs_list], dim=0)
            
            # Get action from policy
            actions = policy.get_action(batch_obs)  # (B, action_dim)
            
            # Convert actions: 10-DOF (pos + 6D rot + gripper) → 7-DOF (pos + euler + gripper)
            actions_np = TorchUtils.to_numpy(actions)
            pos = actions_np[:, :3]
            rot_6d = actions_np[:, 3:9]
            gripper = actions_np[:, 9:10]
            
            # Convert 6D rotation to Euler
            rot_euler = rot_6d_to_euler(rot_6d)
            
            # Combine for LIBERO environment (expects 7-DOF)
            env_actions = np.concatenate([pos, rot_euler, gripper], axis=1)
            
            # Step environment
            obs, reward, done, info = env.step(env_actions)
            
            # Track done
            for k in range(num_episodes):
                dones[k] = dones[k] or done[k]
            
            if all(dones):
                print(f"  All episodes finished at step {steps}")
                break
            
            if steps % 50 == 0:
                success_count = sum(dones)
                print(f"  Step {steps}/{max_steps}: {success_count}/{num_episodes} succeeded")
    
    # Calculate success rate
    num_success = sum(dones)
    success_rate = num_success / num_episodes
    
    print(f"  Result: {num_success}/{num_episodes} succeeded ({success_rate*100:.1f}%)")
    
    env.close()
    
    return success_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate robomimic policy on LIBERO")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to robomimic checkpoint")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_10", "libero_object", "libero_goal"],
        help="LIBERO benchmark to evaluate on"
    )
    parser.add_argument("--task_id", type=int, default=None, help="Specific task ID to evaluate (0-9), or None for all")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes per task")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--save_videos", action="store_true", help="Save videos of rollouts")
    parser.add_argument("--video_folder", type=str, default=None, help="Folder to save videos")
    
    args = parser.parse_args()
    
    if not LIBERO_AVAILABLE:
        print("[Error] LIBERO is not installed. Please install LIBERO first.")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(path=args.checkpoint)
    
    # Load config
    config = config_factory(ckpt_dict["config"]["algo_name"])
    with config.values_unlocked():
        config.update(ckpt_dict["config"])
    
    # Get observation keys
    obs_keys = []
    for modality_name, modality_list in config.observation.modalities.obs.items():
        obs_keys.extend(modality_list)
    
    print(f"Observation keys: {obs_keys}")
    
    # Initialize observation utilities
    ObsUtils.initialize_obs_utils_with_config(config)
    
    # Create policy
    device = torch.device(args.device)
    policy = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes={k: (1,) for k in obs_keys if k not in ["agentview_rgb", "eye_in_hand_rgb"]},
        ac_dim=config.train.action_config.actions.shape[0],
        device=device,
    )
    
    # Load weights
    policy.deserialize(ckpt_dict["model"])
    policy.set_eval()
    
    print(f"Policy loaded successfully")
    
    # Get benchmark
    benchmark = get_benchmark(BENCHMARK_MAP[args.benchmark])(0)
    
    # Evaluate
    if args.task_id is not None:
        # Single task
        task = benchmark.get_task(args.task_id)
        success_rate = evaluate_policy_on_task(
            policy=policy,
            task=task,
            task_id=args.task_id,
            benchmark_name=args.benchmark,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            obs_keys=obs_keys,
            device=device,
            save_videos=args.save_videos,
            video_folder=args.video_folder,
        )
        
        print(f"\n{'='*60}")
        print(f"Final Result: Task {args.task_id} - {success_rate*100:.1f}% success")
        print(f"{'='*60}")
    
    else:
        # All tasks
        results = {}
        for task_id in range(10):
            task = benchmark.get_task(task_id)
            success_rate = evaluate_policy_on_task(
                policy=policy,
                task=task,
                task_id=task_id,
                benchmark_name=args.benchmark,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                obs_keys=obs_keys,
                device=device,
                save_videos=args.save_videos,
                video_folder=args.video_folder,
            )
            results[task_id] = success_rate
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Evaluation Summary - {args.benchmark}")
        print(f"{'='*60}")
        for task_id, sr in results.items():
            task_name = benchmark.get_task_names()[task_id]
            print(f"Task {task_id}: {sr*100:5.1f}% - {task_name}")
        
        avg_success = np.mean(list(results.values()))
        print(f"\nAverage Success Rate: {avg_success*100:.1f}%")
        print(f"{'='*60}")
        
        # Save results
        save_path = args.checkpoint.replace(".pth", "_eval_results.json")
        with open(save_path, "w") as f:
            json.dump({
                "benchmark": args.benchmark,
                "num_episodes": args.num_episodes,
                "results": results,
                "average": avg_success,
            }, f, indent=2)
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()

