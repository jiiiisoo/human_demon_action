"""
Goal relabeling helpers mirrored from Octo with additional offset support.
"""
from typing import Any, Dict

import tensorflow as tf

from .data_utils import tree_merge


def _inject_goal(traj: Dict[str, Any], goal: Dict[str, Any], goal_timesteps: tf.Tensor):
    goal = tree_merge(goal, {"timestep": goal_timesteps})
    traj["task"] = tree_merge(traj["task"], goal)
    return traj


def uniform(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Relabel each step with a uniformly sampled future state.
    """
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]

    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    return _inject_goal(traj, goal, goal_idxs)


def last(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use the final observation in the trajectory as the goal for every step.
    """
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
    goal_idx = tf.fill([traj_len], traj_len - 1)
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idx), traj["observation"])
    return _inject_goal(traj, goal, goal_idx)


def offset(traj: Dict[str, Any], offset: int) -> Dict[str, Any]:
    """
    Relabel with an observation that is `offset` steps into the future (clamped).
    """
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
    indices = tf.range(traj_len) + offset
    indices = tf.clip_by_value(indices, 0, traj_len - 1)
    goal = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj["observation"])
    return _inject_goal(traj, goal, indices)
