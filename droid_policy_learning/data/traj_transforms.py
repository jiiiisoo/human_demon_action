"""
Subset of Octo's trajectory-level transforms used by the RLDS pipeline.
"""
import tensorflow as tf

from .data_utils import make_neutral_actions


def chunk_act_obs(
    traj: dict,
    window_size: int,
    future_action_window_size: int = 0,
) -> dict:
    """
    Chunk observations and actions into sliding windows so each step
    has shape `[window_size, ...]`.
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1), [traj_len, window_size]
    ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, window_size])

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), goal_timestep[:, None]
    )

    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_chunk_indices), traj["observation"]
    )
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    traj["observation"]["pad_mask"] = chunk_indices >= 0

    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        absolute_action_mask = tf.zeros([traj_len, action_dim], dtype=tf.bool)
    else:
        absolute_action_mask = traj.get(
            "absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool)
        )
    neutral_actions = make_neutral_actions(traj["action"], absolute_action_mask)

    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(
        action_past_goal[:, :, None], neutral_actions, traj["action"]
    )
    return traj


def subsample(traj: dict, subsample_length: int) -> dict:
    """
    Randomly subsample each trajectory down to `subsample_length` steps.
    """
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    return traj


def add_pad_mask_dict(traj: dict) -> dict:
    """
    Track which entries should be treated as padding.
    """
    traj_len = tf.shape(traj["action"])[0]
    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey, tensor in traj[key].items():
            if tf.debugging.is_numeric_tensor(tensor):
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
            elif tensor.dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(tensor) != 0
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
        traj[key]["pad_mask_dict"] = pad_mask_dict
    return traj
