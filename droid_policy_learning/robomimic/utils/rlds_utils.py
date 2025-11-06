"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import tensorflow as tf
import torch
import tensorflow_graphics.geometry.transformation as tfg
import numpy as np

def filter_success(trajectory: dict[str, any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def mat_to_rot6d(mat):
    mat = tf.convert_to_tensor(mat)
    cols = tf.gather(mat, indices=[0, 1], axis=-1)
    target_shape = tf.concat([tf.shape(cols)[:-2], [6]], axis=0)
    return tf.reshape(cols, target_shape)


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    cartesian = tf.cast(trajectory["action_dict"]["cartesian_position"], tf.float32)
    T = cartesian[:, :3]
    R = mat_to_rot6d(euler_to_rmat(cartesian[:, 3:6]))
    gripper = tf.cast(trajectory["action_dict"]["gripper_position"], tf.float32)
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            gripper,
        ),
        axis=-1,
    )
    return trajectory


def robomimic_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    obs = {
        "camera/image/varied_camera_1_left_image": tf.cast(
            trajectory["observation"]["image_primary"], tf.float32
        )
        / 255.0,
        "camera/image/varied_camera_2_left_image": tf.cast(
            trajectory["observation"]["image_secondary"], tf.float32
        )
        / 255.0,
        "raw_language": trajectory["task"]["language_instruction"],
        "robot_state/cartesian_position": trajectory["observation"]["proprio"][..., :6],
        "robot_state/gripper_position": trajectory["observation"]["proprio"][..., -1:],
        "pad_mask": trajectory["observation"]["pad_mask"][..., None],
    }

    goal_obs = {}
    task_dict = trajectory.get("task", {})
    if "image_primary" in task_dict:
        goal_obs["camera/image/varied_camera_1_left_image"] = tf.cast(
            task_dict["image_primary"], tf.float32
        ) / 255.0
    if "image_secondary" in task_dict:
        goal_obs["camera/image/varied_camera_2_left_image"] = tf.cast(
            task_dict["image_secondary"], tf.float32
        ) / 255.0
    if "proprio" in task_dict:
        goal_obs["robot_state/cartesian_position"] = task_dict["proprio"][..., :6]
        goal_obs["robot_state/gripper_position"] = task_dict["proprio"][..., -1:]
    if goal_obs:
        goal_obs["pad_mask"] = tf.ones_like(obs["pad_mask"])

    result = {
        "obs": obs,
        "actions": trajectory["action"][1:],
    }
    if goal_obs:
        result["goal_obs"] = goal_obs
    return result

DROID_TO_RLDS_OBS_KEY_MAP = {
    "camera/image/varied_camera_1_left_image": "exterior_image_1_left",
    "camera/image/varied_camera_2_left_image": "exterior_image_2_left"
}

DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP = {
    "robot_state/cartesian_position": "cartesian_position",
    "robot_state/gripper_position": "gripper_position",
}

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        # lengths = np.array(
        #     [
        #         stats["num_transitions"]
        #         for stats in self._rlds_dataset.dataset_statistics
        #     ]
        # )
        if hasattr(self._rlds_dataset, "dataset_statistics"):
            lengths = np.array(
                [stats["num_transitions"] for stats in self._rlds_dataset.dataset_statistics]
            )
        elif hasattr(self._rlds_dataset, "_info") and "num_transitions" in self._rlds_dataset._info:
            lengths = np.array([self._rlds_dataset._info["num_transitions"]])
        else:
            # fallback: try to estimate length from iterator or samples
            lengths = np.array([len(list(self._rlds_dataset.as_numpy_iterator()))])
            
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)
