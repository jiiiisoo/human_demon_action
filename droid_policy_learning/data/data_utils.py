import hashlib
import json
import logging
import os
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import dlimp as dl
import numpy as np
import tensorflow as tf
import tqdm


def tree_map(fn: Callable, tree: dict) -> dict:
    """
    Apply `fn` recursively to every leaf in a nested dict.
    """
    return {
        key: tree_map(fn, value) if isinstance(value, dict) else fn(value)
        for key, value in tree.items()
    }


def tree_merge(*trees: dict) -> dict:
    """
    Recursively merge dictionaries, later entries overriding earlier keys.
    """
    merged: Dict[str, Any] = {}
    for tree in trees:
        for key, value in tree.items():
            if isinstance(value, dict):
                merged[key] = tree_merge(merged.get(key, {}), value)
            else:
                merged[key] = value
    return merged


class NormalizationType(str, Enum):
    """
    Supported normalization schemes for action / proprio features.
    """

    NORMAL = "normal"
    BOUNDS = "bounds"


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    if tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    raise ValueError(f"Cannot create padding for dtype {tensor.dtype}.")


def make_neutral_actions(action: tf.Tensor, absolute_action_mask: tf.Tensor) -> tf.Tensor:
    """
    Zero relative action dims while keeping absolute dims unchanged.
    """
    mask = tf.cast(absolute_action_mask, tf.bool)
    action_rank = action.shape.rank
    mask_rank = mask.shape.rank

    if mask_rank is None or action_rank is None:
        action_rank_tensor = tf.rank(action)
        mask_rank_tensor = tf.rank(mask)
        expand = tf.maximum(action_rank_tensor - mask_rank_tensor, 0)

        def body(mask_tensor, i):
            return tf.expand_dims(mask_tensor, axis=1), i + 1

        mask, _ = tf.while_loop(
            lambda _mask, i: i < expand,
            body,
            [mask, tf.constant(0, dtype=expand.dtype)],
            shape_invariants=[tf.TensorShape(None), tf.TensorShape([])],
        )
    else:
        for _ in range(action_rank - mask_rank):
            mask = tf.expand_dims(mask, axis=1)

    mask = tf.broadcast_to(mask, tf.shape(action))
    return tf.where(mask, action, tf.zeros_like(action))


def pprint_data_mixture(dataset_kwargs_list, dataset_weights):
    print(
        "\n######################################################################################"
    )
    print(
        f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #"
    )
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print(
        "######################################################################################\n"
    )


def get_dataset_statistics(
    dataset: dl.DLataset,
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
) -> dict:
    """
    Either computes dataset statistics or loads them from cache.
    """
    unique_hash = hashlib.sha256(
        "".join(hash_dependencies).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()

    local_path = os.path.expanduser(
        os.path.join("~", ".cache", "octo", f"dataset_statistics_{unique_hash}.json")
    )
    if save_dir is not None:
        path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json")
    else:
        path = local_path

    if tf.io.gfile.exists(path):
        logging.info("Loading existing dataset statistics from %s.", path)
        with tf.io.gfile.GFile(path, "r") as file:
            return json.load(file)

    if os.path.exists(local_path):
        logging.info("Loading existing dataset statistics from %s.", local_path)
        with open(local_path, "r") as file:
            return json.load(file)

    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            "proprio": traj["observation"].get("proprio", tf.zeros_like(traj["action"])),
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute statistics for an infinite dataset.")

    logging.info("Computing dataset statistics. This may take a while on first run.")
    actions = []
    proprios = []
    num_transitions = 0
    num_trajectories = 0
    for traj in tqdm.tqdm(
        dataset.iterator(),
        total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
    ):
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1
    actions_arr = np.concatenate(actions)
    proprios_arr = np.concatenate(proprios)
    metadata = {
        "action": {
            "mean": actions_arr.mean(0).tolist(),
            "std": actions_arr.std(0).tolist(),
            "max": actions_arr.max(0).tolist(),
            "min": actions_arr.min(0).tolist(),
        },
        "proprio": {
            "mean": proprios_arr.mean(0).tolist(),
            "std": proprios_arr.std(0).tolist(),
            "max": proprios_arr.max(0).tolist(),
            "min": proprios_arr.min(0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    try:
        with tf.io.gfile.GFile(path, "w") as file:
            json.dump(metadata, file)
    except tf.errors.PermissionDeniedError:
        logging.warning(
            "Could not write dataset statistics to %s. Writing to %s instead.",
            path,
            local_path,
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as file:
            json.dump(metadata, file)

    return metadata


def combine_dataset_statistics(all_dataset_statistics: Sequence[dict]) -> dict:
    """
    Merge statistics from multiple datasets into a single set.
    """
    merge_stat_keys = ["action", "proprio"]

    num_transitions = [stat["num_transitions"] for stat in all_dataset_statistics]
    stat_weights = [
        transitions / sum(num_transitions) for transitions in num_transitions
    ]

    combined_dataset_statistics = {}
    for key in merge_stat_keys:
        combined_mean = np.array(
            [
                np.array(stat[key]["mean"]) * weight
                for stat, weight in zip(all_dataset_statistics, stat_weights)
            ]
        ).sum(0)
        combined_std = np.sqrt(
            np.array(
                [
                    transitions * (np.array(stat[key]["std"]) ** 2)
                    + transitions
                    * (np.array(stat[key]["mean"]) - combined_mean) ** 2
                    for stat, transitions in zip(all_dataset_statistics, num_transitions)
                ]
            ).sum(0)
            / sum(num_transitions)
        )
        combined_dataset_statistics[key] = {
            "min": np.array([stat[key]["min"] for stat in all_dataset_statistics])
            .min(0)
            .tolist(),
            "max": np.array([stat[key]["max"] for stat in all_dataset_statistics])
            .max(0)
            .tolist(),
            "mean": combined_mean.tolist(),
            "std": combined_std.tolist(),
        }

    combined_dataset_statistics["num_trajectories"] = [
        stat["num_trajectories"] for stat in all_dataset_statistics
    ]
    combined_dataset_statistics["num_transitions"] = num_transitions
    return combined_dataset_statistics


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """
    Allocate integer thread counts proportional to dataset weights.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    if len(weights) > n:
        raise ValueError(
            "Number of threads must be at least the number of datasets being allocated."
        )
    weights = np.array(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    weights = weights / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)
        weights[mask] = 0
        weights = weights / weights.sum()

    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= int(integral.sum())
    for idx in np.argsort(fractional)[::-1][: int(n)]:
        allocation[idx] += 1
    return allocation


def normalize_action_and_proprio(
    traj: dict,
    metadata: dict,
    normalization_type: NormalizationType,
    skip_keys=None,
):
    """
    Normalize action / proprio sequences using cached metadata.
    """
    skip_keys = set(skip_keys or [])
    keys_to_normalize = {
        "action": ("action",),
        "proprio": ("observation", "proprio"),
    }

    for key, tree_path in keys_to_normalize.items():
        if key not in metadata or key in skip_keys:
            continue

        ref = metadata[key]
        node = traj
        for part in tree_path[:-1]:
            node = node.setdefault(part, {})
        leaf_key = tree_path[-1]
        if leaf_key not in node:
            continue
        value = node[leaf_key]
        ref_mean = tf.convert_to_tensor(ref["mean"], dtype=value.dtype)
        ref_std = tf.convert_to_tensor(ref["std"], dtype=value.dtype)
        ref_min = tf.convert_to_tensor(ref["min"], dtype=value.dtype)
        ref_max = tf.convert_to_tensor(ref["max"], dtype=value.dtype)
        if normalization_type == NormalizationType.NORMAL:
            node[leaf_key] = (value - ref_mean) / ref_std
        elif normalization_type == NormalizationType.BOUNDS:
            denom = ref_max - ref_min
            denom = tf.where(denom == 0, tf.ones_like(denom), denom)
            node[leaf_key] = 2.0 * (value - ref_min) / denom - 1.0
        else:
            raise ValueError(f"Unknown normalization type {normalization_type}.")

    return traj
