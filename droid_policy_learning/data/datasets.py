from functools import partial
import inspect
import json
import logging
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from . import goal_relabeling, obs_transforms, traj_transforms
from .data_utils import (
    NormalizationType,
    allocate_threads,
    combine_dataset_statistics,
    get_dataset_statistics,
    normalize_action_and_proprio,
    pprint_data_mixture,
    tree_map,
)
from .spec import ModuleSpec


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: Optional[dict] = None,
    window_size: int = 1,
    future_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: Optional[dict] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Apply trajectory-level transforms (goal relabeling, chunking, subsampling).
    """
    task_augment_kwargs = task_augment_kwargs or {}
    goal_relabeling_kwargs = goal_relabeling_kwargs or {}

    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError(
                "skip_unlabeled=True but dataset does not have language labels."
            )
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )

    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )

    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    if goal_relabeling_strategy is not None:
        if not hasattr(goal_relabeling, goal_relabeling_strategy):
            raise ValueError(
                f"Unknown goal relabeling strategy '{goal_relabeling_strategy}'."
            )
        dataset = dataset.traj_map(
            partial(
                getattr(goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

    if task_augment_strategy is not None:
        raise NotImplementedError(
            "Task augmentation is not currently vendored into the local pipeline."
        )

    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[dict, Mapping[str, dict]] = {},
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Apply frame-level transforms (decode / resize / augment).
    """

    dataset = dataset.frame_map(
        partial(
            obs_transforms.apply_obs_transform,
            partial(
                obs_transforms.decode_and_resize,
                resize_size=resize_size,
                depth_resize_size=depth_resize_size,
            ),
        ),
        num_parallel_calls,
    )

    if train:
        def aug(frame: dict):
            seed = tf.random.uniform(
                [2], maxval=tf.dtypes.int32.max, dtype=tf.int32
            )
            aug_fn = partial(
                obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs
            )
            return obs_transforms.apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    depth_obs_keys: Mapping[str, Optional[str]] = {},
    state_obs_keys: Sequence[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[Sequence[bool]] = None,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    norm_skip_keys: Optional[Sequence[str]] = None,
    filter_functions: Sequence[ModuleSpec] = (),
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """
    Load an RLDS dataset and return a standardized DLataset plus statistics.
    """

    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            missing = REQUIRED_KEYS - set(traj.keys())
            raise ValueError(
                f"Trajectory is missing keys: {missing}. Did you supply a valid standardize_fn?"
            )

        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            pieces = []
            for key in state_obs_keys:
                if key is None:
                    pieces.append(tf.zeros((traj_len, 1), dtype=tf.float32))
                else:
                    pieces.append(tf.cast(old_obs[key], tf.float32))
            new_obs["proprio"] = tf.concat(pieces, axis=1)

        new_obs["timestep"] = tf.range(traj_len)

        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} must be tf.string (found {traj[language_key].dtype})."
                )
            task["language_instruction"] = traj.pop(language_key)

        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj

    builder = tfds.builder(name, data_dir=data_dir)

    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as file:
            dataset_statistics = json.load(file)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads
        )
        for filter_fcn_spec in filter_functions:
            full_dataset = full_dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
        full_dataset = full_dataset.traj_map(restructure, num_parallel_calls)
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
                *map(ModuleSpec.to_string, filter_functions),
            ),
            save_dir=builder.data_dir,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    if action_normalization_mask is not None:
        if (
            len(action_normalization_mask)
            != dataset_statistics["action"]["mean"].shape[-1]
        ):
            raise ValueError(
                f"Length of action_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )
    for filter_fcn_spec in filter_functions:
        dataset = dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
            skip_keys=norm_skip_keys,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
) -> dl.DLataset:
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    dataset = dataset.with_ram_budget(1)
    dataset.dataset_statistics = dataset_statistics
    return dataset


def make_interleaved_dataset(
    dataset_kwargs_list: Sequence[dict],
    sample_weights: Optional[Sequence[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    dataset_statistics: Optional[Union[dict, str]] = None,
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    dataset_sizes = []
    all_dataset_statistics = []
    for dataset_kwargs in dataset_kwargs_list:
        _, per_dataset_stats = make_dataset_from_rlds(**dataset_kwargs, train=train)
        dataset_sizes.append(per_dataset_stats["num_transitions"])
        all_dataset_statistics.append(per_dataset_stats)

    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights, dtype=float)
    sample_weights = sample_weights / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    logging.info("Threads per dataset: %s", threads_per_dataset)
    logging.info("Reads per dataset: %s", reads_per_dataset)

    datasets = []
    for dataset_kwargs, per_dataset_stats, threads, reads in zip(
        dataset_kwargs_list,
        all_dataset_statistics,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=dataset_statistics
            if dataset_statistics is not None
            else per_dataset_stats,
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)
        datasets.append(dataset)

    dataset = dl.DLataset.sample_from_datasets(
        datasets, sample_weights
    ).shuffle(shuffle_buffer_size)

    dataset = apply_frame_transforms(
        dataset, **frame_transform_kwargs, train=train
    )

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.with_ram_budget(1)
    dataset.dataset_statistics = (
        dataset_statistics if dataset_statistics is not None else all_dataset_statistics
    )
    dataset.sample_weights = sample_weights
    return dataset
