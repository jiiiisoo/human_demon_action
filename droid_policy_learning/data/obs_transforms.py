"""
Frame-level observation transforms used in the RLDS pipeline.
"""
from functools import partial
from typing import Mapping, Tuple, Union

import dlimp as dl
from absl import logging
import tensorflow as tf


def augment(
    obs: dict, seed: tf.Tensor, augment_kwargs: Union[dict, Mapping[str, dict]]
) -> dict:
    """
    Augment images while skipping padding frames.
    """
    image_names = {key[6:] for key in obs if key.startswith("image_")}

    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for idx, name in enumerate(image_names):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]
        logging.debug("Augmenting image_%s with kwargs %s", name, kwargs)
        obs[f"image_{name}"] = tf.cond(
            obs["pad_mask_dict"][f"image_{name}"],
            lambda: dl.transforms.augment_image(
                obs[f"image_{name}"],
                **kwargs,
                seed=seed + idx,
            ),
            lambda: obs[f"image_{name}"],
        )

    return obs


def decode_and_resize(
    obs: dict,
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]],
) -> dict:
    """
    Decode encoded images and optionally resize them.
    """
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    depth_names = {key[6:] for key in obs if key.startswith("depth_")}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(
                "No resize_size provided for image_%s. Padding will be 1x1.", name
            )
        image = obs[f"image_{name}"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(f"Unsupported dtype for image_{name}: {image.dtype}")
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f"image_{name}"] = image

    for name in depth_names:
        if name not in depth_resize_size:
            logging.warning(
                "No depth_resize_size provided for depth_%s. Padding will be 1x1.", name
            )
        depth = obs[f"depth_{name}"]
        if depth.dtype == tf.string:
            if tf.strings.length(depth) == 0:
                depth = tf.zeros(
                    (*depth_resize_size.get(name, (1, 1)), 1), dtype=tf.float32
                )
            else:
                depth = tf.io.decode_image(
                    depth, expand_animations=False, dtype=tf.float32
                )[..., 0]
        elif depth.dtype != tf.float32:
            raise ValueError(f"Unsupported dtype for depth_{name}: {depth.dtype}")
        if name in depth_resize_size:
            depth = dl.transforms.resize_depth_image(depth, size=depth_resize_size[name])
        obs[f"depth_{name}"] = depth

    return obs


def apply_obs_transform(fn, frame: dict) -> dict:
    """
    Convenience wrapper used by datasets.apply_frame_transforms.
    """
    frame["task"] = fn(frame["task"])
    frame["observation"] = dl.vmap(fn)(frame["observation"])
    return frame
