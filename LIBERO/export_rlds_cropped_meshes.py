#!/usr/bin/env python3
"""
Replay RLDS episodes by applying stored actions in a LIBERO environment and
export per-step meshes (whole scene and table-centered cube crop).

Output layout:
  <output_root>/<dataset>/episode_00000/whole_scene/step_0000.obj
  <output_root>/<dataset>/episode_00000/cropped_scene/step_0000.obj
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import dlimp as dl

from extract_frames import (
    DatasetInput,
    Feature,
    collect_dataset_inputs,
    iter_tfrecord_records,
    parse_sequence_example,
)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from export_gt_pointcloud import (  # type: ignore
    collect_world_meshes,
    get_reference_center,
    center_and_crop_meshes,
    save_meshes_as_obj,
)
from tqdm import tqdm


def _actions_from_record_fields(
    context: Dict[str, Feature], feature_lists: Dict[str, List[Feature]]
) -> np.ndarray:
    actions: List[np.ndarray] = []

    def add_from_feature_list(feats: List[Feature]):
        for feat in feats:
            if feat.float_list:
                actions.append(np.array(feat.float_list, dtype=np.float32))

    # SequenceExample paths
    if "steps/action" in feature_lists:
        add_from_feature_list(feature_lists["steps/action"])
    elif "action" in feature_lists:
        add_from_feature_list(feature_lists["action"])
    elif "steps" in feature_lists:
        add_from_feature_list(feature_lists["steps"])

    # Context-encoded steps as serialized Features protos
    if not actions and "steps" in context and context["steps"].bytes_list:
        for step_bytes in context["steps"].bytes_list:
            feats = parse_features(step_bytes)
            act_feat = feats.get("action") or feats.get("steps/action")
            if act_feat and act_feat.float_list:
                actions.append(np.array(act_feat.float_list, dtype=np.float32))

    return np.stack(actions)


def _actions_from_example(features: Dict[str, Feature]) -> np.ndarray:
    actions: List[np.ndarray] = []
    for key in ("steps/action", "action"):
        feat = features.get(key)
        if feat and feat.float_list:
            actions.append(np.array(feat.float_list, dtype=np.float32))
        elif feat and feat.bytes_list:
            for b in feat.bytes_list:
                sub = parse_features(b)
                act_feat = sub.get("action") or sub.get("steps/action")
                if act_feat and act_feat.float_list:
                    actions.append(np.array(act_feat.float_list, dtype=np.float32))
    if not actions and "steps" in features and features["steps"].bytes_list:
        for step_bytes in features["steps"].bytes_list:
            feats = parse_features(step_bytes)
            act_feat = feats.get("action") or feats.get("steps/action")
            if act_feat and act_feat.float_list:
                actions.append(np.array(act_feat.float_list, dtype=np.float32))

    return np.stack(actions)


def actions_from_record_bytes(record_bytes: bytes) -> np.ndarray:
    # Try tensorflow's SequenceExample parser first
    seq = tf.train.SequenceExample()
    try:
        seq.ParseFromString(record_bytes)
    except Exception:
        seq = None

    def _extract_from_tf_feature_list(name: str) -> Optional[np.ndarray]:
        if seq is None:
            return None
        if name not in seq.feature_lists.feature_list:
            return None
        feats = seq.feature_lists.feature_list[name].feature
        acts = []
        for feat in feats:
            if feat.float_list.value:
                acts.append(np.array(feat.float_list.value, dtype=np.float32))
        if acts:
            return np.stack(acts)
        return None

    # SequenceExample paths
    for key in ("steps/action", "action", "steps"):
        arr = _extract_from_tf_feature_list(key)
        if arr is not None and arr.size > 0:
            print('arr is not None and arr.size > 0')
            return arr

    # Fallback: try TensorFlow Example parsing
    ex = tf.train.Example()
    try:
        ex.ParseFromString(record_bytes)
        fmap = ex.features.feature
        acts = []
        for key in ("steps/action", "action", "steps"):
            if key in fmap and fmap[key].float_list.value:
                acts.append(np.array(fmap[key].float_list.value, dtype=np.float32))
        if acts:
            return np.stack(acts)
    except Exception:
        pass

    # Last resort: manual parser; ignore decode errors
    try:
        context, feature_lists = parse_sequence_example(record_bytes)
        acts = _actions_from_record_fields(context, feature_lists)
        if acts.size > 0:
            return acts
    except Exception:
        pass

    return np.zeros((0, 7), dtype=np.float32)


def ensure_action_shape(actions: np.ndarray, expected_dim: int = 7) -> np.ndarray:
    if actions.size == 0:
        raise ValueError("No actions found in record")
    if actions.ndim == 1:
        if actions.size % expected_dim != 0:
            raise ValueError(f"Flat action array length {actions.size} not divisible by {expected_dim}")
        actions = actions.reshape(-1, expected_dim)
    if actions.shape[1] != expected_dim:
        raise ValueError(f"Action dim mismatch: expected {expected_dim}, got {actions.shape[1]}")
    return actions


def _patch_dlimp_options():
    """Avoid setting Options.experimental_warm_start on TF builds that don't support it."""
    try:
        import dlimp.dataset as ds
    except Exception:
        return

    def _safe_apply_options(self):
        options = tf.data.Options()
        options.autotune.enabled = True
        options.deterministic = True
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.map_and_filter_fusion = True
        options.experimental_optimization.inject_prefetch = False
        return self.with_options(options)

    ds.DLataset._apply_options = _safe_apply_options


def _dataset_builder_from_shards(shards: List[Path]):
    if not shards:
        raise ValueError("No shards provided to build dataset")
    version_dir = shards[0].parent
    dataset_dir = version_dir.parent
    data_root = dataset_dir.parent
    dataset_name = dataset_dir.name
    if not dataset_name:
        raise ValueError(f"Could not infer dataset name from shard path: {shards[0]}")
    return tfds.builder(dataset_name, data_dir=str(data_root))


def load_rlds_dataset(
    shards: List[Path],
    split: str = "train",
    num_parallel_reads: int = tf.data.AUTOTUNE,
):
    _patch_dlimp_options()
    builder = _dataset_builder_from_shards(shards)
    return dl.DLataset.from_rlds(
        builder,
        split=split,
        shuffle=False,
        num_parallel_reads=num_parallel_reads,
    )


def _int_list_from_feature(feat) -> List[int]:
    if feat is None:
        return []
    if hasattr(feat, "int64_list") and getattr(feat.int64_list, "value", None) is not None:
        return [int(v) for v in feat.int64_list.value]
    if hasattr(feat, "int64_list") and isinstance(feat.int64_list, list):
        return [int(v) for v in feat.int64_list]
    if hasattr(feat, "bytes_list") and getattr(feat.bytes_list, "value", None) is not None:
        vals = []
        for b in feat.bytes_list.value:
            try:
                vals.append(int(b.decode("utf-8")))
            except Exception:
                continue
        return vals
    return []


def _int_from_feature(feat) -> Optional[int]:
    vals = _int_list_from_feature(feat)
    if vals:
        return vals[0]
    return None


def _frame_indices_from_feature_list(fl_map) -> List[int]:
    for key in ("steps/_frame_index", "_frame_index"):
        if key not in fl_map:
            continue
        fl = fl_map[key]
        vals: List[int] = []
        for feat in fl.feature:
            vals.extend(_int_list_from_feature(feat))
        if vals:
            return vals
    return []


def _frame_indices_from_feature_dict(feat_map) -> List[int]:
    for key in ("steps/_frame_index", "_frame_index"):
        if key in feat_map:
            vals = _int_list_from_feature(feat_map[key])
            if vals:
                return vals
    return []


def _traj_metadata_from_sequence_example(record_bytes: bytes) -> Tuple[Optional[int], List[int], Optional[int]]:
    seq = tf.train.SequenceExample()
    try:
        seq.ParseFromString(record_bytes)
    except Exception:
        return None, [], None
    traj_index = _int_from_feature(seq.context.feature.get("_traj_index"))
    traj_len = _int_from_feature(seq.context.feature.get("_len"))
    frame_indices = _frame_indices_from_feature_list(seq.feature_lists.feature_list)
    if not frame_indices:
        frame_indices = _frame_indices_from_feature_dict(seq.context.feature)
    return traj_index, frame_indices, traj_len


def _traj_metadata_from_example(record_bytes: bytes) -> Tuple[Optional[int], List[int], Optional[int]]:
    ex = tf.train.Example()
    try:
        ex.ParseFromString(record_bytes)
    except Exception:
        return None, [], None
    fmap = ex.features.feature
    traj_index = _int_from_feature(fmap.get("_traj_index"))
    traj_len = _int_from_feature(fmap.get("_len"))
    frame_indices = _frame_indices_from_feature_dict(fmap)
    return traj_index, frame_indices, traj_len


def _traj_metadata_manual(record_bytes: bytes) -> Tuple[Optional[int], List[int], Optional[int]]:
    try:
        context, feature_lists = parse_sequence_example(record_bytes)
    except Exception:
        return None, [], None
    traj_index = _int_from_feature(context.get("_traj_index"))
    traj_len = _int_from_feature(context.get("_len"))
    frame_indices = []
    for key in ("steps/_frame_index", "_frame_index"):
        if key in feature_lists:
            for feat in feature_lists[key]:
                if feat.int64_list:
                    frame_indices.extend([int(v) for v in feat.int64_list])
        if key in context and context[key].int64_list:
            frame_indices.extend([int(v) for v in context[key].int64_list])
    return traj_index, frame_indices, traj_len


def traj_metadata_from_record_bytes(record_bytes: bytes) -> Tuple[int, List[int], int]:
    traj_index, frame_indices, traj_len = _traj_metadata_from_sequence_example(record_bytes)
    if traj_index is None and not frame_indices:
        traj_index, frame_indices, traj_len = _traj_metadata_from_example(record_bytes)
    if traj_index is None and not frame_indices:
        traj_index, frame_indices, traj_len = _traj_metadata_manual(record_bytes)

    if traj_index is None:
        raise ValueError("Missing _traj_index in record")
    if not frame_indices:
        raise ValueError("Missing _frame_index in record")
    if traj_len is None:
        traj_len = len(frame_indices)
    return traj_index, frame_indices, traj_len


def replay_episode(
    env,
    actions: np.ndarray,
    out_cropped: Path,
    cube_half: float,
    camera_names: List[str],
    traj_index: int,
    frame_indices: List[int],
    traj_len: int,
    init_obs: Optional[dict] = None,
    include_table: bool = True,
) -> None:
    out_cropped.mkdir(parents=True, exist_ok=True)

    if len(actions) != len(frame_indices):
        raise ValueError(f"Action length {len(actions)} does not match provided _frame_index length {len(frame_indices)}")
    for frame_idx, act in zip(frame_indices, actions):
        obs, _, done, _ = env.step(act.tolist())
        meshes = collect_world_meshes(env, include_robot=True, include_statics=True, exclude_body_substrings=())
        ref_center = get_reference_center(meshes, keyword="table")
        filtered = [m for m in meshes if include_table or "table" not in m["name"].lower()]
        cropped = center_and_crop_meshes(filtered, ref_center, cube_half)
        mesh_path = out_cropped / f"step_{frame_idx:04d}.obj"
        if not mesh_path.exists():
            print(f"Saving mesh to {mesh_path}")
            save_meshes_as_obj(cropped, mesh_path)
        # if done:
        #     break

    actions_path = out_cropped.parent / "actions.npy"
    if not actions_path.exists():
        np.save(actions_path, actions[: len(frame_indices)])

    # record indices for downstream loaders (_traj_index/_frame_index expectations)
    meta_path = out_cropped.parent / "metadata.json"
    metadata = {
        "_traj_index": int(traj_index),
        "_frame_index": frame_indices,
        "_len": traj_len,
        "actions_file": actions_path.name,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def process_dataset(
    dataset: DatasetInput,
    output_root: Path,
    task_suite_name: str,
    task_id: int,
    max_episodes: Optional[int],
    max_steps: Optional[int],
    cube_half: float,
    include_table: bool,
    split: str,
    episode_offset: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> int:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = Path(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    camera_names = ["agentview", "robot0_eye_in_hand"]
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": 128,
        "camera_widths": 128,
        "camera_names": camera_names,
        "camera_depths": False,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    dataset_output = output_root / dataset.display_path
    episode_idx = 0
    global_ep = 0
    rlds_dataset = load_rlds_dataset(dataset.shards, split=split)
    for traj_i, traj in enumerate(rlds_dataset):
        # shard episodes across workers
        if traj_i % num_shards != shard_index:
            global_ep += 1
            continue
        if max_episodes is not None and episode_idx >= max_episodes:
            env.close()
            return episode_idx
        actions = np.asarray(traj["action"])
        actions = ensure_action_shape(actions, expected_dim=actions.shape[-1])
        frame_indices = np.asarray(traj["_frame_index"]).astype(int).tolist()
        traj_len = int(traj['_len'][0])
        if traj_len != len(frame_indices):
            raise ValueError(f"traj_len {traj_len} does not match frame_indices length {len(frame_indices)} for traj {traj_index_raw}")
        traj_index_raw = int(np.asarray(traj["_traj_index"][0]))
        if max_steps is not None:
            actions = actions[:max_steps]
            frame_indices = frame_indices[: len(actions)]
            traj_len = min(traj_len, len(frame_indices))
        env.reset()
        init_obs = env.set_init_state(init_states[0])
        traj_index = episode_offset + traj_index_raw
        if len(actions) != len(frame_indices):
            raise ValueError(
                f"Action length {len(actions)} does not match _frame_index length {len(frame_indices)} for traj {traj_index}"
            )
        episode_dir_name = f"episode_{traj_index:05d}"
        ep_dir = dataset_output / episode_dir_name
        cropped_dir = ep_dir / "cropped_scene"
        replay_episode(
            env=env,
            actions=actions,
            out_cropped=cropped_dir,
            cube_half=cube_half,
            camera_names=camera_names,
            traj_index=traj_index,
            frame_indices=frame_indices,
            traj_len=traj_len,
            init_obs=init_obs,
            include_table=include_table,
        )
        episode_idx += 1
        global_ep += 1
    env.close()
    return episode_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["/mnt/data/modified_libero_rlds/libero_goal_no_noops/1.0.0/libero_goal-train.tfrecord-*"],
        help="Glob patterns or dataset directories containing TFRecord shards.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/data/libero/modified_libero_whole_mesh",
        help="Base directory to write meshes into.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load via dlimp/tfds (e.g., train, val, all).",
    )
    parser.add_argument(
        "--task-suite",
        default="libero_goal",
        help="LIBERO task suite name (e.g., libero_10, libero_spatial).",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task id within the suite to replay.",
    )
    parser.add_argument(
        "--cube-half",
        type=float,
        default=0.5,
        help="Half-edge length (meters) of the cube crop around the table center.",
    )
    parser.add_argument(
        "--exclude-table",
        action="store_true",
        help="If set, remove meshes whose name contains 'table' from the saved outputs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Worker index [0, num_shards-1] for sharding episodes.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of workers to shard episodes across.",
    )
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Offset to add to episode numbering (useful when sharding across multiple workers).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on number of episodes to export.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = collect_dataset_inputs(args.inputs)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for dataset in tqdm(datasets):
        exported = process_dataset(
            dataset,
            output_root,
            args.task_suite,
            args.task_id,
            args.max_episodes,
            args.max_steps,
            args.cube_half,
            split=args.split,
            include_table=not args.exclude_table,
            episode_offset=args.episode_offset,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        print(f"{dataset.display_path}: exported {exported} episodes to {output_root/dataset.display_path}")
        total += exported
    print(f"Done. Total episodes exported: {total}")


if __name__ == "__main__":
    main()
