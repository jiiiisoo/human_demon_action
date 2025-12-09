#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl
from tqdm import tqdm


def _patch_dlimp_options():
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


def load_rlds(shards, split="train"):
    _patch_dlimp_options()
    shards = list(map(str, shards))

    # TFDS builder 자동 추론
    version_dir = Path(shards[0]).parent
    dataset_dir = version_dir.parent
    data_root = dataset_dir.parent
    dataset_name = dataset_dir.name

    builder = tfds.builder(dataset_name, data_dir=str(data_root))

    return dl.DLataset.from_rlds(builder, split=split, shuffle=False)


def compute_episode_hash(traj):
    actions = traj["action"].numpy()
    frame_idx = np.asarray(traj["_frame_index"])
    h = hashlib.sha1()
    h.update(actions.tobytes())
    h.update(frame_idx.tobytes())
    return h.hexdigest()[:16]


def rewrite_rlds_with_trackfile(old_shards, new_folder):
    old_shards = sorted(list(map(Path, old_shards)))
    new_folder = Path(new_folder)
    new_folder.mkdir(parents=True, exist_ok=True)

    # 새 TFRecordWriter를 shard 개수만큼 생성
    writers = []
    for old_path in old_shards:
        new_path = new_folder / old_path.name  # ← same filename
        writers.append(tf.io.TFRecordWriter(str(new_path)))
        print(f"[writer] {new_path}")

    dataset = load_rlds(old_shards)

    shard_idx = 0
    for traj in tqdm(dataset, desc="Adding track_file"):
        writer = writers[shard_idx]
        shard_idx = (shard_idx + 1) % len(writers)

        # hash 생성
        h = compute_episode_hash(traj)
        track_str = f"episode_{h}".encode("utf-8")

        # 기존 example 불러오기
        ex = traj.to_tf_example()
        feat = ex.features.feature

        # track_file 추가 (bytes field)
        feat["observation/track_file"].bytes_list.value[:] = [track_str]

        writer.write(ex.SerializeToString())

    for w in writers:
        w.close()

    print("[Done] Finished writing modified RLDS.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/mnt/data/modified_libero_rlds/libero_goal_no_noops/1.0.0/libero_goal-train.tfrecord-*",
        help="glob for original RLDS",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/modified_libero_rlds_track/libero_goal_no_noops/1.0.0/",
        help="Folder to write new RLDS",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    old_shards = sorted(tf.io.gfile.glob(args.input))

    print("[Input shards]")
    for s in old_shards:
        print(" ", s)

    rewrite_rlds_with_trackfile(old_shards, args.output_dir)


if __name__ == "__main__":
    main()
