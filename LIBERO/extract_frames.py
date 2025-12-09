#!/usr/bin/env python3
"""Extract RGB frames from LIBERO TFRecord shards without TensorFlow.

This script parses TFRecord SequenceExamples directly in Python so that we
can export the jpeg-encoded camera observations (`steps/observation/image`
and `steps/observation/wrist_image`) into a local directory for quick
inspection (defaults to ./frames_js).
"""

from __future__ import annotations

import argparse
import glob
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class Feature:
    """Subset of tf.train.Feature for the fields we need."""

    bytes_list: Optional[List[bytes]] = None
    float_list: Optional[List[float]] = None
    int64_list: Optional[List[int]] = None


@dataclass
class DatasetInput:
    """Represents a dataset directory and the shards it contains."""

    display_path: Path
    shards: List[Path]


def _relative_dataset_path(dataset_root: Path) -> Path:
    resolved = dataset_root.resolve()
    cwd = Path.cwd().resolve()
    try:
        relative = resolved.relative_to(cwd)
        if not relative.parts:
            return Path(resolved.name)
        return relative
    except ValueError:
        return Path(resolved.name)


def collect_dataset_inputs(patterns: List[str]) -> List[DatasetInput]:
    if not patterns:
        raise ValueError("At least one input path or pattern is required")
    dataset_map: Dict[Path, DatasetInput] = {}
    dataset_order: List[Path] = []
    seen_shards: set[Path] = set()
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched input pattern: {pattern}")
        for match in matches:
            if match.is_dir():
                dataset_root = match.resolve()
                shards = sorted(dataset_root.glob("*.tfrecord-*"))
                if not shards:
                    raise FileNotFoundError(
                        f"No TFRecord shards found in directory: {match}"
                    )
            elif match.is_file():
                dataset_root = match.parent.resolve()
                shards = [match.resolve()]
            else:
                continue
            if dataset_root not in dataset_map:
                dataset_map[dataset_root] = DatasetInput(
                    display_path=_relative_dataset_path(dataset_root), shards=[]
                )
                dataset_order.append(dataset_root)
            entry = dataset_map[dataset_root]
            for shard in shards:
                shard_resolved = shard.resolve()
                if not shard_resolved.is_file():
                    continue
                if shard_resolved in seen_shards:
                    continue
                seen_shards.add(shard_resolved)
                entry.shards.append(shard_resolved)
    dataset_inputs: List[DatasetInput] = []
    for dataset_root in dataset_order:
        entry = dataset_map[dataset_root]
        if not entry.shards:
            raise FileNotFoundError(
                f"No TFRecord shards collected for dataset {entry.display_path}"
            )
        entry.shards.sort()
        dataset_inputs.append(entry)
    return dataset_inputs


def read_varint(buffer: bytes, idx: int) -> Tuple[int, int]:
    """Decode a protobuf varint starting at ``idx``."""

    shift = 0
    value = 0
    while True:
        if idx >= len(buffer):
            raise ValueError("Malformed varint: truncated input")
        byte = buffer[idx]
        idx += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, idx
        shift += 7
        if shift >= 64:
            raise ValueError("Malformed varint: exceeds 64 bits")


def skip_value(buffer: bytes, idx: int, wire_type: int) -> int:
    """Skip over an unknown protobuf field."""

    if wire_type == 0:  # varint
        while True:
            if idx >= len(buffer):
                raise ValueError("Malformed varint while skipping")
            if not (buffer[idx] & 0x80):
                return idx + 1
            idx += 1
    elif wire_type == 1:  # 64-bit
        return idx + 8
    elif wire_type == 2:  # length-delimited
        length, idx = read_varint(buffer, idx)
        return idx + length
    elif wire_type == 5:  # 32-bit
        return idx + 4
    else:
        raise ValueError(f"Unsupported wire type {wire_type}")


def parse_bytes_list(buffer: bytes) -> List[bytes]:
    values: List[bytes] = []
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 1 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            values.append(buffer[idx : idx + length])
            idx += length
        else:
            idx = skip_value(buffer, idx, wire_type)
    return values


def parse_float_list(buffer: bytes) -> List[float]:
    values: List[float] = []
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number != 1:
            idx = skip_value(buffer, idx, wire_type)
            continue
        if wire_type == 5:
            values.append(struct.unpack("<f", buffer[idx : idx + 4])[0])
            idx += 4
        elif wire_type == 2:
            length, idx = read_varint(buffer, idx)
            chunk = buffer[idx : idx + length]
            idx += length
            for start in range(0, len(chunk), 4):
                values.append(struct.unpack("<f", chunk[start : start + 4])[0])
        else:
            idx = skip_value(buffer, idx, wire_type)
    return values


def parse_int64_list(buffer: bytes) -> List[int]:
    values: List[int] = []
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number != 1:
            idx = skip_value(buffer, idx, wire_type)
            continue
        if wire_type == 0:
            value, idx = read_varint(buffer, idx)
            if value >= 1 << 63:
                value -= 1 << 64
            values.append(value)
        elif wire_type == 2:
            length, idx = read_varint(buffer, idx)
            end = idx + length
            while idx < end:
                value, idx = read_varint(buffer, idx)
                if value >= 1 << 63:
                    value -= 1 << 64
                values.append(value)
        else:
            idx = skip_value(buffer, idx, wire_type)
    return values


def parse_feature(buffer: bytes) -> Feature:
    feature = Feature()
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 1 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            feature.bytes_list = parse_bytes_list(buffer[idx : idx + length])
            idx += length
        elif field_number == 2 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            feature.float_list = parse_float_list(buffer[idx : idx + length])
            idx += length
        elif field_number == 3 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            feature.int64_list = parse_int64_list(buffer[idx : idx + length])
            idx += length
        else:
            idx = skip_value(buffer, idx, wire_type)
    return feature


def parse_features(buffer: bytes) -> Dict[str, Feature]:
    result: Dict[str, Feature] = {}
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number != 1 or wire_type != 2:
            idx = skip_value(buffer, idx, wire_type)
            continue
        length, idx = read_varint(buffer, idx)
        entry_buf = buffer[idx : idx + length]
        idx += length
        key, value = parse_feature_entry(entry_buf)
        if key is not None and value is not None:
            result[key] = value
    return result


def parse_feature_entry(buffer: bytes) -> Tuple[Optional[str], Optional[Feature]]:
    key: Optional[str] = None
    value: Optional[Feature] = None
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 1 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            key = buffer[idx : idx + length].decode("utf-8")
            idx += length
        elif field_number == 2 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            value = parse_feature(buffer[idx : idx + length])
            idx += length
        else:
            idx = skip_value(buffer, idx, wire_type)
    return key, value


def parse_feature_lists(buffer: bytes) -> Dict[str, List[Feature]]:
    result: Dict[str, List[Feature]] = {}
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number != 1 or wire_type != 2:
            idx = skip_value(buffer, idx, wire_type)
            continue
        length, idx = read_varint(buffer, idx)
        entry_buf = buffer[idx : idx + length]
        idx += length
        key, features = parse_feature_list_entry(entry_buf)
        if key is not None and features is not None:
            result[key] = features
    return result


def parse_feature_list_entry(buffer: bytes) -> Tuple[Optional[str], Optional[List[Feature]]]:
    key: Optional[str] = None
    value: Optional[List[Feature]] = None
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 1 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            key = buffer[idx : idx + length].decode("utf-8")
            idx += length
        elif field_number == 2 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            value = parse_feature_list(buffer[idx : idx + length])
            idx += length
        else:
            idx = skip_value(buffer, idx, wire_type)
    return key, value


def parse_feature_list(buffer: bytes) -> List[Feature]:
    features: List[Feature] = []
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 1 and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            features.append(parse_feature(buffer[idx : idx + length]))
            idx += length
        else:
            idx = skip_value(buffer, idx, wire_type)
    return features


def parse_sequence_example(buffer: bytes) -> Tuple[Dict[str, Feature], Dict[str, List[Feature]]]:
    context: Dict[str, Feature] = {}
    feature_lists: Dict[str, List[Feature]] = {}
    idx = 0
    while idx < len(buffer):
        tag, idx = read_varint(buffer, idx)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number in (1, 2) and wire_type == 2:
            length, idx = read_varint(buffer, idx)
            segment = buffer[idx : idx + length]
            idx += length
            if field_number == 1:
                context = parse_features(segment)
            else:
                feature_lists = parse_feature_lists(segment)
        else:
            idx = skip_value(buffer, idx, wire_type)
    return context, feature_lists


def write_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("Varint encoding expects non-negative integers")
    pieces = []
    while True:
        to_write = value & 0x7F
        value >>= 7
        if value:
            pieces.append(to_write | 0x80)
        else:
            pieces.append(to_write)
            break
    return bytes(pieces)


def _serialize_bytes_list(values: List[bytes]) -> bytes:
    parts = []
    for value in values:
        parts.append(b"\x0a" + write_varint(len(value)) + value)
    return b"".join(parts)


def _serialize_float_list(values: List[float]) -> bytes:
    if not values:
        return b""
    chunk = b"".join(struct.pack("<f", v) for v in values)
    return b"\x0a" + write_varint(len(chunk)) + chunk


def _serialize_int64_list(values: List[int]) -> bytes:
    if not values:
        return b""
    chunk_parts = []
    for value in values:
        if value < 0:
            encoded = (value + (1 << 64)) & ((1 << 64) - 1)
        else:
            encoded = value
        chunk_parts.append(write_varint(encoded))
    chunk = b"".join(chunk_parts)
    return b"\x0a" + write_varint(len(chunk)) + chunk


def serialize_feature(feature: Feature) -> bytes:
    parts = []
    if feature.bytes_list:
        payload = _serialize_bytes_list(feature.bytes_list)
        parts.append(b"\x0a" + write_varint(len(payload)) + payload)
    if feature.float_list:
        payload = _serialize_float_list(feature.float_list)
        parts.append(b"\x12" + write_varint(len(payload)) + payload)
    if feature.int64_list:
        payload = _serialize_int64_list(feature.int64_list)
        parts.append(b"\x1a" + write_varint(len(payload)) + payload)
    return b"".join(parts)


def _serialize_key_value(key: str, value_bytes: bytes) -> bytes:
    key_bytes = key.encode("utf-8")
    payload = b"\x0a" + write_varint(len(key_bytes)) + key_bytes
    payload += b"\x12" + write_varint(len(value_bytes)) + value_bytes
    return payload


def serialize_features(features: Dict[str, Feature]) -> bytes:
    parts = []
    for key, feature in features.items():
        value_bytes = serialize_feature(feature)
        entry = _serialize_key_value(key, value_bytes)
        parts.append(b"\x0a" + write_varint(len(entry)) + entry)
    return b"".join(parts)


def _serialize_feature_list(feature_list: List[Feature]) -> bytes:
    parts = []
    for feature in feature_list:
        feature_bytes = serialize_feature(feature)
        parts.append(b"\x0a" + write_varint(len(feature_bytes)) + feature_bytes)
    return b"".join(parts)


def serialize_feature_lists(feature_lists: Dict[str, List[Feature]]) -> bytes:
    parts = []
    for key, feature_list in feature_lists.items():
        list_payload = _serialize_feature_list(feature_list)
        entry = _serialize_key_value(key, list_payload)
        parts.append(b"\x0a" + write_varint(len(entry)) + entry)
    return b"".join(parts)


def serialize_sequence_example(
    context: Dict[str, Feature], feature_lists: Dict[str, List[Feature]]
) -> bytes:
    parts = []
    if context:
        ctx_bytes = serialize_features(context)
        parts.append(b"\x0a" + write_varint(len(ctx_bytes)) + ctx_bytes)
    if feature_lists:
        fl_bytes = serialize_feature_lists(feature_lists)
        parts.append(b"\x12" + write_varint(len(fl_bytes)) + fl_bytes)
    return b"".join(parts)


def iter_tfrecord_records(path: Path) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            length_bytes = f.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord in {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            f.seek(4, io.SEEK_CUR)  # skip length CRC
            data = f.read(length)
            if len(data) != length:
                raise ValueError(f"TFRecord data truncated in {path}")
            f.seek(4, io.SEEK_CUR)  # skip data CRC
            yield data


def _build_crc32c_table() -> List[int]:
    poly = 0x1EDC6F41
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        table.append(crc)
    return table


_CRC32C_TABLE = _build_crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data) & 0xFFFFFFFF
    return (((crc >> 15) | ((crc & 0xFFFFFFFF) << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def write_tfrecord_record(f, record_bytes: bytes) -> None:
    length = len(record_bytes)
    length_bytes = struct.pack("<Q", length)
    f.write(length_bytes)
    f.write(struct.pack("<I", masked_crc32c(length_bytes)))
    f.write(record_bytes)
    f.write(struct.pack("<I", masked_crc32c(record_bytes)))


CAMERA_FIELDS = {
    "main": "steps/observation/image",
    "wrist": "steps/observation/wrist_image",
}


def feature_bytes_sequence(feature_list: List[Feature]) -> List[bytes]:
    values: List[bytes] = []
    for feature in feature_list:
        if feature.bytes_list:
            values.append(feature.bytes_list[0])
    return values


def feature_bytes_values(feature: Optional[Feature]) -> List[bytes]:
    if feature is None or not feature.bytes_list:
        return []
    return list(feature.bytes_list)


def ensure_output_dir(base: Path, episode_idx: int, camera: str) -> Path:
    episode_dir = base / f"episode_{episode_idx:05d}"
    camera_dir = episode_dir / camera
    camera_dir.mkdir(parents=True, exist_ok=True)
    return camera_dir


def extract_frames_for_dataset(
    shard_paths: Iterable[Path],
    dataset_output: Path,
    cameras: List[str],
    max_episodes: Optional[int],
    max_steps: Optional[int],
) -> int:
    dataset_output.mkdir(parents=True, exist_ok=True)
    episode_idx = 0
    for shard_path in shard_paths:
        for record_bytes in iter_tfrecord_records(shard_path):
            if max_episodes is not None and episode_idx >= max_episodes:
                return episode_idx
            context, feature_lists = parse_sequence_example(record_bytes)
            for camera in cameras:
                field_name = CAMERA_FIELDS[camera]
                if field_name in feature_lists:
                    frames = feature_bytes_sequence(feature_lists[field_name])
                else:
                    frames = feature_bytes_values(context.get(field_name))
                if not frames:
                    continue
                camera_dir = ensure_output_dir(dataset_output, episode_idx, camera)
                for step_idx, frame_bytes in enumerate(frames):
                    if max_steps is not None and step_idx >= max_steps:
                        break
                    frame_path = camera_dir / f"{camera}_step_{step_idx:04d}.jpg"
                    with frame_path.open("wb") as img_file:
                        img_file.write(frame_bytes)
            episode_idx += 1
    return episode_idx


def extract_frames(
    datasets: List[DatasetInput],
    output_root: Path,
    cameras: List[str],
    max_episodes: Optional[int],
    max_steps: Optional[int],
) -> Dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}
    for dataset in datasets:
        dataset_output = output_root / dataset.display_path
        exported = extract_frames_for_dataset(
            dataset.shards, dataset_output, cameras, max_episodes, max_steps
        )
        summary[str(dataset.display_path)] = exported
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-*"]
        ,
        help="Glob patterns or dataset directories that contain TFRecord shards.",
    )
    parser.add_argument(
        "--output-dir",
        default="frames_js",
        help="Where to place the exported frames.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=sorted(CAMERA_FIELDS),
        default=["main", "wrist"],
        help="Which camera streams to export.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episodes to export.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on the number of steps per episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_inputs = collect_dataset_inputs(args.inputs)
    summary = extract_frames(
        dataset_inputs,
        Path(args.output_dir),
        args.cameras,
        args.max_episodes,
        args.max_steps,
    )
    total = sum(summary.values())
    output_base = Path(args.output_dir)
    if len(summary) == 1:
        dataset, count = next(iter(summary.items()))
        destination = output_base / Path(dataset)
        print(f"Exported frames for {count} episodes from {dataset} into {destination}")
    else:
        for dataset, count in summary.items():
            destination = output_base / Path(dataset)
            print(f"- {dataset}: {count} episodes → {destination}")
        print(
            f"Exported {total} episodes across {len(summary)} datasets into {args.output_dir}"
        )


if __name__ == "__main__":
    main()
