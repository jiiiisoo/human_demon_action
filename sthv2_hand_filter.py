#!/usr/bin/env python3
"""
Scan Something-Something V2 videos or extracted frame folders, run hand detection
per frame, and write a JSONL with counts:

{
  "video_id": "12345",
  "path": "/path/to/12345.webm" or folder,
  "num_frames": 120,
  "num_hand_frames": 87,
  "ratio": 0.725,
  "fps": 12.0
}

Supports two input modes:
1) --videos_dir: directory of .webm/.mp4 files (original SthV2)
2) --frames_dir: directory where each subfolder contains frames of a video

Requires: mediapipe, opencv-python
pip install mediapipe opencv-python
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

# Suppress verbose TF/MediaPipe logs (harmless warnings)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow C++ logs
os.environ.setdefault("GLOG_minloglevel", "2")      # absl/glog

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _mp_err = e


def get_hand_detector(static_image_mode: bool = False):
    if mp is None:
        raise RuntimeError(
            f"mediapipe is not available. Install with `pip install mediapipe`. Original error: {_mp_err}"
        )
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def detect_hands_in_frame(detector, bgr_image) -> bool:
    # Convert BGR to RGB for mediapipe
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    return results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0


def process_video(video_path: Path, sample_stride: int, detector) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_id": video_path.stem,
            "path": str(video_path),
            "num_frames": 0,
            "num_hand_frames": 0,
            "ratio": 0.0,
            "fps": 0.0,
            "error": "cannot_open",
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = 0
    hand = 0
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % sample_stride != 0:
            idx += 1
            continue
        ret, frame = cap.retrieve()
        if not ret:
            break
        total += 1
        try:
            if detect_hands_in_frame(detector, frame):
                hand += 1
        except Exception:
            pass
        idx += 1

    cap.release()
    ratio = float(hand) / float(total) if total > 0 else 0.0
    return {
        "video_id": video_path.stem,
        "path": str(video_path),
        "num_frames": total,
        "num_hand_frames": hand,
        "ratio": ratio,
        "fps": float(fps),
    }


def process_frames_folder(folder: Path, sample_stride: int, detector, img_glob: str = "*.jpg") -> dict:
    # Collect frame files
    frames = sorted(folder.glob(img_glob))
    if not frames:
        frames = sorted(folder.glob("*.png"))
    total = 0
    hand = 0
    for i, f in enumerate(frames):
        if i % sample_stride != 0:
            continue
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        total += 1
        try:
            if detect_hands_in_frame(detector, img):
                hand += 1
        except Exception:
            pass
    ratio = float(hand) / float(total) if total > 0 else 0.0
    return {
        "video_id": folder.name,
        "path": str(folder),
        "num_frames": total,
        "num_hand_frames": hand,
        "ratio": ratio,
        "fps": None,
    }


def write_jsonl_line(out_fp, record: dict):
    out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _video_worker(args_tuple):
    (video_path, sample_stride) = args_tuple
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    detector = get_hand_detector(static_image_mode=False)
    with detector:
        return process_video(video_path, sample_stride, detector)


def _frames_worker(args_tuple):
    (folder, sample_stride, img_glob) = args_tuple
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    detector = get_hand_detector(static_image_mode=False)
    with detector:
        return process_frames_folder(folder, sample_stride, detector, img_glob=img_glob)


def main():
    ap = argparse.ArgumentParser(description="SthV2 hand presence filter with MediaPipe Hands")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos_dir", type=str, help="Directory containing video files (.webm/.mp4)")
    src.add_argument("--frames_dir", type=str, help="Directory of frame folders (one folder per video)")
    ap.add_argument("--output_jsonl", type=str, required=True, help="Where to write summary JSONL")
    ap.add_argument("--sample_stride", type=int, default=1, help="Analyze every N-th frame (speed/accuracy tradeoff)")
    ap.add_argument("--img_glob", type=str, default="*.jpg", help="Glob pattern for frames when using --frames_dir")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers across videos/folders")
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    with open(args.output_jsonl, "w", encoding="utf-8") as out_fp:
        if args.videos_dir:
            root = Path(args.videos_dir)
            videos = list(root.glob("*.webm")) + list(root.glob("*.mp4")) + list(root.glob("*.mkv"))
            if args.workers <= 1:
                # single process
                for v in tqdm(videos):
                    rec = _video_worker((v, args.sample_stride))
                    write_jsonl_line(out_fp, rec)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = [ex.submit(_video_worker, (v, args.sample_stride)) for v in videos]
                    for f in tqdm(as_completed(futs), total=len(futs), desc="Videos"):
                        write_jsonl_line(out_fp, f.result())
        else:
            root = Path(args.frames_dir)
            folders = sorted([p for p in root.iterdir() if p.is_dir()])
            if args.workers <= 1:
                for folder in tqdm(folders):
                    rec = _frames_worker((folder, args.sample_stride, args.img_glob))
                    write_jsonl_line(out_fp, rec)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = [ex.submit(_frames_worker, (folder, args.sample_stride, args.img_glob)) for folder in folders]
                    for f in tqdm(as_completed(futs), total=len(futs), desc="Folders"):
                        write_jsonl_line(out_fp, f.result())

    print(f"Wrote results to {args.output_jsonl}")


if __name__ == "__main__":
    main()


