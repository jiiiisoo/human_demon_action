#!/usr/bin/env python3
"""
Extract frames from webm videos for LAQ training.
Creates folder structure expected by ImageVideoDataset.
"""
import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, max_frames=100, sample_rate=1):
    """Extract frames from a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # print(f"Video: {video_path.name}, FPS: {fps:.1f}, Duration: {duration:.1f}s, Frames: {total_frames}")
    
    frame_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample frames based on sample_rate
        if frame_count % sample_rate == 0:
            frame_path = output_dir / f"frame{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_idx += 1
            
            if frame_idx >= max_frames:
                break
                
        frame_count += 1
    
    cap.release()
    
    # Calculate effective FPS
    effective_fps = fps / sample_rate if fps > 0 else 0
    # print(f"Extracted {frame_idx} frames at {effective_fps:.1f} FPS (sample_rate={sample_rate})")
    
    return frame_idx > 0

def process_dataset(input_dir, output_dir, max_videos=1000, max_frames_per_video=100, sample_rate=1):
    """Process the entire dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all webm files
    video_files = list(input_path.glob("*.webm"))
    print(f"Found {len(video_files)} video files")
    
    # Limit number of videos for initial training
    # video_files = video_files[:max_videos]
    print(f"Processing {len(video_files)} videos with sample_rate={sample_rate}")
    
    successful = 0
    for video_file in tqdm(video_files, desc="Extracting frames"):
        # Create folder for this video
        video_name = video_file.stem
        video_output_dir = output_path / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        # Extract frames
        if extract_frames_from_video(video_file, video_output_dir, max_frames_per_video, sample_rate):
            successful += 1
    
    print(f"Successfully processed {successful}/{len(video_files)} videos")
    print(f"Output directory: {output_path}")
    print(f"Sample rate: {sample_rate} (every {sample_rate} frame(s))")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from webm videos")
    parser.add_argument("--input_dir", required=True, help="Input directory with webm files")
    parser.add_argument("--output_dir", required=True, help="Output directory for frame sequences")
    parser.add_argument("--max_videos", type=int, default=1000, help="Maximum number of videos to process")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum frames per video")
    parser.add_argument("--sample_rate", type=int, default=1, help="Sample every N frames (1=all frames, 5=every 5th frame)")
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames,
        sample_rate=args.sample_rate
    )
