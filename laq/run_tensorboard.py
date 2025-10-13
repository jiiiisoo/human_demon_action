#!/usr/bin/env python3
"""
Simple script to run TensorBoard for LAQ training logs.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_tensorboard(log_dir="./runs", port=6006):
    """Run TensorBoard with the specified log directory."""
    
    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"❌ Log directory '{log_dir}' does not exist!")
        print("Make sure you've run training with TensorBoard enabled first.")
        return
    
    # Check if there are any log files
    log_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not log_files:
        print(f"⚠️  No TensorBoard log files found in '{log_dir}'")
        print("Training may not have started yet or TensorBoard logging is disabled.")
        return
    
    print(f"🚀 Starting TensorBoard...")
    print(f"📁 Log directory: {os.path.abspath(log_dir)}")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"📊 Found {len(log_files)} log file(s)")
    print("\nPress Ctrl+C to stop TensorBoard")
    
    try:
        # Run TensorBoard
        subprocess.run([
            sys.executable, "-m", "tensorboard.main", 
            "--logdir", log_dir,
            "--port", str(port),
            "--bind_all"  # Allow external access
        ])
    except KeyboardInterrupt:
        print("\n👋 TensorBoard stopped")
    except Exception as e:
        print(f"❌ Error running TensorBoard: {e}")
        print("Make sure TensorBoard is installed: pip install tensorboard")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TensorBoard for LAQ training")
    parser.add_argument("--log_dir", default="./runs", help="TensorBoard log directory")
    parser.add_argument("--port", type=int, default=6006, help="Port for TensorBoard")
    
    args = parser.parse_args()
    
    run_tensorboard(args.log_dir, args.port)
