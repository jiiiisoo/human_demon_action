#!/usr/bin/env python3
"""
Read a JSONL file with entries like:
  {"video_id": ..., "num_frames": N, "num_hand_frames": M, "ratio": M/N}
and plot a histogram of the ratio distribution.

Usage:
  python plot_hand_ratio_hist.py \
    --jsonl /workspace/human_demon_action/sthv2_hand_counts_frames.jsonl \
    --out /workspace/human_demon_action/hand_ratio_hist.png \
    --bins 20
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ratios(jsonl_path: Path):
    ratios = []
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ratio = obj.get("ratio")
            num_frames = obj.get("num_frames", 0)
            if ratio is None:
                # derive if possible
                hand = obj.get("num_hand_frames")
                if hand is not None and num_frames:
                    ratio = float(hand) / float(num_frames)
            if ratio is None:
                continue
            # keep within [0,1]
            try:
                r = float(ratio)
            except Exception:
                continue
            if math.isfinite(r) and 0.0 <= r <= 1.0 and num_frames and num_frames > 0:
                ratios.append(r)
                total += 1
    return ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Input JSONL with ratio field")
    ap.add_argument("--out", required=True, help="Output PNG path for histogram")
    ap.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    args = ap.parse_args()

    ratios = load_ratios(Path(args.jsonl))
    if not ratios:
        print("No valid ratios found.")
        return

    plt.figure(figsize=(8, 5))
    counts, bin_edges, patches = plt.hist(
        ratios, bins=args.bins, range=(0.0, 1.0), color="#4C78A8", edgecolor="white"
    )

    # Highlight videos with exact 0 ratio in red
    zeros_count = sum(1 for r in ratios if r <= 1e-9)
    if patches:
        patches[0].set_facecolor("#E45756")  # red for the first bin (includes 0)
        # Place a red label above the first bar indicating the zero-count
        x0 = 0.5 * (bin_edges[0] + bin_edges[1])
        y0 = counts[0]
        if y0 > 0:
            plt.text(x0, y0, f"0: {int(zeros_count)}", color="#E45756", fontsize=9,
                     ha="center", va="bottom")
    plt.title("Hand Presence Ratio Distribution")
    plt.xlabel("ratio (num_hand_frames / num_frames)")
    plt.ylabel("# videos")
    plt.grid(alpha=0.2, linestyle=":")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved histogram to {args.out}; N={len(ratios)}")


if __name__ == "__main__":
    main()


