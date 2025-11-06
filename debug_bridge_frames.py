#!/usr/bin/env python3
"""
Debug script to examine actual frame content in the bridge video
"""
import sys
from pathlib import Path
import cv2

def analyze_bridge_and_clips(bridge_path, clip2_path, context=24):
    """Compare frames in bridge vs clip2 to see what's actually preserved"""

    print("="*80)
    print("BRIDGE FRAME ANALYSIS")
    print("="*80)

    # Open videos
    bridge_cap = cv2.VideoCapture(str(bridge_path))
    clip2_cap = cv2.VideoCapture(str(clip2_path))

    bridge_frame_count = int(bridge_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip2_frame_count = int(clip2_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nBridge: {bridge_frame_count} frames")
    print(f"Clip2: {clip2_frame_count} frames")

    # Extract all frames from bridge
    bridge_frames = []
    while True:
        ret, frame = bridge_cap.read()
        if not ret:
            break
        bridge_frames.append(frame)

    # Extract first 30 frames from clip2 for comparison
    clip2_frames = []
    for i in range(min(30, clip2_frame_count)):
        clip2_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = clip2_cap.read()
        if ret:
            clip2_frames.append(frame)

    bridge_cap.release()
    clip2_cap.release()

    print(f"\nExtracted {len(bridge_frames)} bridge frames")
    print(f"Extracted {len(clip2_frames)} clip2 frames for comparison")

    # Compare last 20 frames of bridge with clip2 frames
    print(f"\n{'='*80}")
    print("COMPARING BRIDGE END WITH CLIP2")
    print("="*80)

    start_bridge_idx = max(0, len(bridge_frames) - 20)

    for bridge_idx in range(start_bridge_idx, len(bridge_frames)):
        bridge_frame = bridge_frames[bridge_idx]

        # Try to find matching frame in clip2
        best_match_idx = None
        best_match_score = float('inf')

        for clip2_idx in range(len(clip2_frames)):
            clip2_frame = clip2_frames[clip2_idx]

            # Calculate MSE
            diff = cv2.absdiff(bridge_frame, clip2_frame)
            mse = (diff ** 2).mean()

            if mse < best_match_score:
                best_match_score = mse
                best_match_idx = clip2_idx

        match_quality = "EXACT" if best_match_score < 1 else "CLOSE" if best_match_score < 100 else "DIFFERENT"

        print(f"  Bridge[{bridge_idx:2d}] matches Clip2[{best_match_idx:2d}] (MSE: {best_match_score:8.2f}) [{match_quality}]")

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print("="*80)
    print("Look at the last 3 frames of the bridge.")
    print("They should match clip2[21:24] if gap=13 calculation is correct.")
    print("If they match different frames, that tells us what's actually in the bridge!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_bridge_frames.py <bridge.mp4> <clip2.mp4>")
        sys.exit(1)

    bridge_path = Path(sys.argv[1])
    clip2_path = Path(sys.argv[2])

    if not bridge_path.exists():
        print(f"Error: Bridge not found: {bridge_path}")
        sys.exit(1)

    if not clip2_path.exists():
        print(f"Error: Clip2 not found: {clip2_path}")
        sys.exit(1)

    analyze_bridge_and_clips(bridge_path, clip2_path)
