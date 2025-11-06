"""
Test different frame offsets when attaching two clips.

This script takes two video files and generates multiple test videos,
each with a different frame offset for the second clip, to help identify
the correct alignment.
"""

import sys
import subprocess
from pathlib import Path
import cv2
import numpy as np

def get_video_frame_count(video_path):
    """Get the total frame count of a video."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps

def extract_frames(video_path, start_frame=0, num_frames=None):
    """Extract frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_read = num_frames if num_frames is not None else (total_frames - start_frame)
    
    frames = []
    for i in range(frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def create_video_from_frames(frames, output_path, fps):
    """Create a video from a list of frames."""
    if not frames:
        return None
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path

def add_text_overlay(frame, text, position='top'):
    """Add text overlay to a frame."""
    frame_copy = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    font_thickness = 4
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Calculate position
    if position == 'top':
        x = (frame.shape[1] - text_width) // 2
        y = text_height + 50
    else:  # bottom
        x = (frame.shape[1] - text_width) // 2
        y = frame.shape[0] - 50
    
    # Draw background rectangle
    padding = 20
    cv2.rectangle(frame_copy, 
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + padding),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame_copy, text, (x, y), font, font_scale, (255, 255, 255), font_thickness)
    
    return frame_copy

def test_offset(clip1_path, clip2_path, offset, output_dir, blend_frames=3):
    """
    Test a specific offset for attaching clip2 to clip1.
    
    Args:
        clip1_path: Path to first clip
        clip2_path: Path to second clip
        offset: Frame offset to apply (negative = earlier, positive = later)
        output_dir: Directory to save output
        blend_frames: Number of frames to blend at boundary
    """
    print(f"\n{'='*80}")
    print(f"Testing offset: {offset:+d} frames")
    print(f"{'='*80}")
    
    # Get video info
    clip1_frame_count, fps = get_video_frame_count(clip1_path)
    clip2_frame_count, _ = get_video_frame_count(clip2_path)
    
    print(f"Clip1: {clip1_frame_count} frames @ {fps} fps")
    print(f"Clip2: {clip2_frame_count} frames @ {fps} fps")
    
    # Extract all frames from clip1
    print("Extracting frames from clip1...")
    clip1_frames = extract_frames(clip1_path)
    
    # Calculate clip2 start frame based on offset
    # Positive offset = skip more frames (shift forward)
    # Negative offset = skip fewer frames (shift backward)
    base_skip = 0  # You can adjust this based on what the "correct" offset should be near
    clip2_start_frame = base_skip + offset
    
    if clip2_start_frame < 0:
        print(f"WARNING: Offset {offset} results in negative start frame ({clip2_start_frame}). Clamping to 0.")
        clip2_start_frame = 0
    
    if clip2_start_frame >= clip2_frame_count:
        print(f"ERROR: Offset {offset} results in start frame beyond video length. Skipping.")
        return None
    
    print(f"Clip2 will start at frame {clip2_start_frame} (offset: {offset:+d})")
    
    # Extract frames from clip2 starting at the offset
    print("Extracting frames from clip2...")
    clip2_frames = extract_frames(clip2_path, start_frame=clip2_start_frame)
    
    # Blend the boundary
    print(f"Blending {blend_frames} frames at boundary...")
    combined_frames = []
    
    # Add all clip1 frames except the last blend_frames
    combined_frames.extend(clip1_frames[:-blend_frames])
    
    # Crossfade the overlapping region
    for i in range(blend_frames):
        alpha = (i + 1) / float(blend_frames)
        
        frame1 = clip1_frames[-(blend_frames - i)].astype(np.float32)
        frame2 = clip2_frames[i].astype(np.float32)
        
        # Ensure same dimensions
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        blended = ((1 - alpha) * frame1 + alpha * frame2).astype(np.uint8)
        combined_frames.append(blended)
    
    # Add remaining clip2 frames
    combined_frames.extend(clip2_frames[blend_frames:])
    
    # Add text overlay to first and last few frames to show offset
    overlay_text = f"Offset: {offset:+d}"
    for i in range(min(30, len(combined_frames))):  # First 30 frames
        combined_frames[i] = add_text_overlay(combined_frames[i], overlay_text, 'top')
    
    # Output filename
    output_path = output_dir / f"test_offset_{offset:+03d}.mp4"
    
    print(f"Creating output video: {output_path}")
    print(f"Total frames: {len(combined_frames)}")
    
    # Create video
    create_video_from_frames(combined_frames, output_path, fps)
    
    print(f"✓ Created: {output_path}")
    return output_path

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_clip_offsets.py <clip1.mp4> <clip2.mp4> [blend_frames]")
        print("\nExample:")
        print("  python test_clip_offsets.py Wan2GP/outputs/clip-1.mp4 bridge.mp4")
        print("  python test_clip_offsets.py Wan2GP/outputs/clip-1.mp4 bridge.mp4 5")
        sys.exit(1)
    
    clip1_path = Path(sys.argv[1])
    clip2_path = Path(sys.argv[2])
    blend_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    if not clip1_path.exists():
        print(f"Error: Clip1 not found: {clip1_path}")
        sys.exit(1)
    
    if not clip2_path.exists():
        print(f"Error: Clip2 not found: {clip2_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("test_offsets")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("CLIP OFFSET TESTING")
    print("="*80)
    print(f"Clip1: {clip1_path}")
    print(f"Clip2: {clip2_path}")
    print(f"Blend frames: {blend_frames}")
    print(f"Testing offsets: -4 to +4")
    print(f"Output directory: {output_dir}")
    
    # Test offsets from -4 to +4
    results = []
    for offset in range(-4, 5):
        result = test_offset(clip1_path, clip2_path, offset, output_dir, blend_frames)
        if result:
            results.append((offset, result))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Generated {len(results)} test videos:")
    for offset, path in results:
        print(f"  Offset {offset:+2d}: {path}")
    print(f"\nAll videos saved to: {output_dir.absolute()}")
    print("\nReview each video to find the offset with the smoothest transition.")

if __name__ == "__main__":
    main()
