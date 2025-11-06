"""
Debug script to verify frame alignment at transition boundaries.
This extracts actual frames and compares them to identify misalignments.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def extract_frame(video_path, frame_idx):
    """Extract a single frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def compare_frames(frame1, frame2):
    """Calculate similarity between two frames (MSE and PSNR)."""
    if frame1 is None or frame2 is None:
        return None, None
    if frame1.shape != frame2.shape:
        return None, None
    
    mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    return mse, psnr

def debug_transition_alignment(
    transition_video_path,
    clip2_path,
    blend_frames=3,
    context_frame_count=8,
    regenerate_anchors=True,
    num_anchor_frames=3
):
    """
    Debug the transition→clip2 boundary alignment.
    
    This checks if the preserved frames in the transition match the corresponding
    frames in clip2.
    """
    
    transition_video_path = Path(transition_video_path)
    clip2_path = Path(clip2_path)
    
    if not transition_video_path.exists():
        print(f"❌ Transition video not found: {transition_video_path}")
        return
    
    if not clip2_path.exists():
        print(f"❌ Clip2 not found: {clip2_path}")
        return
    
    # Get frame counts
    cap_trans = cv2.VideoCapture(str(transition_video_path))
    total_transition_frames = int(cap_trans.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_trans.release()
    
    cap_clip2 = cv2.VideoCapture(str(clip2_path))
    total_clip2_frames = int(cap_clip2.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_clip2.release()
    
    print("=" * 80)
    print("TRANSITION → CLIP2 BOUNDARY ALIGNMENT DEBUG")
    print("=" * 80)
    print(f"Transition video: {transition_video_path.name}")
    print(f"Transition frames: {total_transition_frames}")
    print(f"Clip2: {clip2_path.name}")
    print(f"Clip2 frames: {total_clip2_frames}")
    print(f"Blend frames: {blend_frames}")
    print()
    
    # Calculate expected preserved section in transition
    if regenerate_anchors:
        num_anchor_frames_after = min(num_anchor_frames, context_frame_count)
        preserved_section_start_in_clip2 = num_anchor_frames_after
    else:
        preserved_section_start_in_clip2 = 0
    
    preserved_section_end_in_clip2 = context_frame_count
    
    # Calculate where preserved section appears in transition
    # It should be at the end of the transition
    gap_frame_count = total_transition_frames - (context_frame_count * 2)
    
    if gap_frame_count >= 0:
        # INSERT mode
        preserved_start_in_transition = total_transition_frames - (context_frame_count - num_anchor_frames_after)
    else:
        print("⚠️  Unable to determine mode - frame count doesn't match expected INSERT mode")
        return
    
    print(f"Expected preserved section:")
    print(f"  Transition frames [{preserved_start_in_transition}:{total_transition_frames}]")
    print(f"  Should match clip2[{preserved_section_start_in_clip2}:{preserved_section_end_in_clip2}]")
    print()
    
    # Verify preserved frames
    print("Verifying preserved frame correspondence:")
    print("-" * 80)
    
    all_match = True
    for i in range(context_frame_count - num_anchor_frames_after):
        trans_idx = preserved_start_in_transition + i
        clip2_idx = preserved_section_start_in_clip2 + i
        
        frame_trans = extract_frame(transition_video_path, trans_idx)
        frame_clip2 = extract_frame(clip2_path, clip2_idx)
        
        mse, psnr = compare_frames(frame_trans, frame_clip2)
        
        if mse is None:
            print(f"Frame {i}: ❌ Failed to extract or compare")
            all_match = False
        elif mse < 1.0:  # Nearly identical (accounting for compression)
            print(f"Frame {i}: ✓ transition[{trans_idx}] ≈ clip2[{clip2_idx}] (MSE: {mse:.2f}, PSNR: {psnr:.1f}dB)")
        else:
            print(f"Frame {i}: ⚠️  transition[{trans_idx}] ≠ clip2[{clip2_idx}] (MSE: {mse:.2f}, PSNR: {psnr:.1f}dB)")
            all_match = False
    
    print()
    
    # Now check what frames_to_skip_clip2 should be for correct blending
    frames_to_skip_clip2 = context_frame_count - blend_frames
    
    print(f"Blend region check:")
    print(f"  Transition last {blend_frames} frames: [{total_transition_frames - blend_frames}:{total_transition_frames}]")
    print(f"  Clip2 trimmed starts at: clip2[{frames_to_skip_clip2}]")
    print(f"  Clip2 blend region: clip2[{frames_to_skip_clip2}:{frames_to_skip_clip2 + blend_frames}]")
    print()
    
    # Check if blend regions reference the same source frames
    print("Checking blend correspondence:")
    print("-" * 80)
    
    for i in range(blend_frames):
        trans_idx = total_transition_frames - blend_frames + i
        clip2_idx = frames_to_skip_clip2 + i
        
        # Calculate what clip2 frame the transition frame should contain
        offset_in_preserved = trans_idx - preserved_start_in_transition
        expected_clip2_frame = preserved_section_start_in_clip2 + offset_in_preserved
        
        print(f"Blend frame {i}:")
        print(f"  Transition[{trans_idx}] should contain clip2[{expected_clip2_frame}]")
        print(f"  Clip2_trimmed[{i}] = clip2[{clip2_idx}]")
        
        if expected_clip2_frame == clip2_idx:
            print(f"  ✓ MATCH")
        else:
            print(f"  ✗ MISMATCH! Off by {clip2_idx - expected_clip2_frame} frames")
            all_match = False
        
        # Also compare the actual frames
        frame_trans = extract_frame(transition_video_path, trans_idx)
        frame_clip2 = extract_frame(clip2_path, clip2_idx)
        
        mse, psnr = compare_frames(frame_trans, frame_clip2)
        if mse is not None:
            if mse < 10.0:
                print(f"  Frame comparison: Similar (MSE: {mse:.2f}, PSNR: {psnr:.1f}dB)")
            else:
                print(f"  Frame comparison: Different (MSE: {mse:.2f}, PSNR: {psnr:.1f}dB)")
    
    print()
    if all_match:
        print("✓ All alignments correct!")
    else:
        print("✗ Alignment issues detected!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_frame_alignment.py <transition_video> <clip2_video> [blend_frames] [context_frames] [regen_anchors] [num_anchors]")
        sys.exit(1)
    
    transition = sys.argv[1]
    clip2 = sys.argv[2]
    blend = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    context = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    regen = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else True
    num_anch = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    
    debug_transition_alignment(transition, clip2, blend, context, regen, num_anch)
