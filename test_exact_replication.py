#!/usr/bin/env python3

import requests
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

def download_video(url, filename):
    """Download video from URL"""
    print(f"Downloading {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

def get_video_info(video_path):
    """Get frame count and FPS from video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return frame_count, fps

def extract_frames_from_video(video_path):
    """Extract all frames from video as numpy arrays"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def sm_cross_fade_overlap_frames(frames_prev_segment, frames_curr_segment, current_overlap_val, mode, crossfade_sharp_amt):
    """Replicate the exact cross-fade logic from the source code"""
    if current_overlap_val <= 0:
        return []

    if len(frames_prev_segment) < current_overlap_val or len(frames_curr_segment) < current_overlap_val:
        print(f"WARNING: Not enough frames for overlap. prev: {len(frames_prev_segment)}, curr: {len(frames_curr_segment)}, overlap: {current_overlap_val}")
        return []

    # Get the overlapping sections
    prev_tail = frames_prev_segment[-current_overlap_val:]  # Last N frames of previous video
    curr_head = frames_curr_segment[:current_overlap_val]   # First N frames of current video
    
    faded_frames = []
    for i in range(current_overlap_val):
        alpha = (i + 1) / current_overlap_val  # Linear blend from 0 to 1
        
        # Blend the frames
        prev_frame = prev_tail[i].astype(np.float32)
        curr_frame = curr_head[i].astype(np.float32)
        
        blended = (1 - alpha) * prev_frame + alpha * curr_frame
        faded_frames.append(blended.astype(np.uint8))
    
    return faded_frames

def create_video_from_frames_list(frames, output_path, final_fps, parsed_res_wh):
    """Create video from list of frames"""
    if not frames:
        return None
    
    height, width = parsed_res_wh[1], parsed_res_wh[0]  # Note: parsed_res_wh is (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, final_fps, (width, height))
    
    for frame in frames:
        # Ensure frame is the right size
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    
    writer.release()
    return output_path

def replicate_orchestrator_quantization(segment_frames, frame_overlaps):
    """Replicate the exact quantization logic from the orchestrator"""
    print(f"\n=== ORCHESTRATOR QUANTIZATION LOGIC ===")
    print(f"Input segment_frames: {segment_frames}")
    print(f"Input frame_overlaps: {frame_overlaps}")
    
    # Quantize segment frames to 4*N+1 format
    quantized_segment_frames = []
    for i, frames in enumerate(segment_frames):
        new_frames = (frames // 4) * 4 + 1
        if new_frames != frames:
            print(f"Quantized segment {i} length from {frames} to {new_frames} (4*N+1 format).")
        quantized_segment_frames.append(new_frames)
    
    print(f"Quantized segment_frames: {quantized_segment_frames}")
    
    # Quantize overlaps
    quantized_frame_overlap = []
    num_overlaps_to_process = len(quantized_segment_frames) - 1
    
    if num_overlaps_to_process > 0:
        for i in range(num_overlaps_to_process):
            if i < len(frame_overlaps):
                original_overlap = frame_overlaps[i]
            else:
                original_overlap = 0
            
            # Overlap cannot be larger than the shorter of the two segments
            max_possible_overlap = min(quantized_segment_frames[i], quantized_segment_frames[i+1])
            
            # Quantize overlap to be even, then cap it
            new_overlap = (original_overlap // 2) * 2
            new_overlap = min(new_overlap, max_possible_overlap)
            if new_overlap < 0: 
                new_overlap = 0
            
            if new_overlap != original_overlap:
                print(f"Adjusted overlap between segments {i}-{i+1} from {original_overlap} to {new_overlap}.")
            
            quantized_frame_overlap.append(new_overlap)
    
    print(f"Final quantized_frame_overlap: {quantized_frame_overlap}")
    return quantized_segment_frames, quantized_frame_overlap

def test_exact_replication():
    """Test that replicates the exact stitching issue"""
    
    # Video URLs from the user's issue
    video_urls = [
        "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/s0_colormatched.mp4",
        "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/s1_colormatched.mp4", 
        "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/s2_colormatched.mp4"
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download all videos
        video_paths = []
        for i, url in enumerate(video_urls):
            video_file = temp_path / f"segment_{i}.mp4"
            download_video(url, video_file)
            video_paths.append(video_file)
        
        # Get actual frame counts
        actual_segment_frames = []
        for i, video_path in enumerate(video_paths):
            frame_count, fps = get_video_info(video_path)
            actual_segment_frames.append(frame_count)
            print(f"Video {i}: {frame_count} frames @ {fps} FPS")
        
        # Test different overlap scenarios that might have occurred
        
        print(f"\n=== SCENARIO 1: User Expected 10 Frame Overlap ===")
        original_overlaps = [10, 10]  # What user expected
        quantized_segment_frames, quantized_overlaps = replicate_orchestrator_quantization(
            actual_segment_frames, original_overlaps
        )
        
        print(f"\n=== SCENARIO 2: What if overlap was quantized to 0? ===")
        # Test if overlap got reduced to 0 due to quantization
        test_overlaps_zero = [0, 0]
        expected_frames_zero = sum(actual_segment_frames)  # No overlap = simple concatenation
        print(f"If overlaps were [0, 0]: Expected frames = {expected_frames_zero}")
        
        print(f"\n=== SCENARIO 3: Test various overlap values ===")
        # Test different overlap values to see which one gives 55 frames
        for test_overlap in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
            test_overlaps = [test_overlap, test_overlap]
            expected_frames = sum(actual_segment_frames) - (len(test_overlaps) * test_overlap)
            print(f"Overlap {test_overlap}: Expected frames = {expected_frames}")
            if expected_frames == 55:
                print(f"*** FOUND IT! Overlap {test_overlap} gives exactly 55 frames! ***")
        
        # Extract frames and test actual stitching with the problematic overlap
        print(f"\n=== TESTING ACTUAL STITCHING ===")
        all_segment_frames = []
        for i, video_path in enumerate(video_paths):
            frames = extract_frames_from_video(video_path)
            all_segment_frames.append(frames)
            print(f"Extracted {len(frames)} frames from video {i}")
        
        # Calculate which overlap value gives 55 frames
        target_frames = 55
        total_original_frames = sum(len(frames) for frames in all_segment_frames)
        overlap_needed = (total_original_frames - target_frames) // 2
        print(f"To get {target_frames} frames from {total_original_frames} total, overlap should be: {overlap_needed}")
        
        # Test stitching with this overlap
        print(f"\n=== TESTING STITCHING WITH OVERLAP {overlap_needed} ===")
        actual_overlaps_for_stitching = [overlap_needed, overlap_needed]
        
        # Replicate the exact stitching logic from the source
        final_stitched_frames = []
        num_stitch_points = len(all_segment_frames) - 1
        
        print(f"Number of stitch points: {num_stitch_points}")
        print(f"Overlap values: {actual_overlaps_for_stitching}")
        
        # Add frames from first segment (minus overlap with next)
        overlap_for_first_join = actual_overlaps_for_stitching[0] if actual_overlaps_for_stitching else 0
        if len(all_segment_frames[0]) > overlap_for_first_join:
            frames_to_add = all_segment_frames[0][:-overlap_for_first_join if overlap_for_first_join > 0 else len(all_segment_frames[0])]
            final_stitched_frames.extend(frames_to_add)
            print(f"Added {len(frames_to_add)} frames from segment 0 (total: {len(all_segment_frames[0])}, overlap: {overlap_for_first_join})")
        else:
            final_stitched_frames.extend(all_segment_frames[0])
            print(f"Added all {len(all_segment_frames[0])} frames from segment 0 (not enough for overlap)")
        
        # Process each stitch point
        for i in range(num_stitch_points):
            frames_prev_segment = all_segment_frames[i]
            frames_curr_segment = all_segment_frames[i+1]
            current_overlap_val = actual_overlaps_for_stitching[i]
            
            print(f"\nProcessing stitch point {i}: segments {i} -> {i+1}")
            print(f"  Previous segment frames: {len(frames_prev_segment)}")
            print(f"  Current segment frames: {len(frames_curr_segment)}")
            print(f"  Overlap: {current_overlap_val}")
            
            if current_overlap_val > 0:
                faded_frames = sm_cross_fade_overlap_frames(
                    frames_prev_segment, frames_curr_segment, 
                    current_overlap_val, "linear_sharp", 0.3
                )
                final_stitched_frames.extend(faded_frames)
                print(f"  Added {len(faded_frames)} cross-faded frames")
            
            # Add remaining frames from current segment
            if (i + 1) < num_stitch_points:
                # Not the last segment - need to account for next overlap
                overlap_for_next_join = actual_overlaps_for_stitching[i+1]
                start_index = current_overlap_val
                end_index = len(frames_curr_segment) - (overlap_for_next_join if overlap_for_next_join > 0 else 0)
                if end_index > start_index:
                    frames_to_add = frames_curr_segment[start_index:end_index]
                    final_stitched_frames.extend(frames_to_add)
                    print(f"  Added {len(frames_to_add)} middle frames from segment {i+1} (indices {start_index}:{end_index})")
            else:
                # Last segment - add all remaining frames
                start_index = current_overlap_val
                if len(frames_curr_segment) > start_index:
                    frames_to_add = frames_curr_segment[start_index:]
                    final_stitched_frames.extend(frames_to_add)
                    print(f"  Added {len(frames_to_add)} tail frames from segment {i+1} (indices {start_index}:)")
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Original total frames: {total_original_frames}")
        print(f"Target frames (like in logs): 55")
        print(f"Actual stitched frames: {len(final_stitched_frames)}")
        print(f"Overlap used: {overlap_needed}")
        print(f"Match? {len(final_stitched_frames) == 55}")
        
        # Create the stitched video for verification
        output_video = temp_path / f"stitched_test_overlap_{overlap_needed}.mp4"
        if len(final_stitched_frames) > 0:
            created_video = create_video_from_frames_list(final_stitched_frames, output_video, 16, (512, 512))
            if created_video and created_video.exists():
                final_frame_count, final_fps = get_video_info(output_video)
                print(f"Created test video: {final_frame_count} frames @ {final_fps} FPS")
                
                # Copy to current directory for inspection
                shutil.copy2(output_video, f"stitched_exact_replication_overlap_{overlap_needed}.mp4")
                print(f"Copied stitched video to: stitched_exact_replication_overlap_{overlap_needed}.mp4")

if __name__ == "__main__":
    test_exact_replication() 