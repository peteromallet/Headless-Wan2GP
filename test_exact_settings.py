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

def replicate_orchestrator_quantization_exact(segment_frames, frame_overlaps):
    """Replicate the exact quantization logic from the orchestrator"""
    print(f"\n=== EXACT ORCHESTRATOR QUANTIZATION LOGIC ===")
    print(f"Input segment_frames: {segment_frames}")
    print(f"Input frame_overlaps: {frame_overlaps}")
    
    # Preserve a copy of the original overlap list
    _orig_frame_overlap = list(frame_overlaps)
    
    # Quantize segment frames to 4*N+1 format
    quantized_segment_frames = []
    print(f"Quantizing frame counts. Original segment_frames_expanded: {segment_frames}")
    for i, frames in enumerate(segment_frames):
        new_frames = (frames // 4) * 4 + 1
        if new_frames != frames:
            print(f"Quantized segment {i} length from {frames} to {new_frames} (4*N+1 format).")
        quantized_segment_frames.append(new_frames)
    print(f"Finished quantizing frame counts. New quantized_segment_frames: {quantized_segment_frames}")
    
    # Quantize overlaps
    quantized_frame_overlap = []
    num_overlaps_to_process = len(quantized_segment_frames) - 1
    print(f"num_overlaps_to_process: {num_overlaps_to_process}")
    
    if num_overlaps_to_process > 0:
        for i in range(num_overlaps_to_process):
            if i < len(frame_overlaps):
                original_overlap = frame_overlaps[i]
            else:
                print(f"Overlap at index {i} missing. Defaulting to 0.")
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
    
    # Replace original lists with the new quantized ones
    expanded_segment_frames = quantized_segment_frames
    expanded_frame_overlap = quantized_frame_overlap
    
    # If quantisation resulted in an empty overlap list but the original DID contain an overlap value, restore that
    if (not expanded_frame_overlap) and _orig_frame_overlap:
        expanded_frame_overlap = _orig_frame_overlap
        print(f"Restored original overlap list: {expanded_frame_overlap}")
    
    print(f"Final expanded_segment_frames: {expanded_segment_frames}")
    print(f"Final expanded_frame_overlap: {expanded_frame_overlap}")
    
    return expanded_segment_frames, expanded_frame_overlap

def test_exact_settings_replication():
    """Test that replicates the exact settings from the user's failing case"""
    
    # EXACT settings from the user's orchestrator payload
    original_settings = {
        "segment_frames_expanded": [72, 72, 72],
        "frame_overlap_expanded": [10, 10, 10],  # NOTE: 3 values for 3 segments!
        "parsed_resolution_wh": "902x508",
        "num_new_segments_to_generate": 3
    }
    
    print(f"=== EXACT USER SETTINGS REPLICATION ===")
    print(f"Original settings: {json.dumps(original_settings, indent=2)}")
    
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
        
        # Get actual frame counts (these should be the quantized values)
        actual_segment_frames = []
        for i, video_path in enumerate(video_paths):
            frame_count, fps = get_video_info(video_path)
            actual_segment_frames.append(frame_count)
            print(f"Video {i}: {frame_count} frames @ {fps} FPS")
        
        # Apply the exact orchestrator quantization logic
        quantized_segment_frames, quantized_overlaps = replicate_orchestrator_quantization_exact(
            original_settings["segment_frames_expanded"], 
            original_settings["frame_overlap_expanded"]
        )
        
        print(f"\n=== STITCHING LOGIC TEST ===")
        print(f"Expected quantized segments: {quantized_segment_frames}")
        print(f"Actual video segments: {actual_segment_frames}")
        print(f"Quantized overlaps: {quantized_overlaps}")
        
        # Extract frames from all videos
        all_segment_frames = []
        for i, video_path in enumerate(video_paths):
            frames = extract_frames_from_video(video_path)
            all_segment_frames.append(frames)
            print(f"Extracted {len(frames)} frames from video {i}")
        
        # Test stitching with the quantized overlaps
        print(f"\n=== STITCHING WITH QUANTIZED OVERLAPS ===")
        
        # From the stitching code: actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points]
        num_stitch_points = len(all_segment_frames) - 1
        actual_overlaps_for_stitching = quantized_overlaps[:num_stitch_points]
        
        print(f"num_stitch_points: {num_stitch_points}")
        print(f"quantized_overlaps: {quantized_overlaps}")
        print(f"actual_overlaps_for_stitching: {actual_overlaps_for_stitching}")
        
        # Apply the exact stitching logic from the source
        final_stitched_frames = []
        
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
        
        total_original_frames = sum(len(frames) for frames in all_segment_frames)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Original settings - segment_frames: {original_settings['segment_frames_expanded']}")
        print(f"Original settings - frame_overlap: {original_settings['frame_overlap_expanded']}")
        print(f"After quantization - segment_frames: {quantized_segment_frames}")  
        print(f"After quantization - frame_overlap: {quantized_overlaps}")
        print(f"Actual videos total frames: {total_original_frames}")
        print(f"Expected final frames (manual calc): {sum(actual_segment_frames) - sum(actual_overlaps_for_stitching)}")
        print(f"Actual stitched frames: {len(final_stitched_frames)}")
        print(f"Target from logs: 55")
        print(f"Match? {len(final_stitched_frames) == 55}")
        
        # Additional analysis
        print(f"\n=== PROBLEM ANALYSIS ===")
        if len(quantized_overlaps) != num_stitch_points:
            print(f"🚨 FOUND ISSUE: quantized_overlaps has {len(quantized_overlaps)} values but need {num_stitch_points} for {len(all_segment_frames)} segments!")
            print(f"This means actual_overlaps_for_stitching = {actual_overlaps_for_stitching}")

if __name__ == "__main__":
    test_exact_settings_replication() 