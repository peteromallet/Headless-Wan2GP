#!/usr/bin/env python3
"""
Example: Multi-VACE with reference images and guidance video.
This script takes file paths for reference images and a guidance video,
processes them, and constructs the multi-VACE task structure ready for wgp.py.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Add Wan2GP to path for imports if running as a script
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2GP"))

def extract_frames_from_video(video_path, max_frames=None):
    """
    Extracts frames from a video file.
    """
    if not Path(video_path).exists():
        print(f"Error: Video file not found at {video_path}")
        return []
        
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
            
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def load_reference_images(image_paths):
    """
    Loads reference images from file paths.
    """
    loaded_images = []
    for img_path in image_paths:
        if not Path(img_path).exists():
            print(f"Error: Image file not found at {img_path}")
            continue
        try:
            pil_img = Image.open(img_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            loaded_images.append(pil_img)
            print(f"Loaded reference image: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            
    return loaded_images

def parse_context_frames(context_frames_str, max_frame_index):
    """
    Parses a string like '0:16,32,54' into a set of frame indices.
    If the string is empty, it defaults to using all available frames.
    """
    if not context_frames_str:
        return set(range(max_frame_index + 1))  # Default to all frames

    indices = set()
    parts = context_frames_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            try:
                start, end = map(int, part.split(':'))
                for i in range(start, end + 1):
                    if i <= max_frame_index:
                        indices.add(i)
            except ValueError:
                print(f"Warning: Could not parse frame range '{part}'. Skipping.")
        else:
            try:
                index = int(part)
                if index <= max_frame_index:
                    indices.add(index)
            except ValueError:
                print(f"Warning: Could not parse frame index '{part}'. Skipping.")
    return indices

def create_multi_vace_task(ref_image_paths, guidance_video_path, context_frames_str=""):
    """
    Takes image and video paths, processes them, and builds the final task
    dictionary ready for generation.

    Args:
        ref_image_paths (list): List of paths to reference images.
        guidance_video_path (str): Path to the guidance video.
        context_frames_str (str): A string specifying which frames to use from the 
                                  guidance video (e.g., "0:16,32,54"). Other frames
                                  will be replaced with grey.
    """
    print("--- Starting Task Creation ---")
    
    # 1. Process inputs from paths into PIL Images
    ref_images_pil = load_reference_images(ref_image_paths)
    all_guidance_frames = extract_frames_from_video(guidance_video_path, max_frames=81)

    guidance_frames_pil = []
    if all_guidance_frames:
        max_idx = len(all_guidance_frames) - 1
        context_indices = parse_context_frames(context_frames_str, max_idx)
        
        # Get dimensions from the first frame to create a matching grey frame
        width, height = all_guidance_frames[0].size
        grey_frame = Image.new('RGB', (width, height), color=(128, 128, 128))
        
        for i, frame in enumerate(all_guidance_frames):
            if i in context_indices:
                guidance_frames_pil.append(frame)  # Use the real frame
            else:
                guidance_frames_pil.append(grey_frame)  # Use a grey frame
        
        print(f"Processed guidance video: {len(context_indices)} frames used as context, {len(all_guidance_frames) - len(context_indices)} replaced with grey.")
    else:
        print("No guidance frames were extracted.")

    if not ref_images_pil and not guidance_frames_pil:
        print("Error: No valid reference images or guidance frames were loaded. Aborting.")
        return None

    # 2. Structure the multi-VACE inputs with the processed PIL images
    multi_vace_inputs = []
    if ref_images_pil:
        multi_vace_inputs.append({
            'frames': None,
            'masks': None,
            'ref_images': ref_images_pil,
            'strength': 0.25,
            'start_percent': 0.0,
            'end_percent': 1.0
        })
    
    if guidance_frames_pil:
        multi_vace_inputs.append({
            'frames': guidance_frames_pil,
            'masks': None,
            'ref_images': None,
            'strength': 1.0,
            'start_percent': 0.0,
            'end_percent': 0.9
        })

    # 3. Build the complete parameter dictionary for wgp.py's `generate_video`
    generation_params = {
        "input_prompt": "A person dancing in a beautiful garden with flowers",
        "resolution": "1280x720",
        "frame_num": 81,
        "sampling_steps": 30,
        "guide_scale": 5.0,
        "seed": 42,
        "multi_vace_inputs": multi_vace_inputs,  # The final, processed structure
        # --- Standard default parameters ---
        "negative_prompt": "ugly, blurry, distorted",
        "shift": 5.0,
        "offload_model": True,
        "VAE_tile_size": 0,
        "enable_RIFLEx": True,
        "joint_pass": False,
    }
    
    print("\n--- Task Creation Complete ---")
    if ref_images_pil:
        print(f"Stream 1 (Reference): {len(ref_images_pil)} images at strength 0.25")
    if guidance_frames_pil:
        print(f"Stream 2 (Guidance): {len(guidance_frames_pil)} frames at strength 1.0")

    return generation_params

if __name__ == "__main__":
    # --- Point to the input files in the current directory ---
    reference_paths = [
        "frame_1.png",
        "frame_2.png"
    ]
    video_path = "input.mp4"
    context_frames_spec = "0:16,32,54"  #<-- Specify which frames to use
    
    # --- RUN THE TASK CREATION ---
    final_task_parameters = create_multi_vace_task(
        reference_paths, 
        video_path, 
        context_frames_spec
    )
    
    if final_task_parameters:
        print("\n--- Final Parameters for wgp.py ---")
        import json
        
        # We can't print the PIL images, so we show a summary
        summary_params = {k: v for k, v in final_task_parameters.items() if k != 'multi_vace_inputs'}
        summary_params['multi_vace_inputs_summary'] = [
            {
                'ref_images_count': len(s.get('ref_images', []) or []),
                'frames_count': len(s.get('frames', []) or []),
                'strength': s['strength'],
                'timing': f"{s['start_percent']}-{s['end_percent']}"
            } for s in final_task_parameters['multi_vace_inputs']
        ]
        
        print(json.dumps(summary_params, indent=2))
        
        print("\nThis `generation_params` dictionary is now ready to be passed to a loaded WanT2V model.")
        
    # No longer creating or cleaning up dummy files. 