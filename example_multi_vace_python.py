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

def parse_simple_indices(indices_str):
    """
    Parses a string like '0,32,54' into a list of frame indices.
    """
    if not indices_str:
        return []
    indices = []
    parts = indices_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError:
            print(f"Warning: Could not parse frame index '{part}'. Skipping.")
    return indices

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

def create_multi_vace_task(
    guidance_video_path,
    output_dir,
    max_frames_to_process,
    context_frames_str,
    reference_stream_config,
    guidance_stream_config,
    use_causvid_lora=False,
    apply_reward_lora=True,
):
    """
    Takes video path and configs, processes them, and builds the final task
    dictionary ready for generation.

    Args:
        guidance_video_path (str): Path to the guidance video.
        output_dir (Path): Directory to save extracted reference frames.
        max_frames_to_process (int): The number of frames to trim the video to.
        context_frames_str (str): A string specifying which context frames to use from
                                  the guidance video (e.g., "0:16,32,54").
        reference_stream_config (dict): Config for the reference image stream.
        guidance_stream_config (dict): Config for the guidance video stream.
        use_causvid_lora (bool): Whether to enable and apply the CausVid LoRA.
        apply_reward_lora (bool): Whether to enable and apply the reward LoRA.
    """
    print("--- Starting Task Creation ---")

    # 1. Extract frames from video, trimming to the specified max_frames
    all_guidance_frames = extract_frames_from_video(guidance_video_path, max_frames=max_frames_to_process)

    if not all_guidance_frames:
        print("Error: No guidance frames were extracted. Aborting.")
        return None

    # 2. Prepare the multi-VACE inputs list
    multi_vace_inputs = []

    # --- Stream 1: Reference images extracted from the video ---
    ref_images_pil = []
    if reference_stream_config:
        ref_indices_str = reference_stream_config.get("frame_indices", "")
        ref_indices = parse_simple_indices(ref_indices_str)
        
        print(f"\n--- Preparing Reference Stream ---")
        for i in ref_indices:
            if 0 <= i < len(all_guidance_frames):
                frame_pil = all_guidance_frames[i]
                ref_images_pil.append(frame_pil)
                # Save the extracted frame to the specified output directory
                save_path = output_dir / f"ref_frame_{i}.png"
                frame_pil.save(save_path)
                print(f"Extracted and saved reference frame {i} to {save_path}")
            else:
                print(f"Warning: Reference frame index {i} is out of bounds (video has {len(all_guidance_frames)} frames). Skipping.")
        
        if ref_images_pil:
            multi_vace_inputs.append({
                'frames': None,
                'masks': None,
                'ref_images': ref_images_pil,
                'strength': reference_stream_config.get('strength', 0.25),
                'start_percent': reference_stream_config.get('start_percent', 0.0),
                'end_percent': reference_stream_config.get('end_percent', 1.0),
            })

    # --- Stream 2: Main guidance video with non-context frames replaced ---
    guidance_frames_pil = []
    if guidance_stream_config:
        print(f"\n--- Preparing Guidance Stream ---")
        max_idx = len(all_guidance_frames) - 1
        context_indices = parse_context_frames(context_frames_str, max_idx)
        
        width, height = all_guidance_frames[0].size
        grey_frame = Image.new('RGB', (width, height), color=(128, 128, 128))
        
        for i, frame in enumerate(all_guidance_frames):
            if i in context_indices:
                guidance_frames_pil.append(frame)
            else:
                guidance_frames_pil.append(grey_frame)
        
        print(f"Processed guidance video: {len(context_indices)} frames used as context, {len(all_guidance_frames) - len(context_indices)} replaced with grey.")

        if guidance_frames_pil:
            multi_vace_inputs.append({
                'frames': guidance_frames_pil,
                'masks': None,
                'ref_images': None,
                'strength': guidance_stream_config.get('strength', 1.0),
                'start_percent': guidance_stream_config.get('start_percent', 0.0),
                'end_percent': guidance_stream_config.get('end_percent', 0.9),
            })

    if not multi_vace_inputs:
        print("Error: No valid VACE streams were created. Aborting.")
        return None

    # 3. Build the complete parameter dictionary
    generation_params = {
        "input_prompt": "A person dancing in a beautiful garden with flowers",
        "resolution": "1280x720",
        "frame_num": len(all_guidance_frames),
        "sampling_steps": 30,
        "guide_scale": 5.0,
        "seed": 42,
        "multi_vace_inputs": multi_vace_inputs,
        "use_causvid_lora": use_causvid_lora,
        "apply_reward_lora": apply_reward_lora,
        "negative_prompt": "ugly, blurry, distorted",
        "shift": 5.0,
        "offload_model": True,
        "VAE_tile_size": 0,
        "enable_RIFLEx": True,
        "joint_pass": False,
    }
    
    print("\n--- Task Creation Complete ---")
    if ref_images_pil:
        print(f"Stream 1 (Reference): {len(ref_images_pil)} images at strength {reference_stream_config.get('strength', 0.25)}")
    if guidance_frames_pil:
        print(f"Stream 2 (Guidance): {len(guidance_frames_pil)} frames at strength {guidance_stream_config.get('strength', 1.0)}")
    if use_causvid_lora:
        print("LoRA Mode: CausVid LoRA enabled.")
    if apply_reward_lora:
        print("LoRA Mode: Reward LoRA enabled.")

    return generation_params

if __name__ == "__main__":
    # --- Create a directory for test outputs ---
    output_directory = Path("./tests")
    output_directory.mkdir(exist_ok=True)

    max_frames = 40  # Trim the input video to this many frames for the task
    # --- Point to the input video ---
    video_path = "input.mp4"

        # Stream 2: The main guidance video (with non-context frames greyed out)
    guidance_stream_settings = {
        "strength": 1.0,
        "start_percent": 0.0,
        "end_percent": 0.9,
    }
        
    context_frames_spec = "0:16,32,54"  # <-- Specify which frames to use for guidance context

    # Stream 1: Reference frames extracted from the video
    reference_stream_settings = {
        "frame_indices": "0, 39",  # Which frames to extract as references
        "strength": 0.25,
        "start_percent": 0.0,
        "end_percent": 1.0,
    }
    
    # --- LoRA settings ---
    use_causvid = True
    apply_reward = True
    
    # --- RUN THE TASK CREATION ---
    final_task_parameters = create_multi_vace_task(
        guidance_video_path=video_path,
        output_dir=output_directory,
        max_frames_to_process=max_frames,
        context_frames_str=context_frames_spec,
        reference_stream_config=reference_stream_settings,
        guidance_stream_config=guidance_stream_settings,
        use_causvid_lora=use_causvid,
        apply_reward_lora=apply_reward,
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