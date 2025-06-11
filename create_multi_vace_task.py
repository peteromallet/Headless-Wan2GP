#!/usr/bin/env python3
"""
Create Multi-VACE Task for Headless System
==========================================

This script creates a multi-VACE task and adds it to the SQLite database for processing
by the headless system. It contains all the necessary multi-VACE functionality.
"""

import os
import sys
import json
import sqlite3
import datetime
import uuid
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Add the source directory to the path to import database functions
sys.path.insert(0, str(Path(__file__).parent / "source"))
from db_operations import add_task_to_db, init_db, STATUS_QUEUED

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
        "input_prompt": "plants",
        "resolution": "720x720",
        "frame_num": len(all_guidance_frames),        
        "guide_scale": 1.0,
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

def create_multi_vace_database_task(
    guidance_video_path,
    output_dir=None,
    max_frames_to_process=40,
    context_frames_str="0:16,32,54",
    reference_frame_indices="0, 39",
    reference_strength=0.25,
    guidance_strength=1.0,
    sampling_steps=9,
    use_causvid_lora=True,
    apply_reward_lora=True,
    db_path="tasks.db"
):
    """
    Creates a multi-VACE task and adds it to the database for headless processing.
    """
    
    # Set default output directory
    if output_dir is None:
        output_dir = Path("./tests")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Check if video exists
    if not Path(guidance_video_path).exists():
        print(f"Error: Video file not found: {guidance_video_path}")
        return None
    
    # Create reference and guidance stream configs
    reference_stream_config = {
        "frame_indices": reference_frame_indices,
        "strength": reference_strength,
        "start_percent": 0.0,
        "end_percent": 1.0,
    }
    
    guidance_stream_config = {
        "strength": guidance_strength,
        "start_percent": 0.0,
        "end_percent": 0.9,
    }
    
    print("Creating multi-VACE task parameters...")
    
    # Create the multi-VACE parameters using the function above
    generation_params = create_multi_vace_task(
        guidance_video_path=guidance_video_path,
        output_dir=output_dir,
        max_frames_to_process=max_frames_to_process,
        context_frames_str=context_frames_str,
        reference_stream_config=reference_stream_config,
        guidance_stream_config=guidance_stream_config,
        use_causvid_lora=use_causvid_lora,
        apply_reward_lora=apply_reward_lora,
    )
    
    if not generation_params:
        print("Failed to create multi-VACE parameters")
        return None
    
    # Generate a unique task ID
    task_id = f"multi_vace_{int(datetime.datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
    
    # Convert PIL images to file paths for database storage
    # Save any PIL images that were created and replace them with file paths
    processed_multi_vace_inputs = []
    for i, vace_input in enumerate(generation_params.get("multi_vace_inputs", [])):
        processed_input = vace_input.copy()
        
        # Handle reference images
        if vace_input.get("ref_images"):
            ref_image_paths = []
            for j, pil_img in enumerate(vace_input["ref_images"]):
                ref_path = output_dir / f"ref_img_stream{i}_frame{j}.png"
                pil_img.save(ref_path)
                ref_image_paths.append(str(ref_path))
                print(f"Saved reference image to: {ref_path}")
            processed_input["ref_image_paths"] = ref_image_paths
            del processed_input["ref_images"]  # Remove PIL objects
        
        # Handle guidance frames (save individual frame images, not video)
        if vace_input.get("frames"):
            frame_paths = []
            
            print(f"Saving {len(vace_input['frames'])} guidance frames as individual images...")
            for j, frame_pil in enumerate(vace_input["frames"]):
                frame_path = output_dir / f"guidance_stream{i}_frame{j:04d}.png"
                frame_pil.save(frame_path)
                frame_paths.append(str(frame_path))
            
            processed_input["frame_paths"] = frame_paths
            del processed_input["frames"]  # Remove PIL objects
            print(f"Saved guidance frames to: {len(frame_paths)} individual image files")
        
        # Ensure reference-only streams have dummy frame paths to be considered valid by headless system
        if processed_input.get("ref_image_paths") and not processed_input.get("frame_paths"):
            # Reference-only streams need dummy frame paths to be considered valid by headless system
            # Create a single transparent/black frame as placeholder
            dummy_frame_path = output_dir / f"ref_stream{i}_dummy_frame.png"
            
            # Create a small dummy frame (the actual reference images will be used)
            dummy_img = Image.new('RGB', (64, 64), color=(0, 0, 0))  # Small black frame
            dummy_img.save(dummy_frame_path)
            
            processed_input["frame_paths"] = [str(dummy_frame_path)]
            print(f"Created dummy frame for reference-only stream: {dummy_frame_path}")
        
        processed_multi_vace_inputs.append(processed_input)
    
    # Create the task payload for the database
    task_payload = {
        "task_id": task_id,
        "prompt": generation_params["input_prompt"],
        "negative_prompt": generation_params["negative_prompt"],
        "resolution": generation_params["resolution"],
        "video_length": generation_params["frame_num"],
        "num_inference_steps": sampling_steps,
        "seed": generation_params["seed"],
        "guidance_scale": generation_params["guide_scale"],
        "flow_shift": generation_params["shift"],
        "use_causvid_lora": generation_params["use_causvid_lora"],
        "apply_reward_lora": generation_params["apply_reward_lora"],
        "multi_vace_inputs": processed_multi_vace_inputs,
        "offload_model": generation_params["offload_model"],
        "VAE_tile_size": generation_params["VAE_tile_size"],
        "RIFLEx_setting": 1 if generation_params["enable_RIFLEx"] else 0,
        "joint_pass": generation_params["joint_pass"],
        "project_id": "multi_vace_project"
    }
    
    # Initialize database if it doesn't exist
    init_db()
    
    # Add task to database
    try:
        add_task_to_db(task_payload, "standard_wgp_task")  # Use standard task type
        print(f"\n✅ Multi-VACE task created successfully!")
        print(f"Task ID: {task_id}")
        print(f"Database: {db_path}")
        print(f"Status: Queued")
        print(f"\nReference frames saved to: {output_dir}")
        print(f"Task will be processed by headless.py when it runs.")
        return task_id
        
    except Exception as e:
        print(f"❌ Failed to add task to database: {e}")
        return None

def main():
    """
    Example usage of the multi-VACE task creation
    """
    # Configuration - modify these as needed
    video_path = "input.mp4"  # Change this to your video path
    
    task_config = {
        "guidance_video_path": video_path,
        "output_dir": "./multi_vace_tests",
        "max_frames_to_process": 40,
        "context_frames_str": "0:16,39",  # Which frames to use for guidance
        "reference_frame_indices": "0, 39",  # Which frames to extract as references
        "reference_strength": 0.25,
        "guidance_strength": 1.0,
        "sampling_steps": 9,
        "use_causvid_lora": True,
        "apply_reward_lora": True
    }
    
    print("Creating Multi-VACE Task for Headless Processing")
    print("=" * 50)
    print(f"Video: {task_config['guidance_video_path']}")
    print(f"Frames to process: {task_config['max_frames_to_process']}")
    print(f"Context frames: {task_config['context_frames_str']}")
    print(f"Reference frames: {task_config['reference_frame_indices']}")
    print("=" * 50)
    
    task_id = create_multi_vace_database_task(**task_config)
    
    if task_id:
        print(f"\n🎬 Run 'python headless.py' to process the task!")
    else:
        print(f"\n❌ Task creation failed. Check the error messages above.")

if __name__ == "__main__":
    main() 