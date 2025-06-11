#!/usr/bin/env python3
"""
Multi-VACE Creator - Consolidated Single Script
==============================================

This script creates multi-VACE tasks with selective encoding options:
- Reference images (first/last frame)  
- Context frames with grey fill
- Both encodings together

Supports both direct usage and database queuing for headless processing.
"""

import os
import sys
import json
import sqlite3
import datetime
import uuid
import cv2
import numpy as np
import argparse
from pathlib import Path
from PIL import Image

# Add paths for imports
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2GP"))
    sys.path.insert(0, str(Path(__file__).parent / "source"))

def extract_frames_from_video(video_path, max_frames=None, target_size=(720, 720)):
    """Extracts frames from a video file and resizes them."""
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
        
        # Resize frame to target size to prevent dimension mismatch issues
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        frames.append(pil_image)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
            
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def parse_simple_indices(indices_str):
    """Parses a string like '0,32,54' into a list of frame indices."""
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
    """Parses a string like '0:16,32,54' into a set of frame indices."""
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
    encoding_mode="both",
    context_frames_str="0:16,39",
    reference_strength=0.25,
    guidance_strength=1.0,
    use_causvid_lora=True,
    apply_reward_lora=True,
    sampling_steps=9
):
    """
    Creates a multi-VACE task with selective encoding options.
    
    Args:
        encoding_mode: "both", "refs_only", or "context_only"
    """
    print(f"--- Creating Multi-VACE Task ({encoding_mode.upper().replace('_', ' ')}) ---")

    # Extract frames from video
    all_frames = extract_frames_from_video(guidance_video_path, max_frames_to_process)
    if not all_frames:
        print("Error: No frames extracted. Aborting.")
        return None

    if encoding_mode in ["refs_only", "both"] and len(all_frames) < 2:
        print("Error: Need at least 2 frames for reference encoding. Aborting.")
        return None

    multi_vace_inputs = []

    # --- Reference Images Encoding ---
    if encoding_mode in ["refs_only", "both"]:
        print(f"\n🎯 Creating Reference Images Encoding...")
        
        first_frame = all_frames[0]
        last_frame = all_frames[-1]
        
        # Save reference frames
        first_path = output_dir / "first_frame_ref.png"
        last_path = output_dir / "last_frame_ref.png"
        first_frame.save(first_path)
        last_frame.save(last_path)
        
        print(f"Saved: {first_path}")
        print(f"Saved: {last_path}")
        
        ref_images_list = [first_frame, last_frame]

        # Create dummy black masks to prevent downstream filtering of None.
        # Use the same size as the reference images to avoid resize issues.
        width, height = ref_images_list[0].size
        dummy_mask = Image.new('RGB', (width, height), (0, 0, 0))  # Use RGB for channel consistency
        dummy_masks = [dummy_mask] * len(ref_images_list)
        
        # Create dummy frames to align with reference images and masks
        dummy_frame = Image.new('RGB', (width, height), color=(0, 0, 0))
        dummy_frames = [dummy_frame] * len(ref_images_list)
        
        multi_vace_inputs.append({
            'frames': dummy_frames,
            'masks': dummy_masks,
            'ref_images': ref_images_list,
            'strength': reference_strength,
            'start_percent': 0.0,
            'end_percent': 1.0,
        })

    # --- Context Frames Encoding ---
    if encoding_mode in ["context_only", "both"]:
        print(f"\n🎥 Creating Context Frames Encoding...")
        
        context_indices = parse_context_frames(context_frames_str, len(all_frames) - 1)
        width, height = all_frames[0].size
        grey_frame = Image.new('RGB', (width, height), color=(128, 128, 128))
        
        guidance_frames = []
        context_count = 0
        grey_count = 0
        
        for i, frame in enumerate(all_frames):
            if i in context_indices:
                guidance_frames.append(frame)
                context_count += 1
            else:
                guidance_frames.append(grey_frame)
                grey_count += 1
        
        # Create dummy black masks for every context frame to ensure data alignment
        context_dummy_mask = Image.new('RGB', (width, height), (0, 0, 0)) # Use RGB for channel consistency
        context_masks = [context_dummy_mask] * len(guidance_frames)
        
        print(f"Context frames: {context_count}")
        print(f"Grey frames: {grey_count}")
        print(f"Context indices: {sorted(context_indices)}")
        
        multi_vace_inputs.append({
            'frames': guidance_frames,
            'masks': context_masks,
            'ref_images': None,
            'strength': guidance_strength,
            'start_percent': 0.0,
            'end_percent': 0.9,
        })

    if not multi_vace_inputs:
        print("Error: No valid VACE encodings created. Aborting.")
        return None

    # Build complete parameters
    generation_params = {
        "input_prompt": "high quality video",
        "resolution": "720x720",
        "frame_num": len(all_frames),
        "sampling_steps": sampling_steps,
        "guide_scale": 1.0,
        "seed": 42,
        "multi_vace_inputs": multi_vace_inputs,
        "use_causvid_lora": use_causvid_lora,
        "apply_reward_lora": apply_reward_lora,
        "negative_prompt": "ugly, blurry, distorted, low quality",
        "shift": 5.0,
        "offload_model": True,
        "VAE_tile_size": 0,
        "enable_RIFLEx": True,
        "joint_pass": False,
    }
    
    print("\n--- Task Creation Complete ---")
    print(f"✅ Mode: {encoding_mode.upper().replace('_', ' ')}")
    print(f"✅ Total frames: {len(all_frames)}")
    if encoding_mode in ["refs_only", "both"]:
        print(f"✅ Reference encoding: strength {reference_strength}")
    if encoding_mode in ["context_only", "both"]:
        print(f"✅ Context encoding: strength {guidance_strength}")
    print(f"✅ CausVid LoRA: {'enabled' if use_causvid_lora else 'disabled'}")
    print(f"✅ Reward LoRA: {'enabled' if apply_reward_lora else 'disabled'}")

    return generation_params

def queue_task_for_headless(
    guidance_video_path,
    output_dir,
    max_frames_to_process,
    encoding_mode="both",
    context_frames_str="0:16,39",
    reference_strength=0.25,
    guidance_strength=1.0,
    sampling_steps=9,
    use_causvid_lora=True,
    apply_reward_lora=True,
    model_name="vace_14B"  # Add model parameter with VACE default
):
    """
    Creates a multi-VACE task and queues it for headless processing.
    """
    try:
        from db_operations import add_task_to_db, init_db
    except ImportError:
        print("❌ Error: Cannot import database functions. Make sure you're in the right directory.")
        return None
    
    # Create the multi-VACE parameters
    generation_params = create_multi_vace_task(
        guidance_video_path=guidance_video_path,
        output_dir=output_dir,
        max_frames_to_process=max_frames_to_process,
        encoding_mode=encoding_mode,
        context_frames_str=context_frames_str,
        reference_strength=reference_strength,
        guidance_strength=guidance_strength,
        use_causvid_lora=use_causvid_lora,
        apply_reward_lora=apply_reward_lora,
        sampling_steps=sampling_steps
    )
    
    if not generation_params:
        return None
    
    # Convert PIL images to file paths for database storage
    processed_multi_vace_inputs = []
    for i, vace_input in enumerate(generation_params["multi_vace_inputs"]):
        processed_input = vace_input.copy()
        
        # Handle reference images
        if vace_input.get("ref_images"):
            ref_image_paths = []
            for j, pil_img in enumerate(vace_input["ref_images"]):
                ref_path = output_dir / f"ref_img_stream{i}_frame{j}.png"
                pil_img.save(ref_path)
                ref_image_paths.append(str(ref_path))
            processed_input["ref_images"] = ref_image_paths
        
        # Handle masks by overwriting the key with paths
        if vace_input.get("masks"):
            mask_paths = []
            for j, pil_img in enumerate(vace_input["masks"]):
                if pil_img:
                    mask_path = output_dir / f"mask_img_stream{i}_frame{j}.png"
                    pil_img.save(mask_path)
                    mask_paths.append(str(mask_path))
                else:
                    mask_paths.append(None)
            processed_input["masks"] = mask_paths
        
        # Handle guidance frames by overwriting the key with paths
        if vace_input.get("frames"):
            frame_paths = []
            for j, frame_pil in enumerate(vace_input["frames"]):
                frame_path = output_dir / f"guidance_stream{i}_frame{j:04d}.png"
                frame_pil.save(frame_path)
                frame_paths.append(str(frame_path))
            processed_input["frames"] = frame_paths
        
        processed_multi_vace_inputs.append(processed_input)
    
    # Generate task ID
    task_id = f"multi_vace_{encoding_mode}_{int(datetime.datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
    
    # Create task payload
    task_payload = {
        "task_id": task_id,
        "model": model_name,
        "prompt": generation_params["input_prompt"],
        "negative_prompt": generation_params["negative_prompt"],
        "resolution": generation_params["resolution"],
        "video_length": generation_params["frame_num"],
        "num_inference_steps": generation_params["sampling_steps"],
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
        "project_id": f"multi_vace_{encoding_mode}_project"
    }
    
    # Add to database
    init_db()
    add_task_to_db(task_payload, f"multi_vace_{encoding_mode}_task")
    
    print(f"\n🚀 Task {task_id} queued for headless processing!")
    return task_id

def main():
    parser = argparse.ArgumentParser(description="Multi-VACE Creator - Consolidated Script")
    parser.add_argument("--video", type=str, default="input.mp4", help="Input video file (default: input.mp4)")
    parser.add_argument("--frames", type=int, default=40, help="Max frames to process (default: 40)")
    parser.add_argument("--context", type=str, default="0:16,39", help="Context frames spec (default: 0:16,39)")
    parser.add_argument("--ref-strength", type=float, default=0.25, help="Reference image strength (default: 0.25)")
    parser.add_argument("--guide-strength", type=float, default=1.0, help="Guidance video strength (default: 1.0)")
    parser.add_argument("--steps", type=int, default=9, help="Sampling steps (default: 9)")
    
    # VACE encoding selection flags
    encoding_group = parser.add_mutually_exclusive_group()
    encoding_group.add_argument("--refs-only", action="store_true", help="🎯 Only first/last frame references")
    encoding_group.add_argument("--context-only", action="store_true", help="🎥 Only context frames with grey fill")
    encoding_group.add_argument("--both", action="store_true", default=True, help="🎯🎥 Both encodings (default)")
    
    parser.add_argument("--no-causvid", action="store_true", help="Disable CausVid LoRA")
    parser.add_argument("--no-reward", action="store_true", help="Disable Reward LoRA")
    parser.add_argument("--model", type=str, default="vace_14B", help="Model to use (default: vace_14B)")
    parser.add_argument("--output-dir", type=str, default="./multi_vace_output", help="Output directory")
    parser.add_argument("--no-queue", action="store_true", help="Don't queue for headless processing")
    
    args = parser.parse_args()
    
    # Determine encoding mode
    if args.refs_only:
        encoding_mode = "refs_only"
        print("🎯 MODE: Reference images only (first + last frame)")
    elif args.context_only:
        encoding_mode = "context_only"
        print("🎥 MODE: Context frames only (with grey fill)")
    else:
        encoding_mode = "both"
        print("🎯🎥 MODE: Both encodings (references + context frames)")
    
    # Create output directory
    output_directory = Path(args.output_dir)
    output_directory.mkdir(exist_ok=True)
    
    # Display configuration
    print("=" * 60)
    print(f"🎬 Video: {args.video}")
    print(f"📊 Frames: {args.frames}")
    print(f"📋 Context: {args.context}")
    if encoding_mode in ["refs_only", "both"]:
        print(f"🎯 Ref strength: {args.ref_strength}")
    if encoding_mode in ["context_only", "both"]:
        print(f"🎥 Guide strength: {args.guide_strength}")
    print(f"🤖 Model: {args.model}")
    print(f"🧬 CausVid LoRA: {'✅' if not args.no_causvid else '❌'}")
    print(f"🏆 Reward LoRA: {'✅' if not args.no_reward else '❌'}")
    print(f"🔢 Steps: {args.steps}")
    print("=" * 60)
    
    # Create the task
    if args.no_queue:
        # Just create the parameters without queuing
        generation_params = create_multi_vace_task(
            guidance_video_path=args.video,
            output_dir=output_directory,
            max_frames_to_process=args.frames,
            encoding_mode=encoding_mode,
            context_frames_str=args.context,
            reference_strength=args.ref_strength,
            guidance_strength=args.guide_strength,
            use_causvid_lora=not args.no_causvid,
            apply_reward_lora=not args.no_reward,
            sampling_steps=args.steps
        )
        
        if generation_params:
            print("\n📋 Task parameters created successfully!")
            print("Use these parameters with your WanT2V model for generation.")
        else:
            print("\n❌ Failed to create task parameters.")
    else:
        # Queue for headless processing
        task_id = queue_task_for_headless(
            guidance_video_path=args.video,
            output_dir=output_directory,
            max_frames_to_process=args.frames,
            encoding_mode=encoding_mode,
            context_frames_str=args.context,
            reference_strength=args.ref_strength,
            guidance_strength=args.guide_strength,
            sampling_steps=args.steps,
            use_causvid_lora=not args.no_causvid,
            apply_reward_lora=not args.no_reward,
            model_name=args.model
        )
        
        if task_id:
            print(f"🚀 Run 'python headless.py' to process the task!")
        else:
            print("❌ Failed to queue task.")

if __name__ == "__main__":
    main() 