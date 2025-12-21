"""
VLM Utilities for Image Analysis and Prompt Generation

This module provides helper functions for using Qwen VLM to analyze images
and generate descriptive prompts for video generation.
"""

import sys
from pathlib import Path
from typing import Optional, Union, List, Tuple
from PIL import Image
import torch


def download_qwen_vlm_if_needed(model_dir: Path) -> Path:
    """
    Download Qwen2.5-VL-7B-Instruct model if not already present.
    Uses the same download pattern as WanGP's other models.

    Args:
        model_dir: Directory to download the model to (should be ckpts/Qwen2.5-VL-7B-Instruct)

    Returns:
        Path to the downloaded model directory
    """
    # Check if we have the standard HuggingFace format (multiple model files)
    model_files = [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "config.json",
        "tokenizer_config.json"
    ]

    has_all_files = all((model_dir / f).exists() for f in model_files)

    if not has_all_files:
        print(f"[VLM_DOWNLOAD] Downloading Qwen2.5-VL-7B-Instruct to {model_dir}...")
        print(f"[VLM_DOWNLOAD] This is a one-time download (~16GB). Future runs will use the cached model.")

        try:
            from huggingface_hub import snapshot_download

            # Download the model in standard HuggingFace format
            snapshot_download(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"[VLM_DOWNLOAD] ✅ Download complete: {model_dir}")
        except Exception as e:
            print(f"[VLM_DOWNLOAD] ❌ Download failed: {e}")
            raise

    return model_dir


def generate_transition_prompt(
    start_image_path: str,
    end_image_path: str,
    base_prompt: Optional[str] = None,
    num_frames: Optional[int] = None,
    fps: int = 16,
    device: str = "cuda",
    dprint=print
) -> str:
    """
    Use QwenVLM to generate a descriptive prompt for the transition between two images.

    The model is automatically offloaded to CPU after inference to free VRAM.

    Args:
        start_image_path: Path to the starting image
        end_image_path: Path to the ending image
        base_prompt: Optional base prompt to append after VLM-generated description
        num_frames: Number of frames in the video segment (for duration calculation)
        fps: Frames per second (default: 16)
        device: Device to run the model on ('cuda' or 'cpu')
        dprint: Print function for logging

    Returns:
        Generated prompt describing the transition, with base_prompt appended if provided

    Example outputs:
        "She runs from the kitchen to the playground as the camera pans"
        "It zooms in on the eye to reveal a horse"
        "The scene transitions from day to night as clouds roll across the sky"

    Note:
        - VLM is loaded on first call and offloaded to CPU after each inference
        - VRAM usage: ~14GB during inference (Qwen2.5-VL-7B)
        - Inference time: ~5-10 seconds per transition on GPU
    """
    try:
        # Add Wan2GP to path for imports
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from Wan2GP.wan.utils.prompt_extend import QwenPromptExpander

        dprint(f"[VLM_TRANSITION] Generating transition prompt from {Path(start_image_path).name} → {Path(end_image_path).name}")

        # Load both images
        start_img = Image.open(start_image_path).convert("RGB")
        end_img = Image.open(end_image_path).convert("RGB")

        # Combine images side by side for VLM to see both
        combined_width = start_img.width + end_img.width
        combined_height = max(start_img.height, end_img.height)
        combined_img = Image.new('RGB', (combined_width, combined_height))
        combined_img.paste(start_img, (0, 0))
        combined_img.paste(end_img, (start_img.width, 0))

        # Initialize VLM with Qwen2.5-VL-7B
        # Use local model from ckpts directory, download if needed
        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"

        # Ensure model is downloaded
        dprint(f"[VLM_TRANSITION] Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)

        dprint(f"[VLM_TRANSITION] Initializing Qwen2.5-VL-7B-Instruct from local path: {local_model_path}")
        extender = QwenPromptExpander(
            model_name=str(local_model_path),
            device=device,
            is_vl=True  # CRITICAL: Enable VL mode
        )

        # Craft the query prompt with examples
        base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "the objects/people inside the scene move excitingly and things transform or shift with the camera"

        # Add duration info if available
        duration_text = ""
        if num_frames and fps:
            duration_seconds = num_frames / fps
            duration_text = f" This transition occurs over approximately {duration_seconds:.1f} seconds ({num_frames} frames at {fps} FPS)."

        query = f"""You are viewing two images side by side: the left image shows the starting frame, and the right image shows the ending frame of a video sequence.

{duration_text} Your goal is to create a THREE-SENTENCE prompt that describes the MOTION and CHANGES in this transition based on the user's description: '{base_prompt_text}'

FOCUS ON MOTION: Describe what MOVES, what CHANGES, and HOW things transition between these frames. Everything should be described in terms of motion and transformation, not static states.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

SENTENCE 1 (PRIMARY MOTION): Describe the main action, camera movement, and major scene transitions. What is the dominant movement happening?

SENTENCE 2 (MOVING ELEMENTS): Describe how the characters, objects, and environment are moving or changing. Focus on what's in motion and how it moves through space.

SENTENCE 3 (MOTION DETAILS): Describe the subtle motion details - secondary movements, environmental dynamics, particles, lighting shifts, and small-scale motions.

Examples of MOTION-FOCUSED descriptions:

- "The sun rises rapidly above the jagged peaks as the camera tilts upward from the dark valley floor. The silhouette pine trees sway gently against the shifting violet and gold sky as the entire landscape brightens. Wisps of morning mist evaporate and drift upward from the river surface while distant birds circle and glide through the upper left corner."

- "A woman sprints from the kitchen into the bright exterior sunlight as the camera pans right to track her accelerating path. Her vintage floral dress flows and ripples in the wind while colorful playground equipment blurs past in the background. Her hair whips back dynamically and dust particles kick up and swirl around her sneakers as she impacts the gravel."

- "The camera zooms aggressively inward into a macro shot of an eye as the brown horse reflection grows larger and more detailed. The iris textures shift under the changing warm lighting while the biological details come into sharper focus. The pupil constricts and contracts in reaction to the light while the tiny reflected horse tosses its mane and shifts position."

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""

        system_prompt = "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES following this structure: 1) PRIMARY MOTION, 2) MOVING ELEMENTS, 3) MOTION DETAILS. Focus exclusively on what moves and changes, not static descriptions."

        dprint(f"[VLM_TRANSITION] Running inference...")
        result = extender.extend_with_img(
            prompt=query,
            system_prompt=system_prompt,
            image=combined_img
        )

        vlm_prompt = result.prompt.strip()
        dprint(f"[VLM_TRANSITION] Generated: {vlm_prompt}")

        return vlm_prompt

    except Exception as e:
        dprint(f"[VLM_TRANSITION] ERROR: Failed to generate transition prompt: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to base_prompt if VLM fails
        if base_prompt and base_prompt.strip():
            dprint(f"[VLM_TRANSITION] Falling back to base prompt: {base_prompt}")
            return base_prompt
        else:
            dprint(f"[VLM_TRANSITION] Falling back to generic prompt")
            return "cinematic transition"


def generate_transition_prompts_batch(
    image_pairs: List[Tuple[str, str]],
    base_prompts: List[Optional[str]],
    num_frames_list: Optional[List[int]] = None,
    fps: int = 16,
    device: str = "cuda",
    dprint=print,
    task_id: Optional[str] = None,
    upload_debug_images: bool = True
) -> List[str]:
    """
    Batch generate transition prompts for multiple image pairs.

    This is much more efficient than calling generate_transition_prompt() multiple times,
    because it loads the VLM model once and reuses it for all pairs.

    Args:
        image_pairs: List of (start_image_path, end_image_path) tuples
        base_prompts: List of base prompts to append (one per pair, can be None)
        num_frames_list: List of frame counts for each segment (for duration calculation)
        fps: Frames per second (default: 16)
        device: Device to run the model on ('cuda' or 'cpu')
        dprint: Print function for logging
        task_id: Optional task ID for organizing debug image uploads
        upload_debug_images: Whether to upload debug combined images to storage (default: True)

    Returns:
        List of generated prompts (one per image pair)

    Example:
        image_pairs = [
            ("img1.jpg", "img2.jpg"),
            ("img2.jpg", "img3.jpg"),
            ("img3.jpg", "img4.jpg")
        ]
        base_prompts = ["cinematic", "cinematic", "cinematic"]
        prompts = generate_transition_prompts_batch(image_pairs, base_prompts)
    """
    if len(image_pairs) != len(base_prompts):
        raise ValueError(f"image_pairs and base_prompts must have same length ({len(image_pairs)} != {len(base_prompts)})")

    if not image_pairs:
        return []

    try:
        # Add Wan2GP to path for imports
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from Wan2GP.wan.utils.prompt_extend import QwenPromptExpander

        # Log memory BEFORE loading
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
            dprint(f"[VLM_BATCH] GPU memory BEFORE: {gpu_mem_before:.2f} GB")

        dprint(f"[VLM_BATCH] Initializing Qwen2.5-VL-7B-Instruct for {len(image_pairs)} transitions...")

        # Initialize VLM ONCE for all pairs
        # Use local model from ckpts directory, download if needed
        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"

        # Ensure model is downloaded
        dprint(f"[VLM_BATCH] Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)

        dprint(f"[VLM_BATCH] Using local model from: {local_model_path}")
        extender = QwenPromptExpander(
            model_name=str(local_model_path),
            device=device,
            is_vl=True  # CRITICAL: Enable VL mode
        )

        dprint(f"[VLM_BATCH] Model loaded (initially on CPU)")

        system_prompt = "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES following this structure: 1) PRIMARY MOTION, 2) MOVING ELEMENTS, 3) MOTION DETAILS. Focus exclusively on what moves and changes, not static descriptions."

        results = []
        for i, ((start_path, end_path), base_prompt) in enumerate(zip(image_pairs, base_prompts)):
            try:
                dprint(f"[VLM_BATCH] Processing pair {i+1}/{len(image_pairs)}: {Path(start_path).name} → {Path(end_path).name}")
                
                # [VLM_FILE_DEBUG] Log the actual file paths and verify they exist
                start_exists = Path(start_path).exists() if start_path else False
                end_exists = Path(end_path).exists() if end_path else False
                dprint(f"[VLM_FILE_DEBUG] Pair {i}: start={start_path} (exists={start_exists})")
                dprint(f"[VLM_FILE_DEBUG] Pair {i}: end={end_path} (exists={end_exists})")
                
                # Compute file hashes to verify image identity
                import hashlib
                def get_file_hash(filepath):
                    """Get first 8 chars of MD5 hash for quick file identity check."""
                    try:
                        with open(filepath, 'rb') as f:
                            return hashlib.md5(f.read()).hexdigest()[:8]
                    except:
                        return 'ERROR'
                
                if start_exists:
                    start_size = Path(start_path).stat().st_size
                    start_hash = get_file_hash(start_path)
                    dprint(f"[VLM_FILE_DEBUG] Pair {i}: start file size={start_size} bytes, hash={start_hash}")
                if end_exists:
                    end_size = Path(end_path).stat().st_size
                    end_hash = get_file_hash(end_path)
                    dprint(f"[VLM_FILE_DEBUG] Pair {i}: end file size={end_size} bytes, hash={end_hash}")

                # Load and combine images
                start_img = Image.open(start_path).convert("RGB")
                end_img = Image.open(end_path).convert("RGB")
                
                # [VLM_IMAGE_VERIFY] Log image dimensions to verify correct loading
                dprint(f"[VLM_IMAGE_VERIFY] Pair {i}: start_img dimensions={start_img.size}, end_img dimensions={end_img.size}")
                
                # [VLM_IMAGE_CONTENT] Log image color statistics to help verify content
                import numpy as np
                def get_image_stats(img):
                    """Get average RGB and warmth indicator for image."""
                    arr = np.array(img)
                    avg_r, avg_g, avg_b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
                    # Warmth: positive = warm (more red), negative = cool (more blue)
                    warmth = (avg_r - avg_b) / 255 * 100
                    brightness = (avg_r + avg_g + avg_b) / 3
                    return f"RGB=({avg_r:.0f},{avg_g:.0f},{avg_b:.0f}) brightness={brightness:.0f} warmth={warmth:+.1f}%"
                
                dprint(f"[VLM_IMAGE_CONTENT] Pair {i} START: {get_image_stats(start_img)}")
                dprint(f"[VLM_IMAGE_CONTENT] Pair {i} END: {get_image_stats(end_img)}")

                combined_width = start_img.width + end_img.width
                combined_height = max(start_img.height, end_img.height)
                combined_img = Image.new('RGB', (combined_width, combined_height))
                combined_img.paste(start_img, (0, 0))
                combined_img.paste(end_img, (start_img.width, 0))
                
                # [VLM_DEBUG_SAVE] Save combined image for manual inspection
                # This shows EXACTLY what VLM sees - left=start, right=end
                debug_path = None
                start_debug_path = None
                end_debug_path = None
                try:
                    debug_dir = Path(start_path).parent / "vlm_debug"
                    debug_dir.mkdir(exist_ok=True)
                    debug_path = debug_dir / f"vlm_combined_pair{i}.jpg"
                    combined_img.save(str(debug_path), quality=95)
                    dprint(f"[VLM_DEBUG_SAVE] Saved combined image for pair {i} to: {debug_path}")
                    
                    # Also save individual start/end images to debug folder for clarity
                    start_debug_path = debug_dir / f"vlm_pair{i}_LEFT_start.jpg"
                    end_debug_path = debug_dir / f"vlm_pair{i}_RIGHT_end.jpg"
                    start_img.save(str(start_debug_path), quality=95)
                    end_img.save(str(end_debug_path), quality=95)
                    dprint(f"[VLM_DEBUG_SAVE] Saved individual images: {start_debug_path.name}, {end_debug_path.name}")
                except Exception as e_save:
                    dprint(f"[VLM_DEBUG_SAVE] Could not save debug image: {e_save}")
                
                # [VLM_DEBUG_UPLOAD] Upload debug images to Supabase for remote inspection
                if upload_debug_images and task_id and debug_path and debug_path.exists():
                    try:
                        from .common_utils import upload_intermediate_file_to_storage
                        
                        # Upload the combined image (most important - shows exactly what VLM sees)
                        upload_filename = f"vlm_debug_pair{i}_combined.jpg"
                        upload_url = upload_intermediate_file_to_storage(
                            debug_path,
                            task_id,
                            upload_filename,
                            dprint=dprint
                        )
                        if upload_url:
                            dprint(f"[VLM_DEBUG_UPLOAD] ✅ Pair {i} COMBINED (what VLM sees): {upload_url}")
                        else:
                            dprint(f"[VLM_DEBUG_UPLOAD] ❌ Failed to upload combined image for pair {i}")
                        
                        # Upload individual images too for clarity
                        if start_debug_path and start_debug_path.exists():
                            start_url = upload_intermediate_file_to_storage(
                                start_debug_path, task_id, f"vlm_debug_pair{i}_LEFT.jpg", dprint=dprint
                            )
                            if start_url:
                                dprint(f"[VLM_DEBUG_UPLOAD] ✅ Pair {i} LEFT (start image): {start_url}")
                        
                        if end_debug_path and end_debug_path.exists():
                            end_url = upload_intermediate_file_to_storage(
                                end_debug_path, task_id, f"vlm_debug_pair{i}_RIGHT.jpg", dprint=dprint
                            )
                            if end_url:
                                dprint(f"[VLM_DEBUG_UPLOAD] ✅ Pair {i} RIGHT (end image): {end_url}")
                                
                    except Exception as e_upload:
                        dprint(f"[VLM_DEBUG_UPLOAD] ❌ Failed to upload debug images for pair {i}: {e_upload}")

                # Craft query with base_prompt context
                base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "the objects/people inside the scene move excitingly and things transform or shift with the camera"

                # Add duration info if available
                duration_text = ""
                if num_frames_list and i < len(num_frames_list) and num_frames_list[i]:
                    num_frames = num_frames_list[i]
                    duration_seconds = num_frames / fps
                    duration_text = f" This transition occurs over approximately {duration_seconds:.1f} seconds ({num_frames} frames at {fps} FPS)."

                query = f"""You are viewing two images side by side: the left image shows the starting frame, and the right image shows the ending frame of a video sequence.

{duration_text} Your goal is to create a THREE-SENTENCE prompt that describes the MOTION and CHANGES in this transition based on the user's description: '{base_prompt_text}'

FOCUS ON MOTION: Describe what MOVES, what CHANGES, and HOW things transition between these frames. Everything should be described in terms of motion and transformation, not static states.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

SENTENCE 1 (PRIMARY MOTION): Describe the main action, camera movement, and major scene transitions. What is the dominant movement happening?

SENTENCE 2 (MOVING ELEMENTS): Describe how the characters, objects, and environment are moving or changing. Focus on what's in motion and how it moves through space.

SENTENCE 3 (MOTION DETAILS): Describe the subtle motion details - secondary movements, environmental dynamics, particles, lighting shifts, and small-scale motions.

Examples of MOTION-FOCUSED descriptions:

- "The sun rises rapidly above the jagged peaks as the camera tilts upward from the dark valley floor. The silhouette pine trees sway gently against the shifting violet and gold sky as the entire landscape brightens. Wisps of morning mist evaporate and drift upward from the river surface while distant birds circle and glide through the upper left corner."

- "A woman sprints from the kitchen into the bright exterior sunlight as the camera pans right to track her accelerating path. Her vintage floral dress flows and ripples in the wind while colorful playground equipment blurs past in the background. Her hair whips back dynamically and dust particles kick up and swirl around her sneakers as she impacts the gravel."

- "The camera zooms aggressively inward into a macro shot of an eye as the brown horse reflection grows larger and more detailed. The iris textures shift under the changing warm lighting while the biological details come into sharper focus. The pupil constricts and contracts in reaction to the light while the tiny reflected horse tosses its mane and shifts position."

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""

                # Run inference
                result = extender.extend_with_img(
                    prompt=query,
                    system_prompt=system_prompt,
                    image=combined_img
                )

                vlm_prompt = result.prompt.strip()
                dprint(f"[VLM_BATCH] Generated: {vlm_prompt}")

                results.append(vlm_prompt)

            except Exception as e:
                dprint(f"[VLM_BATCH] ERROR processing pair {i+1}: {e}")
                # Fallback to base_prompt
                if base_prompt and base_prompt.strip():
                    results.append(base_prompt)
                else:
                    results.append("cinematic transition")

        dprint(f"[VLM_BATCH] Completed {len(results)}/{len(image_pairs)} prompts")

        # Log memory BEFORE cleanup (ALWAYS log, not debug-only)
        if torch.cuda.is_available():
            gpu_mem_before_cleanup = torch.cuda.memory_allocated() / 1024**3
            print(f"[VLM_CLEANUP] GPU memory BEFORE cleanup: {gpu_mem_before_cleanup:.2f} GB")

        # Explicitly delete model and processor to free all memory
        print(f"[VLM_CLEANUP] Cleaning up VLM model and processor...")
        try:
            del extender.model
            del extender.processor
            del extender
            print(f"[VLM_CLEANUP] ✅ Successfully deleted VLM objects")
        except Exception as e:
            print(f"[VLM_CLEANUP] ⚠️  Error during deletion: {e}")

        # Force garbage collection
        import gc
        collected = gc.collect()
        print(f"[VLM_CLEANUP] Garbage collected {collected} objects")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
            gpu_freed = gpu_mem_before_cleanup - gpu_mem_after
            print(f"[VLM_CLEANUP] GPU memory AFTER cleanup: {gpu_mem_after:.2f} GB")
            print(f"[VLM_CLEANUP] GPU memory freed: {gpu_freed:.2f} GB")

        print(f"[VLM_CLEANUP] ✅ VLM cleanup complete")

        return results

    except Exception as e:
        dprint(f"[VLM_BATCH] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to base prompts
        return [bp if bp and bp.strip() else "cinematic transition" for bp in base_prompts]


def generate_single_image_prompt(
    image_path: str,
    base_prompt: Optional[str] = None,
    num_frames: Optional[int] = None,
    fps: int = 16,
    device: str = "cuda",
    dprint=print
) -> str:
    """
    Use QwenVLM to generate a descriptive prompt based on a single image.
    
    This is used for single-image video generation where there's no transition
    between images - instead, we describe the image and suggest natural motion.

    Args:
        image_path: Path to the image
        base_prompt: Optional base prompt to incorporate
        num_frames: Number of frames in the video segment (for duration calculation)
        fps: Frames per second (default: 16)
        device: Device to run the model on ('cuda' or 'cpu')
        dprint: Print function for logging

    Returns:
        Generated prompt describing the image and suggesting motion
    """
    try:
        # Add Wan2GP to path for imports
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from Wan2GP.wan.utils.prompt_extend import QwenPromptExpander

        dprint(f"[VLM_SINGLE] Generating prompt for single image: {Path(image_path).name}")

        # Load the image
        img = Image.open(image_path).convert("RGB")

        # Initialize VLM with Qwen2.5-VL-7B
        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"

        # Ensure model is downloaded
        dprint(f"[VLM_SINGLE] Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)

        dprint(f"[VLM_SINGLE] Initializing Qwen2.5-VL-7B-Instruct from local path: {local_model_path}")
        extender = QwenPromptExpander(
            model_name=str(local_model_path),
            device=device,
            is_vl=True
        )

        # Craft the query prompt
        base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "the objects/people inside the scene move excitingly and things transform or shift with the camera"

        # Add duration info if available
        duration_text = ""
        if num_frames and fps:
            duration_seconds = num_frames / fps
            duration_text = f" The video will be approximately {duration_seconds:.1f} seconds long ({num_frames} frames at {fps} FPS)."

        query = f"""You are viewing a single image that will be the starting frame of a video sequence.

{duration_text} Your goal is to create a THREE-SENTENCE prompt that describes the image and suggests NATURAL MOTION based on the user's description: '{base_prompt_text}'

FOCUS ON MOTION: Describe what's in the image and how things could naturally move, animate, or change. Everything should suggest dynamic motion from this starting point.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

SENTENCE 1 (SCENE & CAMERA): Describe the scene and suggest camera movement (pan, zoom, tilt, tracking shot, etc.). What would a cinematographer do?

SENTENCE 2 (SUBJECT MOTION): Describe the main subjects and how they could naturally move or animate. People breathe, blink, shift weight; animals move; plants sway; water flows.

SENTENCE 3 (ENVIRONMENTAL DYNAMICS): Describe ambient motion - wind effects, lighting changes, particles, atmospheric effects, subtle movements that bring the scene to life.

Examples of MOTION-FOCUSED single-image descriptions:

- "The camera slowly pushes forward through the misty forest as the early morning light filters through the canopy. The tall pine trees sway gently in the breeze while a deer in the clearing lifts its head alertly and flicks its ears. Dust motes drift lazily through the golden light beams and fallen leaves rustle and tumble across the forest floor."

- "The camera tracks slowly around the woman as she stands at the window gazing out at the city. She shifts her weight slightly and turns her head, her hair catching the warm light from the sunset outside. The curtains billow softly in a gentle breeze while city lights begin twinkling on in the darkening skyline."

- "The camera zooms gradually into the vintage car parked on the empty desert road as heat waves shimmer off the asphalt. Chrome details on the car glint and sparkle as the harsh sun shifts position overhead. Tumbleweeds roll slowly across the cracked pavement while sand particles drift on the hot wind."

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""

        system_prompt = "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES following this structure: 1) SCENE & CAMERA, 2) SUBJECT MOTION, 3) ENVIRONMENTAL DYNAMICS. Focus on natural motion that could emerge from this single image."

        dprint(f"[VLM_SINGLE] Running inference...")
        result = extender.extend_with_img(
            prompt=query,
            system_prompt=system_prompt,
            image=img
        )

        vlm_prompt = result.prompt.strip()
        dprint(f"[VLM_SINGLE] Generated: {vlm_prompt}")

        # Cleanup
        try:
            del extender.model
            del extender.processor
            del extender
        except:
            pass
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return vlm_prompt

    except Exception as e:
        dprint(f"[VLM_SINGLE] ERROR: Failed to generate single image prompt: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to base_prompt if VLM fails
        if base_prompt and base_prompt.strip():
            dprint(f"[VLM_SINGLE] Falling back to base prompt: {base_prompt}")
            return base_prompt
        else:
            dprint(f"[VLM_SINGLE] Falling back to generic prompt")
            return "cinematic video"


def generate_single_image_prompts_batch(
    image_paths: List[str],
    base_prompts: List[Optional[str]],
    num_frames_list: Optional[List[int]] = None,
    fps: int = 16,
    device: str = "cuda",
    dprint=print
) -> List[str]:
    """
    Batch generate prompts for multiple single images.
    
    This is more efficient than calling generate_single_image_prompt() multiple times,
    because it loads the VLM model once and reuses it for all images.

    Args:
        image_paths: List of image paths
        base_prompts: List of base prompts (one per image, can be None)
        num_frames_list: List of frame counts for each segment (for duration calculation)
        fps: Frames per second (default: 16)
        device: Device to run the model on ('cuda' or 'cpu')
        dprint: Print function for logging

    Returns:
        List of generated prompts (one per image)
    """
    if len(image_paths) != len(base_prompts):
        raise ValueError(f"image_paths and base_prompts must have same length ({len(image_paths)} != {len(base_prompts)})")

    if not image_paths:
        return []

    try:
        # Add Wan2GP to path for imports
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from Wan2GP.wan.utils.prompt_extend import QwenPromptExpander

        # Log memory BEFORE loading
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
            dprint(f"[VLM_SINGLE_BATCH] GPU memory BEFORE: {gpu_mem_before:.2f} GB")

        dprint(f"[VLM_SINGLE_BATCH] Initializing Qwen2.5-VL-7B-Instruct for {len(image_paths)} single images...")

        # Initialize VLM ONCE for all images
        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"

        # Ensure model is downloaded
        dprint(f"[VLM_SINGLE_BATCH] Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)

        dprint(f"[VLM_SINGLE_BATCH] Using local model from: {local_model_path}")
        extender = QwenPromptExpander(
            model_name=str(local_model_path),
            device=device,
            is_vl=True
        )

        dprint(f"[VLM_SINGLE_BATCH] Model loaded")

        system_prompt = "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES following this structure: 1) SCENE & CAMERA, 2) SUBJECT MOTION, 3) ENVIRONMENTAL DYNAMICS. Focus on natural motion that could emerge from this single image."

        results = []
        for i, (image_path, base_prompt) in enumerate(zip(image_paths, base_prompts)):
            try:
                dprint(f"[VLM_SINGLE_BATCH] Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")

                # Load image
                img = Image.open(image_path).convert("RGB")
                dprint(f"[VLM_SINGLE_BATCH] Image {i}: dimensions={img.size}")

                # Craft query with base_prompt context
                base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "the objects/people inside the scene move excitingly and things transform or shift with the camera"

                # Add duration info if available
                duration_text = ""
                if num_frames_list and i < len(num_frames_list) and num_frames_list[i]:
                    num_frames = num_frames_list[i]
                    duration_seconds = num_frames / fps
                    duration_text = f" The video will be approximately {duration_seconds:.1f} seconds long ({num_frames} frames at {fps} FPS)."

                query = f"""You are viewing a single image that will be the starting frame of a video sequence.

{duration_text} Your goal is to create a THREE-SENTENCE prompt that describes the image and suggests NATURAL MOTION based on the user's description: '{base_prompt_text}'

FOCUS ON MOTION: Describe what's in the image and how things could naturally move, animate, or change. Everything should suggest dynamic motion from this starting point.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

SENTENCE 1 (SCENE & CAMERA): Describe the scene and suggest camera movement (pan, zoom, tilt, tracking shot, etc.). What would a cinematographer do?

SENTENCE 2 (SUBJECT MOTION): Describe the main subjects and how they could naturally move or animate. People breathe, blink, shift weight; animals move; plants sway; water flows.

SENTENCE 3 (ENVIRONMENTAL DYNAMICS): Describe ambient motion - wind effects, lighting changes, particles, atmospheric effects, subtle movements that bring the scene to life.

Examples of MOTION-FOCUSED single-image descriptions:

- "The camera slowly pushes forward through the misty forest as the early morning light filters through the canopy. The tall pine trees sway gently in the breeze while a deer in the clearing lifts its head alertly and flicks its ears. Dust motes drift lazily through the golden light beams and fallen leaves rustle and tumble across the forest floor."

- "The camera tracks slowly around the woman as she stands at the window gazing out at the city. She shifts her weight slightly and turns her head, her hair catching the warm light from the sunset outside. The curtains billow softly in a gentle breeze while city lights begin twinkling on in the darkening skyline."

- "The camera zooms gradually into the vintage car parked on the empty desert road as heat waves shimmer off the asphalt. Chrome details on the car glint and sparkle as the harsh sun shifts position overhead. Tumbleweeds roll slowly across the cracked pavement while sand particles drift on the hot wind."

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""

                # Run inference
                result = extender.extend_with_img(
                    prompt=query,
                    system_prompt=system_prompt,
                    image=img
                )

                vlm_prompt = result.prompt.strip()
                dprint(f"[VLM_SINGLE_BATCH] Generated: {vlm_prompt}")

                results.append(vlm_prompt)

            except Exception as e:
                dprint(f"[VLM_SINGLE_BATCH] ERROR processing image {i+1}: {e}")
                # Fallback to base_prompt
                if base_prompt and base_prompt.strip():
                    results.append(base_prompt)
                else:
                    results.append("cinematic video")

        dprint(f"[VLM_SINGLE_BATCH] Completed {len(results)}/{len(image_paths)} prompts")

        # Cleanup
        if torch.cuda.is_available():
            gpu_mem_before_cleanup = torch.cuda.memory_allocated() / 1024**3
            print(f"[VLM_SINGLE_CLEANUP] GPU memory BEFORE cleanup: {gpu_mem_before_cleanup:.2f} GB")

        print(f"[VLM_SINGLE_CLEANUP] Cleaning up VLM model and processor...")
        try:
            del extender.model
            del extender.processor
            del extender
            print(f"[VLM_SINGLE_CLEANUP] ✅ Successfully deleted VLM objects")
        except Exception as e:
            print(f"[VLM_SINGLE_CLEANUP] ⚠️  Error during deletion: {e}")

        import gc
        collected = gc.collect()
        print(f"[VLM_SINGLE_CLEANUP] Garbage collected {collected} objects")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
            gpu_freed = gpu_mem_before_cleanup - gpu_mem_after
            print(f"[VLM_SINGLE_CLEANUP] GPU memory AFTER cleanup: {gpu_mem_after:.2f} GB")
            print(f"[VLM_SINGLE_CLEANUP] GPU memory freed: {gpu_freed:.2f} GB")

        print(f"[VLM_SINGLE_CLEANUP] ✅ VLM cleanup complete")

        return results

    except Exception as e:
        dprint(f"[VLM_SINGLE_BATCH] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to base prompts
        return [bp if bp and bp.strip() else "cinematic video" for bp in base_prompts]


def test_vlm_transition():
    """Test function for VLM transition prompt generation."""
    print("\n" + "="*80)
    print("Testing VLM Transition Prompt Generation")
    print("="*80 + "\n")

    # This is a placeholder test - would need actual test images
    print("To test, call:")
    print("  generate_transition_prompt('path/to/start.jpg', 'path/to/end.jpg')")
    print("\nExample usage in travel orchestrator:")
    print("  if orchestrator_payload.get('enhance_prompt', False):")
    print("      prompt = generate_transition_prompt(start_img, end_img, base_prompt)")


if __name__ == "__main__":
    test_vlm_transition()
