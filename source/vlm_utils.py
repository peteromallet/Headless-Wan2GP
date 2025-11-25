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
        base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "a cinematic sequence"

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
    dprint=print
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

                # Load and combine images
                start_img = Image.open(start_path).convert("RGB")
                end_img = Image.open(end_path).convert("RGB")

                combined_width = start_img.width + end_img.width
                combined_height = max(start_img.height, end_img.height)
                combined_img = Image.new('RGB', (combined_width, combined_height))
                combined_img.paste(start_img, (0, 0))
                combined_img.paste(end_img, (start_img.width, 0))

                # Craft query with base_prompt context
                base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "a cinematic sequence"

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
