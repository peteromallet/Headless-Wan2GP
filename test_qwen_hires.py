#!/usr/bin/env python3
"""
Test script for qwen_image_hires two-pass generation.

Runs tasks directly (no database needed) with varying hires pass settings.

Usage:
    python test_qwen_hires.py
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

# Test images
TEST_IMAGES = [
    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/thumbnails/thumb_1759060119274_jiq13e.jpg",
    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/17feb9ad-0252-4de2-8661-a5adb2d71738-u2_1ba92b67-d100-4d2f-9c99-370eb0163c04.jpeg",
    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/thumbnails/thumb_1759059607755_lpq4rg.jpg",
]

# Base task parameters
BASE_TASK = {
    "seed": 2062757810,
    "resolution": "1152x864",
    "num_inference_steps": 6,  # Pass 1 steps
    "guidance_scale": 1,
    "in_this_scene": True,
    "add_in_position": False,

    # LoRAs with phase-based multipliers (pass1;pass2)
    # The system auto-converts simple strengths like "0.5" to "0.5;0.5"
    # Lightning LoRAs are auto-detected and converted to "X;0" (disabled in pass 2)
    # You can manually specify phase multipliers for fine control:
    #   "0.5;0.5" = same strength both passes
    #   "1.0;0"   = only in pass 1
    #   "0;0.8"   = only in pass 2
    #   "0.5;0.8" = different strengths per pass
    "additional_loras": {
        "https://huggingface.co/peteromallet/random_junk/resolve/main/in_scene_different_object_000010500.safetensors": "0.5;0.5"
    },

    "in_this_scene_strength": 0.0,
    "scene_reference_strength": 0.0,
    "style_reference_strength": 1.1,  # Will be auto-converted to "1.1;1.1"
}

# Test prompts (applied to each image)
TEST_PROMPTS = [
    "A serene mountain landscape at golden hour, with snow-capped peaks reflecting in a crystal clear alpine lake, pine trees in the foreground, dramatic clouds in the sky",
    "A bustling city street at night, neon signs reflecting on wet pavement, people with umbrellas walking past illuminated shop windows, rain creating a dreamy atmosphere",
    "An underwater coral reef scene, colorful tropical fish swimming among vibrant coral formations, rays of sunlight penetrating the crystal blue water, creating dancing light patterns on the ocean floor"
]

PROMPT_NAMES = ["mountain", "citynight", "underwater"]

# Hires settings (fixed for all tests)
HIRES_SCALE = 1.0  # 1x = same resolution (refinement only, no upscale)
HIRES_STEPS = 6  # Pass 2 steps
HIRES_DENOISE = 0.6  # Noise amount: 0.0 (no change) to 1.0 (full denoise)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "test_results" / "qwen_hires"


# ============================================================================
# Text Overlay
# ============================================================================

def apply_text_overlay(image: Image.Image, text: str, position: str = "bottom-left") -> Image.Image:
    """Apply text overlay to an image with a semi-transparent background."""
    if not text:
        return image
    
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    font_size = max(18, min(image.width, image.height) // 35)
    font = None
    font_paths = [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()
    
    lines = text.strip().split('\n')
    line_height = font_size + 6
    max_line_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_line_width = max(max_line_width, bbox[2] - bbox[0])
    
    text_height = line_height * len(lines)
    padding = 12
    
    if position == "bottom-left":
        x = padding
        y = image.height - text_height - padding * 2
    elif position == "top-left":
        x = padding
        y = padding
    else:
        x = padding
        y = image.height - text_height - padding * 2
    
    bg_bbox = (x - padding, y - padding, x + max_line_width + padding, y + text_height + padding)
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_height), line, font=font, fill=(255, 255, 255, 255))
    
    return image.convert('RGB')


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Qwen Image Hires Test - Two-Pass Generation (Direct Execution)")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Base resolution: {BASE_TASK['resolution']}")
    print(f"  Pass 1 steps: {BASE_TASK['num_inference_steps']}")
    print(f"  Pass 2 steps: {HIRES_STEPS}")
    print(f"  Hires scale: {HIRES_SCALE}x")
    print(f"  Hires denoise: {HIRES_DENOISE}")
    print(f"  Number of test images: {len(TEST_IMAGES)}")
    print(f"  Prompts per image: {len(TEST_PROMPTS)}")
    print(f"  Total tests: {len(TEST_IMAGES) * len(TEST_PROMPTS)}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print()

    # Import the task processing machinery
    print("Loading model and task system...")
    from source.task_conversion import db_task_to_generation_task
    from headless_model_management import HeadlessTaskQueue

    wan2gp_path = str(PROJECT_ROOT / "Wan2GP")

    # Initialize and start task queue
    task_queue = HeadlessTaskQueue(
        wan_dir=wan2gp_path,
        debug_mode=True,
        main_output_dir=str(OUTPUT_DIR)
    )
    task_queue.start()

    print("Queue started!")
    print()
    print("-" * 70)

    completed = 0
    failed = 0

    try:
        for img_idx, image_url in enumerate(TEST_IMAGES, 1):
            image_name = image_url.split('/')[-1][:30]

            for prompt_idx, prompt in enumerate(TEST_PROMPTS, 1):
                print(f"\n🖼️  Image {img_idx}/{len(TEST_IMAGES)} × Prompt {prompt_idx}/{len(TEST_PROMPTS)}: {image_name}")
                print(f"  📝 {prompt[:80]}...")

                task_id = f"hires_test_img{img_idx}_p{prompt_idx}_{datetime.now().strftime('%H%M%S')}"

                # Build task params
                task_params = {
                    **BASE_TASK,
                    "prompt": prompt,
                    "style_reference_image": image_url,
                    "subject_reference_image": image_url,
                    "hires_scale": HIRES_SCALE,
                    "hires_steps": HIRES_STEPS,
                    "hires_denoise": HIRES_DENOISE,
                    "hires_upscale_method": "bicubic",
                }

                print(f"  🔄 Running: {BASE_TASK['num_inference_steps']} steps → {HIRES_STEPS} steps @ {HIRES_SCALE}x")

                try:
                    # Convert to generation task
                    gen_task = db_task_to_generation_task(
                        db_task_params=task_params,
                        task_id=task_id,
                        task_type="qwen_image_style",
                        wan2gp_path=wan2gp_path
                    )

                    # Submit task and wait for completion
                    task_queue.submit_task(gen_task)
                    result = task_queue.wait_for_completion(task_id, timeout=600.0)

                    if result.get("success") and result.get("output_path"):
                        output_path = Path(result["output_path"])

                        # Load, overlay, and save
                        if output_path.exists():
                            image = Image.open(output_path)

                            overlay_text = (
                                f"Pass1: {BASE_TASK['num_inference_steps']} steps @ {BASE_TASK['resolution']}\n"
                                f"Pass2: {HIRES_STEPS} steps @ {HIRES_DENOISE} denoise\n"
                                f"Scale: {HIRES_SCALE}x"
                            )
                            image = apply_text_overlay(image, overlay_text)

                            # Save to organized folder
                            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                            save_path = OUTPUT_DIR / f"img{img_idx}_{PROMPT_NAMES[prompt_idx - 1]}_hires.jpg"
                            image.save(save_path, quality=95)

                            print(f"  ✅ Saved: {save_path.relative_to(PROJECT_ROOT)}")
                            completed += 1
                        else:
                            print(f"  ⚠️ Output file not found: {output_path}")
                            failed += 1
                    else:
                        error = result.get("error", "Unknown error")
                        print(f"  ❌ Failed: {error}")
                        failed += 1

                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    failed += 1
    finally:
        # Always stop the queue
        print("\nStopping queue...")
        task_queue.stop()

    print()
    print("=" * 70)
    print(f"Complete: {completed} succeeded, {failed} failed")
    print(f"Results in: {OUTPUT_DIR}/")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
