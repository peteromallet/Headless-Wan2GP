#!/usr/bin/env python3
"""
Test script for qwen_image_hires two-pass generation.

Runs tasks directly via WanOrchestrator (no queue, no database).

Usage:
    python test_qwen_hires.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

# Base task parameters
BASE_TASK = {
    "seed": 2062757810,
    "resolution": "1152x864",
    "num_inference_steps": 6,
    "guidance_scale": 1,
    "in_this_scene": True,
    "add_in_position": False,
    "additional_loras": {
        "https://huggingface.co/peteromallet/random_junk/resolve/main/in_scene_different_object_000010500.safetensors": 0.5
    },
    "style_reference_image": "https://ujlwuvkrxlvoswwkerdf.supabase.co/storage/v1/object/public/dataset/images/2474514122-4-d113aab8.png?",
    "in_this_scene_strength": 0.5,
    "subject_reference_image": "https://ujlwuvkrxlvoswwkerdf.supabase.co/storage/v1/object/public/dataset/images/2474514122-4-d113aab8.png?",
    "scene_reference_strength": 0.5,
    "style_reference_strength": 1.1,
}

# Test prompts
PROMPTS = [
    "A woman is skydiving through dramatic clouds at sunset, her colorful parachute unfurling behind her, the golden hour light casting long shadows across the landscape below",
    "A majestic tiger prowling through a misty bamboo forest at dawn, dappled sunlight filtering through the leaves, creating an ethereal atmosphere",
    "An astronaut floating in the vast emptiness of space, Earth visible in the background, stars scattered across the infinite darkness",
    "A vintage red sports car racing along a coastal highway, waves crashing against rocky cliffs, seagulls flying overhead in the golden afternoon light",
]

# Hires pass 2 step variations to test
HIRES_STEPS_TO_TEST = [1, 2, 3, 4, 5, 6]

# Hires settings
HIRES_SCALE = 2.0
HIRES_DENOISE = 0.5

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
    print("Qwen Image Hires Test - Two-Pass Generation")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Base resolution: {BASE_TASK['resolution']}")
    print(f"  Pass 1 steps: {BASE_TASK['num_inference_steps']}")
    print(f"  Hires scale: {HIRES_SCALE}x")
    print(f"  Hires denoise: {HIRES_DENOISE}")
    print(f"  Pass 2 steps to test: {HIRES_STEPS_TO_TEST}")
    print(f"  Number of prompts: {len(PROMPTS)}")
    print(f"  Total tasks: {len(PROMPTS) * len(HIRES_STEPS_TO_TEST)}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print()
    
    # Import and initialize WanOrchestrator
    print("Initializing WanOrchestrator...")
    from headless_wgp import WanOrchestrator
    
    wan2gp_path = str(PROJECT_ROOT / "Wan2GP")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    orchestrator = WanOrchestrator(
        wan_root=wan2gp_path,
        main_output_dir=str(OUTPUT_DIR)
    )
    
    print("Orchestrator ready!")
    print()
    print("-" * 70)
    
    completed = 0
    failed = 0
    
    for prompt_idx, prompt in enumerate(PROMPTS, 1):
        prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"\n📝 Prompt {prompt_idx}: \"{prompt_preview}\"")
        
        for hires_steps in HIRES_STEPS_TO_TEST:
            print(f"\n  🔄 Running: hires_steps={hires_steps}")
            
            try:
                # Build hires config
                hires_config = {
                    "enabled": True,
                    "scale": HIRES_SCALE,
                    "hires_steps": hires_steps,
                    "denoising_strength": HIRES_DENOISE,
                    "upscale_method": "bicubic",
                }
                
                # Call generate directly
                output_path = orchestrator.generate(
                    prompt=prompt,
                    model_type="qwen_image_edit_20B",
                    resolution=BASE_TASK["resolution"],
                    num_inference_steps=BASE_TASK["num_inference_steps"],
                    guidance_scale=BASE_TASK["guidance_scale"],
                    seed=BASE_TASK["seed"],
                    video_prompt_type="KI",
                    hires_config=hires_config,
                )
                
                if output_path and Path(output_path).exists():
                    image = Image.open(output_path)
                    
                    overlay_text = (
                        f"Pass1: {BASE_TASK['num_inference_steps']} steps @ {BASE_TASK['resolution']}\n"
                        f"Pass2: {hires_steps} steps @ {HIRES_DENOISE} denoise\n"
                        f"Scale: {HIRES_SCALE}x"
                    )
                    image = apply_text_overlay(image, overlay_text)
                    
                    # Save to organized folder
                    save_dir = OUTPUT_DIR / f"prompt{prompt_idx}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"hires_steps_{hires_steps}.jpg"
                    image.save(save_path, quality=95)
                    
                    print(f"  ✅ Saved: {save_path.relative_to(PROJECT_ROOT)}")
                    completed += 1
                else:
                    print(f"  ❌ No output returned")
                    failed += 1
                    
            except Exception as e:
                import traceback
                print(f"  ❌ Error: {e}")
                traceback.print_exc()
                failed += 1
    
    print()
    print("=" * 70)
    print(f"Complete: {completed} succeeded, {failed} failed")
    print(f"Results in: {OUTPUT_DIR}/")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
