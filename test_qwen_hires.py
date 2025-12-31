#!/usr/bin/env python3
"""
Test script for qwen_image_hires two-pass generation.

Submits tasks with varying hires pass settings (steps 1-6 for pass 2)
across 4 different prompts. Monitors completion and applies text overlays.

Usage:
    python test_qwen_hires.py
"""

import os
import sys
import uuid
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from supabase import create_client, Client

# ============================================================================
# Configuration
# ============================================================================

# Base task parameters (from user's example)
BASE_TASK = {
    "seed": 2062757810,
    "resolution": "1152x864",  # Base resolution for pass 1
    "num_inference_steps": 6,  # Pass 1 steps
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
HIRES_SCALE = 2.0  # 2x upscale
HIRES_DENOISE = 0.5  # 50% denoise for pass 2

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
    
    # Try to use a nice font, fall back to default
    font_size = max(18, min(image.width, image.height) // 35)
    font = None
    font_paths = [
        "/System/Library/Fonts/Menlo.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",  # Windows
    ]
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()
    
    # Split text into lines and calculate bbox
    lines = text.strip().split('\n')
    line_height = font_size + 6
    max_line_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_line_width = max(max_line_width, bbox[2] - bbox[0])
    
    text_height = line_height * len(lines)
    padding = 12
    
    # Calculate position
    if position == "bottom-left":
        x = padding
        y = image.height - text_height - padding * 2
    elif position == "top-left":
        x = padding
        y = padding
    elif position == "bottom-right":
        x = image.width - max_line_width - padding * 2
        y = image.height - text_height - padding * 2
    else:  # top-right
        x = image.width - max_line_width - padding * 2
        y = padding
    
    # Draw semi-transparent background
    bg_bbox = (x - padding, y - padding, x + max_line_width + padding, y + text_height + padding)
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
    
    # Composite overlay onto image
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    
    # Draw text
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_height), line, font=font, fill=(255, 255, 255, 255))
    
    return image.convert('RGB')


# ============================================================================
# Supabase Setup
# ============================================================================

def get_supabase_client() -> Client:
    """Get Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    
    if not url or not key:
        # Try loading from .env file
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError(
            "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
        )
    
    return create_client(url, key)


# ============================================================================
# Task Management
# ============================================================================

def generate_task_id(prefix: str = "hires_test") -> str:
    """Generate a unique task ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{short_uuid}"


def create_hires_task(prompt: str, hires_steps: int, prompt_idx: int) -> dict:
    """Create a qwen_image_style task with hires fix parameters."""
    task_id = generate_task_id(f"hires_p{prompt_idx}_s{hires_steps}")
    
    # Create output subdirectory for this specific test run
    output_subdir = f"{OUTPUT_DIR}/prompt{prompt_idx}/hires_steps_{hires_steps}"
    
    # Overlay text for this task
    overlay_text = f"Pass1: {BASE_TASK['num_inference_steps']} steps @ {BASE_TASK['resolution']}\nPass2: {hires_steps} steps @ {HIRES_DENOISE} denoise\nScale: {HIRES_SCALE}x"
    
    task = {
        "id": task_id,
        "task_type": "qwen_image_style",  # Use style task which now supports hires
        "status": "Pending",
        "prompt": prompt,
        "prompt_idx": prompt_idx,
        "hires_steps": hires_steps,
        "overlay_text": overlay_text,
        "params": {
            **BASE_TASK,
            "prompt": prompt,
            # Hires configuration (triggers two-pass when hires_scale is set)
            "hires_scale": HIRES_SCALE,
            "hires_steps": hires_steps,
            "hires_denoise": HIRES_DENOISE,
            "hires_upscale_method": "bicubic",
            # Output directory
            "output_dir": output_subdir,
        },
    }
    
    return task


def submit_task(supabase: Client, task: dict) -> bool:
    """Submit a task to Supabase."""
    try:
        result = supabase.table("tasks").insert({
            "id": task["id"],
            "task_type": task["task_type"],
            "status": task["status"],
            "params": task["params"],
        }).execute()
        
        print(f"  ✓ Submitted: {task['id']}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to submit {task['id']}: {e}")
        return False


def check_task_status(supabase: Client, task_ids: list) -> dict:
    """Check status of multiple tasks."""
    try:
        result = supabase.table("tasks").select("id, status, output_url").in_("id", task_ids).execute()
        return {r["id"]: r for r in result.data}
    except Exception as e:
        print(f"Error checking task status: {e}")
        return {}


def download_and_overlay(task: dict, output_path: Path) -> bool:
    """Download task output and apply overlay."""
    import requests
    
    output_url = task.get("output_url")
    if not output_url:
        return False
    
    try:
        # Download image
        response = requests.get(output_url, timeout=30)
        response.raise_for_status()
        
        # Load image
        from io import BytesIO
        image = Image.open(BytesIO(response.content))
        
        # Apply overlay
        overlay_text = task.get("overlay_text", "")
        if overlay_text:
            image = apply_text_overlay(image, overlay_text)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"  Error processing {task['id']}: {e}")
        return False


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
    
    # Connect to Supabase
    print("Connecting to Supabase...")
    try:
        supabase = get_supabase_client()
        print("  ✓ Connected")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        sys.exit(1)
    
    print()
    print("Submitting tasks...")
    print("-" * 70)
    
    # Create and submit all tasks
    all_tasks = []
    submitted = 0
    failed = 0
    
    for prompt_idx, prompt in enumerate(PROMPTS, 1):
        prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
        print(f"\nPrompt {prompt_idx}: \"{prompt_preview}\"")
        
        for hires_steps in HIRES_STEPS_TO_TEST:
            task = create_hires_task(prompt, hires_steps, prompt_idx)
            
            if submit_task(supabase, task):
                all_tasks.append(task)
                submitted += 1
            else:
                failed += 1
    
    print()
    print("-" * 70)
    print(f"Submitted: {submitted}, Failed: {failed}")
    
    if submitted == 0:
        print("No tasks submitted. Exiting.")
        return 1
    
    # Wait for completion and process results
    print()
    print("Waiting for tasks to complete...")
    print("(Press Ctrl+C to skip waiting - you can run this script again later)")
    print()
    
    task_ids = [t["id"] for t in all_tasks]
    task_lookup = {t["id"]: t for t in all_tasks}
    completed_ids = set()
    failed_ids = set()
    
    try:
        while len(completed_ids) + len(failed_ids) < len(task_ids):
            time.sleep(5)
            
            statuses = check_task_status(supabase, task_ids)
            
            for task_id, status_info in statuses.items():
                if task_id in completed_ids or task_id in failed_ids:
                    continue
                
                status = status_info.get("status")
                
                if status == "Completed":
                    # Merge output_url into our task dict
                    task_lookup[task_id]["output_url"] = status_info.get("output_url")
                    
                    # Download and apply overlay
                    task = task_lookup[task_id]
                    prompt_idx = task["prompt_idx"]
                    hires_steps = task["hires_steps"]
                    output_path = OUTPUT_DIR / f"prompt{prompt_idx}" / f"hires_steps_{hires_steps}" / f"{task_id}.jpg"
                    
                    if download_and_overlay(task, output_path):
                        print(f"  ✓ {task_id} → {output_path.relative_to(PROJECT_ROOT)}")
                    else:
                        print(f"  ⚠ {task_id} completed but no output URL")
                    
                    completed_ids.add(task_id)
                    
                elif status == "Failed":
                    print(f"  ✗ {task_id} failed")
                    failed_ids.add(task_id)
            
            pending = len(task_ids) - len(completed_ids) - len(failed_ids)
            if pending > 0:
                print(f"  ... {pending} tasks still pending")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted! You can re-run this script to check remaining tasks.")
    
    print()
    print("=" * 70)
    print(f"Final: {len(completed_ids)} completed, {len(failed_ids)} failed")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)
    
    return 0 if len(failed_ids) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
