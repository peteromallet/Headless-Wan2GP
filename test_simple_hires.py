#!/usr/bin/env python3
"""
Simple test to verify hires fix works without NaN errors.
Tests just one prompt with hires_steps=1.
"""
import os
import sys

# Enable debug mode to save Pass 1 output
os.environ["DEBUG_HIRES_SAVE_PASS1"] = "1"

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.headless_task_queue import HeadlessTaskQueue
from source.task_registry import TaskRegistry
import asyncio

def main():
    print("=" * 70)
    print("Simple Hires Fix Test - Verify NaN Bug is Fixed")
    print("=" * 70)
    print()

    # Configuration
    config = {
        "prompt": "A cat sitting on a red cushion in warm sunlight",
        "model": "qwen_image_edit_20B",
        "resolution": "1152x864",
        "num_inference_steps": 6,
        "guidance_scale": 1,
        "seed": 12345,
        "hires_config": {
            "enabled": True,
            "scale": 2.0,
            "hires_steps": 1,  # This was causing NaN with old code
            "denoising_strength": 0.5,
            "upscale_method": "bicubic"
        }
    }

    print(f"Configuration:")
    print(f"  Prompt: {config['prompt']}")
    print(f"  Base resolution: {config['resolution']}")
    print(f"  Pass 1 steps: {config['num_inference_steps']}")
    print(f"  Pass 2 steps: {config['hires_config']['hires_steps']}")
    print(f"  Hires denoise: {config['hires_config']['denoising_strength']}")
    print(f"  Debug output: Enabled (Pass 1 will be saved)")
    print()

    # Initialize task queue
    wan2gp_path = os.path.join(os.path.dirname(__file__), "Wan2GP")
    output_dir = os.path.join(os.path.dirname(__file__), "test_results", "simple_hires")
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing task queue...")
    queue = HeadlessTaskQueue(
        wan2gp_path=wan2gp_path,
        output_directory=output_dir,
        num_workers=1
    )
    queue.start()
    print("Queue started!")
    print()

    # Create and submit task
    print("Submitting task...")
    task_id = queue.submit_task(
        task_type="qwen_image_style",
        priority=0,
        **config
    )
    print(f"Task submitted: {task_id}")
    print()

    # Wait for completion
    print("Waiting for task to complete...")
    result = queue.wait_for_completion(task_id, timeout=600)

    if result and result.get("status") == "completed":
        print()
        print("=" * 70)
        print("✅ SUCCESS! Hires fix completed without errors")
        print("=" * 70)
        print(f"Output: {result.get('output_path')}")
        print(f"Status: {result.get('status')}")
        print()
        print("Check the debug output (debug_pass1_*.png) to see Pass 1 result")
        return 0
    else:
        print()
        print("=" * 70)
        print("❌ FAILED! Task did not complete successfully")
        print("=" * 70)
        print(f"Result: {result}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
