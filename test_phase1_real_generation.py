#!/usr/bin/env python3
"""
Phase 1 Real Generation Test

This script tests actual image/video generation to verify files are saved
to the configured output directory.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from headless_model_management import HeadlessTaskQueue, GenerationTask

def test_real_generation(test_output_dir: str, wan_dir: str):
    """
    Test actual generation to verify files are saved correctly.
    """
    print(f"\n{'='*60}")
    print("Phase 1 Real Generation Test")
    print(f"{'='*60}\n")

    # Create test output directory
    test_output_path = Path(test_output_dir).resolve()
    test_output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Test output directory: {test_output_path}")

    # Count files before generation
    files_before = list(test_output_path.glob("*"))
    print(f"✓ Files in output dir before: {len(files_before)}")

    # Initialize HeadlessTaskQueue with custom output directory
    print(f"\n1. Initializing HeadlessTaskQueue...")
    try:
        queue = HeadlessTaskQueue(
            wan_dir=wan_dir,
            max_workers=1,
            debug_mode=True,
            main_output_dir=str(test_output_path)
        )
        print("✓ HeadlessTaskQueue initialized")
    except Exception as e:
        print(f"✗ Failed to initialize HeadlessTaskQueue: {e}")
        return False

    # Start the queue
    print("\n2. Starting queue...")
    try:
        # Don't preload any model - let it load on demand
        queue.start(preload_model=None)
        print("✓ Queue started")
    except Exception as e:
        print(f"✗ Failed to start queue: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Submit a simple image generation task (Flux is fastest)
    print("\n3. Submitting test image generation task (Flux t2i)...")
    try:
        # Use Flux for fastest generation (single image)
        task = GenerationTask(
            id="test_phase1_generation",
            model="flux",  # Flux is fast for testing
            prompt="a red apple on a table",
            parameters={
                "video_length": 1,  # Single image
                "resolution": "512x512",  # Small for speed
                "num_inference_steps": 4,  # Minimal steps for speed
                "seed": 42
            }
        )

        task_id = queue.submit_task(task)
        print(f"✓ Task submitted: {task_id}")
    except Exception as e:
        print(f"✗ Failed to submit task: {e}")
        import traceback
        traceback.print_exc()
        queue.stop()
        return False

    # Wait for task completion
    print("\n4. Waiting for task completion (timeout: 300s)...")
    try:
        result = queue.wait_for_completion(task_id, timeout=300.0)

        if result["success"]:
            output_path = result["output_path"]
            print(f"✓ Task completed successfully")
            print(f"  Output path: {output_path}")

            # Verify output file exists
            output_file = Path(output_path)
            if output_file.exists():
                print(f"✓ Output file exists: {output_file}")

                # Verify it's in our test directory
                try:
                    output_file.relative_to(test_output_path)
                    print(f"✓ Output file is in configured directory!")

                    # Clean up test file
                    output_file.unlink()
                    print(f"✓ Cleaned up test file")

                    queue.stop()

                    print(f"\n{'='*60}")
                    print("✅ Phase 1 Real Generation Test: PASSED")
                    print(f"{'='*60}\n")
                    print("Files are being saved to the configured output directory!")
                    return True

                except ValueError:
                    print(f"✗ Output file is NOT in configured directory!")
                    print(f"   Expected: {test_output_path}")
                    print(f"   Got:      {output_file.parent}")
                    queue.stop()
                    return False
            else:
                print(f"✗ Output file does not exist: {output_file}")
                queue.stop()
                return False
        else:
            error = result.get("error", "Unknown error")
            print(f"✗ Task failed: {error}")
            queue.stop()
            return False

    except Exception as e:
        print(f"✗ Error during task execution: {e}")
        import traceback
        traceback.print_exc()
        queue.stop()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test real generation with Phase 1 config")
    parser.add_argument(
        "--test-output-dir",
        type=str,
        default="./test_phase1_real_outputs",
        help="Test output directory to use"
    )
    parser.add_argument(
        "--wan-dir",
        type=str,
        default="./Wan2GP",
        help="Path to Wan2GP directory"
    )

    args = parser.parse_args()

    # Resolve paths
    wan_dir = Path(args.wan_dir).resolve()
    if not wan_dir.exists():
        print(f"✗ Wan2GP directory not found: {wan_dir}")
        sys.exit(1)

    # Run test
    success = test_real_generation(args.test_output_dir, str(wan_dir))

    if not success:
        print(f"\n{'='*60}")
        print("❌ Phase 1 Real Generation Test: FAILED")
        print(f"{'='*60}\n")
        print("Files are NOT being saved to the configured directory.")
        print("DO NOT proceed to production!")
        sys.exit(1)

if __name__ == "__main__":
    main()
