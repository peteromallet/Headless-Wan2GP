#!/usr/bin/env python3
"""
Phase 1 Production Directory Test

Test with the ACTUAL production outputs/ directory (not a test directory).
Also runs multiple generations to verify consistency.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from headless_model_management import HeadlessTaskQueue, GenerationTask

def test_production_directory(wan_dir: str, num_generations: int = 2):
    """
    Test with actual production outputs/ directory.
    """
    print(f"\n{'='*60}")
    print("Phase 1 Production Directory Test")
    print(f"Testing with ACTUAL production outputs/ directory")
    print(f"{'='*60}\n")

    # Use the REAL production directory
    production_output_dir = Path("./outputs").resolve()
    print(f"✓ Production output directory: {production_output_dir}")
    print(f"  (This is the REAL directory that will be used in production)")

    # Initialize HeadlessTaskQueue with production directory
    print(f"\n1. Initializing HeadlessTaskQueue with production outputs/ dir...")
    try:
        queue = HeadlessTaskQueue(
            wan_dir=wan_dir,
            max_workers=1,
            debug_mode=True,
            main_output_dir=str(production_output_dir)
        )
        print("✓ HeadlessTaskQueue initialized")
    except Exception as e:
        print(f"✗ Failed to initialize HeadlessTaskQueue: {e}")
        return False

    # Start the queue
    print("\n2. Starting queue...")
    try:
        queue.start(preload_model=None)
        print("✓ Queue started")
    except Exception as e:
        print(f"✗ Failed to start queue: {e}")
        return False

    # Run multiple generations
    all_passed = True
    output_files = []

    for i in range(num_generations):
        print(f"\n{'='*60}")
        print(f"Generation {i+1} of {num_generations}")
        print(f"{'='*60}")

        # Submit task
        print(f"\n3.{i+1}. Submitting test image generation task...")
        try:
            task = GenerationTask(
                id=f"test_production_{i+1}",
                model="flux",
                prompt=f"test image number {i+1}",
                parameters={
                    "video_length": 1,
                    "resolution": "512x512",
                    "num_inference_steps": 4,
                    "seed": 42 + i  # Different seed for each
                }
            )

            task_id = queue.submit_task(task)
            print(f"✓ Task submitted: {task_id}")
        except Exception as e:
            print(f"✗ Failed to submit task: {e}")
            all_passed = False
            continue

        # Wait for completion
        print(f"\n4.{i+1}. Waiting for task completion...")
        try:
            result = queue.wait_for_completion(task_id, timeout=300.0)

            if result["success"]:
                output_path = result["output_path"]
                print(f"✓ Task completed successfully")
                print(f"  Output path: {output_path}")

                # Verify output file exists
                output_file = Path(output_path)
                if output_file.exists():
                    print(f"✓ Output file exists")

                    # Verify it's in production directory
                    try:
                        rel_path = output_file.relative_to(production_output_dir)
                        print(f"✓ Output file is in production outputs/ directory!")
                        print(f"  Relative path: {rel_path}")

                        # Verify it's NOT in old Wan2GP/outputs/
                        wan2gp_outputs = Path(wan_dir) / "outputs"
                        try:
                            output_file.relative_to(wan2gp_outputs)
                            print(f"✗ WARNING: File is in OLD Wan2GP/outputs/ location!")
                            all_passed = False
                        except ValueError:
                            print(f"✓ File is NOT in old Wan2GP/outputs/ location")

                        output_files.append(output_file)

                    except ValueError:
                        print(f"✗ Output file is NOT in production outputs/ directory!")
                        print(f"   Expected: {production_output_dir}")
                        print(f"   Got:      {output_file.parent}")
                        all_passed = False
                else:
                    print(f"✗ Output file does not exist: {output_file}")
                    all_passed = False
            else:
                error = result.get("error", "Unknown error")
                print(f"✗ Task failed: {error}")
                all_passed = False

        except Exception as e:
            print(f"✗ Error during task execution: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Stop queue
    queue.stop()

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Total generations: {num_generations}")
    print(f"Successful: {len(output_files)}")
    print(f"\nGenerated files:")
    for f in output_files:
        print(f"  - {f}")

    # Clean up test files
    print(f"\nCleaning up test files...")
    for f in output_files:
        try:
            f.unlink()
            print(f"  ✓ Deleted {f.name}")
        except Exception as e:
            print(f"  ✗ Failed to delete {f.name}: {e}")

    print(f"\n{'='*60}")
    if all_passed:
        print("✅ Phase 1 Production Directory Test: PASSED")
        print(f"{'='*60}\n")
        print("All files saved to production outputs/ directory!")
        print("Phase 1 is working correctly with the real production directory.")
        return True
    else:
        print("❌ Phase 1 Production Directory Test: FAILED")
        print(f"{'='*60}\n")
        print("Some files were NOT saved to the correct directory.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test with production outputs/ directory")
    parser.add_argument(
        "--wan-dir",
        type=str,
        default="./Wan2GP",
        help="Path to Wan2GP directory"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help="Number of generations to test"
    )

    args = parser.parse_args()

    wan_dir = Path(args.wan_dir).resolve()
    if not wan_dir.exists():
        print(f"✗ Wan2GP directory not found: {wan_dir}")
        sys.exit(1)

    success = test_production_directory(str(wan_dir), args.num_generations)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
