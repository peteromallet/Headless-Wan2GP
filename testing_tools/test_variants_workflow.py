#!/usr/bin/env python3
"""
One-shot test script to demonstrate the variants.json workflow
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from base_tester import scan_experiments, process_experiment
from headless_model_management import HeadlessTaskQueue

def test_variants_workflow():
    """Test the variants.json workflow end-to-end"""
    print("🔍 Testing variants.json workflow end-to-end...\n")

    # Setup paths
    script_dir = Path(__file__).parent
    testing_dir = script_dir / "testing"
    experiments_path = testing_dir

    print(f"Experiments path: {experiments_path}")
    print(f"Scanning for experiments in: {list(experiments_path.iterdir())}")

    # Initialize task queue
    wan_root = script_dir / "Wan2GP"
    print(f"\n🔄 Initializing task queue with WAN root: {wan_root}")

    task_queue = HeadlessTaskQueue(wan_dir=str(wan_root), max_workers=1)
    task_queue.start()
    print("✅ Task queue initialized")

    try:
        # Scan for pending tests
        pending_tests = scan_experiments(experiments_path)
        print(f"\n✅ Found {len(pending_tests)} pending tests total")

        # Filter for variant tests
        variant_tests = [t for t in pending_tests if t.get('is_variant', False)]
        print(f"✅ Found {len(variant_tests)} variant-based tests")

        if not variant_tests:
            print("❌ No variant tests found. Make sure variant_test_demo has variants.json")
            return

        # Process first variant test
        for i, test_info in enumerate(variant_tests[:1], 1):  # Just test first one
            test_name = test_info["test_name"]
            generation_num = test_info["generation_num"]
            is_variant = test_info.get("is_variant", False)

            print(f"\n--- Processing Test {i}: {test_name} #{generation_num} ---")
            print(f"Is variant: {is_variant}")

            if 'input_set' in test_info and test_info['input_set']:
                input_set = test_info['input_set']
                print(f"Variant prompt: {input_set.get('prompt', 'Unknown')}")
                print(f"Variant length: {input_set.get('length', 'Unknown')} frames")
                print(f"Variant resolution: {input_set.get('resolution', 'Unknown')}")

            # Process the experiment
            success = process_experiment(task_queue, test_info)

            if success:
                print(f"✅ Test {test_name} #{generation_num} completed successfully")

                # Check output file
                test_folder = test_info["folder"]
                output_file = test_folder / f"{generation_num}_output.mp4"
                if output_file.exists():
                    file_size = output_file.stat().st_size
                    print(f"📁 Output file: {output_file.name} ({file_size:,} bytes)")
                else:
                    print("❌ Output file not found")
            else:
                print(f"❌ Test {test_name} #{generation_num} failed")

            break  # Only test one for demonstration

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n🔄 Shutting down task queue...")
        task_queue.stop(timeout=10.0)
        print("✅ Task queue shutdown complete")

    print("\n🎯 Variants workflow test completed!")

if __name__ == "__main__":
    test_variants_workflow()