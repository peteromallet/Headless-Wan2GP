#!/usr/bin/env python3
"""
Phase 1 Output Directory Validation Test

This script directly tests the HeadlessTaskQueue to verify that the main_output_dir
parameter is correctly flowing through to WGP's output configuration.

Usage:
    python test_phase1_output_dir.py --test-output-dir ./test_outputs
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from headless_model_management import HeadlessTaskQueue, GenerationTask

def test_output_directory_config(test_output_dir: str, wan_dir: str):
    """
    Test that output directory configuration is working correctly.

    This test:
    1. Creates a HeadlessTaskQueue with a custom output directory
    2. Initializes the orchestrator (which sets up WGP config)
    3. Checks that WGP's server_config has the correct paths
    """
    print(f"\n{'='*60}")
    print("Phase 1 Output Directory Validation Test")
    print(f"{'='*60}\n")

    # Create test output directory
    test_output_path = Path(test_output_dir).resolve()
    test_output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created test output directory: {test_output_path}")

    # Initialize HeadlessTaskQueue with custom output directory
    print(f"\n1. Initializing HeadlessTaskQueue with main_output_dir={test_output_path}")
    try:
        queue = HeadlessTaskQueue(
            wan_dir=wan_dir,
            max_workers=1,
            debug_mode=True,
            main_output_dir=str(test_output_path)
        )
        print("✓ HeadlessTaskQueue initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize HeadlessTaskQueue: {e}")
        return False

    # Force orchestrator initialization to trigger WGP setup
    print("\n2. Initializing WanOrchestrator (this sets up WGP configuration)...")
    try:
        queue._ensure_orchestrator()
        print("✓ WanOrchestrator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check WGP configuration
    print("\n3. Verifying WGP server_config output paths...")
    try:
        # Access wgp through the orchestrator's wgp module (already imported and configured)
        # Don't import wgp fresh - use the one that was already configured!
        import sys

        # The orchestrator already imported wgp and configured it
        # We can access it from sys.modules
        if 'wgp' in sys.modules:
            wgp = sys.modules['wgp']
            print("   ✓ Using wgp from sys.modules (already configured by orchestrator)")
        else:
            print("   ✗ wgp not found in sys.modules - orchestrator may not have imported it yet")
            return False

        # Check BOTH module-level variables AND dictionary
        # (wgp.py uses module-level variables, not the dictionary!)
        module_save_path = getattr(wgp, 'save_path', "NOT_SET")
        module_image_save_path = getattr(wgp, 'image_save_path', "NOT_SET")
        dict_save_path = wgp.server_config.get("save_path", "NOT_SET")
        dict_image_save_path = wgp.server_config.get("image_save_path", "NOT_SET")

        print(f"   wgp.save_path (module-level): {module_save_path}")
        print(f"   wgp.image_save_path (module-level): {module_image_save_path}")
        print(f"   wgp.server_config['save_path'] (dict): {dict_save_path}")
        print(f"   wgp.server_config['image_save_path'] (dict): {dict_image_save_path}")

        # Use module-level variables for validation (this is what wgp.py actually uses)
        save_path = module_save_path
        image_save_path = module_image_save_path

        # Validate paths match our test directory
        expected_path = str(test_output_path)

        success = True
        if save_path == expected_path:
            print(f"✓ save_path correctly configured")
        else:
            print(f"✗ save_path mismatch!")
            print(f"   Expected: {expected_path}")
            print(f"   Got:      {save_path}")
            success = False

        if image_save_path == expected_path:
            print(f"✓ image_save_path correctly configured")
        else:
            print(f"✗ image_save_path mismatch!")
            print(f"   Expected: {expected_path}")
            print(f"   Got:      {image_save_path}")
            success = False

        return success

    except Exception as e:
        print(f"✗ Failed to verify WGP configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Phase 1 output directory configuration")
    parser.add_argument(
        "--test-output-dir",
        type=str,
        default="./test_phase1_outputs",
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
    success = test_output_directory_config(args.test_output_dir, str(wan_dir))

    # Report results
    print(f"\n{'='*60}")
    if success:
        print("✅ Phase 1 Validation: PASSED")
        print("\nThe output directory configuration is working correctly!")
        print("Both save_path (videos) and image_save_path (images) are configured.")
        print("\nYou can proceed with production validation tomorrow.")
    else:
        print("❌ Phase 1 Validation: FAILED")
        print("\nThe output directory configuration has issues.")
        print("Please review the errors above before deploying to production.")
        sys.exit(1)
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
