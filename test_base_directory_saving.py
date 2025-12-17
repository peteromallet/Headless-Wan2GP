#!/usr/bin/env python3
"""
Test to verify all files save to base directory (no subdirectories).
"""

import sys
import tempfile
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "source"))

from common_utils import prepare_output_path, prepare_output_path_with_upload


def test_base_directory_saving():
    """Test that files save to base directory when task_type is not specified."""
    print("\n" + "="*60)
    print("Testing Base Directory Saving (No Subdirectories)")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Test 1: prepare_output_path without task_type
        print("\n1. Testing prepare_output_path() without task_type...")
        output_path, db_location = prepare_output_path(
            task_id="test_task_001",
            filename="test_output.mp4",
            main_output_dir_base=base_dir,
            # task_type=None (default)
        )

        # Verify output is in base directory (no subdirectories)
        # Note: Function adds task_id prefix for uniqueness
        expected_path = base_dir / "test_task_001_test_output.mp4"
        if output_path == expected_path:
            print(f"   ✓ File saves to base directory: {output_path.relative_to(base_dir)}")
        else:
            print(f"   ✗ FAILED: Expected {expected_path}, got {output_path}")
            return False

        # Test 2: prepare_output_path_with_upload without task_type
        print("\n2. Testing prepare_output_path_with_upload() without task_type...")
        output_path2, db_location2 = prepare_output_path_with_upload(
            task_id="test_task_002",
            filename="test_frame.png",
            main_output_dir_base=base_dir,
            # task_type=None (default)
        )

        expected_path2 = base_dir / "test_task_002_test_frame.png"
        if output_path2 == expected_path2:
            print(f"   ✓ File saves to base directory: {output_path2.relative_to(base_dir)}")
        else:
            print(f"   ✗ FAILED: Expected {expected_path2}, got {output_path2}")
            return False

        # Test 3: Verify custom_output_dir still works
        print("\n3. Testing custom_output_dir (should still work)...")
        custom_dir = base_dir / "custom" / "path"
        output_path3, db_location3 = prepare_output_path(
            task_id="test_task_003",
            filename="custom_output.mp4",
            main_output_dir_base=base_dir,
            custom_output_dir=custom_dir
        )

        if output_path3.parent == custom_dir:
            print(f"   ✓ custom_output_dir respected: {output_path3.relative_to(base_dir)}")
        else:
            print(f"   ✗ FAILED: Expected parent {custom_dir}, got {output_path3.parent}")
            return False

        # Test 4: Verify task_type parameter is truly optional
        print("\n4. Testing explicit task_type=None...")
        output_path4, db_location4 = prepare_output_path(
            task_id="test_task_004",
            filename="explicit_none.mp4",
            main_output_dir_base=base_dir,
            task_type=None  # Explicitly pass None
        )

        expected_path4 = base_dir / "test_task_004_explicit_none.mp4"
        if output_path4 == expected_path4:
            print(f"   ✓ Explicit task_type=None works: {output_path4.relative_to(base_dir)}")
        else:
            print(f"   ✗ FAILED: Expected {expected_path4}, got {output_path4}")
            return False

    print("\n" + "="*60)
    print("✅ All Base Directory Tests PASSED")
    print("="*60)
    print("\nVerified behavior:")
    print("• Files save directly to base directory")
    print("• No task-type subdirectories created")
    print("• custom_output_dir parameter still respected")
    print("• task_type parameter is optional (defaults to None)")

    return True


if __name__ == "__main__":
    success = test_base_directory_saving()
    sys.exit(0 if success else 1)
