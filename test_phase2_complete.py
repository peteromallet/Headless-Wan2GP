#!/usr/bin/env python3
"""
Comprehensive Phase 2 Test

Tests:
1. Single-level directory structure (outputs/{task_type}/)
2. Standardized filenames ({task_id}_{purpose})
3. Filename conflict handling (_1, _2, etc.)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "source"))

from common_utils import prepare_output_path, prepare_output_path_with_upload


def test_single_level_directories():
    """Test that task_type creates single-level directories."""
    print("\n1. Testing single-level directory structure...")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        tests = [
            ("vace", "vace"),
            ("travel_orchestrator", "travel_orchestrator"),
            ("extract_frame", "extract_frame"),
            ("magic_edit", "magic_edit"),
        ]

        all_passed = True
        for task_type, expected_subdir in tests:
            output_path, _ = prepare_output_path(
                task_id=f"test_{task_type}",
                filename="output.mp4",
                main_output_dir_base=base_dir,
                task_type=task_type
            )

            # Check that it's directly under base_dir/{task_type}/
            expected_path = base_dir / expected_subdir / f"test_{task_type}_output.mp4"
            if output_path == expected_path:
                print(f"   ✓ {task_type} → {expected_subdir}/")
            else:
                print(f"   ✗ {task_type}: Expected {expected_path}, got {output_path}")
                all_passed = False

        return all_passed


def test_standardized_filenames():
    """Test that filenames follow {task_id}_{purpose} pattern."""
    print("\n2. Testing standardized filename format...")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        tests = [
            ("frame_0.png", "task_001_frame_0.png"),  # Should add task_id prefix
            ("task_002_interpolated.mp4", "task_002_interpolated.mp4"),  # Already has task_id
            ("output.mp4", "task_003_output.mp4"),  # Should add task_id prefix
        ]

        all_passed = True
        for idx, (input_filename, expected_filename) in enumerate(tests, 1):
            task_id = f"task_{idx:03d}"
            output_path, _ = prepare_output_path(
                task_id=task_id,
                filename=input_filename,
                main_output_dir_base=base_dir,
                task_type="test"
            )

            if output_path.name == expected_filename:
                print(f"   ✓ {input_filename} → {expected_filename}")
            else:
                print(f"   ✗ {input_filename}: Expected {expected_filename}, got {output_path.name}")
                all_passed = False

        return all_passed


def test_filename_conflict_handling():
    """Test that filename conflicts are resolved with _1, _2, etc."""
    print("\n3. Testing filename conflict handling...")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        task_type_dir = base_dir / "test_task"
        task_type_dir.mkdir(parents=True, exist_ok=True)

        # Create a test file to cause conflict
        existing_file = task_type_dir / "task_001_output.mp4"
        existing_file.touch()

        # Try to create a file with the same name
        output_path1, _ = prepare_output_path(
            task_id="task_001",
            filename="output.mp4",
            main_output_dir_base=base_dir,
            task_type="test_task"
        )

        if output_path1.name == "task_001_output_1.mp4":
            print(f"   ✓ First conflict: task_001_output.mp4 → task_001_output_1.mp4")
        else:
            print(f"   ✗ First conflict: Expected task_001_output_1.mp4, got {output_path1.name}")
            return False

        # Create the conflicting file
        output_path1.touch()

        # Try again - should get _2
        output_path2, _ = prepare_output_path(
            task_id="task_001",
            filename="output.mp4",
            main_output_dir_base=base_dir,
            task_type="test_task"
        )

        if output_path2.name == "task_001_output_2.mp4":
            print(f"   ✓ Second conflict: task_001_output.mp4 → task_001_output_2.mp4")
        else:
            print(f"   ✗ Second conflict: Expected task_001_output_2.mp4, got {output_path2.name}")
            return False

        return True


def test_task_type_as_directory():
    """Test that _get_task_type_directory returns task_type directly."""
    print("\n4. Testing task type directory mapping...")

    from common_utils import _get_task_type_directory

    tests = [
        ("vace", "vace"),
        ("travel_orchestrator", "travel_orchestrator"),
        ("unknown_task", "unknown_task"),  # Unknown types use their name as directory
        (None, "misc"),  # Only None/empty returns 'misc'
        ("", "misc"),
    ]

    all_passed = True
    for task_type, expected_dir in tests:
        actual_dir = _get_task_type_directory(task_type)
        if actual_dir == expected_dir:
            print(f"   ✓ {task_type} → {expected_dir}")
        else:
            print(f"   ✗ {task_type}: Expected {expected_dir}, got {actual_dir}")
            all_passed = False

    return all_passed


def test_prepare_output_path_with_upload():
    """Test that prepare_output_path_with_upload forwards task_type correctly."""
    print("\n5. Testing prepare_output_path_with_upload...")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        output_path, _ = prepare_output_path_with_upload(
            task_id="test_upload",
            filename="output.mp4",
            main_output_dir_base=base_dir,
            task_type="upload_test"
        )

        expected_path = base_dir / "upload_test" / "test_upload_output.mp4"
        if output_path == expected_path:
            print(f"   ✓ task_type forwarded correctly: {output_path.relative_to(base_dir)}")
            return True
        else:
            print(f"   ✗ Expected {expected_path}, got {output_path}")
            return False


def main():
    print("="*70)
    print("Phase 2 Complete Implementation Test")
    print("="*70)

    results = []

    # Run all tests
    results.append(("Single-level directories", test_single_level_directories()))
    results.append(("Standardized filenames", test_standardized_filenames()))
    results.append(("Filename conflict handling", test_filename_conflict_handling()))
    results.append(("Task type directory mapping", test_task_type_as_directory()))
    results.append(("prepare_output_path_with_upload", test_prepare_output_path_with_upload()))

    # Summary
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✅ Phase 2 Complete Implementation: ALL TESTS PASSED")
        print("\nVerified:")
        print("• outputs/{task_type}/ directory structure")
        print("• {task_id}_{purpose} filename pattern")
        print("• Filename conflict resolution (_1, _2, etc.)")
        print("• prepare_output_path_with_upload forwards task_type")
        return 0
    else:
        print("\n❌ Phase 2 Complete Implementation: SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
