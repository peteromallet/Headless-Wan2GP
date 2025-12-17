#!/usr/bin/env python3
"""
Complete File Organization Test Suite

Tests all task types to verify files save to correct directories.
This test can run WITHOUT a full worker/WGP setup by mocking the key functions.
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "source"))

from common_utils import prepare_output_path, prepare_output_path_with_upload, _get_task_type_directory


def test_wgp_post_processing():
    """Test that worker.py post-processing moves WGP outputs correctly."""
    print("\n" + "="*80)
    print("TEST: WGP Post-Processing Logic")
    print("="*80)

    # Import the move function from worker
    sys.path.insert(0, str(Path(__file__).parent))
    from worker import move_wgp_output_to_task_type_dir

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Test each WGP task type
        wgp_task_types = [
            "vace", "vace_21", "vace_22",
            "flux", "t2v", "t2v_22", "i2v", "i2v_22",
            "hunyuan", "ltxv", "inpaint_frames",
            "qwen_image_edit", "qwen_image_style",
            "image_inpaint", "annotated_image_edit",
            "generate_video"
        ]

        all_passed = True
        for task_type in wgp_task_types:
            task_id = f"{task_type}_test_001"

            # Simulate WGP generating to base directory
            base_output = base_dir / f"{task_id}_output.mp4"
            base_output.write_text("fake video")

            # Run post-processing
            new_path = move_wgp_output_to_task_type_dir(
                output_path=str(base_output),
                task_type=task_type,
                task_id=task_id,
                main_output_dir_base=base_dir
            )

            # Verify file was moved to task-type directory
            expected_dir = base_dir / task_type
            new_path_obj = Path(new_path)

            if new_path_obj.parent != expected_dir:
                print(f"❌ {task_type}: File NOT in correct directory")
                print(f"   Expected parent: {expected_dir}")
                print(f"   Actual parent: {new_path_obj.parent}")
                all_passed = False
                continue

            if not new_path_obj.exists():
                print(f"❌ {task_type}: File doesn't exist after move")
                print(f"   Expected: {new_path_obj}")
                all_passed = False
                continue

            if base_output.exists():
                print(f"❌ {task_type}: Original file still exists (should be moved)")
                all_passed = False
                continue

            print(f"✅ {task_type:25s} → {task_type}/{new_path_obj.name}")

        print()
        return all_passed


def test_specialized_handlers():
    """Test that specialized handlers use correct task_type directories."""
    print("\n" + "="*80)
    print("TEST: Specialized Handler Directories")
    print("="*80)

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Test each specialized handler task type
        handler_tests = [
            ("extract_frame", "frame_0.png"),
            ("rife_interpolate_images", "interpolated.mp4"),
            ("magic_edit", "edited.webp"),
            ("create_visualization", "visualization.mp4"),
            ("join_clips_segment", "joined.mp4"),
        ]

        all_passed = True
        for task_type, filename in handler_tests:
            task_id = f"{task_type}_test_001"

            # Test prepare_output_path_with_upload
            output_path, db_location = prepare_output_path_with_upload(
                task_id=task_id,
                filename=filename,
                main_output_dir_base=base_dir,
                task_type=task_type
            )

            # Verify path structure
            expected_dir = base_dir / task_type

            if output_path.parent != expected_dir:
                print(f"❌ {task_type}: Wrong directory")
                print(f"   Expected: {expected_dir}")
                print(f"   Got: {output_path.parent}")
                all_passed = False
                continue

            # Verify filename includes task_id
            if not output_path.name.startswith(task_id):
                print(f"❌ {task_type}: Filename doesn't start with task_id")
                print(f"   Expected prefix: {task_id}")
                print(f"   Got: {output_path.name}")
                all_passed = False
                continue

            print(f"✅ {task_type:25s} → {task_type}/{output_path.name}")

        print()
        return all_passed


def test_travel_tasks():
    """Test that travel tasks use correct directories."""
    print("\n" + "="*80)
    print("TEST: Travel Task Directories")
    print("="*80)

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Test travel_segment files
        segment_files = [
            ("guide video", "seg00_guide_123456_abc123.mp4"),
            ("mask video", "seg00_mask_123456_abc123.mp4"),
            ("saturated", "seg00_saturated.mp4"),
            ("brightened", "seg00_brightened.mp4"),
            ("colormatched", "seg00_colormatched.mp4"),
            ("banner", "seg00_banner.mp4"),
        ]

        all_passed = True
        task_id = "travel_test_001"

        print("\nTravel Segment Files:")
        for description, filename in segment_files:
            output_path, _ = prepare_output_path(
                task_id=task_id,
                filename=f"{task_id}_{filename}",
                main_output_dir_base=base_dir,
                task_type="travel_segment"
            )

            expected_dir = base_dir / "travel_segment"

            if output_path.parent != expected_dir:
                print(f"❌ {description}: Wrong directory")
                all_passed = False
                continue

            print(f"✅ {description:20s} → travel_segment/{output_path.name}")

        # Test travel_stitch files
        stitch_files = [
            ("single video", "single_video.mp4"),
            ("stitched intermediate", "stitched_intermediate.mp4"),
            ("final output", "output_123456_abc123.mp4"),
        ]

        print("\nTravel Stitch Files:")
        for description, filename in stitch_files:
            output_path, _ = prepare_output_path(
                task_id=task_id,
                filename=f"{task_id}_{filename}",
                main_output_dir_base=base_dir,
                task_type="travel_stitch"
            )

            expected_dir = base_dir / "travel_stitch"

            if output_path.parent != expected_dir:
                print(f"❌ {description}: Wrong directory")
                all_passed = False
                continue

            print(f"✅ {description:20s} → travel_stitch/{output_path.name}")

        print()
        return all_passed


def test_filename_conflict_resolution():
    """Test that filename conflicts are resolved with _1, _2, etc."""
    print("\n" + "="*80)
    print("TEST: Filename Conflict Resolution")
    print("="*80)

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        task_id = "conflict_test_001"
        filename = "output.mp4"

        # Create first file
        path1, _ = prepare_output_path(
            task_id=task_id,
            filename=filename,
            main_output_dir_base=base_dir,
            task_type="vace"
        )
        path1.parent.mkdir(parents=True, exist_ok=True)
        path1.write_text("file 1")

        # Create second file with same name
        path2, _ = prepare_output_path(
            task_id=task_id,
            filename=filename,
            main_output_dir_base=base_dir,
            task_type="vace"
        )

        # Should have _1 suffix
        if "_1" not in path2.stem:
            print(f"❌ Conflict resolution failed")
            print(f"   First file: {path1.name}")
            print(f"   Second file: {path2.name} (expected _1 suffix)")
            return False

        print(f"✅ First file:  {path1.name}")
        print(f"✅ Second file: {path2.name} (conflict resolved)")

        # Create third file
        path2.parent.mkdir(parents=True, exist_ok=True)
        path2.write_text("file 2")

        path3, _ = prepare_output_path(
            task_id=task_id,
            filename=filename,
            main_output_dir_base=base_dir,
            task_type="vace"
        )

        if "_2" not in path3.stem:
            print(f"❌ Second conflict resolution failed")
            print(f"   Third file: {path3.name} (expected _2 suffix)")
            return False

        print(f"✅ Third file:  {path3.name} (conflict resolved)")
        print()
        return True


def test_directory_creation():
    """Test that directories are created automatically."""
    print("\n" + "="*80)
    print("TEST: Automatic Directory Creation")
    print("="*80)

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Directory shouldn't exist yet
        vace_dir = base_dir / "vace"
        if vace_dir.exists():
            print("❌ Test setup error: directory already exists")
            return False

        # Call prepare_output_path
        output_path, _ = prepare_output_path(
            task_id="test_001",
            filename="output.mp4",
            main_output_dir_base=base_dir,
            task_type="vace"
        )

        # Directory should be created
        if not output_path.parent.exists():
            print(f"❌ Directory not created: {output_path.parent}")
            return False

        print(f"✅ Directory created: {output_path.parent}")
        print()
        return True


def test_backward_compatibility():
    """Test that task_type=None still works (backward compatibility)."""
    print("\n" + "="*80)
    print("TEST: Backward Compatibility (task_type=None)")
    print("="*80)

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Call without task_type (old behavior)
        output_path, _ = prepare_output_path(
            task_id="test_001",
            filename="output.mp4",
            main_output_dir_base=base_dir
            # NO task_type parameter
        )

        # Should save to base directory
        if output_path.parent != base_dir:
            print(f"❌ Backward compatibility broken")
            print(f"   Expected: {base_dir}")
            print(f"   Got: {output_path.parent}")
            return False

        print(f"✅ No task_type → saves to base directory")
        print(f"   Path: {output_path}")
        print()
        return True


def test_no_move_for_non_wgp_tasks():
    """Test that non-WGP tasks don't get post-processed."""
    print("\n" + "="*80)
    print("TEST: Non-WGP Tasks Skip Post-Processing")
    print("="*80)

    sys.path.insert(0, str(Path(__file__).parent))
    from worker import move_wgp_output_to_task_type_dir

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Test non-WGP task types
        non_wgp_tasks = [
            "extract_frame",
            "magic_edit",
            "join_clips_segment",
            "travel_orchestrator"
        ]

        all_passed = True
        for task_type in non_wgp_tasks:
            task_id = f"{task_type}_test_001"

            # Create file in base directory
            base_output = base_dir / f"{task_id}_output.mp4"
            base_output.write_text("fake file")
            original_path = str(base_output)

            # Run post-processing
            returned_path = move_wgp_output_to_task_type_dir(
                output_path=original_path,
                task_type=task_type,
                task_id=task_id,
                main_output_dir_base=base_dir
            )

            # Should return original path (not moved)
            if returned_path != original_path:
                print(f"❌ {task_type}: File was moved (shouldn't be)")
                all_passed = False
                continue

            # File should still exist in base directory
            if not base_output.exists():
                print(f"❌ {task_type}: File was deleted/moved")
                all_passed = False
                continue

            print(f"✅ {task_type:25s} → skipped (not WGP task)")

        print()
        return all_passed


def main():
    print("="*80)
    print("Complete File Organization Test Suite")
    print("="*80)
    print("\nTesting all task types and directory organization logic...")

    results = []

    # Run all tests
    results.append(("WGP Post-Processing", test_wgp_post_processing()))
    results.append(("Specialized Handlers", test_specialized_handlers()))
    results.append(("Travel Tasks", test_travel_tasks()))
    results.append(("Filename Conflicts", test_filename_conflict_resolution()))
    results.append(("Directory Creation", test_directory_creation()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Non-WGP Task Skipping", test_no_move_for_non_wgp_tasks()))

    # Summary
    print("="*80)
    print("Test Results Summary")
    print("="*80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        print("\nFile organization is working correctly:")
        print("• WGP outputs post-processed to task-type directories")
        print("• Specialized handlers save directly to task-type directories")
        print("• Travel tasks organize all intermediate + final files")
        print("• Filename conflicts resolved with _1, _2, etc.")
        print("• Directories created automatically")
        print("• Backward compatible (task_type=None → base directory)")
        print("• Non-WGP tasks skip post-processing")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
