#!/usr/bin/env python3
"""
Reset Test Script - Clean test outputs and results

This script removes all generated files from test experiments to allow fresh runs.
Removes variant outputs, logs, metadata, temp_variants directories, and batch-level files.
Supports both single test folders and batch-level experiments directories.
"""

import argparse
from pathlib import Path
import sys


def reset_test_folder(test_folder: Path):
    """Reset a single test folder by removing all generated files."""
    if not test_folder.exists():
        print(f"❌ Test folder does not exist: {test_folder}")
        return False

    if not test_folder.is_dir():
        print(f"❌ Not a directory: {test_folder}")
        return False

    # Files to remove
    patterns_to_remove = [
        "*_output.mp4",
        "*_metadata.json",
        "*_run_details.json",
        "*_run.log",
        "*_progress",
        "*_error.json",  # Numbered error files
        "error.json",
        "run.log",  # Single experiment log files
        "output.mp4",  # Text-to-video fallback output
        "metadata.json",  # Text-to-video fallback metadata
        "run_details.json"  # Text-to-video fallback run details
    ]

    files_removed = 0

    print(f"🧹 Cleaning test folder: {test_folder}")

    for pattern in patterns_to_remove:
        for file_path in test_folder.glob(pattern):
            try:
                file_path.unlink()
                print(f"  ✅ Removed: {file_path.name}")
                files_removed += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {file_path.name}: {e}")

    # Clean temp_variants directory if it exists
    temp_variants_dir = test_folder / "temp_variants"
    if temp_variants_dir.exists() and temp_variants_dir.is_dir():
        print(f"🧹 Cleaning temp_variants directory")
        temp_files_removed = 0
        for temp_file in temp_variants_dir.iterdir():
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                    print(f"  ✅ Removed: temp_variants/{temp_file.name}")
                    temp_files_removed += 1
                except Exception as e:
                    print(f"  ❌ Failed to remove temp_variants/{temp_file.name}: {e}")

        # Remove the empty temp_variants directory
        if temp_files_removed > 0:
            try:
                temp_variants_dir.rmdir()
                print(f"  ✅ Removed empty temp_variants directory")
            except Exception as e:
                print(f"  ❌ Failed to remove temp_variants directory: {e}")

        files_removed += temp_files_removed

    if files_removed == 0:
        print(f"  ℹ️  No files to remove (already clean)")
    else:
        print(f"  ✅ Removed {files_removed} file(s)")

    return True


def reset_experiments_folder(experiments_folder: Path):
    """Reset all test folders in an experiments directory."""
    if not experiments_folder.exists():
        print(f"❌ Experiments folder does not exist: {experiments_folder}")
        return False

    test_folders = [d for d in experiments_folder.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not test_folders:
        print(f"❌ No test folders found in: {experiments_folder}")
        return False

    print(f"🧪 Found {len(test_folders)} test folder(s) to reset:")
    for folder in test_folders:
        print(f"  - {folder.name}")

    print()

    success_count = 0
    for test_folder in test_folders:
        if reset_test_folder(test_folder):
            success_count += 1
        print()

    # Clean temp_assets folder
    assets_dir = experiments_folder / "temp_assets"
    assets_files_removed = 0
    if assets_dir.exists() and assets_dir.is_dir():
        print(f"🧹 Cleaning temp_assets folder")
        try:
            import shutil
            shutil.rmtree(assets_dir)
            print(f"  ✅ Removed temp_assets folder and all contents")
            assets_files_removed += 1
        except Exception as e:
            print(f"  ❌ Failed to remove temp_assets folder: {e}")

    # Also clean legacy assets folder if it exists
    legacy_assets_dir = experiments_folder / "assets"
    if legacy_assets_dir.exists() and legacy_assets_dir.is_dir():
        print(f"🧹 Cleaning legacy assets folder")
        try:
            import shutil
            shutil.rmtree(legacy_assets_dir)
            print(f"  ✅ Removed legacy assets folder and all contents")
            assets_files_removed += 1
        except Exception as e:
            print(f"  ❌ Failed to remove legacy assets folder: {e}")

    # Clean batch-level files
    batch_files_removed = 0
    batch_patterns = [
        "consolidated_logs.txt",
        "summaries.json"
    ]

    print(f"🧹 Cleaning batch-level files")
    for pattern in batch_patterns:
        for file_path in experiments_folder.glob(pattern):
            try:
                file_path.unlink()
                print(f"  ✅ Removed: {file_path.name}")
                batch_files_removed += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {file_path.name}: {e}")

    if batch_files_removed == 0:
        print(f"  ℹ️  No batch-level files to remove")
    else:
        print(f"  ✅ Removed {batch_files_removed} batch-level file(s)")

    if assets_files_removed == 0:
        print(f"  ℹ️  No temp_assets folder to remove")
    else:
        print(f"  ✅ Removed temp_assets folder")

    print()
    print(f"📊 Reset Summary: {success_count}/{len(test_folders)} test folders processed successfully")
    print(f"📊 Batch-level files: {batch_files_removed} file(s) removed")
    if assets_files_removed > 0:
        print(f"📊 Temp_assets folder: removed")
    return success_count == len(test_folders)


def main():
    parser = argparse.ArgumentParser(description='Reset test experiments by removing generated files')
    parser.add_argument('path', help='Path to test folder or experiments directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')

    args = parser.parse_args()

    target_path = Path(args.path).resolve()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be removed")
        print()

    print("🧪 RESET TEST - CLEAN EXPERIMENT OUTPUTS")
    print("=" * 50)
    print(f"Target path: {target_path}")
    print()

    if not target_path.exists():
        print(f"❌ Path does not exist: {target_path}")
        sys.exit(1)

    if target_path.is_file():
        print(f"❌ Path is a file, expected directory: {target_path}")
        sys.exit(1)

    # Check if this looks like a single test folder (contains settings.json)
    if (target_path / "settings.json").exists():
        print("📁 Detected single test folder")
        success = reset_test_folder(target_path)
    else:
        print("📁 Detected experiments directory")
        success = reset_experiments_folder(target_path)

    print()
    if success:
        print("✅ Reset completed successfully!")
        sys.exit(0)
    else:
        print("❌ Reset completed with errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()