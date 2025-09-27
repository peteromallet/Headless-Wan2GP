#!/usr/bin/env python3
"""Base tester script for continuous experiment processing.
Processes batch-level variants.json across multiple experiment configurations,
with fallback to single text-to-video generation when no variants are defined.
"""

import json
import sys
import argparse
import time
import os
import logging
import io
from pathlib import Path
from datetime import datetime

# Add our project to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.common_utils import generate_unique_task_id
from headless_wgp import WanOrchestrator
from testing_tools.variant_utils import load_variants_json, process_variant


def update_consolidated_log(experiments_path):
    """Update the consolidated log with all actual log content concatenated"""
    experiments_dir = Path(experiments_path)
    consolidated_file = experiments_dir / "consolidated_logs.txt"

    # Find all log files and sort by modification time
    log_files = []
    for pattern in ["**/*_run.log", "**/run.log"]:
        for file_path in experiments_dir.glob(pattern):
            if file_path.exists():
                log_files.append((file_path.stat().st_mtime, file_path))

    # Sort by modification time (oldest first)
    log_files.sort(key=lambda x: x[0])

    # Write all logs concatenated
    with open(consolidated_file, 'w') as consolidated:
        consolidated.write(f"CONSOLIDATED LOGS - {datetime.now().isoformat()}\n")
        consolidated.write("=" * 80 + "\n\n")

        for _, log_file in log_files:
            try:
                relative_path = str(log_file.relative_to(experiments_dir))
                consolidated.write(f"\n{'='*80}\n")
                consolidated.write(f"LOG FILE: {relative_path}\n")
                consolidated.write(f"MODIFIED: {datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()}\n")
                consolidated.write(f"{'='*80}\n\n")

                with open(log_file, 'r') as f:
                    consolidated.write(f.read())

                consolidated.write(f"\n\n{'='*80}\n")
                consolidated.write(f"END OF {relative_path}\n")
                consolidated.write(f"{'='*80}\n\n")

            except Exception as e:
                consolidated.write(f"ERROR reading {log_file}: {e}\n\n")

    print(f"📊 Updated consolidated logs: {len(log_files)} log files concatenated")




def scan_experiments(experiments_path, target_variant=None):
    """Scan experiments folder for test folders with settings.json and find pending generations.

    Args:
        experiments_path: Path to experiments directory
        target_variant: If specified, only process this variant number (1-based)
    """
    experiments_dir = Path(experiments_path)

    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_path}")

    pending_tests = []

    # Check for batch-level variants.json
    batch_variants_file = experiments_dir / "variants.json"
    batch_variants_data = None

    if batch_variants_file.exists():
        print(f"✅ Found batch-level variants.json in {experiments_dir.name}")
        batch_variants_data = load_variants_json(batch_variants_file)

    if batch_variants_data and 'variants' in batch_variants_data:
        print(f"📋 Batch has {len(batch_variants_data['variants'])} variants")

        # Create central temp_assets folder and clean it if it exists
        assets_dir = experiments_dir / "temp_assets"
        if assets_dir.exists():
            print(f"🧹 Cleaning existing temp_assets folder")
            import shutil
            shutil.rmtree(assets_dir)
        assets_dir.mkdir(exist_ok=True)
        print(f"📁 Created central temp_assets folder: {assets_dir}")

        # Process all variants once and store in temp_assets folder
        if target_variant:
            print(f"🔄 Processing variant {target_variant} only into temp_assets folder...")
            variant_range = [target_variant - 1]  # Convert to 0-based index
        else:
            print(f"🔄 Processing all variants into temp_assets folder...")
            variant_range = range(len(batch_variants_data['variants']))

        variant_assets = {}
        for idx in variant_range:
            i = idx + 1  # Convert back to 1-based for display
            variant_data = batch_variants_data['variants'][idx]
            print(f"  Processing variant {i}/{len(batch_variants_data['variants'])}: {variant_data['prompt'][:50]}...")

            processed_files = process_variant(variant_data, i, experiments_dir, assets_dir)

            if processed_files:
                variant_assets[i] = {
                    'video': Path(processed_files['video']),
                    'mask': Path(processed_files['mask']),
                    'prompt_file': Path(processed_files['prompt_file']),
                    'resolution': processed_files['resolution'],
                    'length': processed_files['length'],
                    'prompt': processed_files['prompt']
                }
                print(f"    ✅ Variant {i} assets created")
            else:
                print(f"    ❌ Variant {i} processing failed")

        # Process variants across all experiment folders using shared assets
        # Sort folders by modification time (most recent first)
        test_folders = sorted(
            [f for f in experiments_dir.iterdir() if f.is_dir() and (f / "settings.json").exists()],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        for test_folder in test_folders:
                settings_file = test_folder / "settings.json"
                print(f"📁 Found experiment: {test_folder.name}")

                # Process each variant using shared assets
                if target_variant:
                    # Only process the specified variant
                    variant_keys = [target_variant] if target_variant in variant_assets else []
                else:
                    variant_keys = variant_assets.keys()

                for i in variant_keys:
                    output_file = test_folder / f"{test_folder.name}_{i}_output.mp4"
                    error_file = test_folder / f"{i}_error.json"

                    if not output_file.exists() and not error_file.exists():
                        variant_data = batch_variants_data['variants'][i-1]  # variants list is 0-indexed
                        print(f"  Variant {i}: {variant_data['prompt'][:50]}...")
                        print(f"    ✅ Using shared temp_assets")

                        # Use shared assets from temp_assets folder
                        pending_tests.append({
                            "folder": test_folder,
                            "settings_file": settings_file,
                            "test_name": test_folder.name,
                            "generation_num": i,
                            "input_set": variant_assets[i],
                            "is_variant": True
                        })
                    else:
                        if output_file.exists():
                            print(f"  Variant {i}: Already completed")
                        if error_file.exists():
                            print(f"  Variant {i}: Has error - skipping")
    else:
        # Fallback to text-to-video mode when no batch-level variants.json exists
        # Sort folders by modification time (most recent first)
        test_folders = sorted(
            [f for f in experiments_dir.iterdir() if f.is_dir()],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        for test_folder in test_folders:
                settings_file = test_folder / "settings.json"

                if settings_file.exists():
                    # Check for single output.mp4 (text-to-video mode)
                    output_file = test_folder / f"{test_folder.name}_output.mp4"
                    error_file = test_folder / "error.json"

                    if not output_file.exists() and not error_file.exists():
                        print(f"Found pending test (text-to-video): {test_folder.name}")
                        pending_tests.append({
                            "folder": test_folder,
                            "settings_file": settings_file,
                            "test_name": test_folder.name,
                            "generation_num": None,
                            "input_set": {},
                            "is_variant": False
                        })
                    elif error_file.exists():
                        print(f"Skipping test {test_folder.name} (has error.json)")

    return pending_tests


def load_settings(settings_file):
    """Load settings.json configuration."""
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        print(f"Error loading settings from {settings_file}: {e}")
        return None


class LogCapture:
    """Context manager to capture logs and console output."""

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_stream = io.StringIO()
        self.original_stdout = None
        self.original_stderr = None
        self.logger = None
        self.handler = None

    def __enter__(self):
        # Setup file handler for logging
        self.logger = logging.getLogger()
        self.handler = logging.FileHandler(self.log_file_path, mode='w')
        self.handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

        # Capture stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create a tee that writes to both original and log file
        class TeeWriter:
            def __init__(self, original, log_file):
                self.original = original
                self.log_file = log_file

            def write(self, text):
                self.original.write(text)
                self.log_file.write(text)
                self.log_file.flush()

            def flush(self):
                self.original.flush()
                self.log_file.flush()

        self.log_file = open(self.log_file_path, 'a')
        sys.stdout = TeeWriter(self.original_stdout, self.log_file)
        sys.stderr = TeeWriter(self.original_stderr, self.log_file)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Close log file
        if hasattr(self, 'log_file'):
            self.log_file.close()

        # Remove handler
        if self.handler and self.logger:
            self.logger.removeHandler(self.handler)
            self.handler.close()


def process_experiment(task_queue, test_info, output_base_dir=None):
    """Process a single experiment based on its settings.json."""
    test_folder = test_info["folder"]
    settings_file = test_info["settings_file"]
    test_name = test_info["test_name"]
    generation_num = test_info.get("generation_num")
    input_set = test_info.get("input_set", {})
    is_variant = test_info.get("is_variant", False)

    # Setup log file for this generation
    if generation_num is not None:
        log_file = test_folder / f"{generation_num}_run.log"
        display_name = f"{test_name} #{generation_num}"
    else:
        log_file = test_folder / "run.log"
        display_name = test_name

    with LogCapture(log_file):
        print(f"Processing generation: {display_name}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Log file: {log_file}")
        print("-" * 60)

        # Load settings
        settings = load_settings(settings_file)
        if not settings:
            print("❌ Failed to load settings")
            return False

        print(f"Model type: {settings.get('model_type', 'Unknown')}")
        if is_variant:
            print(f"Variant mode: Using generated guide/mask from variants.json")
            print(f"Variant resolution: {input_set.get('resolution', 'Unknown')}")
            print(f"Variant length: {input_set.get('length', 'Unknown')} frames")
            print(f"Variant prompt: {input_set.get('prompt', 'Unknown')}")
        print(f"Settings: {json.dumps(settings, indent=2)}")

        # Load prompt from variant if available
        if is_variant and 'prompt' in input_set:
            # For variants, use the prompt directly from the variant data
            settings['prompt'] = input_set['prompt']
            print(f"Using variant prompt: {settings['prompt']}")

            # Also update video_length from variant if not set in settings
            if 'video_length' not in settings and 'length' in input_set:
                settings['video_length'] = input_set['length']
                print(f"Using variant length: {settings['video_length']} frames")
        elif 'prompt_file' in input_set:
            try:
                prompt_file_path = input_set['prompt_file']
                with open(prompt_file_path, 'r') as f:
                    if prompt_file_path.suffix == '.txt':
                        # Handle plain text files
                        settings['prompt'] = f.read().strip()
                        print(f"Loaded prompt from {prompt_file_path.name}: {settings['prompt']}")
                    else:
                        # Handle JSON files
                        prompt_data = json.load(f)
                        if isinstance(prompt_data, dict) and 'prompt' in prompt_data:
                            settings['prompt'] = prompt_data['prompt']
                            print(f"Loaded prompt from {prompt_file_path.name}: {settings['prompt']}")
                        elif isinstance(prompt_data, str):
                            settings['prompt'] = prompt_data
                            print(f"Loaded prompt from {prompt_file_path.name}: {settings['prompt']}")
            except Exception as e:
                print(f"⚠️ Could not load prompt file: {e}")

        print()

        try:
            # Use task queue for generation (like worker.py does)
            from headless_model_management import GenerationTask

            start_time = time.time()
            print(f"Starting generation at: {datetime.now().isoformat()}")

            # Determine model from settings
            model_type = settings.get("model_type")
            temp_model_file = None
            temp_model_name = None

            if not model_type and "model" in settings:
                # Create temporary model file and register with WGP
                temp_model_name = "base_tester_model"
                temp_model_file = Path(__file__).parent / "Wan2GP" / "defaults" / f"{temp_model_name}.json"

                model_file_content = {"model": settings["model"]}
                with open(temp_model_file, 'w') as f:
                    json.dump(model_file_content, f, indent=2)

                print(f"Created temporary model file: {temp_model_file}")

                # Register with WGP so the task queue can find it
                try:
                    import wgp
                    temp_model_def = settings["model"].copy()
                    temp_model_def["vace_class"] = True
                    wgp.models_def[temp_model_name] = temp_model_def
                    print(f"Registered temporary model with WGP: {temp_model_name}")
                except Exception as e:
                    print(f"⚠️ Could not register model with WGP: {e}")

                model_type = temp_model_name
            elif model_type:
                print(f"Using model type: {model_type}")
            else:
                model_type = "vace_14B_cocktail_2_2"  # Default model
                print(f"Using default model type: {model_type}")

            # Extract generation parameters
            generation_params = {k: v for k, v in settings.items() if k not in ["model_type", "model"]}

            # Add video inputs if available
            if 'video' in input_set:
                generation_params['video_guide'] = str(input_set['video'])
                print(f"Video guide: {input_set['video']}")

            if 'mask' in input_set:
                generation_params['video_mask'] = str(input_set['mask'])
                print(f"Video mask: {input_set['mask']}")

            print(f"Generation parameters: {json.dumps(generation_params, indent=2)}")

            # Create and submit task to queue
            task_id = f"test_{test_name}_{generation_num or 'single'}_{int(time.time())}"
            prompt = generation_params.pop('prompt', '')

            generation_task = GenerationTask(
                id=task_id,
                model=model_type,
                prompt=prompt,
                parameters=generation_params
            )

            print("Submitting to persistent task queue...")
            submitted_task_id = task_queue.submit_task(generation_task)

            # Wait for completion
            max_wait_time = 1800  # 30 minutes
            wait_interval = 2
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                status = task_queue.get_task_status(task_id)
                if status is None:
                    print("❌ Task status became None")
                    return False

                if status.status == "completed":
                    result_path = status.result_path
                    break
                elif status.status == "failed":
                    print(f"❌ Generation failed: {status.error_message}")
                    return False

                time.sleep(wait_interval)
                elapsed_time += wait_interval
            else:
                print(f"❌ Generation timeout after {max_wait_time}s")
                return False

            generation_time = time.time() - start_time
            print(f"Generation completed in {generation_time:.1f}s")

            if result_path:
                print(f"Result path: {result_path}")

                # Move result to experiment folder with appropriate name
                wan_root = Path(__file__).parent / "Wan2GP"
                abs_result_path = (wan_root / result_path).resolve()

                if generation_num is not None:
                    output_path = test_folder / f"{test_name}_{generation_num}_output.mp4"
                else:
                    output_path = test_folder / f"{test_name}_output.mp4"

                print(f"Moving from: {abs_result_path}")
                print(f"Moving to: {output_path}")

                if abs_result_path.exists():
                    import shutil
                    shutil.move(str(abs_result_path), str(output_path))

                    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                    print(f"✅ File moved successfully - Size: {file_size:.1f}MB")

                    # Save generation metadata
                    metadata = {
                        "test_name": display_name,
                        "generation_num": generation_num,
                        "generation_time": generation_time,
                        "file_size_mb": file_size,
                        "timestamp": datetime.now().isoformat(),
                        "settings": settings,
                        "input_set": {k: str(v) for k, v in input_set.items()},
                        "status": "success",
                        "log_file": str(log_file)
                    }

                    if generation_num is not None:
                        metadata_file = test_folder / f"{generation_num}_metadata.json"
                    else:
                        metadata_file = test_folder / "metadata.json"

                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    # Save run details (timing focused)
                    run_details = {
                        "test_name": display_name,
                        "generation_num": generation_num,
                        "generation_time_seconds": generation_time,
                        "generation_time_minutes": generation_time / 60,
                        "start_timestamp": datetime.fromtimestamp(start_time).isoformat(),
                        "end_timestamp": datetime.now().isoformat(),
                        "status": "completed"
                    }

                    if generation_num is not None:
                        run_details_file = test_folder / f"{generation_num}_run_details.json"
                    else:
                        run_details_file = test_folder / "run_details.json"

                    with open(run_details_file, 'w') as f:
                        json.dump(run_details, f, indent=2)

                    print(f"✅ Metadata saved to: {metadata_file}")
                    print(f"✅ Run details saved to: {run_details_file}")

                    # Keep temporary model file for reuse across tests
                    print(f"📦 Keeping model file for reuse: {temp_model_file}")

                    print("=" * 60)
                    print("GENERATION COMPLETED SUCCESSFULLY")

                    # Update consolidated log after successful generation
                    try:
                        update_consolidated_log(test_folder.parent)
                    except Exception as e:
                        print(f"⚠️ Could not update consolidated log: {e}")

                    # Keep model file and registration for reuse
                    print(f"📦 Keeping model file and registration for reuse")

                    return True
                else:
                    print(f"❌ Generation completed but output file not found: {abs_result_path}")
                    # Keep temporary model file for reuse across tests
                    print(f"📦 Keeping model file for reuse: {temp_model_file}")
                    return False
            else:
                print("❌ Generation failed - no output generated")
                # Keep temporary model file for reuse
                print(f"📦 Keeping model file for reuse")
                return False

        except Exception as e:
            print(f"❌ Error processing experiment: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()

            # Save error metadata
            error_metadata = {
                "test_name": display_name,
                "generation_num": generation_num,
                "timestamp": datetime.now().isoformat(),
                "settings": settings,
                "input_set": {k: str(v) for k, v in input_set.items()},
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "log_file": str(log_file)
            }

            if generation_num is not None:
                error_file = test_folder / f"{generation_num}_error.json"
            else:
                error_file = test_folder / "error.json"

            with open(error_file, 'w') as f:
                json.dump(error_metadata, f, indent=2)

            # Save run details for failed run
            failed_time = time.time() - start_time if 'start_time' in locals() else 0
            run_details = {
                "test_name": display_name,
                "generation_num": generation_num,
                "generation_time_seconds": failed_time,
                "generation_time_minutes": failed_time / 60,
                "start_timestamp": datetime.fromtimestamp(start_time).isoformat() if 'start_time' in locals() else None,
                "end_timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__
            }

            if generation_num is not None:
                run_details_file = test_folder / f"{generation_num}_run_details.json"
            else:
                run_details_file = test_folder / "run_details.json"

            with open(run_details_file, 'w') as f:
                json.dump(run_details, f, indent=2)

            print(f"❌ Error metadata saved to: {error_file}")
            print(f"❌ Run details saved to: {run_details_file}")

            # Keep temporary model file for reuse
            print(f"📦 Keeping model file for reuse")

            print("=" * 60)
            print("GENERATION FAILED")

            # Update consolidated log after failed generation
            try:
                update_consolidated_log(test_folder.parent)
            except Exception as e:
                print(f"⚠️ Could not update consolidated log: {e}")

            # Keep temporary model file and registration for reuse
            print(f"📦 Keeping model file and registration for reuse")

            return False


def run_base_tester(experiments_subfolder, passthrough=False, target_variant=None, direct_input=None):
    """Run base tester on specified experiments subfolder.

    Args:
        experiments_subfolder: Name of subfolder in testing/ to process
        passthrough: Whether to use passthrough mode
        target_variant: If specified, only process this variant number (1-based)
        direct_input: Dict with video_guide, video_mask, and prompt for direct input mode
    """

    # Construct testing path
    script_dir = Path(__file__).parent
    testing_dir = script_dir / "testing"
    experiments_path = testing_dir / experiments_subfolder

    # Create experiments directory if it doesn't exist
    experiments_path.mkdir(parents=True, exist_ok=True)

    print("🧪 BASE TESTER - CONTINUOUS EXPERIMENT PROCESSOR")
    print("=" * 50)
    print(f"Experiments path: {experiments_path}")
    if direct_input:
        print("🎬 Direct video/mask input mode")
        print(f"Video guide: {direct_input['video_guide']}")
        print(f"Video mask: {direct_input['video_mask']}")
        print(f"Prompt: {direct_input['prompt']}")
    elif target_variant:
        print(f"🎯 Single variant mode: Processing variant {target_variant} only")
    else:
        print("🔄 Polling mode: Will continuously check for new experiments...")
    print("   Press Ctrl+C to stop")
    print()

    # Initialize task queue once (like worker.py does for persistent model loading)
    wan_root = Path(__file__).parent / "Wan2GP"

    # Import HeadlessTaskQueue for persistent model management
    from headless_model_management import HeadlessTaskQueue

    print("🔄 Initializing persistent model queue...")
    task_queue = HeadlessTaskQueue(wan_dir=str(wan_root), max_workers=1)
    task_queue.start()
    print("✅ Task queue initialized - models will stay loaded between tests")

    batch_number = 1

    try:
        while True:
            print(f"\n🔍 SCAN #{batch_number} - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 40)

            # Scan for pending tests
            try:
                pending_tests = scan_experiments(experiments_path, target_variant)
            except FileNotFoundError as e:
                print(f"❌ {e}")
                time.sleep(30)
                continue

            if not pending_tests:
                print("✅ No pending tests found - waiting for new experiments...")
                print("   (Add experiments and they will be auto-processed)")
                time.sleep(30)  # Wait 30 seconds before next scan
                batch_number += 1
                continue

            # Process only the first (most recent) experiment, then rescan
            test_info = pending_tests[0]
            print(f"Found {len(pending_tests)} pending test(s)")
            print(f"--- Processing most recent: {test_info['test_name']} ---")

            success = process_experiment(task_queue, test_info)

            # Create individual result record
            results = {
                "successful": 1 if success else 0,
                "failed": 0 if success else 1,
                "tests": [{
                    "name": test_info["test_name"],
                    "status": "success" if success else "failed"
                }]
            }

            # Save summary for this single test
            _save_batch_summary(experiments_path, experiments_subfolder, [test_info], results)

            batch_number += 1
            print(f"\n⏱️ Test complete - rescanning immediately for new experiments...")
            # No sleep - immediately rescan to pick up any new experiments

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        # Cleanup task queue
        try:
            print("🔄 Shutting down task queue...")
            task_queue.stop(timeout=30.0)
            print("✅ Task queue shutdown complete")
        except Exception as e:
            print(f"⚠️ Error during task queue shutdown: {e}")

        print("🏁 BASE TESTER STOPPED")

def _save_batch_summary(experiments_path, experiments_subfolder, pending_tests, results):
    """Save summary results for a batch."""

    # Save summary results to top-level summaries.json
    summary_file = experiments_path / "summaries.json"

    # Load existing summaries or create new list
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            all_summaries = json.load(f)
    else:
        all_summaries = []

    # Append new summary
    new_summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments_subfolder": experiments_subfolder,
        "total_pending": len(pending_tests),
        "successful": results["successful"],
        "failed": results["failed"],
        "tests": results["tests"]
    }
    all_summaries.append(new_summary)

    # Write back to file
    with open(summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    # Final consolidated log update
    try:
        update_consolidated_log(experiments_path)
    except Exception as e:
        print(f"⚠️ Could not update final consolidated log: {e}")

    print("🏁 BATCH COMPLETE")
    print(f"✅ Successful: {results['successful']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"Summary saved: {summary_file}")
    print(f"Consolidated log: {experiments_path}/consolidated_logs.txt")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Base tester for processing experiments")
    parser.add_argument("--experiments", required=True,
                       help="Subfolder name within /testing directory to process")
    parser.add_argument("--no-passthrough", action="store_true",
                       help="Disable passthrough mode (enable parameter processing)")
    parser.add_argument("--variant", type=int,
                       help="Process only the specified variant number (1-based)")
    parser.add_argument("--video-guide", type=str,
                       help="Path to video guide file (for direct video/mask input)")
    parser.add_argument("--video-mask", type=str,
                       help="Path to video mask file (for direct video/mask input)")
    parser.add_argument("--prompt", type=str,
                       help="Prompt text (for direct video/mask input)")
    # Default to passthrough mode to preserve JSON configs exactly as-is

    args = parser.parse_args()

    # Check if direct video/mask mode is requested
    direct_input = None
    if args.video_guide or args.video_mask or args.prompt:
        if not (args.video_guide and args.video_mask and args.prompt):
            print("❌ For direct video/mask input, you must provide --video-guide, --video-mask, and --prompt")
            sys.exit(1)

        direct_input = {
            'video_guide': args.video_guide,
            'video_mask': args.video_mask,
            'prompt': args.prompt
        }

    run_base_tester(args.experiments, passthrough=not args.no_passthrough, target_variant=args.variant, direct_input=direct_input)


if __name__ == "__main__":
    main()