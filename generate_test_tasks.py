#!/usr/bin/env python3
"""
Generate a set of end-to-end headless-server tasks that exercise the
most important travel-segment scenarios:

1. single image, no continue-video
2. 3 images, no continue-video
3. continue-video + single image
4. continue-video + 2 images

For each test a JSON payload (compatible with add_task.py) is written
under   tests/<test_name>/<test_name>_task.json  and the referenced
input files (images / video) are copied into that same directory with a
clean, descriptive name so the folder is self-contained and easy to
inspect.

After writing the JSON the script *optionally* enqueues the task by
invoking   python add_task.py --type travel_orchestrator --params
@<json_file>

Set  --enqueue   to actually perform the enqueue.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import sqlite3
from typing import Dict, List, Tuple
import time

# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------
PROJECT_ID = "test_suite"
BASE_PROMPT = "Car driving through a city, sky morphing"
NEG_PROMPT = "chaotic"
MODEL_NAME = "vace_14B"
RESOLUTION = "500x500"
FPS = 16
SEGMENT_FRAMES_DEFAULT = 81  # will be quantised downstream (4n+1)
FRAME_OVERLAP_DEFAULT = 12
SEED_BASE = 11111
OUTPUT_DIR_DEFAULT = "./outputs"

SAMPLES_DIR = Path("samples")
ASSET_IMAGES = [
    SAMPLES_DIR / "1.png",
    SAMPLES_DIR / "2.png",
    SAMPLES_DIR / "3.png",
]
ASSET_VIDEO = SAMPLES_DIR / "test.mp4"

TESTS_ROOT = Path("tests")

# ---------------------------------------------------------------------

def make_orchestrator_payload(*, run_id: str,
                              images: list[Path],
                              continue_video: Path | None,
                              num_segments: int) -> dict:
    """Create the orchestrator_details dict used by headless server."""
    payload: dict = {
        "run_id": run_id,
        "input_image_paths_resolved": [str(p) for p in images],
        "parsed_resolution_wh": RESOLUTION,
        "model_name": MODEL_NAME,
        "use_causvid_lora": True,
        "num_new_segments_to_generate": num_segments,
        "base_prompts_expanded": [BASE_PROMPT] * num_segments,
        "negative_prompts_expanded": [NEG_PROMPT] * num_segments,
        "segment_frames_expanded": [SEGMENT_FRAMES_DEFAULT] * num_segments,
        "frame_overlap_expanded": [FRAME_OVERLAP_DEFAULT] * num_segments,
        "fps_helpers": FPS,
        "fade_in_params_json_str": json.dumps({
            "low_point": 0.0, "high_point": 1.0,
            "curve_type": "ease_in_out", "duration_factor": 0.0
        }),
        "fade_out_params_json_str": json.dumps({
            "low_point": 0.0, "high_point": 1.0,
            "curve_type": "ease_in_out", "duration_factor": 0.0
        }),
        "seed_base": SEED_BASE,
        "main_output_dir_for_run": OUTPUT_DIR_DEFAULT,
    }
    if continue_video is not None:
        payload["continue_from_video_resolved_path"] = str(continue_video)
    return payload


def write_test_case(name: str,
                    images: list[Path],
                    continue_video: Path | None,
                    num_segments: int,
                    enqueue: bool) -> None:
    """Materialise a test folder with JSON and asset copies."""
    test_dir = TESTS_ROOT / name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy / link assets with descriptive names and record new paths
    copied_images: list[Path] = []
    for idx, img in enumerate(images):
        dst = test_dir / f"{name}_image{idx+1}{img.suffix}"
        if dst.exists():
            dst.unlink()
        shutil.copy2(img, dst)
        copied_images.append(dst)
    if continue_video is not None:
        dst_vid = test_dir / f"{name}_continue_video{continue_video.suffix}"
        if dst_vid.exists():
            dst_vid.unlink()
        shutil.copy2(continue_video, dst_vid)
        continue_video_path = dst_vid
    else:
        continue_video_path = None

    # Build JSON payload (wrapper around orchestrator_details)
    run_id = f"{name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    orch_payload = make_orchestrator_payload(
        run_id=run_id,
        images=copied_images,
        continue_video=continue_video_path,
        num_segments=num_segments,
    )
    task_json: dict = {
        "project_id": PROJECT_ID,
        "orchestrator_details": orch_payload,
    }

    json_path = test_dir / f"{name}_task.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(task_json, fp, indent=2)

    try:
        print(f"[WRITE] {json_path.resolve().relative_to(Path.cwd().resolve())}")
    except ValueError:
        # Fallback if paths are on different drives or unrelated
        print(f"[WRITE] {json_path.resolve()}")

    if enqueue:
        cmd = [sys.executable, "add_task.py", "--type", "travel_orchestrator",
               "--params", f"@{json_path}"]
        print("[ENQUEUE]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] add_task failed: {e}")


def copy_results_for_comparison():
    """Copy all test inputs and outputs to a single comparison directory."""
    comparison_dir = Path("test_results_comparison")
    comparison_dir.mkdir(exist_ok=True)
    
    print(f"[COMPARISON] Creating comparison directory: {comparison_dir}")
    
    # Get task outputs from database
    task_outputs = get_task_outputs_from_db()
    
    # Copy inputs and outputs for each test
    for test_name in ["test_1_single_image", "test_2_three_images", "test_3_continue_plus_one", "test_4_continue_plus_two"]:
        test_dir = Path("tests") / test_name
        if not test_dir.exists():
            print(f"[WARNING] Test directory not found: {test_dir}")
            continue
            
        # Create test-specific comparison subdirectory
        comp_test_dir = comparison_dir / test_name
        comp_test_dir.mkdir(exist_ok=True)
        
        # Copy input files
        inputs_dir = comp_test_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        
        for input_file in test_dir.glob("*"):
            if input_file.is_file() and not input_file.name.endswith("_task.json"):
                dst = inputs_dir / input_file.name
                shutil.copy2(input_file, dst)
                print(f"[COPY INPUT] {input_file} -> {dst}")
        
        # Copy task JSON for reference
        task_json = test_dir / f"{test_name}_task.json"
        if task_json.exists():
            shutil.copy2(task_json, comp_test_dir / "task_config.json")
            print(f"[COPY CONFIG] {task_json} -> {comp_test_dir / 'task_config.json'}")
        
        # Copy output files from database paths
        outputs_dir = comp_test_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        # Copy outputs for this test from database
        test_pattern = test_name.replace("test_", "").replace("_", "")
        copied_count = 0
        for task_id, output_path, status in task_outputs:
            if test_pattern in task_id and output_path and status == "Complete":
                # Convert database path (files/xyz.mp4) to full path (public/files/xyz.mp4)
                if output_path.startswith("files/"):
                    full_output_path = Path("public") / output_path
                else:
                    full_output_path = Path(output_path)
                
                if full_output_path.exists():
                    dst = outputs_dir / f"{test_name}_{full_output_path.name}"
                    shutil.copy2(full_output_path, dst)
                    print(f"[COPY OUTPUT] {full_output_path} -> {dst}")
                    copied_count += 1
                else:
                    print(f"[WARNING] Output file not found: {full_output_path}")
        
        if copied_count == 0:
            print(f"[INFO] No completed outputs found for {test_name}")
    
    print(f"[COMPARISON] Results comparison ready in: {comparison_dir}")
    return comparison_dir


def get_task_outputs_from_db() -> List[Tuple[str, str, str]]:
    """Query tasks.db for task outputs. Returns list of (task_id, output_location, status)."""
    try:
        conn = sqlite3.connect("tasks.db")
        cursor = conn.cursor()
        
        # Get all tasks with their output locations and status
        cursor.execute("""
            SELECT id, output_location, status 
            FROM tasks 
            WHERE project_id = 'test_suite'
            ORDER BY created_at DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        print(f"[DB] Found {len(results)} test suite tasks in database")
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to query database: {e}")
        return []


def wait_for_task_completion(max_wait_minutes: int = 30) -> None:
    """Wait for all test_suite tasks to complete."""
    print(f"[WAIT] Monitoring task completion (max {max_wait_minutes} minutes)...")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while True:
        try:
            conn = sqlite3.connect("tasks.db")
            cursor = conn.cursor()
            
            # Check for queued or in-progress test suite tasks
            cursor.execute("""
                SELECT COUNT(*) 
                FROM tasks 
                WHERE project_id = 'test_suite' 
                AND status IN ('Queued', 'In Progress')
            """)
            
            pending_count = cursor.fetchone()[0]
            
            # Get completed/failed counts
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM tasks 
                WHERE project_id = 'test_suite' 
                GROUP BY status
            """)
            
            status_counts = dict(cursor.fetchall())
            conn.close()
            
            if pending_count == 0:
                print(f"[WAIT] All tasks completed! Status summary: {status_counts}")
                break
                
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                print(f"[WAIT] Timeout reached. Still pending: {pending_count}, Status: {status_counts}")
                break
                
            print(f"[WAIT] {pending_count} tasks still pending... (elapsed: {elapsed/60:.1f}min)")
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            print(f"[ERROR] Failed to check task status: {e}")
            break


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def main(args) -> None:
    TESTS_ROOT.mkdir(exist_ok=True)

    # --- scenario definitions ------------------------------------------------
    scenarios = [
        ("test_1_single_image",          [ASSET_IMAGES[0]],                 None,                 1),
        ("test_2_three_images",          ASSET_IMAGES,                      None,                 3),
        ("test_3_continue_plus_one",     [ASSET_IMAGES[0]],                 ASSET_VIDEO,          1),
        ("test_4_continue_plus_two",     ASSET_IMAGES[:2],                  ASSET_VIDEO,          2),
    ]

    for name, imgs, cont_vid, segments in scenarios:
        write_test_case(name, imgs, cont_vid, segments, enqueue=args.enqueue)

    print("\nAll test cases generated under", TESTS_ROOT.resolve())

    if args.compare:
        if not args.no_wait:
            print("[INFO] Waiting for task completion before comparison (use --no-wait to skip)")
            wait_for_task_completion(args.wait_minutes)
        copy_results_for_comparison()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enqueue", action="store_true",
                        help="Actually call add_task.py for each generated test case")
    parser.add_argument("--compare", action="store_true", help="Copy results for comparison")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting for task completion (compare immediately)")
    parser.add_argument("--wait-minutes", type=int, default=30, help="Max minutes to wait for completion")
    args = parser.parse_args()

    if args.compare:
        if not args.no_wait:
            print("[INFO] Waiting for task completion before comparison (use --no-wait to skip)")
            wait_for_task_completion(args.wait_minutes)
        copy_results_for_comparison()
    else:
        main(args) 