"""Steerable Motion orchestrator CLI script.

This entry-point parses command-line arguments for the two high-level tasks
(`travel_between_images` and `different_pose`), initialises logging/debug
behaviour, ensures the local SQLite `tasks` database exists, and then
delegates the heavy-lifting to modular handlers living in
`sm_functions.travel_between_images` and `sm_functions.different_pose`.

The script itself therefore coordinates the overall workflow, keeps global
state (e.g. DEBUG_MODE), and performs lightweight orchestration rather than
image/video processing.
"""

import argparse
import sqlite3
import json
import time
import uuid
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys 
import traceback

# --- Import from our new sm_functions package --- 
from sm_functions import (
    run_travel_between_images_task,
    run_different_pose_task,
    # Common utilities that steerable_motion.py might directly use (e.g. for init)
    DEBUG_MODE as SM_DEBUG_MODE, # Alias to avoid conflict if main script defines its own DEBUG_MODE
    DEFAULT_DB_TABLE_NAME,
    dprint,
    parse_resolution
)
# Expose DEBUG_MODE from common_utils to the global scope of this script
# This allows dprint within this file (if any) to work as expected.
# The actual value will be set after parsing args.
global DEBUG_MODE
DEBUG_MODE = SM_DEBUG_MODE # Initialize with the value from common_utils

# --- Constants for DB interaction and defaults (specific to steerable_motion.py argparser) ---
# These were NOT moved to common_utils as they are tied to CLI parsing here.
DEFAULT_MODEL_NAME = "vace_14B"
DEFAULT_SEGMENT_FRAMES = 81 
DEFAULT_FPS_HELPERS = 25 
DEFAULT_SEED = 12345

# ----------------------------------------------------
# Helper: Ensure the SQLite DB and tasks table exist
# ----------------------------------------------------

def _ensure_db_initialized(db_path_str: str, table_name: str = DEFAULT_DB_TABLE_NAME):
    """Creates the tasks table (and helpful index) in the SQLite DB if it doesn't exist."""
    conn = sqlite3.connect(db_path_str)
    cursor = conn.cursor()
        cursor.execute(
        f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                    task_id TEXT PRIMARY KEY,
                    params TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Queued',
                    output_location TEXT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
    )
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_status_created_at ON {table_name} (status, created_at)")
            conn.commit()
            conn.close()

# ----------------------------------------------------

def main():
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--resolution", type=str, required=True, help="Output resolution, e.g., '960x544'.")
    common_parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name for headless.py tasks.")
    common_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed for tasks, incremented per segment/step.")
    common_parser.add_argument("--db_path", type=str, default="tasks.db", help="Path to the SQLite database for headless.py.")
    common_parser.add_argument("--output_dir", type=str, default="./steerable_motion_output", help="Base directory for all outputs.")
    common_parser.add_argument("--fps_helpers", type=int, default=DEFAULT_FPS_HELPERS, help="FPS for generated mask/guide videos.")
    common_parser.add_argument("--output_video_frames", type=int, default=30, help="Number of frames for the final generated video output (e.g., in different_pose). Default 30.")
    common_parser.add_argument("--poll_interval", type=int, default=15, help="Polling interval (seconds) for task status.")
    common_parser.add_argument("--poll_timeout", type=int, default=30 * 60, help="Timeout (seconds) for polling a single task.")
    common_parser.add_argument("--skip_cleanup", action="store_true", help="Skip cleanup of intermediate segment/task directories.")
    common_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging and prevent cleanup of intermediate files.")
    common_parser.add_argument("--use_causvid_lora", action="store_true", help="Enable CausVid LoRA for video generation tasks.")

    parser = argparse.ArgumentParser(description="Generates steerable motion videos or performs other image tasks using headless.py.")
    subparsers = parser.add_subparsers(title="tasks", dest="task", required=True, help="Specify the task to perform.")

    parser_travel = subparsers.add_parser("travel_between_images", help="Generate video interpolating between multiple anchor images.", parents=[common_parser])
    parser_travel.add_argument("--input_images", nargs='+', required=True, help="List of input anchor image paths.")
    parser_travel.add_argument("--base_prompts", nargs='+', required=True, help="List of base prompts for each segment.")
    parser_travel.add_argument("--frame_overlap", type=int, default=16, help="Number of frames to overlap between segments.")
    parser_travel.add_argument("--segment_frames", type=int, default=DEFAULT_SEGMENT_FRAMES, help="Total frames for each headless.py segment.")

    parser_pose = subparsers.add_parser("different_pose", help="Generate an image with a different pose based on an input image and prompt.", parents=[common_parser])
    parser_pose.add_argument("--input_image", type=str, required=True, help="Path to the single input image.")
    parser_pose.add_argument("--prompt", type=str, required=True, help="Prompt to guide the generation.")

    args = parser.parse_args()

    # Set the global DEBUG_MODE based on parsed arguments
    # This ensures dprint in this file and in common_utils (via its DEBUG_MODE) honors the CLI flag
    global DEBUG_MODE
    from sm_functions import common_utils as sm_common_utils
    DEBUG_MODE = args.debug
    sm_common_utils.DEBUG_MODE = args.debug # Explicitly set it in the imported module as well
    
    if DEBUG_MODE:
        dprint("Debug mode enabled by CLI arguments.")
        dprint(f"Parsed arguments: {args}")

    try:
        # parse_resolution is now imported from sm_functions (which gets it from common_utils)
        parsed_resolution_val = parse_resolution(args.resolution) 
    except ValueError as e:
        parser.error(str(e))

    main_output_dir = Path(args.output_dir) 
    main_output_dir.mkdir(parents=True, exist_ok=True)
    dprint(f"Main output directory: {main_output_dir.resolve()}")
    
    db_file_path_str = args.db_path 
    try:
        _ensure_db_initialized(db_file_path_str, DEFAULT_DB_TABLE_NAME)
    except Exception as e_db_init:
        print(f"Fatal: Could not initialize database: {e_db_init}")
        return 1

    # Propagate the debug flag to task modules so their local DEBUG_MODE copies stay in sync
    import sm_functions.travel_between_images as _travel_mod
    import sm_functions.different_pose as _diff_pose_mod
    _travel_mod.DEBUG_MODE = args.debug
    _diff_pose_mod.DEBUG_MODE = args.debug

    exit_code = 0
    if args.task == "travel_between_images":
        exit_code = run_travel_between_images_task(args, args, parsed_resolution_val, main_output_dir, db_file_path_str)
    elif args.task == "different_pose":
        exit_code = run_different_pose_task(args, args, parsed_resolution_val, main_output_dir, db_file_path_str)
    else:
        parser.error(f"Unknown task: {args.task}")

    print(f"\nSteerable motion script finished for task '{args.task}'. Exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())