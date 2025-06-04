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
import shlex # Add shlex import
from dotenv import load_dotenv

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
DEFAULT_FPS_HELPERS = 16 
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
            id TEXT PRIMARY KEY,
            params TEXT NOT NULL,
            task_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Queued',
            output_location TEXT NULL,
            dependant_on TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_status_created_at ON {table_name} (status, created_at)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_dependant_on ON {table_name}(dependant_on)")
    conn.commit()
    conn.close()

# ----------------------------------------------------

def main():
    # --- Load .env file for potential SQLITE_DB_PATH_ENV ---
    # This is a simple way to allow .env to influence db_path before arg parsing sets defaults.
    # A more robust solution might integrate dotenv earlier or pass it explicitly.
    load_dotenv()
    env_sqlite_db_path = os.getenv("SQLITE_DB_PATH_ENV")
    # --- End .env load ---

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
    common_parser.add_argument("--skip_cleanup", action="store_true", help="Skip cleanup of intermediate files and folders (useful for debugging). This applies to orchestrator-based tasks like travel_between_images.")
    common_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging and prevent cleanup of intermediate files.")
    common_parser.add_argument("--use_causvid_lora", action="store_true", help="Enable CausVid LoRA for video generation tasks.")
    common_parser.add_argument("--upscale_model_name", type=str, default="ltxv_13B", help="Model name for headless.py to use for upscaling tasks (default: ltxv_13B).")
    common_parser.add_argument("--final_image_strength", type=float, default=0.0, help="Strength of the final anchor image when used as a VACE reference (0.0 to 1.0). 1.0 is full strength (opaque), 0.0 is fully transparent. Using this implicitly makes the end anchor a VACE ref. Values outside [0,1] will be clamped.")
    common_parser.add_argument("--initial_image_strength", type=float, default=0.0, help="Strength of the initial anchor image when used as a VACE reference (0.0 to 1.0). 1.0 is full strength (opaque), 0.0 is fully transparent. Similar to --final_image_strength but for the start anchor. Values outside [0,1] will be clamped.")
    common_parser.add_argument(
        "--fade_in_duration", 
        type=str, 
        default='{"low_point": 0.0, "high_point": 1.0, "curve_type": "ease_in_out", "duration_factor": 0.0}',
        help="JSON string for fade-in settings of the end anchor in the guide video. Example: '{\"low_point\": 0.0, \"high_point\": 1.0, \"curve_type\": \"ease_in_out\", \"duration_factor\": 0.5}'. 'duration_factor' is proportion of available frames (0.0-1.0)."
    )
    common_parser.add_argument(
        "--fade_out_duration", 
        type=str, 
        default='{"low_point": 0.0, "high_point": 1.0, "curve_type": "ease_in_out", "duration_factor": 0.0}',
        help="JSON string for fade-out settings of the start anchor in the guide video. Example: '{\"low_point\": 0.0, \"high_point\": 1.0, \"curve_type\": \"ease_in_out\", \"duration_factor\": 0.5}'. 'duration_factor' is proportion of available frames (0.0-1.0)."
    )
    common_parser.add_argument(
        "--subsequent_starting_strength_adjustment",
        type=float,
        default=0.0,
        help="Float value to adjust the starting strength of subsequent/overlapped segment's initial frames. Applied after fade-out calculations. E.g., -0.2 to reduce strength, 0.1 to increase."
    )
    common_parser.add_argument(
        "--desaturate_subsequent_starting_frames",
        type=float,
        default=0.0,
        help="Float value (0.0 to 1.0) to desaturate the initial frames of subsequent/overlapped segments. 0.0 is no desaturation, 1.0 is fully grayscale. Applied after strength adjustments."
    )
    common_parser.add_argument(
        "--adjust_brightness_subsequent_starting_frames",
        type=float,
        default=0.0,
        help="Float value to adjust brightness of initial frames of subsequent/overlapped segments. Positive values make it darker (e.g., 0.1 for 10% darker), negative values make it brighter (e.g., -0.1 for 10% brighter). 0.0 means no change. Applied after strength and desaturation."
    )
    common_parser.add_argument(
        "--cfg_star_switch",
        type=int,
        default=0, # Default to off
        choices=[0, 1],
        help="Enable CFG* (Zero-Star) technique (0 for Off, 1 for On). Modulates CFG application."
    )
    common_parser.add_argument(
        "--cfg_zero_step",
        type=int,
        default=-1, # Default to -1 (inactive or wgp.py's internal default)
        help="Number of initial steps (0-indexed) where conditional guidance is zeroed if CFG* is active. -1 might mean inactive or use wgp.py default."
    )
    common_parser.add_argument(
        "--params_json_str",
        type=str, 
        default=None, 
        help="JSON string with additional wgp.py parameters to override (use with caution). Example: --params_json_str '{\"flow_shift\": 2.0, \"steps\": 25}'"
    )
    common_parser.add_argument(
        "--after_first_post_generation_saturation",
        type=float,
        default=None,
        help="Saturation level (eq=saturation=<value>) applied via FFmpeg to every generated video segment AFTER the first one. 1.0 means no change. Values <1.0 desaturate, >1.0 increase saturation. If omitted, no saturation change is applied."
    )

    parser = argparse.ArgumentParser(description="Generates steerable motion videos or performs other image tasks using headless.py.")
    subparsers = parser.add_subparsers(title="tasks", dest="task", required=True, help="Specify the task to perform.")

    parser_travel = subparsers.add_parser("travel_between_images", help="Generate video interpolating between multiple anchor images.", parents=[common_parser])
    parser_travel.add_argument("--input_images", nargs='+', required=True, help="List of input anchor image paths.")
    parser_travel.add_argument("--base_prompts", nargs='+', required=True, help="List of base prompts for each segment.")
    parser_travel.add_argument("--negative_prompts", nargs='+', default=[""], help="List of negative prompts for each segment, or a single one for all. Defaults to an empty string for no negative prompt.")
    parser_travel.add_argument("--frame_overlap", nargs='+', type=int, default=[16], help="Number of frames to overlap between segments. Can be one value for all, or one per segment (number of images - 1, or number of images if --continue_from_video is used).")
    parser_travel.add_argument("--segment_frames", nargs='+', type=int, default=[DEFAULT_SEGMENT_FRAMES], help="Total frames for each headless.py segment. Can be one value for all, or one per segment (number of images - 1, or number of images if --continue_from_video is used).")
    parser_travel.add_argument("--upscale_factor", type=float, default=0.0, help="Factor to upscale final video (e.g., 1.5, 2.0). Use 0.0 or 1.0 to disable. Uses model specified by --upscale_model_name.")
    parser_travel.add_argument("--continue_from_video", type=str, default=None, help="Path or URL to a video to continue from. If provided, this video acts as the first segment.")

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
    
    executed_command_str = None
    if DEBUG_MODE:
        script_name = Path(sys.argv[0]).name
        # Properly quote each argument, especially those with spaces
        quoted_args = [shlex.quote(arg) for arg in sys.argv[1:]]
        remaining_args = " ".join(quoted_args)
        executed_command_str = f"python {script_name} {remaining_args}"
        dprint(f"Executed command: {executed_command_str}")

    try:
        # parse_resolution is now imported from sm_functions (which gets it from common_utils)
        parsed_resolution_val = parse_resolution(args.resolution) 
    except ValueError as e:
        parser.error(str(e))

    # --- Argument Validation specific to travel_between_images ---
    if args.task == "travel_between_images":
        if args.continue_from_video:
            if not args.input_images:
                parser.error("For 'travel_between_images' with --continue_from_video, at least one input image is required.")
            num_segments_expected = len(args.input_images)
        else:
            if len(args.input_images) < 2:
                # This check is somewhat redundant with num_segments_expected > 0 later, but good for clarity.
                parser.error("For 'travel_between_images' (without --continue_from_video), at least two input images are required.")
            num_segments_expected = len(args.input_images) - 1
        
        if num_segments_expected <= 0 and args.input_images: # Allow num_segments_expected to be 0 if no input_images for continue_from_video (already errored)
             pass # Error already handled or will be if num_segments_expected is critical for a check below and is zero.
        elif num_segments_expected <=0 :
             parser.error("No segments to generate based on input images and --continue_from_video flag.")

        # Validate --segment_frames
        if len(args.segment_frames) > 1 and len(args.segment_frames) != num_segments_expected:
            parser.error(
                f"Number of --segment_frames values ({len(args.segment_frames)}) must match the number of segments "
                f"({num_segments_expected}), or be a single value."
            )
        for i, val in enumerate(args.segment_frames):
            if val <= 0:
                parser.error(f"--segment_frames value at index {i} ('{val}') must be positive.")

        # Validate --frame_overlap
        if len(args.frame_overlap) > 1 and len(args.frame_overlap) != num_segments_expected:
            parser.error(
                f"Number of --frame_overlap values ({len(args.frame_overlap)}) must match the number of segments "
                f"({num_segments_expected}), or be a single value."
            )
        for i, val in enumerate(args.frame_overlap):
            if val < 0:
                parser.error(f"--frame_overlap value at index {i} ('{val}') cannot be negative.")
        
        # Validate --base_prompts
        if len(args.base_prompts) > 1 and len(args.base_prompts) != num_segments_expected:
             parser.error(
                f"Number of --base_prompts values ({len(args.base_prompts)}) must match the number of segments "
                f"({num_segments_expected}), or be a single value."
            )

        # Validate --negative_prompts
        if len(args.negative_prompts) > 1 and len(args.negative_prompts) != num_segments_expected:
            parser.error(
                f"Number of --negative_prompts values ({len(args.negative_prompts)}) must match the number of segments "
                f"({num_segments_expected}), or be a single value."
            )

        # Cross-validation: segment_frames vs frame_overlap for each segment
        if num_segments_expected > 0: # Only run if there are segments to validate
            for i in range(num_segments_expected):
                # Determine current segment's frame and overlap values
                current_segment_frames_val = args.segment_frames[0] if len(args.segment_frames) == 1 else args.segment_frames[i]
                current_frame_overlap_val = args.frame_overlap[0] if len(args.frame_overlap) == 1 else args.frame_overlap[i]

                if current_frame_overlap_val > 0 and current_segment_frames_val <= current_frame_overlap_val:
                    start_image_name_for_error = ""
                    if args.continue_from_video:
                        if i == 0:
                            start_image_name_for_error = "the continued video"
                        else:
                            start_image_name_for_error = f"image {args.input_images[i-1]}"
                    else:
                        start_image_name_for_error = f"image {args.input_images[i]}"
                    
                    end_image_name_for_error = f"image {args.input_images[i]}" if args.continue_from_video else f"image {args.input_images[i+1]}"

                    parser.error(
                        f"For segment {i+1} (transitioning from {start_image_name_for_error} to {end_image_name_for_error}), --segment_frames ({current_segment_frames_val}) "
                        f"must be greater than --frame_overlap ({current_frame_overlap_val}) when overlap is used."
                    )
    # --- End Argument Validation ---

    main_output_dir = Path(args.output_dir) 
    main_output_dir.mkdir(parents=True, exist_ok=True)
    dprint(f"Main output directory: {main_output_dir.resolve()}")
    
    # Determine the database path: .env variable, then CLI arg, then default.
    # args.db_path already holds the CLI value or its default ("tasks.db").
    db_file_path_str = env_sqlite_db_path if env_sqlite_db_path else args.db_path
    dprint(f"Using database path: {Path(db_file_path_str).resolve()}")

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
        exit_code = run_travel_between_images_task(args, args, parsed_resolution_val, main_output_dir, db_file_path_str, executed_command_str)
    elif args.task == "different_pose":
        exit_code = run_different_pose_task(args, args, parsed_resolution_val, main_output_dir, db_file_path_str, executed_command_str)
    else:
        parser.error(f"Unknown task: {args.task}")

    print(f"\nSteerable motion script finished for task '{args.task}'. Exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())