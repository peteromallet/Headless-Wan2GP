"""Wan2GP Headless Server.

This long-running process polls the `tasks` table (SQLite by default, or
Supabase-backed Postgres when configured), claims queued tasks, and executes
them using the Wan2GP `wgp.py` video generator.  Besides standard generation
tasks it also contains specialised handlers for:

• `generate_openpose` – creates OpenPose skeleton images using dwpose.
• `rife_interpolate_images` – does frame interpolation between two stills.

The server configures global overrides for `wgp.py`, manages temporary
directories, moves or uploads finished artefacts, and updates task status in
the database before looping again.  It serves as the runtime backend that
`steerable_motion.py` (and other clients) rely upon to perform heavy
generation work.
"""

import argparse
import sys
import os
import types
from pathlib import Path
from PIL import Image
import json
import time
# import shutil # No longer moving files, tasks are removed from tasks.json
import traceback
import requests # For downloading the LoRA
import inspect # Added import
import sqlite3 # Added for SQLite database
import urllib.parse # Added for URL encoding
import threading
import uuid # Added import for UUID

# --- Add imports for OpenPose generation ---
import numpy as np
try:
    # Import from Wan2GP submodule
    import sys
    from pathlib import Path
    wan2gp_path = Path(__file__).parent / "Wan2GP"
    if str(wan2gp_path) not in sys.path:
        sys.path.insert(0, str(wan2gp_path))
    from preprocessing.dwpose.pose import PoseBodyFaceVideoAnnotator
except ImportError:
    PoseBodyFaceVideoAnnotator = None # Allow script to load if module not found, error out at runtime
# --- End OpenPose imports ---

from dotenv import load_dotenv # For .env file
# import psycopg2 # For PostgreSQL - REMOVED
# import psycopg2.extras # For dictionary cursor with PostgreSQL - REMOVED
from supabase import create_client, Client as SupabaseClient # For Supabase
import tempfile # For temporary directories
import shutil # For file operations
import cv2 # Added for RIFE interpolation

# --- SM_RESTRUCTURE: Import moved/new utilities ---
from Wan2GP.sm_functions.common_utils import (
    # dprint is already defined locally in headless.py
    generate_unique_task_id as sm_generate_unique_task_id, # Alias to avoid conflict if headless has its own
    add_task_to_db as sm_add_task_to_db,
    # poll_task_status, # headless doesn't poll for sub-tasks this way
    get_video_frame_count_and_fps as sm_get_video_frame_count_and_fps,
    _get_unique_target_path as sm_get_unique_target_path,
    image_to_frame as sm_image_to_frame,
    create_color_frame as sm_create_color_frame,
    _adjust_frame_brightness as sm_adjust_frame_brightness,
    _copy_to_folder_with_unique_name as sm_copy_to_folder_with_unique_name,
    _apply_strength_to_image as sm_apply_strength_to_image,
    parse_resolution as sm_parse_resolution # For parsing resolution string from orchestrator
)
from Wan2GP.sm_functions.video_utils import (
    extract_frames_from_video as sm_extract_frames_from_video,
    create_video_from_frames_list as sm_create_video_from_frames_list,
    cross_fade_overlap_frames as sm_cross_fade_overlap_frames,
    _apply_saturation_to_video_ffmpeg as sm_apply_saturation_to_video_ffmpeg,
    # color_match_video_to_reference # If needed by stitch/segment tasks
)
from Wan2GP.sm_functions.travel_between_images import (
    get_easing_function as sm_get_easing_function # For guide video fades
)
# --- End SM_RESTRUCTURE imports ---

# -----------------------------------------------------------------------------
# Global DB Configuration (will be set in main)
# -----------------------------------------------------------------------------
DB_TYPE = "sqlite" # Default to sqlite, will be changed to "supabase" if configured
# PG_DSN = None # REMOVED - Supabase client handles connection
PG_TABLE_NAME = "tasks" # Still needed for RPC calls (table name in Supabase/Postgres)
SQLITE_DB_PATH = "tasks.db"
SUPABASE_URL = None
SUPABASE_SERVICE_KEY = None
SUPABASE_VIDEO_BUCKET = "videos" # Default bucket name, can be overridden by .env
SUPABASE_CLIENT: SupabaseClient | None = None # Global Supabase client instance

# Add SQLite connection lock for thread safety
sqlite_lock = threading.Lock()

# SQLite retry configuration
SQLITE_MAX_RETRIES = 5
SQLITE_RETRY_DELAY = 0.5  # seconds

# Global variable for ComfyUI Output Path
COMFYUI_OUTPUT_PATH_CONFIG: Path | None = None

# -----------------------------------------------------------------------------
# Status Constants
# -----------------------------------------------------------------------------
STATUS_QUEUED = "Queued"
STATUS_IN_PROGRESS = "In Progress"
STATUS_COMPLETE = "Complete"
STATUS_FAILED = "Failed"

# -----------------------------------------------------------------------------
# Debug / Verbose Logging Helpers
# -----------------------------------------------------------------------------

debug_mode = False  # This will be toggled on via the --debug CLI flag in main()

def dprint(msg: str):
    """Print a debug message if --debug flag is enabled."""
    if debug_mode:
        # Prefix with timestamp for easier tracing
        print(f"[DEBUG {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# -----------------------------------------------------------------------------
# 1. Parse arguments for the server
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("WanGP Headless Server")

    pgroup_server = parser.add_argument_group("Server Settings")
    pgroup_server.add_argument("--db-file", type=str, default="tasks.db",
                               help="Path to the SQLite database file (if not using PostgreSQL via .env).")
    pgroup_server.add_argument("--main-output-dir", type=str, default="./outputs",
                               help="Base directory where outputs for each task will be saved (in subdirectories)")
    pgroup_server.add_argument("--poll-interval", type=int, default=10,
                               help="How often (in seconds) to check tasks.json for new tasks.")
    pgroup_server.add_argument("--debug", action="store_true",
                               help="Enable verbose debug logging (prints additional diagnostics)")

    # Advanced wgp.py Global Config Overrides (Optional) - Applied once at server start
    pgroup_wgp_globals = parser.add_argument_group("WGP Global Config Overrides (Applied at Server Start)")
    pgroup_wgp_globals.add_argument("--wgp-attention-mode", type=str, default=None,
                                choices=["auto", "sdpa", "sage", "sage2", "flash", "xformers"])
    pgroup_wgp_globals.add_argument("--wgp-compile", type=str, default=None, choices=["", "transformer"])
    pgroup_wgp_globals.add_argument("--wgp-profile", type=int, default=None)
    pgroup_wgp_globals.add_argument("--wgp-vae-config", type=int, default=None)
    pgroup_wgp_globals.add_argument("--wgp-boost", type=int, default=None)
    pgroup_wgp_globals.add_argument("--wgp-transformer-quantization", type=str, default=None, choices=["int8", "bf16"])
    pgroup_wgp_globals.add_argument("--wgp-transformer-dtype-policy", type=str, default=None, choices=["", "fp16", "bf16"])
    pgroup_wgp_globals.add_argument("--wgp-text-encoder-quantization", type=str, default=None, choices=["int8", "bf16"])
    pgroup_wgp_globals.add_argument("--wgp-vae-precision", type=str, default=None, choices=["16", "32"])
    pgroup_wgp_globals.add_argument("--wgp-mixed-precision", type=str, default=None, choices=["0", "1"])
    pgroup_wgp_globals.add_argument("--wgp-preload-policy", type=str, default=None,
                                help="Set wgp.py's preload_model_policy (e.g., 'P,S' or 'P'. Avoid 'U' to keep models loaded longer).")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# 2. Gradio Monkey-Patching (same as before)
# -----------------------------------------------------------------------------
def patch_gradio():
    import gradio as gr
    gr.Info = lambda msg: print(f"[INFO] {msg}")
    gr.Warning = lambda msg: print(f"[WARNING] {msg}")
    class _GrError(RuntimeError): pass
    def _raise(msg, *a, **k): raise _GrError(msg)
    gr.Error = _raise
    class _Dummy:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return None
        def update(self, **kwargs): return None
    dummy_attrs = ["HTML", "DataFrame", "Gallery", "Button", "Row", "Column", "Accordion", "Progress", "Dropdown", "Slider", "Textbox", "Checkbox", "Radio", "Image", "Video", "Audio", "DownloadButton", "UploadButton", "Markdown", "Tabs", "State", "Text", "Number"]
    for attr in dummy_attrs:
        setattr(gr, attr, _Dummy)
    gr.update = lambda *a, **k: None
    def dummy_event_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    gr.on = dummy_event_handler
    gr.Request = type('Request', (object,), {'client': type('Client', (object,), {'host': 'localhost'})})
    gr.SelectData = type('SelectData', (), {'index': None, '_data': None})
    gr.EventData = type('EventData', (), {'target':None, '_data':None})

# -----------------------------------------------------------------------------
# Database Helper Functions with Improved SQLite Handling
# -----------------------------------------------------------------------------

def execute_sqlite_with_retry(db_path_str: str, operation_func, *args, **kwargs):
    """Execute SQLite operations with retry logic for handling locks and I/O errors"""
    for attempt in range(SQLITE_MAX_RETRIES):
        try:
            with sqlite_lock:  # Ensure only one thread accesses SQLite at a time
                conn = sqlite3.connect(db_path_str, timeout=30.0)  # 30 second timeout
                conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance
                conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
                try:
                    result = operation_func(conn, *args, **kwargs)
                    conn.commit()
                    return result
                finally:
                    conn.close()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            error_msg = str(e).lower()
            if "database is locked" in error_msg or "disk i/o error" in error_msg or "database disk image is malformed" in error_msg:
                if attempt < SQLITE_MAX_RETRIES - 1:
                    wait_time = SQLITE_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"SQLite error on attempt {attempt + 1}: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"SQLite error after {SQLITE_MAX_RETRIES} attempts: {e}")
                    raise
            else:
                # For other SQLite errors, don't retry
                raise
        except Exception as e:
            # For non-SQLite errors, don't retry
            raise
    
    raise sqlite3.OperationalError(f"Failed to execute SQLite operation after {SQLITE_MAX_RETRIES} attempts")

def _migrate_sqlite_schema(db_path_str: str):
    """Applies necessary schema migrations to an existing SQLite database."""
    dprint(f"SQLite Migration: Checking schema for {db_path_str}...")
    try:
        def migration_operations(conn):
            cursor = conn.cursor()
            
            # Check if task_type column exists
            cursor.execute(f"PRAGMA table_info(tasks)")
            columns = [row[1] for row in cursor.fetchall()]
            task_type_column_exists = 'task_type' in columns

            if not task_type_column_exists:
                dprint("SQLite Migration: 'task_type' column not found. Adding it.")
                cursor.execute("ALTER TABLE tasks ADD COLUMN task_type TEXT") # Add as nullable first
                conn.commit() # Commit alter table before data migration
                dprint("SQLite Migration: 'task_type' column added.")
            else:
                dprint("SQLite Migration: 'task_type' column already exists.")

            # Populate task_type from params if it's NULL (for old rows or newly added column)
            dprint("SQLite Migration: Attempting to populate NULL 'task_type' from 'params' JSON...")
            cursor.execute("SELECT task_id, params FROM tasks WHERE task_type IS NULL")
            rows_to_migrate = cursor.fetchall()
            
            migrated_count = 0
            for task_id, params_json_str in rows_to_migrate:
                try:
                    params_dict = json.loads(params_json_str)
                    # Attempt to get task_type from common old locations within params
                    # The user might need to adjust these keys if their old storage was different
                    old_task_type = params_dict.get("task_type") # Most likely if it was in params
                    
                    if old_task_type:
                        dprint(f"SQLite Migration: Found task_type '{old_task_type}' in params for task_id {task_id}. Updating row.")
                        cursor.execute("UPDATE tasks SET task_type = ? WHERE task_id = ?", (old_task_type, task_id))
                        migrated_count += 1
                    else:
                        # If task_type is not in params, it might be inferred from 'model' or other fields
                        # For instance, if 'model' field implied the task type for older tasks.
                        # This part is highly dependent on previous conventions.
                        # As a simple default, if not found, it will remain NULL unless a default is set.
                        # For 'travel_between_images' and 'different_pose', these are typically set by steerable_motion.py
                        # and wouldn't exist as 'task_type' inside params for headless.py's default processing.
                        # Headless tasks like 'generate_openpose' *did* use task_type in params.
                        dprint(f"SQLite Migration: No 'task_type' key in params for task_id {task_id}. It will remain NULL or needs manual/specific migration logic if it was inferred differently.")
                except json.JSONDecodeError:
                    dprint(f"SQLite Migration: Could not parse params JSON for task_id {task_id}. Skipping 'task_type' population for this row.")
                except Exception as e_row:
                    dprint(f"SQLite Migration: Error processing row for task_id {task_id}: {e_row}")
            
            if migrated_count > 0:
                conn.commit()
            dprint(f"SQLite Migration: Populated 'task_type' for {migrated_count} rows from params.")

            # Default remaining NULL task_types for old standard tasks
            # This ensures rows that didn't have an explicit 'task_type' in their params (e.g. old default WGP tasks)
            # get a value, respecting the NOT NULL constraint if the table is new or fully validated.
            default_task_type_for_old_rows = "standard_wgp_task" 
            cursor.execute(
                f"UPDATE tasks SET task_type = ? WHERE task_type IS NULL", 
                (default_task_type_for_old_rows,)
            )
            updated_to_default_count = cursor.rowcount
            if updated_to_default_count > 0:
                conn.commit()
            dprint(f"SQLite Migration: Updated {updated_to_default_count} older rows with NULL task_type to default '{default_task_type_for_old_rows}'.")

            dprint("SQLite Migration: Schema check and population attempt complete.")
            return True

        execute_sqlite_with_retry(db_path_str, migration_operations)
        
    except Exception as e:
        print(f"[ERROR] SQLite Migration: Failed to migrate schema for {db_path_str}: {e}")
        traceback.print_exc()
        # Depending on severity, you might want to sys.exit(1)

def init_db(db_path_str: str):
    """Initialize the SQLite database with proper error handling"""
    def _init_operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'Queued',
                params TEXT NOT NULL,
                task_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                output_location TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_created ON tasks(status, created_at)")
        return True
    
    try:
        execute_sqlite_with_retry(db_path_str, _init_operation)
        print(f"SQLite database initialized: {db_path_str}")
    except Exception as e:
        print(f"Failed to initialize SQLite database: {e}")
        sys.exit(1)

def get_oldest_queued_task(db_path_str: str):
    """Get the oldest queued task with proper error handling"""
    def _get_operation(conn):
        cursor = conn.cursor()
        cursor.execute("SELECT task_id, params, task_type FROM tasks WHERE status = 'Queued' ORDER BY created_at ASC LIMIT 1")
        task_row = cursor.fetchone()
        if task_row:
            return {"task_id": task_row[0], "params": json.loads(task_row[1]), "task_type": task_row[2]}
        return None
    
    try:
        return execute_sqlite_with_retry(db_path_str, _get_operation)
    except Exception as e:
        print(f"Error getting oldest queued task: {e}")
        return None

def update_task_status(db_path_str: str, task_id: str, status: str, output_location_val: str | None = None):
    """Updates a task's status and updated_at timestamp with proper error handling"""
    def _update_operation(conn, task_id, status, output_location_val):
        cursor = conn.cursor()
        if status == STATUS_COMPLETE and output_location_val is not None:
            cursor.execute("UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP, output_location = ? WHERE task_id = ?", 
                           (status, output_location_val, task_id))
        else:
            cursor.execute("UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?", (status, task_id))
        return True
    
    try:
        execute_sqlite_with_retry(db_path_str, _update_operation, task_id, status, output_location_val)
        dprint(f"SQLite: Updated status of task {task_id} to {status}. Output: {output_location_val if output_location_val else 'N/A'}")
    except Exception as e:
        print(f"Error updating task status for {task_id}: {e}")
        # Don't raise here to avoid crashing the main loop

# -----------------------------------------------------------------------------
# 3. Minimal send_cmd implementation (task_id instead of task_name)
# -----------------------------------------------------------------------------
def make_send_cmd(task_id):
    def _send(cmd, data=None):
        prefix = f"[Task ID: {task_id}]"
        if cmd == "progress":
            if isinstance(data, list) and len(data) >= 2:
                prog, txt = data[0], data[1]
                if isinstance(prog, tuple) and len(prog) == 2: step, total = prog; print(f"{prefix}[Progress] {step}/{total} – {txt}")
                else: print(f"{prefix}[Progress] {txt}")
        elif cmd == "status": print(f"{prefix}[Status] {data}")
        elif cmd == "info": print(f"{prefix}[INFO] {data}")
        elif cmd == "error": print(f"{prefix}[ERROR] {data}"); raise RuntimeError(f"wgp.py error for {task_id}: {data}")
        elif cmd == "output": print(f"{prefix}[Output] video written.")
    
    print(f"DEBUG: Signature of _send for task {task_id}: {inspect.signature(_send)}") # Added diagnostic print
    return _send

# -----------------------------------------------------------------------------
# 4. State builder for a single task (same as before)
# -----------------------------------------------------------------------------
def build_task_state(wgp_mod, model_filename, task_params_dict, all_loras_for_model):
    state = {
        "model_filename": model_filename,
        "validate_success": 1,
        "advanced": True,
        "gen": {"queue": [], "file_list": [], "file_settings_list": [], "prompt_no": 1, "prompts_max": 1},
        "loras": all_loras_for_model,
    }
    model_type_key = wgp_mod.get_model_type(model_filename)
    ui_defaults = wgp_mod.get_default_settings(model_filename).copy()

    # Override with task_params from JSON, but preserve some crucial ones if CausVid is used
    causvid_active = task_params_dict.get("use_causvid_lora", False)

    for key, value in task_params_dict.items():
        if key not in ["output_sub_dir", "model", "task_id", "use_causvid_lora"]:
            if causvid_active and key in ["steps", "guidance_scale", "flow_shift", "activated_loras", "loras_multipliers"]:
                continue # These will be set by causvid logic if flag is true
            ui_defaults[key] = value
    
    ui_defaults["prompt"] = task_params_dict.get("prompt", "Default prompt")
    ui_defaults["resolution"] = task_params_dict.get("resolution", "832x480")
    # Allow task to specify frames/video_length, steps, guidance_scale, flow_shift unless overridden by CausVid
    if not causvid_active:
        ui_defaults["video_length"] = task_params_dict.get("frames", task_params_dict.get("video_length", 81))
        ui_defaults["num_inference_steps"] = task_params_dict.get("steps", task_params_dict.get("num_inference_steps", 30))
        ui_defaults["guidance_scale"] = task_params_dict.get("guidance_scale", ui_defaults.get("guidance_scale", 5.0))
        ui_defaults["flow_shift"] = task_params_dict.get("flow_shift", ui_defaults.get("flow_shift", 3.0))
    else: # CausVid specific defaults if not touched by its logic yet
        ui_defaults["video_length"] = task_params_dict.get("frames", task_params_dict.get("video_length", 81))
        # steps, guidance_scale, flow_shift will be set below by causvid logic

    ui_defaults["seed"] = task_params_dict.get("seed", -1)
    ui_defaults["lset_name"] = "" 

    def load_pil_images(paths_list_or_str, wgp_convert_func):
        if paths_list_or_str is None: return None
        paths_list = paths_list_or_str if isinstance(paths_list_or_str, list) else [paths_list_or_str]
        images = []
        for p_str in paths_list:
            p = Path(p_str.strip())
            if not p.is_file(): print(f"[WARNING] Image file not found: {p}"); continue
            try:
                img = Image.open(p)
                images.append(wgp_convert_func(img))
            except Exception as e:
                print(f"[WARNING] Failed to load image {p}: {e}")
        return images if images else None

    if task_params_dict.get("image_start_paths"):
        loaded = load_pil_images(task_params_dict["image_start_paths"], wgp_mod.convert_image)
        if loaded: ui_defaults["image_start"] = loaded
    if task_params_dict.get("image_end_paths"):
        loaded = load_pil_images(task_params_dict["image_end_paths"], wgp_mod.convert_image)
        if loaded: ui_defaults["image_end"] = loaded
    if task_params_dict.get("image_refs_paths"):
        loaded = load_pil_images(task_params_dict["image_refs_paths"], wgp_mod.convert_image)
        if loaded: ui_defaults["image_refs"] = loaded
    
    for key in ["video_source_path", "video_guide_path", "video_mask_path", "audio_guide_path"]:
        if task_params_dict.get(key):
            ui_defaults[key.replace("_path","")] = task_params_dict[key]

    if task_params_dict.get("prompt_enhancer_mode"):
        ui_defaults["prompt_enhancer"] = task_params_dict["prompt_enhancer_mode"]
        wgp_mod.server_config["enhancer_enabled"] = 1
    elif "prompt_enhancer" not in task_params_dict:
        ui_defaults["prompt_enhancer"] = ""
        wgp_mod.server_config["enhancer_enabled"] = 0

    # --- Custom LoRA Handling (e.g., from lora_name) ---
    custom_lora_name_stem = task_params_dict.get("lora_name")
    task_id_for_dprint = task_params_dict.get('task_id', 'N/A') # For logging

    if custom_lora_name_stem:
        custom_lora_filename = f"{custom_lora_name_stem}.safetensors"
        dprint(f"[Task ID: {task_id_for_dprint}] Custom LoRA specified via lora_name: {custom_lora_filename}")

        # Ensure activated_loras is a list
        activated_loras_val = ui_defaults.get("activated_loras", [])
        if isinstance(activated_loras_val, str):
            # Handles comma-separated string from task_params or previous logic
            current_activated_list = [str(item).strip() for item in activated_loras_val.split(',') if item.strip()]
        elif isinstance(activated_loras_val, list):
            current_activated_list = list(activated_loras_val) # Ensure it's a mutable copy
        else:
            dprint(f"[Task ID: {task_id_for_dprint}] Unexpected type for activated_loras: {type(activated_loras_val)}. Initializing as empty list.")
            current_activated_list = []
        
        if custom_lora_filename not in current_activated_list:
            current_activated_list.append(custom_lora_filename)
            dprint(f"[Task ID: {task_id_for_dprint}] Added '{custom_lora_filename}' to activated_loras list: {current_activated_list}")

            # Handle multipliers: Add a default "1.0" if a LoRA was added and multipliers are potentially mismatched
            loras_multipliers_str = ui_defaults.get("loras_multipliers", "")
            if isinstance(loras_multipliers_str, (list, tuple)):
                loras_multipliers_list = [str(m).strip() for m in loras_multipliers_str if str(m).strip()] # Convert all to string and clean
            elif isinstance(loras_multipliers_str, str):
                loras_multipliers_list = [m.strip() for m in loras_multipliers_str.split(" ") if m.strip()] # Space-separated string
            else:
                dprint(f"[Task ID: {task_id_for_dprint}] Unexpected type for loras_multipliers: {type(loras_multipliers_str)}. Initializing as empty list.")
                loras_multipliers_list = []

            # If number of multipliers is less than activated LoRAs, pad with "1.0"
            while len(loras_multipliers_list) < len(current_activated_list):
                loras_multipliers_list.append("1.0")
                dprint(f"[Task ID: {task_id_for_dprint}] Padded loras_multipliers with '1.0'. Now: {loras_multipliers_list}")
            
            ui_defaults["loras_multipliers"] = " ".join(loras_multipliers_list)
        else:
            dprint(f"[Task ID: {task_id_for_dprint}] Custom LoRA '{custom_lora_filename}' already in activated_loras list.")
            
        ui_defaults["activated_loras"] = current_activated_list # Update ui_defaults
    # --- End Custom LoRA Handling ---

    # --- Handle remove_background_image_ref legacy key ---
    if "remove_background_image_ref" in ui_defaults and "remove_background_images_ref" not in ui_defaults:
        ui_defaults["remove_background_images_ref"] = ui_defaults["remove_background_image_ref"]
    # --- End Handle remove_background_image_ref ---

    # Apply CausVid LoRA specific settings if the flag is true
    if causvid_active:
        print(f"[Task ID: {task_params_dict.get('task_id')}] Applying CausVid LoRA settings.")
        
        # If steps are specified in the task JSON for a CausVid task, use them; otherwise, default to 9.
        if "steps" in task_params_dict:
            ui_defaults["num_inference_steps"] = task_params_dict["steps"]
            print(f"[Task ID: {task_params_dict.get('task_id')}] CausVid task using specified steps: {ui_defaults['num_inference_steps']}")
        elif "num_inference_steps" in task_params_dict:
            ui_defaults["num_inference_steps"] = task_params_dict["num_inference_steps"]
            print(f"[Task ID: {task_params_dict.get('task_id')}] CausVid task using specified num_inference_steps: {ui_defaults['num_inference_steps']}")
        else:
            ui_defaults["num_inference_steps"] = 9 # Default for CausVid if not specified in task
            print(f"[Task ID: {task_params_dict.get('task_id')}] CausVid task defaulting to steps: {ui_defaults['num_inference_steps']}")

        ui_defaults["guidance_scale"] = 1.0 # Still overridden
        ui_defaults["flow_shift"] = 1.0     # Still overridden
        
        causvid_lora_basename = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
        current_activated = ui_defaults.get("activated_loras", [])
        if not isinstance(current_activated, list):
             try: current_activated = [str(item).strip() for item in str(current_activated).split(',') if item.strip()] 
             except: current_activated = []

        if causvid_lora_basename not in current_activated:
            current_activated.append(causvid_lora_basename)
        ui_defaults["activated_loras"] = current_activated

        current_multipliers_str = ui_defaults.get("loras_multipliers", "")
        # Basic handling: if multipliers exist, prepend; otherwise, set directly.
        # More sophisticated merging might be needed if specific order or pairing is critical.
        # This assumes multipliers are space-separated.
        if current_multipliers_str:
            multipliers_list = current_multipliers_str.split()
            lora_names_list = [Path(lora_path).name for lora_path in all_loras_for_model 
                               if Path(lora_path).name in current_activated and Path(lora_path).name != causvid_lora_basename]
            
            final_multipliers = []
            final_loras = []

            # Add CausVid first
            final_loras.append(causvid_lora_basename)
            final_multipliers.append("0.7")

            # Add existing, ensuring no duplicate multiplier for already present CausVid (though it shouldn't be)
            processed_other_loras = set()
            for i, lora_name in enumerate(current_activated):
                if lora_name == causvid_lora_basename: continue # Already handled
                if lora_name not in processed_other_loras:
                    final_loras.append(lora_name)
                    if i < len(multipliers_list):
                         final_multipliers.append(multipliers_list[i])
                    else:
                         final_multipliers.append("1.0") # Default if not enough multipliers
                    processed_other_loras.add(lora_name)
            
            ui_defaults["activated_loras"] = final_loras # ensure order matches multipliers
            ui_defaults["loras_multipliers"] = " ".join(final_multipliers)
        else:
            ui_defaults["loras_multipliers"] = "0.7"
            ui_defaults["activated_loras"] = [causvid_lora_basename] # ensure only causvid if no others

    state[model_type_key] = ui_defaults
    return state, ui_defaults

# -----------------------------------------------------------------------------
# 5. Download utility
# -----------------------------------------------------------------------------
def download_file(url, dest_folder, filename):
    dest_path = Path(dest_folder) / filename
    if dest_path.exists():
        print(f"[INFO] File {filename} already exists in {dest_folder}.")
        return True
    try:
        print(f"Downloading {filename} from {url} to {dest_folder}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        if dest_path.exists(): # Attempt to clean up partial download
            try: os.remove(dest_path)
            except: pass
        return False

# -----------------------------------------------------------------------------
# 6. Process a single task dictionary from the tasks.json list
# -----------------------------------------------------------------------------

def process_single_task(wgp_mod, task_params_dict, main_output_dir_base: Path, task_type: str):
    dprint(f"PROCESS_SINGLE_TASK received task_params_dict: {json.dumps(task_params_dict)}") # DEBUG ADDED
    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))
    print(f"--- Processing task ID: {task_id} of type: {task_type} ---")
    output_location_to_db = None # Will store the final path/URL for the DB

    # --- Check for new task type ---
    if task_type == "generate_openpose":
        if PoseBodyFaceVideoAnnotator is None:
            print(f"[ERROR Task ID: {task_id}] PoseBodyFaceVideoAnnotator not imported. Cannot process 'generate_openpose' task.")
            return False, "PoseBodyFaceVideoAnnotator module not available."
        
        print(f"[Task ID: {task_id}] Identified as 'generate_openpose' task.")
        return _handle_generate_openpose_task(task_params_dict, main_output_dir_base, task_id)
    elif task_type == "rife_interpolate_images":
        print(f"[Task ID: {task_id}] Identified as 'rife_interpolate_images' task.")
        return _handle_rife_interpolate_task(wgp_mod, task_params_dict, main_output_dir_base, task_id)
    elif task_type == "comfyui_workflow": # New task type for ComfyUI
        print(f"[Task ID: {task_id}] Identified as 'comfyui_workflow' task.")
        return _handle_comfyui_workflow_task(task_params_dict, main_output_dir_base, task_id)
    # --- SM_RESTRUCTURE: Add new travel task handlers ---
    elif task_type == "travel_orchestrator":
        print(f"[Task ID: {task_id}] Identified as 'travel_orchestrator' task.")
        return _handle_travel_orchestrator_task(task_params_dict, main_output_dir_base, task_id)
    elif task_type == "travel_segment":
        print(f"[Task ID: {task_id}] Identified as 'travel_segment' task.")
        # This will call wgp_mod like a standard task but might have pre/post processing
        # based on orchestrator details passed in its params.
        return _handle_travel_segment_task(wgp_mod, task_params_dict, main_output_dir_base, task_id)
    elif task_type == "travel_stitch":
        print(f"[Task ID: {task_id}] Identified as 'travel_stitch' task.")
        return _handle_travel_stitch_task(task_params_dict, main_output_dir_base, task_id)
    # --- End SM_RESTRUCTURE ---
    
    # Default handling for standard wgp tasks (original logic)
    task_model_type_logical = task_params_dict.get("model", "t2v")
    # Determine the actual model filename before checking/downloading LoRA, as LoRA path depends on it
    model_filename_for_task = wgp_mod.get_model_filename(task_model_type_logical,
                                                         wgp_mod.transformer_quantization,
                                                         wgp_mod.transformer_dtype_policy)
    
    use_causvid = task_params_dict.get("use_causvid_lora", False)
    causvid_lora_basename = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
    causvid_lora_url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

    if use_causvid:
        base_lora_dir_for_model = Path(wgp_mod.get_lora_dir(model_filename_for_task))
        target_causvid_lora_dir = base_lora_dir_for_model
        
        if "14B" in model_filename_for_task and "t2v" in model_filename_for_task.lower():
             pass 
        elif "14B" in model_filename_for_task:
             pass 

        if not Path(target_causvid_lora_dir / causvid_lora_basename).exists():
            print(f"[Task ID: {task_id}] CausVid LoRA not found. Attempting download...")
            if not download_file(causvid_lora_url, target_causvid_lora_dir, causvid_lora_basename):
                print(f"[WARNING Task ID: {task_id}] Failed to download CausVid LoRA. Proceeding without it or with default settings.")
                task_params_dict["use_causvid_lora"] = False
            else:
                 pass 
        if not "14B" in model_filename_for_task or not "t2v" in model_filename_for_task.lower():
            print(f"[WARNING Task ID: {task_id}] CausVid LoRA is intended for 14B T2V models. Current model is {model_filename_for_task}. Results may vary.")

    print(f"[Task ID: {task_id}] Using model file: {model_filename_for_task}")

    # Use a temporary directory for wgp.py to save its output
    # This temporary directory will be created under the system's temp location
    temp_output_dir = tempfile.mkdtemp(prefix=f"wgp_headless_{task_id}_")
    dprint(f"[Task ID: {task_id}] Using temporary output directory: {temp_output_dir}")

    original_wgp_save_path = wgp_mod.save_path
    wgp_mod.save_path = str(temp_output_dir) # wgp.py saves here

    lora_dir_for_active_model = wgp_mod.get_lora_dir(model_filename_for_task)
    all_loras_for_active_model, _, _, _, _, _, _ = wgp_mod.setup_loras(
        model_filename_for_task, None, lora_dir_for_active_model, "", None
    )

    state, ui_params = build_task_state(wgp_mod, model_filename_for_task, task_params_dict, all_loras_for_active_model)
    
    gen_task_placeholder = {"id": 1, "prompt": ui_params.get("prompt"), "params": {}}
    send_cmd = make_send_cmd(task_id)

    tea_cache_value = ui_params.get("tea_cache_setting", ui_params.get("tea_cache", 0.0))

    print(f"[Task ID: {task_id}] Starting generation with effective params: {json.dumps(ui_params, default=lambda o: 'Unserializable' if isinstance(o, Image.Image) else o.__dict__ if hasattr(o, '__dict__') else str(o), indent=2)}")
    generation_success = False

    # --- Determine frame_num for wgp.py --- 
    requested_frames_from_task = ui_params.get("video_length", 81)
    if requested_frames_from_task == 1:
        frame_num_for_wgp = 1
        dprint(f"[Task ID: {task_id}] Single-frame request detected, passing through (1 frame).")
    elif requested_frames_from_task <= 4:
        frame_num_for_wgp = 5  # wgp.py drops to 1 otherwise; use 5 to preserve motion guidance
        dprint(f"[Task ID: {task_id}] Small-frame ({requested_frames_from_task}) request adjusted to 5 frames for wgp.py compatibility.")
    else:
        frame_num_for_wgp = (requested_frames_from_task // 4) * 4 + 1
        dprint(f"[Task ID: {task_id}] Calculated frame count for wgp.py: {frame_num_for_wgp} (requested: {requested_frames_from_task})")
    # --- End frame_num determination ---

    ui_params["video_length"] = frame_num_for_wgp

    try:
        wgp_mod.generate_video(
            task=gen_task_placeholder, send_cmd=send_cmd,
            prompt=ui_params["prompt"],
            negative_prompt=ui_params.get("negative_prompt", ""),
            resolution=ui_params["resolution"],
            video_length=ui_params.get("video_length", 81),
            seed=ui_params["seed"],
            num_inference_steps=ui_params.get("num_inference_steps", 30),
            guidance_scale=ui_params.get("guidance_scale", 5.0),
            audio_guidance_scale=ui_params.get("audio_guidance_scale", 5.0),
            flow_shift=ui_params.get("flow_shift", wgp_mod.get_default_flow(model_filename_for_task, wgp_mod.test_class_i2v(model_filename_for_task))),
            embedded_guidance_scale=ui_params.get("embedded_guidance_scale", 6.0),
            repeat_generation=ui_params.get("repeat_generation", 1),
            multi_images_gen_type=ui_params.get("multi_images_gen_type", 0),
            tea_cache_setting=tea_cache_value,
            tea_cache_start_step_perc=ui_params.get("tea_cache_start_step_perc", 0),
            activated_loras=ui_params.get("activated_loras", []),
            loras_multipliers=ui_params.get("loras_multipliers", ""),
            image_prompt_type=ui_params.get("image_prompt_type", "T"),
            image_start=[wgp_mod.convert_image(img) for img in ui_params.get("image_start", [])],
            image_end=[wgp_mod.convert_image(img) for img in ui_params.get("image_end", [])],
            model_mode=ui_params.get("model_mode", 0),
            video_source=ui_params.get("video_source", None),
            keep_frames_video_source=ui_params.get("keep_frames_video_source", ""),
            
            video_prompt_type=ui_params.get("video_prompt_type", ""),
            image_refs=ui_params.get("image_refs", None),
            video_guide=ui_params.get("video_guide", None),
            keep_frames_video_guide=ui_params.get("keep_frames_video_guide", ""),
            video_mask=ui_params.get("video_mask", None),
            audio_guide=ui_params.get("audio_guide", None),
            sliding_window_size=ui_params.get("sliding_window_size", 81),
            sliding_window_overlap=ui_params.get("sliding_window_overlap", 5),
            sliding_window_overlap_noise=ui_params.get("sliding_window_overlap_noise", 20),
            sliding_window_discard_last_frames=ui_params.get("sliding_window_discard_last_frames", 0),
            remove_background_images_ref=ui_params.get("remove_background_images_ref", ui_params.get("remove_background_image_ref", 1)),
            temporal_upsampling=ui_params.get("temporal_upsampling", ""),
            spatial_upsampling=ui_params.get("spatial_upsampling", ""),
            RIFLEx_setting=ui_params.get("RIFLEx_setting", 0),
            slg_switch=ui_params.get("slg_switch", 0),
            slg_layers=ui_params.get("slg_layers", [9]),
            slg_start_perc=ui_params.get("slg_start_perc", 10),
            slg_end_perc=ui_params.get("slg_end_perc", 90),
            cfg_star_switch=ui_params.get("cfg_star_switch", 0),
            cfg_zero_step=ui_params.get("cfg_zero_step", -1),
            prompt_enhancer=ui_params.get("prompt_enhancer", ""),
            state=state,
            model_filename=model_filename_for_task
        )
        print(f"[Task ID: {task_id}] Generation completed to temporary directory: {wgp_mod.save_path}")
        generation_success = True
    except Exception as e:
        print(f"[ERROR] Task ID {task_id} failed during generation: {e}")
        traceback.print_exc()
        # generation_success remains False
    finally:
        wgp_mod.save_path = original_wgp_save_path # Restore original save path

    if generation_success:
        # Find the generated video file (assuming one .mp4)
        generated_video_file = None
        for item in Path(temp_output_dir).iterdir():
            if item.is_file() and item.suffix.lower() == ".mp4":
                generated_video_file = item
                break
        
        if generated_video_file:
            dprint(f"[Task ID: {task_id}] Found generated video: {generated_video_file}")
            if DB_TYPE == "sqlite":
                dprint(f"HEADLESS SQLITE SAVE: task_params_dict contains output_path? Key: 'output_path', Value: {task_params_dict.get('output_path')}") # DEBUG ADDED
                custom_output_path_str = task_params_dict.get("output_path")
                if custom_output_path_str:
                    # `output_path` is expected to be a full path (absolute or relative). If relative, resolve against cwd.
                    final_video_path = Path(custom_output_path_str).expanduser().resolve()
                    final_video_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_sub_dir_val = task_params_dict.get("output_sub_dir", task_id)
                    output_sub_dir_path = Path(output_sub_dir_val)
                    # If the provided sub-dir is absolute, use it directly; otherwise nest it under the main output dir.
                    if output_sub_dir_path.is_absolute():
                        final_output_dir = output_sub_dir_path
                    else:
                        final_output_dir = main_output_dir_base / output_sub_dir_path
                    final_output_dir.mkdir(parents=True, exist_ok=True)
                    final_video_path = final_output_dir / f"{task_id}.mp4"
                try:
                    shutil.move(str(generated_video_file), str(final_video_path))
                    output_location_to_db = str(final_video_path.resolve())
                    print(f"[Task ID: {task_id}] Output video saved to: {output_location_to_db}")

                    # If a custom output_path was used, there might still be a default
                    # directory (e.g., <main_output_dir_base>/<task_id>) that was created
                    # earlier by wgp.py or previous logic.  Clean it up to avoid clutter.
                    if custom_output_path_str:
                        default_dir_to_clean = (main_output_dir_base / task_id)
                        try:
                            if default_dir_to_clean.exists() and default_dir_to_clean.is_dir():
                                shutil.rmtree(default_dir_to_clean)
                                dprint(f"[Task ID: {task_id}] Removed default output directory that is no longer needed: {default_dir_to_clean}")
                        except Exception as e_cleanup_default:
                            print(f"[WARNING Task ID: {task_id}] Could not remove default output directory {default_dir_to_clean}: {e_cleanup_default}")
                except Exception as e_move:
                    print(f"[ERROR Task ID: {task_id}] Failed to move video to final local destination: {e_move}")
                    generation_success = False # Mark as failed if file handling fails
            
            elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
                # For Supabase, use task_id as part of the object name to ensure uniqueness
                # You might want to include the original filename if it's meaningful
                
                # URL-encode the filename part to handle spaces and special characters
                encoded_file_name = urllib.parse.quote(generated_video_file.name)
                object_name = f"{task_id}/{encoded_file_name}" 
                # Or if you wanted task_id as the filename: object_name = f"{task_id}.mp4"
                # Or if you wanted just the encoded filename: object_name = encoded_file_name
                dprint(f"[Task ID: {task_id}] Original filename: {generated_video_file.name}")
                dprint(f"[Task ID: {task_id}] Encoded filename for Supabase object: {encoded_file_name}")
                dprint(f"[Task ID: {task_id}] Final Supabase object_name: {object_name}")
                
                public_url = upload_to_supabase_storage(generated_video_file, object_name, SUPABASE_VIDEO_BUCKET)
                if public_url:
                    output_location_to_db = public_url
                else:
                    print(f"[WARNING Task ID: {task_id}] Supabase upload failed or no URL returned. No output location will be saved.")
                    generation_success = False # Mark as failed if upload fails
            else:
                print(f"[WARNING Task ID: {task_id}] Output generated but DB_TYPE ({DB_TYPE}) is not sqlite or Supabase client is not configured for upload.")
                generation_success = False # Cannot determine final resting place
        else:
            print(f"[WARNING Task ID: {task_id}] Generation reported success, but no .mp4 file found in {temp_output_dir}")
            generation_success = False # No output to save
    
    # Clean up the temporary directory
    try:
        shutil.rmtree(temp_output_dir)
        dprint(f"[Task ID: {task_id}] Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e_clean:
        print(f"[WARNING Task ID: {task_id}] Failed to clean up temporary directory {temp_output_dir}: {e_clean}")

    print(f"--- Finished task ID: {task_id} (Success: {generation_success}) ---")
    return generation_success, output_location_to_db

# -----------------------------------------------------------------------------
# +++. New function to handle 'generate_openpose' task
# -----------------------------------------------------------------------------
def _handle_generate_openpose_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str):
    """
    Handles the 'generate_openpose' task type.
    Generates an OpenPose image from an input image path specified in task_params_dict.
    Saves the OpenPose image to the output_path also specified in task_params_dict.
    """
    print(f"[Task ID: {task_id}] Handling 'generate_openpose' task.")
    input_image_path_str = task_params_dict.get("input_image_path")
    output_image_path_str = task_params_dict.get("output_path") # Expecting full path from caller

    if not input_image_path_str:
        print(f"[ERROR Task ID: {task_id}] 'input_image_path' not specified for generate_openpose task.")
        return False, "Missing input_image_path"
    
    if not output_image_path_str:
        # Fallback if not specified, though steerable_motion.py should always provide it
        default_output_dir = main_output_dir_base / task_id
        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = default_output_dir / f"{task_id}_openpose.png"
        print(f"[WARNING Task ID: {task_id}] 'output_path' not specified. Defaulting to {output_image_path}")
    else:
        output_image_path = Path(output_image_path_str)

    input_image_path = Path(input_image_path_str)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_image_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image file not found: {input_image_path}")
        return False, f"Input image not found: {input_image_path}"

    try:
        pil_input_image = Image.open(input_image_path).convert("RGB")
        
        # Config for Pose Annotator (similar to wgp.py)
        # Ensure these ckpt paths are accessible relative to where headless.py runs
        pose_cfg_dict = {
            "DETECTION_MODEL": "ckpts/pose/yolox_l.onnx",
            "POSE_MODEL": "ckpts/pose/dw-ll_ucoco_384.onnx",
            "RESIZE_SIZE": 1024 # Internal resize for detection, not necessarily final output size
        }
        if PoseBodyFaceVideoAnnotator is None: # Should have been caught earlier, but double check
             raise ImportError("PoseBodyFaceVideoAnnotator could not be imported.")
             
        pose_annotator = PoseBodyFaceVideoAnnotator(pose_cfg_dict)
        
        # The forward method expects a list of PIL images
        # It returns a list of NumPy arrays (H, W, C) in BGR format by default from dwpose
        openpose_np_frames_bgr = pose_annotator.forward([pil_input_image])

        if not openpose_np_frames_bgr or openpose_np_frames_bgr[0] is None:
            print(f"[ERROR Task ID: {task_id}] OpenPose generation failed or returned no frame.")
            return False, "OpenPose generation returned no data."

        openpose_np_frame_bgr = openpose_np_frames_bgr[0]
        
        # PoseBodyFaceVideoAnnotator output is BGR, convert to RGB for PIL save if needed
        # However, PIL Image.fromarray can often handle BGR if mode is not specified,
        # but explicitly converting is safer for PNG.
        # Or, let's check if the annotator itself returns RGB.
        # dwpose.annotator.py > draw_pose seems to draw on an RGB copy.
        # Let's assume `anno_ins.forward` gives RGB-compatible array or directly RGB.
        # If colors are inverted, we'll need: openpose_np_frame_rgb = cv2.cvtColor(openpose_np_frame_bgr, cv2.COLOR_BGR2RGB)
        # For now, let's assume it's directly usable by PIL.
        
        openpose_pil_image = Image.fromarray(openpose_np_frame_bgr.astype(np.uint8)) # Ensure uint8
        openpose_pil_image.save(output_image_path)
        
        print(f"[Task ID: {task_id}] Successfully generated OpenPose image to: {output_image_path.resolve()}")
        return True, str(output_image_path.resolve())

    except ImportError as ie:
        print(f"[ERROR Task ID: {task_id}] Import error during OpenPose generation: {ie}. Ensure 'preprocessing' module is in PYTHONPATH and dependencies are installed.")
        traceback.print_exc()
        return False, f"Import error: {ie}"
    except FileNotFoundError as fnfe: # For missing ONNX models
        print(f"[ERROR Task ID: {task_id}] ONNX model file not found for OpenPose: {fnfe}. Ensure 'ckpts/pose/*' models are present.")
        traceback.print_exc()
        return False, f"ONNX model not found: {fnfe}"
    except Exception as e:
        print(f"[ERROR Task ID: {task_id}] Failed during OpenPose image generation: {e}")
        traceback.print_exc()
        return False, f"OpenPose generation exception: {e}"

# -----------------------------------------------------------------------------
# +++. New function to handle 'rife_interpolate_images' task
# -----------------------------------------------------------------------------
def _handle_rife_interpolate_task(wgp_mod, task_params_dict: dict, main_output_dir_base: Path, task_id: str):
    """
    Handles the 'rife_interpolate_images' task type using wgp.py's RIFE capabilities.
    """
    print(f"[Task ID: {task_id}] Handling 'rife_interpolate_images' task.")
    
    input_image_path1_str = task_params_dict.get("input_image_path1")
    input_image_path2_str = task_params_dict.get("input_image_path2")
    output_video_path_str = task_params_dict.get("output_path")
    num_rife_frames = task_params_dict.get("frames")
    resolution_str = task_params_dict.get("resolution") # e.g., "960x544"
    # Default model for RIFE context, actual RIFE process might be model-agnostic in wgp.py
    # Using a known model type like t2v for get_model_filename
    default_model_for_context = task_params_dict.get("model_context_for_rife", "vace_14B") 

    # Validate required parameters
    required_params = {
        "input_image_path1": input_image_path1_str,
        "input_image_path2": input_image_path2_str,
        "output_path": output_video_path_str,
        "frames": num_rife_frames,
        "resolution": resolution_str
    }
    missing_params = [key for key, value in required_params.items() if value is None]
    if missing_params:
        error_msg = f"Missing required parameters for rife_interpolate_images: {', '.join(missing_params)}"
        print(f"[ERROR Task ID: {task_id}] {error_msg}")
        return False, error_msg

    input_image1_path = Path(input_image_path1_str)
    input_image2_path = Path(input_image_path2_str)
    output_video_path = Path(output_video_path_str)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    generation_success = False # Initialize
    output_location_to_db = None # Initialize

    dprint(f"[Task ID: {task_id}] Checking input image paths.")
    if not input_image1_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image 1 not found: {input_image1_path}")
        return False, f"Input image 1 not found: {input_image1_path}"
    if not input_image2_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image 2 not found: {input_image2_path}")
        return False, f"Input image 2 not found: {input_image2_path}"
    dprint(f"[Task ID: {task_id}] Input images found.")

    # Create a temporary directory for wgp.py output (though not used by direct RIFE, kept for structure)
    temp_output_dir = tempfile.mkdtemp(prefix=f"wgp_rife_{task_id}_")
    original_wgp_save_path = wgp_mod.save_path
    wgp_mod.save_path = str(temp_output_dir)
    
    try:
        pil_image_start = Image.open(input_image1_path).convert("RGB")
        pil_image_end = Image.open(input_image2_path).convert("RGB")

        # Get a valid model_filename for wgp.py context (RIFE might not use its weights)
        # Assuming default transformer_quantization and transformer_dtype_policy from wgp_mod
        actual_model_filename = wgp_mod.get_model_filename(
            default_model_for_context, # e.g., "vace_14B"
            wgp_mod.transformer_quantization,
            wgp_mod.transformer_dtype_policy
        )
        dprint(f"[Task ID: {task_id}] Using model file for RIFE context: {actual_model_filename}")

        # Prepare minimal state for wgp.py
        # LoRA setup for the context model (RIFE likely won't use these LoRAs)
        lora_dir_for_context_model = wgp_mod.get_lora_dir(actual_model_filename)
        all_loras_for_context_model, _, _, _, _, _, _ = wgp_mod.setup_loras(
             actual_model_filename, None, lora_dir_for_context_model, "", None
        )
        
        # Basic state, RIFE might not need detailed model-specific ui_params
        # The key is that `generate_video` has a valid `model_filename` in `state`
        # and `state[wgp_mod.get_model_type(actual_model_filename)]` exists.
        model_type_key = wgp_mod.get_model_type(actual_model_filename)
        minimal_ui_defaults_for_model_type = wgp_mod.get_default_settings(actual_model_filename).copy()
        # Override crucial RIFE params in the minimal_ui_defaults for clarity if generate_video uses them from here
        minimal_ui_defaults_for_model_type["resolution"] = resolution_str
        minimal_ui_defaults_for_model_type["video_length"] = int(num_rife_frames)


        state = {
            "model_filename": actual_model_filename,
            "loras": all_loras_for_context_model, # Provide loras for the context model
            model_type_key: minimal_ui_defaults_for_model_type, # Default settings for the context model type
            "gen": {"queue": [], "file_list": [], "file_settings_list": [], "prompt_no": 1, "prompts_max": 1} 
        }

        print(f"[Task ID: {task_id}] Starting direct RIFE interpolation (bypassing wgp.py).")
        dprint(f"  Input 1: {input_image1_path}")
        dprint(f"  Input 2: {input_image2_path}")

        # ---- Direct RIFE Implementation ----
        import torch
        import numpy as np
        from rife.inference import temporal_interpolation
        dprint(f"[Task ID: {task_id}] Imported RIFE modules.")

        width_out, height_out = map(int, resolution_str.split("x"))
        dprint(f"[Task ID: {task_id}] Parsed resolution: {width_out}x{height_out}")

        def pil_to_tensor_rgb_norm(pil_im: Image.Image):
            pil_resized = pil_im.resize((width_out, height_out), Image.Resampling.LANCZOS)
            np_rgb = np.asarray(pil_resized).astype(np.float32) / 127.5 - 1.0  # [0,255]->[-1,1]
            tensor = torch.from_numpy(np_rgb).permute(2, 0, 1)  # C H W
            return tensor

        t_start = pil_to_tensor_rgb_norm(pil_image_start)
        t_end   = pil_to_tensor_rgb_norm(pil_image_end)

        sample_in = torch.stack([t_start, t_end], dim=1).unsqueeze(0)  # 1 x 3 x 2 x H x W

        device_for_rife = "cuda" if torch.cuda.is_available() else "cpu"
        sample_in = sample_in.to(device_for_rife)
        dprint(f"[Task ID: {task_id}] Input tensor for RIFE prepared on device: {device_for_rife}, shape: {sample_in.shape}")

        exp_val = 3  # x8 (2^3 + 1 = 9 frames output by this RIFE implementation for 2 inputs)
        flownet_ckpt = os.path.join("ckpts", "flownet.pkl")
        dprint(f"[Task ID: {task_id}] Checking for RIFE model: {flownet_ckpt}")
        if not os.path.exists(flownet_ckpt):
            error_msg_flownet = f"RIFE Error: flownet.pkl not found at {flownet_ckpt}"
            print(f"[ERROR Task ID: {task_id}] {error_msg_flownet}")
            # generation_success remains False, will be returned
            return False, error_msg_flownet # Explicitly return
        dprint(f"[Task ID: {task_id}] RIFE model found: {flownet_ckpt}. Exp_val: {exp_val}")
        
        # Remove batch dimension for rife.inference.temporal_interpolation's internal process_frames
        sample_in_for_rife = sample_in[0] # Shape: C x 2 x H x W

        try:
            # sample_out_from_rife will have shape C x F_out x H x W
            dprint(f"[Task ID: {task_id}] Calling temporal_interpolation with input shape: {sample_in_for_rife.shape}")
            sample_out_from_rife = temporal_interpolation(flownet_ckpt, sample_in_for_rife, exp_val, device=device_for_rife)
            dprint(f"[Task ID: {task_id}] temporal_interpolation call completed.")
            if sample_out_from_rife is not None:
                dprint(f"[Task ID: {task_id}] RIFE output tensor shape: {sample_out_from_rife.shape}")
            else:
                dprint(f"[Task ID: {task_id}] RIFE process returned None for sample_out_from_rife.")
        except Exception as e_rife: # Inner catch for temporal_interpolation
            print(f"[ERROR Task ID: {task_id}] RIFE interpolation failed: {e_rife}")
            traceback.print_exc()
            generation_success = False
            sample_out_from_rife = None

        if sample_out_from_rife is not None:
            sample_out_no_batch = sample_out_from_rife.to("cpu")  # Shape C x F_out x H x W
            total_frames_generated = sample_out_no_batch.shape[1] # F_out is at index 1
            dprint(f"[Task ID: {task_id}] RIFE produced {total_frames_generated} frames.")

            # Trim / pad to desired num_rife_frames
            if total_frames_generated < num_rife_frames:
                print(f"[Task ID: {task_id}] Warning: RIFE produced {total_frames_generated} frames, expected {num_rife_frames}. Padding last frame.")
                pad_frames = num_rife_frames - total_frames_generated
            else:
                pad_frames = 0
            
            frames_list_np = []
            # Iterate up to the smaller of desired frames or actual RIFE output frames
            for idx in range(min(num_rife_frames, total_frames_generated)):
                frame_tensor = sample_out_no_batch[:, idx] # Shape: C x H x W
                frame_np = ((frame_tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)  # H W C RGB
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frames_list_np.append(frame_bgr)
            
            # Pad with last frame if RIFE produced fewer frames than desired *and* we have at least one frame
            if pad_frames > 0 and frames_list_np:
                last_frame_to_pad = frames_list_np[-1].copy()
                frames_list_np.extend([last_frame_to_pad for _ in range(pad_frames)])
            elif not frames_list_np and num_rife_frames > 0: # RIFE produced 0 frames, or processing failed to create any
                 dprint(f"[Task ID: {task_id}] Error: No frames available to write for RIFE video (num_rife_frames: {num_rife_frames}).")
                 # generation_success remains False (from initialization)

            # Only proceed to write if we have frames in frames_list_np
            if frames_list_np:
                try:
                    fps_output = 25 # Or use a parameter if available/needed
                    dprint(f"[Task ID: {task_id}] Writing RIFE output video to: {output_video_path} with {len(frames_list_np)} frames.")
                    output_video_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, float(fps_output), (width_out, height_out))
                    if not out_writer.isOpened():
                        # This will be caught by the outer broad 'except Exception as e'
                        raise IOError(f"Failed to open VideoWriter for output RIFE video: {output_video_path}")
                    for frame_np in frames_list_np:
                        out_writer.write(frame_np)
                    out_writer.release()

                    if output_video_path.exists() and output_video_path.stat().st_size > 0:
                        dprint(f"[Task ID: {task_id}] Video file confirmed exists and has size > 0.")
                        generation_success = True # Set to True only after successful write and check
                        output_location_to_db = str(output_video_path.resolve())
                        print(f"[Task ID: {task_id}] Direct RIFE video saved to: {output_location_to_db}")
                    else:
                        print(f"[ERROR Task ID: {task_id}] RIFE output file missing or empty after writing attempt: {output_video_path}")
                        # generation_success remains False (from initialization)
                except Exception as e_video_write:
                    print(f"[ERROR Task ID: {task_id}] Exception during RIFE video writing: {e_video_write}")
                    traceback.print_exc()
                    # generation_success remains False (from initialization)
            # If frames_list_np was empty, generation_success remains False
        else: # sample_out_from_rife was None (RIFE call itself failed and was caught by e_rife)
            # generation_success remains False (error already printed by e_rife block)
            pass # dprint for this case is already after the e_rife block

        # ---- End Direct RIFE Implementation ----

    except Exception as e: # Broad catch-all for the whole RIFE block
        print(f"[ERROR Task ID: {task_id}] Overall _handle_rife_interpolate_task failed: {e}")
        traceback.print_exc()
        generation_success = False
    finally:
        wgp_mod.save_path = original_wgp_save_path # Restore original save path

    return generation_success, output_location_to_db

# -----------------------------------------------------------------------------
# +++. New function to handle 'comfyui_workflow' task
# -----------------------------------------------------------------------------
def _handle_comfyui_workflow_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str):
    """
    Handles the 'comfyui_workflow' task type.
    Executes a ComfyUI workflow based on parameters in task_params_dict.
    """
    print(f"[Task ID: {task_id}] Handling 'comfyui_workflow' task.")
    generation_success = False
    output_location_to_db = None

    # --- 1. Extract ComfyUI specific parameters from task_params_dict ---
    comfyui_workflow_file_path = task_params_dict.get("comfyui_workflow_path")
    comfyui_server_address = task_params_dict.get("comfyui_server_address", "http://127.0.0.1:8188") # Default if not provided
    comfyui_inputs_for_workflow = task_params_dict.get("comfyui_inputs", {})
    # Example: client_id for ComfyUI API
    comfyui_client_id = str(uuid.uuid4())

    dprint(f"[Task ID: {task_id}] ComfyUI Workflow File: {comfyui_workflow_file_path}")
    dprint(f"[Task ID: {task_id}] ComfyUI Server: {comfyui_server_address}")
    dprint(f"[Task ID: {task_id}] ComfyUI Inputs: {json.dumps(comfyui_inputs_for_workflow, indent=2)}")

    if not comfyui_workflow_file_path:
        print(f"[ERROR Task ID: {task_id}] 'comfyui_workflow_path' not specified.")
        return False, "Missing comfyui_workflow_path"
    
    workflow_path = Path(comfyui_workflow_file_path)
    if not workflow_path.is_file():
        print(f"[ERROR Task ID: {task_id}] ComfyUI workflow file not found: {workflow_path}")
        return False, f"ComfyUI workflow file not found: {workflow_path}"

    # Create a temporary directory for any intermediate files ComfyUI might need or for its direct output
    # before we move it to the final location.
    temp_comfy_output_dir = tempfile.mkdtemp(prefix=f"comfy_headless_{task_id}_")
    dprint(f"[Task ID: {task_id}] Using temporary directory for ComfyUI outputs/staging: {temp_comfy_output_dir}")

    try:
        # --- 2. Load the ComfyUI workflow JSON --- 
        # Placeholder: You'll need to read the JSON file content here.
        # Example:
        # with open(workflow_path, 'r') as f:
        #     loaded_workflow_json = json.load(f)
        loaded_workflow_json = {} # Replace with actual loading
        dprint(f"[Task ID: {task_id}] Successfully loaded workflow (placeholder): {workflow_path.name}")

        # --- 3. Modify the workflow JSON with comfyui_inputs_for_workflow ---
        # Placeholder: This is the core logic for dynamically setting node inputs.
        # You will need to iterate through `comfyui_inputs_for_workflow` and 
        # find the corresponding nodes in `loaded_workflow_json` to update their values.
        # Nodes are often identified by title (in _meta.title) or by class_type and index.
        # Example (highly conceptual):
        # for node_id, node_data in loaded_workflow_json.items(): # ComfyUI API format often has nodes keyed by string ID
        #     node_title = node_data.get("_meta", {}).get("title")
        #     if node_title and node_title in comfyui_inputs_for_workflow:
        #         # Find the correct widget/input field and update it
        #         # For a 'LoadImage' node, this might be node_data['inputs']['image'] = comfyui_inputs_for_workflow[node_title]
        #         # For a 'CLIPTextEncode' (prompt) node, node_data['inputs']['text'] = comfyui_inputs_for_workflow[node_title]
        #         dprint(f"[Task ID: {task_id}] Updated node '{node_title}' in workflow.")
        #         pass # Actual update logic here
        dprint(f"[Task ID: {task_id}] Workflow modification step (placeholder). You need to implement dynamic input injection.")

        # --- 4. Send the workflow to ComfyUI API (/prompt endpoint) ---
        # Placeholder: Use the `requests` library to POST the `loaded_workflow_json`.
        # Example:
        # prompt_payload = {"prompt": loaded_workflow_json, "client_id": comfyui_client_id}
        # response = requests.post(f"{comfyui_server_address}/prompt", json=prompt_payload)
        # response.raise_for_status()
        # prompt_response_data = response.json()
        # submitted_prompt_id = prompt_response_data.get('prompt_id')
        submitted_prompt_id = "dummy_prompt_id_123" # Replace with actual submission
        if not submitted_prompt_id:
            print(f"[ERROR Task ID: {task_id}] Failed to submit prompt to ComfyUI or did not receive prompt_id.")
            # dprint(f"ComfyUI submission response: {prompt_response_data}") # If you had actual response
            raise ValueError("ComfyUI prompt submission failed")
        dprint(f"[Task ID: {task_id}] Workflow submitted to ComfyUI. Prompt ID: {submitted_prompt_id}")

        # --- 5. Poll ComfyUI API for completion status (/history or /queue) ---
        # Placeholder: Loop and poll the /history/{submitted_prompt_id} endpoint.
        # Example:
        # while True:
        #     history_response = requests.get(f"{comfyui_server_address}/history/{submitted_prompt_id}")
        #     history_response.raise_for_status()
        #     history_data = history_response.json()
        #     if submitted_prompt_id in history_data and history_data[submitted_prompt_id].get("outputs"):
        #         dprint(f"[Task ID: {task_id}] ComfyUI workflow execution completed.")
        #         comfyui_outputs = history_data[submitted_prompt_id]["outputs"]
        #         break
        #     # Check for errors in queue or history too
        #     # Add timeout logic
        #     time.sleep(5) # Polling interval
        comfyui_outputs = {} # Replace with actual polling and output retrieval
        dprint(f"[Task ID: {task_id}] Workflow completion polling (placeholder). Outputs (placeholder): {comfyui_outputs}")

        # --- 6. Identify and retrieve output files ---
        # Placeholder: Parse `comfyui_outputs` to find the generated files.
        # ComfyUI outputs often specify filename, subfolder, type relative to ComfyUI's base output directory.
        # You'll need to know ComfyUI's output directory structure or configure it.
        # Example:
        # output_video_files = []
        # for node_id_output, node_output_data in comfyui_outputs.items():
        #     if 'videos' in node_output_data:
        #         for video_info in node_output_data['videos']:
        #             # Construct full path: ComfyUI_output_dir / video_info['subfolder'] / video_info['filename']
        #             # output_video_files.append(full_path_to_video)
        #             pass
        # generated_final_file_path = output_video_files[0] if output_video_files else None
        
        # --- Temporary placeholder for generated_final_file_path using COMFYUI_OUTPUT_PATH_CONFIG --- 
        # This simulates finding an output. Replace with actual parsing of `comfyui_outputs`.
        # For this example, assume ComfyUI API told us about "some_video.mp4" in subfolder "temp_xyz"
        placeholder_comfy_subfolder = "temp_xyz_output_from_comfy"
        placeholder_comfy_filename = "placeholder_comfy_video.mp4"
        
        # Ensure the placeholder subfolder exists within the configured ComfyUI output path for the dummy file creation to succeed
        # In a real scenario, ComfyUI would create this subfolder.
        if COMFYUI_OUTPUT_PATH_CONFIG: # Check added to ensure it's not None
            full_placeholder_comfy_dir = COMFYUI_OUTPUT_PATH_CONFIG / placeholder_comfy_subfolder
            full_placeholder_comfy_dir.mkdir(parents=True, exist_ok=True) # Create if doesn't exist
            generated_final_file_path = full_placeholder_comfy_dir / placeholder_comfy_filename
            
            # For this placeholder, let's simulate a file being created by ComfyUI in its output folder.
            with open(generated_final_file_path, "w") as f: f.write("dummy comfyui output content from its real output dir")
            dprint(f"[Task ID: {task_id}] Output file identification (placeholder). Assumed output from ComfyUI: {generated_final_file_path}")
        else:
            # This case should ideally be caught by the COMFYUI_OUTPUT_PATH_CONFIG check at the start of this section
            # or the task should fail gracefully if it cannot determine the path.
            generated_final_file_path = None # Cannot form path if COMFYUI_OUTPUT_PATH_CONFIG is None
            dprint(f"[Task ID: {task_id}] COMFYUI_OUTPUT_PATH_CONFIG is None, cannot determine placeholder output path.")
        # --- End Temporary placeholder --- 

        if generated_final_file_path and generated_final_file_path.exists() and generated_final_file_path.stat().st_size > 0:
            # --- 7. Move/Upload the final output file ---
            if DB_TYPE == "sqlite":
                custom_output_path_str = task_params_dict.get("output_path")
                if custom_output_path_str:
                    final_video_path = Path(custom_output_path_str).expanduser().resolve()
                    final_video_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_sub_dir_val = task_params_dict.get("output_sub_dir", task_id) # Use task_id as fallback subdir
                    output_sub_dir_path = Path(output_sub_dir_val)
                    if output_sub_dir_path.is_absolute():
                        final_output_dir = output_sub_dir_path
                    else:
                        final_output_dir = main_output_dir_base / output_sub_dir_path
                    final_output_dir.mkdir(parents=True, exist_ok=True)
                    final_video_path = final_output_dir / f"{task_id}_comfy_output{generated_final_file_path.suffix}"
                
                shutil.move(str(generated_final_file_path), str(final_video_path))
                output_location_to_db = str(final_video_path.resolve())
                print(f"[Task ID: {task_id}] ComfyUI Output (SQLite) saved to: {output_location_to_db}")
            
            elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
                encoded_file_name = urllib.parse.quote(generated_final_file_path.name)
                object_name = f"{task_id}/{encoded_file_name}" 
                public_url = upload_to_supabase_storage(generated_final_file_path, object_name, SUPABASE_VIDEO_BUCKET)
                if public_url:
                    output_location_to_db = public_url
                    print(f"[Task ID: {task_id}] ComfyUI Output (Supabase) uploaded. URL: {public_url}")
                else:
                    print(f"[WARNING Task ID: {task_id}] Supabase upload failed for ComfyUI output.")
                    # generation_success remains False if upload fails
            else:
                print(f"[WARNING Task ID: {task_id}] ComfyUI output generated but DB_TYPE ({DB_TYPE}) not supported for auto-handling or Supabase client not configured.")
            
            generation_success = True # Set to true if output handled
        else:
            print(f"[ERROR Task ID: {task_id}] ComfyUI execution did not produce a valid output file at {generated_final_file_path} or file is empty.")
            generation_success = False

    except requests.exceptions.RequestException as e_req:
        print(f"[ERROR Task ID: {task_id}] ComfyUI API request failed: {e_req}")
        traceback.print_exc()
        generation_success = False
        output_location_to_db = f"ComfyUI API Error: {e_req}"
    except ValueError as e_val:
        print(f"[ERROR Task ID: {task_id}] Value error during ComfyUI task processing: {e_val}")
        traceback.print_exc()
        generation_success = False
        output_location_to_db = f"ComfyUI Processing Value Error: {e_val}"
    except Exception as e:
        print(f"[ERROR Task ID: {task_id}] Unhandled exception during ComfyUI workflow execution: {e}")
        traceback.print_exc()
        generation_success = False
        output_location_to_db = f"ComfyUI Unhandled Exception: {e}"
    finally:
        # Clean up the temporary ComfyUI output directory
        try:
            if Path(temp_comfy_output_dir).exists():
                shutil.rmtree(temp_comfy_output_dir)
                dprint(f"[Task ID: {task_id}] Cleaned up temporary ComfyUI directory: {temp_comfy_output_dir}")
        except Exception as e_clean:
            print(f"[WARNING Task ID: {task_id}] Failed to clean up temp ComfyUI directory {temp_comfy_output_dir}: {e_clean}")

    print(f"--- Finished ComfyUI Workflow task ID: {task_id} (Success: {generation_success}) ---")
    return generation_success, output_location_to_db

# --- SM_RESTRUCTURE: New Handler Functions for Travel Tasks ---
def _handle_travel_orchestrator_task(task_params_from_db: dict, main_output_dir_base: Path, orchestrator_task_id_str: str):
    """Handles the main 'travel_orchestrator' task.
    Its primary role is to parse the orchestration details and enqueue the FIRST 'travel_segment' task.
    """
    dprint(f"_handle_travel_orchestrator_task: Starting for {orchestrator_task_id_str}")
    generation_success = False # Represents success of *orchestration step*, not full travel
    output_location_for_orchestrator_db = None # Orchestrator itself doesn't produce a final video

    try:
        # The actual orchestrator payload is nested inside 'params' from the DB row
        if 'orchestrator_details' not in task_params_from_db:
            msg = f"[ERROR Task ID: {orchestrator_task_id_str}] 'orchestrator_details' not found in task_params_from_db."
            print(msg)
            return False, msg
        
        orchestrator_payload = task_params_from_db['orchestrator_details']
        dprint(f"Orchestrator payload for {orchestrator_task_id_str}: {json.dumps(orchestrator_payload, indent=2, default=str)}")

        run_id = orchestrator_payload.get("run_id", orchestrator_task_id_str) # Fallback to task_id if run_id missing
        
        # Ensure main_output_dir_for_run from payload is used as the base for current_run_output_dir
        # Fallback to main_output_dir_base (server's default) if not in payload, though it should be.
        base_dir_for_this_run_str = orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
        current_run_output_dir_name = f"travel_run_{run_id}"
        # Create current_run_output_dir under the directory specified by main_output_dir_for_run from the orchestrator payload
        current_run_output_dir = Path(base_dir_for_this_run_str) / current_run_output_dir_name
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Orchestrator {orchestrator_task_id_str}: Base output directory for this run: {current_run_output_dir.resolve()}")

        # --- Enqueue the first segment task --- 
        segment_index = 0 # Start with the first segment
        num_new_segments = orchestrator_payload.get("num_new_segments_to_generate", 0)

        if num_new_segments <= 0:
            msg = f"[WARNING Task ID: {orchestrator_task_id_str}] No new segments to generate based on orchestrator payload. Orchestration complete (vacuously)."
            print(msg)
            return True, msg

        first_segment_task_id = sm_generate_unique_task_id(f"travel_seg_{run_id}_{segment_index:02d}_")
        
        # Prepare VACE refs for this specific segment
        vace_refs_for_this_segment = [
            ref for ref in orchestrator_payload.get("vace_image_refs_prepared", [])
            if (ref.get("type") == "initial" and ref.get("segment_idx_for_naming") == segment_index) or \
               (ref.get("type") == "final" and ref.get("segment_idx_for_naming") == segment_index)
        ]

        segment_task_params = {
            "segment_task_id": first_segment_task_id,
            "orchestrator_task_id_ref": orchestrator_task_id_str, # Reference to this orchestrator task
            "orchestrator_run_id": run_id, 
            "segment_index": segment_index,
            "is_first_segment": True,
            "is_last_segment": (segment_index == num_new_segments - 1),
            
            "full_orchestrator_payload": orchestrator_payload, # Pass the whole orchestrator payload

            # Per-segment details (extracted for convenience, but segment handler can also get from full_orchestrator_payload)
            "input_image_paths_resolved": orchestrator_payload["input_image_paths_resolved"], # Needed for anchor logic
            "continue_from_video_resolved_path": orchestrator_payload.get("continue_from_video_resolved_path"), # Needed for anchor logic
            "base_prompt": orchestrator_payload["base_prompts_expanded"][segment_index],
            "negative_prompt": orchestrator_payload["negative_prompts_expanded"][segment_index],
            "segment_frames_target": orchestrator_payload["segment_frames_expanded"][segment_index],
            "frame_overlap_with_next": orchestrator_payload["frame_overlap_expanded"][segment_index],
            "frame_overlap_from_previous": orchestrator_payload["frame_overlap_expanded"][0] if orchestrator_payload.get("continue_from_video_resolved_path") and segment_index == 0 else 0,
            "vace_image_refs_for_segment": vace_refs_for_this_segment,

            "current_run_base_output_dir": str(current_run_output_dir.resolve()), # Base for segment's specific output folder
            "parsed_resolution_wh": orchestrator_payload["parsed_resolution_wh"],
            "model_name": orchestrator_payload["model_name"],
            "seed_to_use": orchestrator_payload.get("seed_base", 12345) + segment_index, # Ensure seed_base exists or default
            "execution_engine": orchestrator_payload["execution_engine"],
            # ComfyUI specific workflow paths if engine is ComfyUI, segment handler will use these
            "comfyui_workflow_path_override": orchestrator_payload.get("original_task_args", {}).get("comfyui_workflow_path_for_travel"),
            "comfyui_server_address_override": orchestrator_payload.get("original_task_args", {}).get("comfyui_server_address_for_travel"),

            # Pass through common args that segment task (and its WGP/Comfy sub-task) will need
            "use_causvid_lora": orchestrator_payload.get("use_causvid_lora", False),
            "cfg_star_switch": orchestrator_payload.get("cfg_star_switch", 0),
            "cfg_zero_step": orchestrator_payload.get("cfg_zero_step", -1),
            "params_json_str_override": orchestrator_payload.get("params_json_str_override"),
            "fps_helpers": orchestrator_payload["fps_helpers"],
            "last_frame_duplication": orchestrator_payload["last_frame_duplication"],
            "fade_in_params_json_str": orchestrator_payload["fade_in_params_json_str"],
            "fade_out_params_json_str": orchestrator_payload["fade_out_params_json_str"],
            "subsequent_starting_strength_adjustment": orchestrator_payload.get("subsequent_starting_strength_adjustment", 0.0),
            "desaturate_subsequent_starting_frames": orchestrator_payload.get("desaturate_subsequent_starting_frames", 0.0),
            "adjust_brightness_subsequent_starting_frames": orchestrator_payload.get("adjust_brightness_subsequent_starting_frames", 0.0),
            "after_first_post_generation_saturation": orchestrator_payload.get("after_first_post_generation_saturation"), # For segment post-processing
            "crossfade_sharp_amt": orchestrator_payload.get("crossfade_sharp_amt", 0.3), # For stitcher

            "upscale_factor": orchestrator_payload.get("upscale_factor", 0.0), # For stitcher
            "upscale_model_name": orchestrator_payload.get("upscale_model_name"), # For stitcher

            "debug_mode_enabled": orchestrator_payload.get("debug_mode_enabled", False),
            "skip_cleanup_enabled": orchestrator_payload.get("skip_cleanup_enabled", False) # For stitcher
        }

        # Determine DB path from global config (SQLITE_DB_PATH or SUPABASE_CLIENT implicitly)
        db_path_for_add = SQLITE_DB_PATH if DB_TYPE == "sqlite" else None # Supabase uses client
        
        sm_add_task_to_db(
            task_payload=segment_task_params,
            db_path_str=db_path_for_add, # Pass None if Supabase, it uses global client
            task_type="travel_segment", # This is the task_type for the segment generation
            task_id_override=first_segment_task_id
        )
        msg = f"Orchestrator {orchestrator_task_id_str}: Successfully enqueued first 'travel_segment' task (ID: {first_segment_task_id})."
        print(msg)
        generation_success = True
        output_location_for_orchestrator_db = msg

    except Exception as e:
        msg = f"[ERROR Task ID: {orchestrator_task_id_str}] Failed during travel orchestration: {e}"
        print(msg)
        traceback.print_exc()
        generation_success = False
        output_location_for_orchestrator_db = msg
    
    return generation_success, output_location_for_orchestrator_db

def _handle_travel_segment_task(wgp_mod, task_params_from_db: dict, main_output_dir_base: Path, segment_task_id_str: str):
    dprint(f"_handle_travel_segment_task: Starting for {segment_task_id_str}")
    # task_params_from_db contains what was enqueued for this specific segment, 
    # including potentially 'full_orchestrator_payload'.
    segment_params = task_params_from_db 
    generation_success = False # Success of the WGP/Comfy sub-task for this segment
    final_segment_video_output_path_str = None # Output of the WGP sub-task

    try:
        # --- 1. Initialization & Parameter Extraction ---
        orchestrator_task_id_ref = segment_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = segment_params.get("orchestrator_run_id")
        segment_idx = segment_params.get("segment_index")

        if orchestrator_task_id_ref is None or orchestrator_run_id is None or segment_idx is None:
            msg = f"Segment task {segment_task_id_str} missing critical orchestrator refs or segment_index."
            print(f"[ERROR Task {segment_task_id_str}]: {msg}")
            return False, msg

        full_orchestrator_payload = segment_params.get("full_orchestrator_payload")
        if not full_orchestrator_payload:
            dprint(f"Segment {segment_idx}: full_orchestrator_payload not in direct params. Querying orchestrator task {orchestrator_task_id_ref}")
            orchestrator_task_raw_params_json = None
            if DB_TYPE == "sqlite":
                def _get_orc_params(conn):
                    cursor = conn.cursor()
                    cursor.execute("SELECT params FROM tasks WHERE task_id = ?", (orchestrator_task_id_ref,))
                    row = cursor.fetchone()
                    return row[0] if row else None
                orchestrator_task_raw_params_json = execute_sqlite_with_retry(SQLITE_DB_PATH, _get_orc_params)
            elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
                orc_resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("params").eq("task_id", orchestrator_task_id_ref).execute()
                if orc_resp.data: orchestrator_task_raw_params_json = orc_resp.data[0]["params"]
            
            if orchestrator_task_raw_params_json:
                try: 
                    fetched_params = json.loads(orchestrator_task_raw_params_json) if isinstance(orchestrator_task_raw_params_json, str) else orchestrator_task_raw_params_json
                    full_orchestrator_payload = fetched_params.get("orchestrator_details")
                    if not full_orchestrator_payload:
                        raise ValueError("'orchestrator_details' key missing in fetched orchestrator task params.")
                    dprint(f"Segment {segment_idx}: Successfully fetched orchestrator_details from DB.")
                except Exception as e_fetch_orc:
                    msg = f"Segment {segment_idx}: Failed to fetch/parse orchestrator_details from DB for task {orchestrator_task_id_ref}: {e_fetch_orc}"
                    print(f"[ERROR Task {segment_task_id_str}]: {msg}")
                    return False, msg
            else:
                msg = f"Segment {segment_idx}: Could not retrieve params for orchestrator task {orchestrator_task_id_ref}. Cannot proceed."
                print(f"[ERROR Task {segment_task_id_str}]: {msg}")
                return False, msg
        
        # Now full_orchestrator_payload is guaranteed to be populated or we've exited.
        current_run_base_output_dir_str = segment_params.get("current_run_base_output_dir")
        if not current_run_base_output_dir_str: # Should be passed by orchestrator/prev segment
            current_run_base_output_dir_str = full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
            current_run_base_output_dir_str = str(Path(current_run_base_output_dir_str) / f"travel_run_{orchestrator_run_id}")

        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        segment_processing_dir = current_run_base_output_dir / f"segment_{segment_idx:02d}_{segment_task_id_str[:8]}"
        segment_processing_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Segment {segment_idx} (Task {segment_task_id_str}): Processing in {segment_processing_dir.resolve()}")

        # --- 2. Guide Video Preparation ---
        actual_guide_video_path_for_wgp: Path | None = None
        path_to_previous_segment_video_output_for_guide: str | None = None
        
        is_first_segment = segment_params.get("is_first_segment", segment_idx == 0) # is_first_segment should be reliable
        is_first_segment_from_scratch = is_first_segment and not full_orchestrator_payload.get("continue_from_video_resolved_path")
        is_first_new_segment_after_continue = is_first_segment and full_orchestrator_payload.get("continue_from_video_resolved_path")
        is_subsequent_segment = not is_first_segment

        parsed_res_wh = full_orchestrator_payload["parsed_resolution_wh"]
        fps_helpers = full_orchestrator_payload["fps_helpers"]
        fade_in_duration_str = full_orchestrator_payload["fade_in_params_json_str"]
        fade_out_duration_str = full_orchestrator_payload["fade_out_params_json_str"]
        last_frame_duplication_count = full_orchestrator_payload["last_frame_duplication"]
        
        # Define gray_frame_bgr here for use in subsequent segment strength adjustment
        gray_frame_bgr = sm_create_color_frame(parsed_res_wh, (128, 128, 128))

        try: # Parsing fade params
            fade_in_p = json.loads(fade_in_duration_str)
            fi_low, fi_high, fi_curve, fi_factor = float(fade_in_p.get("low_point",0)), float(fade_in_p.get("high_point",1)), str(fade_in_p.get("curve_type","ease_in_out")), float(fade_in_p.get("duration_factor",0))
        except Exception as e_fade_in:
            fi_low, fi_high, fi_curve, fi_factor = 0.0,1.0,"ease_in_out",0.0
            dprint(f"Seg {segment_idx} Warn: Using default fade-in params due to parse error on '{fade_in_duration_str}': {e_fade_in}")
        try:
            fade_out_p = json.loads(fade_out_duration_str)
            fo_low, fo_high, fo_curve, fo_factor = float(fade_out_p.get("low_point",0)), float(fade_out_p.get("high_point",1)), str(fade_out_p.get("curve_type","ease_in_out")), float(fade_out_p.get("duration_factor",0))
        except Exception as e_fade_out:
            fo_low, fo_high, fo_curve, fo_factor = 0.0,1.0,"ease_in_out",0.0
            dprint(f"Seg {segment_idx} Warn: Using default fade-out params due to parse error on '{fade_out_duration_str}': {e_fade_out}")

        if is_first_new_segment_after_continue:
            path_to_previous_segment_video_output_for_guide = full_orchestrator_payload.get("continue_from_video_resolved_path")
            if not path_to_previous_segment_video_output_for_guide or not Path(path_to_previous_segment_video_output_for_guide).exists():
                msg = f"Seg {segment_idx}: Continue video path {path_to_previous_segment_video_output_for_guide} invalid."
                print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        elif is_subsequent_segment:
            path_to_previous_segment_video_output_for_guide = segment_params.get("path_to_previous_segment_output")
            if not path_to_previous_segment_video_output_for_guide:
                prev_seg_task_id_to_find = get_previous_segment_task_id(orchestrator_run_id, segment_idx - 1)
                if prev_seg_task_id_to_find:
                    path_to_previous_segment_video_output_for_guide = get_task_output_location_from_db(prev_seg_task_id_to_find)
            if not path_to_previous_segment_video_output_for_guide or not Path(path_to_previous_segment_video_output_for_guide).exists():
                msg = f"Seg {segment_idx}: Prev segment output for guide invalid/not found. Expected from prev task output. Path: {path_to_previous_segment_video_output_for_guide}"
                print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        
        try: # Guide Video Creation Block
            guide_video_base_name = f"s{segment_idx}_guide_vid"
            # Ensure guide video is saved in the current segment's processing directory
            actual_guide_video_path_for_wgp = sm_get_unique_target_path(segment_processing_dir, guide_video_base_name, ".mp4")
            
            # segment_frames_target and frame_overlap_from_previous should be in segment_params, passed by orchestrator/prev segment
            base_duration_new_content_for_guide = segment_params.get("segment_frames_target", full_orchestrator_payload["segment_frames_expanded"][segment_idx])
            overlap_connecting_to_previous_for_guide = segment_params.get("frame_overlap_from_previous", 0) # Default to 0 if not set
            guide_video_total_frames = base_duration_new_content_for_guide + overlap_connecting_to_previous_for_guide

            if guide_video_total_frames <= 0:
                dprint(f"Task {segment_task_id_str}: Guide video frames {guide_video_total_frames}. No guide will be created."); actual_guide_video_path_for_wgp = None
            else:
                frames_for_guide_list = [sm_create_color_frame(parsed_res_wh, (128,128,128)).copy() for _ in range(guide_video_total_frames)]
                input_images_resolved = full_orchestrator_payload["input_image_paths_resolved"]
                end_anchor_img_path_str: str
                if full_orchestrator_payload.get("continue_from_video_resolved_path"): # Number of input images matches number of new segments
                    if segment_idx < len(input_images_resolved):
                        end_anchor_img_path_str = input_images_resolved[segment_idx]
                    else:
                        raise ValueError(f"Seg {segment_idx}: End anchor index {segment_idx} out of bounds for input_images ({len(input_images_resolved)}) with continue_from_video.")
                else: # Not continuing from video, so number of input images is num_segments + 1
                    if (segment_idx + 1) < len(input_images_resolved):
                        end_anchor_img_path_str = input_images_resolved[segment_idx + 1]
                    else:
                        raise ValueError(f"Seg {segment_idx}: End anchor index {segment_idx+1} out of bounds for input_images ({len(input_images_resolved)}) when not continuing from video.")
                
                end_anchor_frame_np = sm_image_to_frame(end_anchor_img_path_str, parsed_res_wh)
                if end_anchor_frame_np is None: raise ValueError(f"Failed to load end anchor image: {end_anchor_img_path_str}")
                num_end_anchor_duplicates = last_frame_duplication_count + 1
                start_anchor_frame_np = None

                if is_first_segment_from_scratch:
                    start_anchor_img_path_str = input_images_resolved[0]
                    start_anchor_frame_np = sm_image_to_frame(start_anchor_img_path_str, parsed_res_wh)
                    if start_anchor_frame_np is None: raise ValueError(f"Failed to load start anchor: {start_anchor_img_path_str}")
                    if frames_for_guide_list: frames_for_guide_list[0] = start_anchor_frame_np.copy()
                    # ... (Ported fade logic for start_anchor_frame_np from original file, using fo_low, fo_high, fo_curve, fo_factor) ...
                    pot_max_idx_start_fade = guide_video_total_frames - num_end_anchor_duplicates - 1
                    avail_frames_start_fade = max(0, pot_max_idx_start_fade) 
                    num_start_fade_steps = int(avail_frames_start_fade * fo_factor)
                    if num_start_fade_steps > 0:
                        actual_start_fade_end_idx = min(num_start_fade_steps -1 , pot_max_idx_start_fade -1) 
                        easing_fn_out = sm_get_easing_function(fo_curve)
                        for k_fo in range(num_start_fade_steps):
                            idx_in_guide = 1 + k_fo
                            if idx_in_guide > actual_start_fade_end_idx +1 or idx_in_guide >= guide_video_total_frames: break
                            alpha_lin = 1.0 - ((k_fo + 1) / float(num_start_fade_steps))
                            e_alpha = fo_low + (fo_high - fo_low) * easing_fn_out(alpha_lin)
                            e_alpha = np.clip(e_alpha, 0.0, 1.0)
                            frames_for_guide_list[idx_in_guide] = cv2.addWeighted(frames_for_guide_list[idx_in_guide].astype(np.float32), 1.0 - e_alpha, start_anchor_frame_np.astype(np.float32), e_alpha, 0).astype(np.uint8)
                    
                    # ... (Ported fade logic for end_anchor_frame_np from original file, using fi_low, fi_high, fi_curve, fi_factor) ...
                    min_idx_end_fade = 1 
                    max_idx_end_fade = guide_video_total_frames - num_end_anchor_duplicates - 1
                    avail_frames_end_fade = max(0, max_idx_end_fade - min_idx_end_fade + 1)
                    num_end_fade_steps = int(avail_frames_end_fade * fi_factor)
                    if num_end_fade_steps > 0:
                        actual_end_fade_start_idx = max(min_idx_end_fade, max_idx_end_fade - num_end_fade_steps + 1)
                        easing_fn_in = sm_get_easing_function(fi_curve)
                        for k_fi in range(num_end_fade_steps):
                            idx_in_guide = actual_end_fade_start_idx + k_fi
                            if idx_in_guide > max_idx_end_fade or idx_in_guide >= guide_video_total_frames: break
                            alpha_lin = (k_fi + 1) / float(num_end_fade_steps)
                            e_alpha = fi_low + (fi_high - fi_low) * easing_fn_in(alpha_lin)
                            e_alpha = np.clip(e_alpha, 0.0, 1.0)
                            base_f = frames_for_guide_list[idx_in_guide]
                            frames_for_guide_list[idx_in_guide] = cv2.addWeighted(base_f.astype(np.float32), 1.0 - e_alpha, end_anchor_frame_np.astype(np.float32), e_alpha, 0).astype(np.uint8)
                    elif fi_factor > 0 and avail_frames_end_fade > 0: 
                        for k_fill in range(min_idx_end_fade, max_idx_end_fade + 1):
                            if k_fill < guide_video_total_frames: frames_for_guide_list[k_fill] = end_anchor_frame_np.copy()

                elif path_to_previous_segment_video_output_for_guide: # Continued or Subsequent
                    prev_vid_total_frames, _ = sm_get_video_frame_count_and_fps(path_to_previous_segment_video_output_for_guide)
                    if prev_vid_total_frames is None: raise ValueError("Could not get frame count of previous video for guide.")
                    actual_overlap_to_use = min(overlap_connecting_to_previous_for_guide, prev_vid_total_frames)
                    start_extraction_idx = max(0, prev_vid_total_frames - actual_overlap_to_use)
                    overlap_frames_raw = sm_extract_frames_from_video(path_to_previous_segment_video_output_for_guide, start_extraction_idx, actual_overlap_to_use)
                    frames_read_for_overlap = 0
                    for k, frame_fp in enumerate(overlap_frames_raw): # frame_fp is frame_from_prev
                        if k >= guide_video_total_frames: break
                        if frame_fp.shape[1]!=parsed_res_wh[0] or frame_fp.shape[0]!=parsed_res_wh[1]: frame_fp = cv2.resize(frame_fp, parsed_res_wh, interpolation=cv2.INTER_AREA)
                        frames_for_guide_list[k] = frame_fp.copy()
                        frames_read_for_overlap +=1
                    
                    # ... (Ported strength/desat/brightness logic from original, using segment_params/full_orchestrator_payload for settings) ...
                    strength_adj = full_orchestrator_payload.get("subsequent_starting_strength_adjustment", 0.0)
                    desat_factor = full_orchestrator_payload.get("desaturate_subsequent_starting_frames", 0.0)
                    bright_adj = full_orchestrator_payload.get("adjust_brightness_subsequent_starting_frames", 0.0)
                    if frames_read_for_overlap > 0:
                        if fo_factor > 0.0:
                            num_init_fade_steps = min(int(frames_read_for_overlap * fo_factor), frames_read_for_overlap)
                            easing_fn_fo_ol = sm_get_easing_function(fo_curve)
                            for k_fo_ol in range(num_init_fade_steps):
                                alpha_l = 1.0 - ((k_fo_ol + 1) / float(num_init_fade_steps))
                                eff_s = fo_low + (fo_high - fo_low) * easing_fn_fo_ol(alpha_l)
                                eff_s += strength_adj
                                eff_s = np.clip(eff_s,0,1)
                                base_f=frames_for_guide_list[k_fo_ol]
                                frames_for_guide_list[k_fo_ol] = cv2.addWeighted(gray_frame_bgr.astype(np.float32),1-eff_s,base_f.astype(np.float32),eff_s,0).astype(np.uint8)
                                if desat_factor > 0:
                                    g=cv2.cvtColor(frames_for_guide_list[k_fo_ol],cv2.COLOR_BGR2GRAY)
                                    gb=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
                                    frames_for_guide_list[k_fo_ol]=cv2.addWeighted(frames_for_guide_list[k_fo_ol],1-desat_factor,gb,desat_factor,0)
                                if bright_adj!=0:
                                    frames_for_guide_list[k_fo_ol]=sm_adjust_frame_brightness(frames_for_guide_list[k_fo_ol],bright_adj)
                        else:
                            eff_s=fo_high+strength_adj; eff_s=np.clip(eff_s,0,1)
                            if abs(eff_s-1.0)>1e-5 or desat_factor>0 or bright_adj!=0:
                                for k_s_ol in range(frames_read_for_overlap):
                                    base_f=frames_for_guide_list[k_s_ol];frames_for_guide_list[k_s_ol]=cv2.addWeighted(gray_frame_bgr.astype(np.float32),1-eff_s,base_f.astype(np.float32),eff_s,0).astype(np.uint8)
                                    if desat_factor>0: g=cv2.cvtColor(frames_for_guide_list[k_s_ol],cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR);frames_for_guide_list[k_s_ol]=cv2.addWeighted(frames_for_guide_list[k_s_ol],1-desat_factor,gb,desat_factor,0)
                                    if bright_adj!=0: frames_for_guide_list[k_s_ol]=sm_adjust_frame_brightness(frames_for_guide_list[k_s_ol],bright_adj)
                    # ... (Ported fade-in logic for end_anchor_frame_np for subsequent segments) ...
                    min_idx_efs = frames_read_for_overlap; max_idx_efs = guide_video_total_frames - num_end_anchor_duplicates - 1
                    avail_f_efs = max(0, max_idx_efs - min_idx_efs + 1); num_efs_steps = int(avail_f_efs * fi_factor)
                    if num_efs_steps > 0:
                        actual_efs_start_idx = max(min_idx_efs, max_idx_efs - num_efs_steps + 1)
                        easing_fn_in_s = sm_get_easing_function(fi_curve)
                        for k_fi_s in range(num_efs_steps):
                            idx = actual_efs_start_idx+k_fi_s
                            if idx > max_idx_efs or idx >= guide_video_total_frames: break
                            if idx < min_idx_efs: continue
                            alpha_l=(k_fi_s+1)/float(num_efs_steps);e_alpha=fi_low+(fi_high-fi_low)*easing_fn_in_s(alpha_l);e_alpha=np.clip(e_alpha,0,1)
                            base_f=frames_for_guide_list[idx];frames_for_guide_list[idx]=cv2.addWeighted(base_f.astype(np.float32),1-e_alpha,end_anchor_frame_np.astype(np.float32),e_alpha,0).astype(np.uint8)
                    elif fi_factor > 0 and avail_f_efs > 0:
                        for k_fill in range(min_idx_efs, max_idx_efs + 1):
                            if k_fill < guide_video_total_frames: frames_for_guide_list[k_fill] = end_anchor_frame_np.copy()
                
                # Duplication & final first frame for all types
                if guide_video_total_frames > 0:
                    for k_dup in range(min(num_end_anchor_duplicates, guide_video_total_frames)):
                        idx_s = guide_video_total_frames - 1 - k_dup
                        if idx_s >= 0: 
                            frames_for_guide_list[idx_s] = end_anchor_frame_np.copy()
                        else: 
                            break # break the loop if idx_s is out of bounds
                if is_first_segment_from_scratch and guide_video_total_frames > 0 and start_anchor_frame_np is not None:
                    frames_for_guide_list[0] = start_anchor_frame_np.copy()

                if frames_for_guide_list:
                    guide_video_file_path = sm_create_video_from_frames_list(frames_for_guide_list, actual_guide_video_path_for_wgp, fps_helpers, parsed_res_wh)
                    if guide_video_file_path and guide_video_file_path.exists(): actual_guide_video_path_for_wgp = guide_video_file_path
                    else: print(f"ERROR: Failed to create guide video."); actual_guide_video_path_for_wgp = None
                else: actual_guide_video_path_for_wgp = None; dprint("No frames for guide video.")
        except Exception as e_guide:
            print(f"ERROR Task {segment_task_id_str} guide prep: {e_guide}"); traceback.print_exc(); actual_guide_video_path_for_wgp = None
        
        # --- Enqueue WGP Generation as a Sub-Task ---
        if actual_guide_video_path_for_wgp is None and not (is_first_segment_from_scratch or path_to_previous_segment_video_output_for_guide):
            # If guide creation failed AND it was essential (not first segment that could run guideless, or no prev video for subsequent)
            print(f"[ERROR Task {segment_task_id_str}]: Essential guide video failed or not possible. Cannot proceed with WGP sub-task.")
            return False, "Guide video creation failed for segment that requires it."
            
        base_duration_wgp = segment_params["segment_frames_target"]
        overlap_from_previous_wgp = segment_params["frame_overlap_from_previous"]
        current_segment_total_frames_unquantized_wgp = base_duration_wgp + overlap_from_previous_wgp
        final_frames_for_wgp_generation = current_segment_total_frames_unquantized_wgp
        current_wgp_engine = full_orchestrator_payload["execution_engine"]

        if current_wgp_engine == "wgp": # Quantization specific to wgp engine
            is_ltxv_m = "ltxv" in full_orchestrator_payload["model_name"].lower()
            latent_s = 8 if is_ltxv_m else 4
            quantized_wgp_f = (current_segment_total_frames_unquantized_wgp // latent_s) * latent_s + 1
            if quantized_wgp_f != current_segment_total_frames_unquantized_wgp: dprint(f"Quantizing WGP: {current_segment_total_frames_unquantized_wgp} to {quantized_wgp_f}")
            final_frames_for_wgp_generation = quantized_wgp_f

        wgp_sub_task_id = sm_generate_unique_task_id(f"wgp_sub_{segment_task_id_str[:10]}_")
        wgp_payload = {
            "task_id": wgp_sub_task_id,
            "model": full_orchestrator_payload["model_name"],
            "prompt": segment_params["base_prompt"],
            "negative_prompt": segment_params["negative_prompt"],
            "resolution": f"{parsed_res_wh[0]}x{parsed_res_wh[1]}",
            "frames": final_frames_for_wgp_generation,
            "seed": segment_params["seed_to_use"],
            "output_sub_dir": str(segment_processing_dir.relative_to(main_output_dir_base)),
            "video_guide_path": str(actual_guide_video_path_for_wgp.resolve()) if actual_guide_video_path_for_wgp else None,
            "use_causvid_lora": full_orchestrator_payload.get("use_causvid_lora", False),
            "cfg_star_switch": full_orchestrator_payload.get("cfg_star_switch", 0),
            "cfg_zero_step": full_orchestrator_payload.get("cfg_zero_step", -1),
            "image_refs_paths": [ref["processed_path"] for ref in segment_params.get("vace_image_refs_for_segment", []) if ref.get("processed_path")],
            # ComfyUI specific if engine is comfyui
            "comfyui_workflow_path": full_orchestrator_payload.get("comfyui_workflow_path_override", "placeholder_comfy_workflow_for_travel.json"),
            "comfyui_server_address": full_orchestrator_payload.get("comfyui_server_address_override", "http://127.0.0.1:8188"),
        }
        if full_orchestrator_payload.get("params_json_str_override"):
            try:
                additional_p = json.loads(full_orchestrator_payload["params_json_str_override"])
                additional_p.pop("frames", None); additional_p.pop("video_length", None)
                wgp_payload.update(additional_p)
            except Exception as e_json: dprint(f"Error merging override params: {e_json}")
        
        # If ComfyUI engine, structure inputs for it, especially image paths
        if current_wgp_engine == "comfyui":
            comfy_inputs = {
                "guide_video_path": wgp_payload["video_guide_path"],
                "prompt": wgp_payload["prompt"],
                "negative_prompt": wgp_payload["negative_prompt"],
                "seed": wgp_payload["seed"],
                "frames": wgp_payload["frames"],
                 # Start/End images for ComfyUI. Need to determine correct ones based on segment_idx.
            }
            # Determine start_image_path for ComfyUI
            if is_first_segment_from_scratch:
                comfy_inputs["start_image_path"] = full_orchestrator_payload["input_image_paths_resolved"][0]
            elif path_to_previous_segment_video_output_for_guide: # Continued or subsequent
                # Extract first frame of previous output as start_image for ComfyUI
                # This needs a helper: extract_specific_frame_ffmpeg from common_utils
                # from .common_utils import extract_specific_frame_ffmpeg as sm_extract_specific_frame_ffmpeg
                # temp_start_frame_img = sm_get_unique_target_path(segment_processing_dir, f"s{segment_idx}_comfy_start_frame", ".png")
                # prev_vid_fps = sm_get_video_frame_count_and_fps(path_to_previous_segment_video_output_for_guide)[1] or fps_helpers
                # if sm_extract_specific_frame_ffmpeg(path_to_previous_segment_video_output_for_guide, 0, temp_start_frame_img, prev_vid_fps):
                #     comfy_inputs["start_image_path"] = str(temp_start_frame_img.resolve())
                # else: dprint("Failed to extract start frame for ComfyUI from previous video.")
                # For now, ComfyUI workflow will need to handle getting start frame from prev video if path given, or use first guide frame.
                 comfy_inputs["previous_video_path_for_start_frame"] = path_to_previous_segment_video_output_for_guide

            comfy_inputs["end_image_path"] = end_anchor_img_path_str # Determined earlier for guide
            wgp_payload["comfyui_inputs"] = comfy_inputs

        dprint(f"Seg {segment_idx}: Enqueuing {current_wgp_engine} sub-task {wgp_sub_task_id} with payload: {json.dumps(wgp_payload, default=str)}")
        db_path_for_wgp_add = SQLITE_DB_PATH if DB_TYPE == "sqlite" else None
        sm_add_task_to_db(wgp_payload, db_path_for_wgp_add, current_wgp_engine, wgp_sub_task_id)
        
        print(f"Seg {segment_idx}: Waiting for {current_wgp_engine} sub-task {wgp_sub_task_id}...")
        from Wan2GP.sm_functions.common_utils import poll_task_status as sm_poll_status # Re-import to ensure correct scope
        raw_wgp_output_video_loc = sm_poll_status(wgp_sub_task_id, db_path_for_wgp_add, 
                                                  full_orchestrator_payload.get("poll_interval",15), 
                                                  full_orchestrator_payload.get("poll_timeout", 30*60))

        if raw_wgp_output_video_loc and Path(raw_wgp_output_video_loc).exists():
            dprint(f"Seg {segment_idx}: WGP sub-task {wgp_sub_task_id} OK. Output: {raw_wgp_output_video_loc}")
            raw_out_path = Path(raw_wgp_output_video_loc)
            final_segment_video_output_path_str = str(raw_out_path.resolve())
            # Ensure output is in segment_processing_dir (it should be due to output_sub_dir in wgp_payload)
            if raw_out_path.parent.resolve() != segment_processing_dir.resolve():
                dprint(f"WGP output {raw_out_path} not in seg dir {segment_processing_dir}. This is unexpected.")
                # Copy if somehow it ended up elsewhere, though output_sub_dir should handle this.
                # For safety, let's ensure it's where we expect it for saturation and next step.
                final_dest_in_seg_dir = segment_processing_dir / raw_out_path.name
                try: shutil.copy2(str(raw_out_path), str(final_dest_in_seg_dir)); final_segment_video_output_path_str = str(final_dest_in_seg_dir.resolve())
                except Exception as e_final_copy: print(f"WARN: Failed to copy {raw_out_path} to {final_dest_in_seg_dir}: {e_final_copy}")
            generation_success = True
        else:
            print(f"ERROR Task {segment_task_id_str}: WGP sub-task {wgp_sub_task_id} failed/timed out/output missing.")
            generation_success = False

        # --- Post-generation Saturation ---
        if generation_success and final_segment_video_output_path_str and (is_subsequent_segment or is_first_new_segment_after_continue):
            sat_level = full_orchestrator_payload.get("after_first_post_generation_saturation")
            if sat_level is not None:
                dprint(f"Seg {segment_idx}: Applying post-gen saturation {sat_level}")
                sat_out_base = f"s{segment_idx}_final_sat_{sat_level:.2f}"
                sat_out_path = sm_get_unique_target_path(segment_processing_dir, sat_out_base, Path(final_segment_video_output_path_str).suffix)
                if sm_apply_saturation_to_video_ffmpeg(final_segment_video_output_path_str, sat_out_path, sat_level):
                    final_segment_video_output_path_str = str(sat_out_path.resolve())
                else: dprint(f"Seg {segment_idx}: Failed post-gen saturation.")

        # --- Enqueue Next Segment OR Stitch Task ---
        if generation_success and final_segment_video_output_path_str:
            db_path_next_add = SQLITE_DB_PATH if DB_TYPE == "sqlite" else None
            if not segment_params["is_last_segment"]:
                next_seg_idx = segment_idx + 1
                next_seg_task_id = sm_generate_unique_task_id(f"travel_seg_{orchestrator_run_id}_{next_seg_idx:02d}_")
                next_seg_payload = { # Copying relevant fields, ensuring full_orchestrator_payload is passed
                    "segment_task_id": next_seg_task_id, "orchestrator_task_id_ref": orchestrator_task_id_ref,
                    "orchestrator_run_id": orchestrator_run_id, "segment_index": next_seg_idx,
                    "is_first_segment": False, "is_last_segment": (next_seg_idx == full_orchestrator_payload["num_new_segments_to_generate"] - 1),
                    "path_to_previous_segment_output": final_segment_video_output_path_str,
                    "full_orchestrator_payload": full_orchestrator_payload, # CRITICAL: Pass full orchestrator payload
                     # Fields specific to this segment, derived from full_orchestrator_payload by next segment handler
                    "segment_frames_target": full_orchestrator_payload["segment_frames_expanded"][next_seg_idx],
                    "frame_overlap_from_previous": full_orchestrator_payload["frame_overlap_expanded"][segment_idx],
                    "frame_overlap_with_next": full_orchestrator_payload["frame_overlap_expanded"][next_seg_idx],
                    "base_prompt": full_orchestrator_payload["base_prompts_expanded"][next_seg_idx],
                    "negative_prompt": full_orchestrator_payload["negative_prompts_expanded"][next_seg_idx],
                    "seed_to_use": full_orchestrator_payload["seed_base"] + next_seg_idx,
                    "current_run_base_output_dir": str(current_run_base_output_dir.resolve()),
                    "parsed_resolution_wh": full_orchestrator_payload["parsed_resolution_wh"],
                    "model_name": full_orchestrator_payload["model_name"],
                    "execution_engine": full_orchestrator_payload["execution_engine"],
                    # ... (other necessary params from full_orchestrator_payload for next segment)... 
                    "vace_image_refs_for_segment": [ ref for ref in full_orchestrator_payload.get("vace_image_refs_prepared", []) if (ref.get("type") == "initial" and ref.get("segment_idx_for_naming", -1) == next_seg_idx) or (ref.get("type") == "final" and ref.get("segment_idx_for_naming", -1) == next_seg_idx)],
                    "use_causvid_lora": full_orchestrator_payload["use_causvid_lora"],
                    "cfg_star_switch": full_orchestrator_payload["cfg_star_switch"],
                    "cfg_zero_step": full_orchestrator_payload["cfg_zero_step"],
                    "params_json_str_override": full_orchestrator_payload.get("params_json_str_override"),
                    "fps_helpers": full_orchestrator_payload["fps_helpers"],
                    "last_frame_duplication": full_orchestrator_payload["last_frame_duplication"],
                    "fade_in_params_json_str": full_orchestrator_payload["fade_in_params_json_str"],
                    "fade_out_params_json_str": full_orchestrator_payload["fade_out_params_json_str"],
                    "subsequent_starting_strength_adjustment": full_orchestrator_payload.get("subsequent_starting_strength_adjustment", 0.0),
                    "desaturate_subsequent_starting_frames": full_orchestrator_payload.get("desaturate_subsequent_starting_frames", 0.0),
                    "adjust_brightness_subsequent_starting_frames": full_orchestrator_payload.get("adjust_brightness_subsequent_starting_frames", 0.0),
                    "after_first_post_generation_saturation": full_orchestrator_payload.get("after_first_post_generation_saturation"),
                    "debug_mode_enabled": full_orchestrator_payload.get("debug_mode_enabled", False)
                }
                sm_add_task_to_db(next_seg_payload, db_path_next_add, "travel_segment", next_seg_task_id)
                dprint(f"Seg {segment_idx}: Enqueued next seg ({next_seg_idx}) task {next_seg_task_id}.")
            elif segment_params["is_last_segment"]:
                stitch_task_id = sm_generate_unique_task_id(f"travel_stitch_{orchestrator_run_id}_")
                stitch_payload = { # Payloads for stitch task
                    "stitch_task_id": stitch_task_id, "orchestrator_task_id_ref": orchestrator_task_id_ref, 
                    "orchestrator_run_id": orchestrator_run_id,
                    "num_total_segments_generated": full_orchestrator_payload["num_new_segments_to_generate"],
                    "current_run_base_output_dir": str(current_run_base_output_dir.resolve()),
                    "frame_overlap_settings_expanded": full_orchestrator_payload["frame_overlap_expanded"],
                    "crossfade_sharp_amt": full_orchestrator_payload.get("crossfade_sharp_amt", 0.3),
                    "parsed_resolution_wh": full_orchestrator_payload["parsed_resolution_wh"],
                    "fps_final_video": full_orchestrator_payload["fps_helpers"],
                    "upscale_factor": full_orchestrator_payload.get("upscale_factor", 0.0),
                    "upscale_model_name": full_orchestrator_payload.get("upscale_model_name"),
                    "seed_for_upscale": full_orchestrator_payload["seed_base"] + 5000,
                    "debug_mode_enabled": full_orchestrator_payload.get("debug_mode_enabled", False),
                    "skip_cleanup_enabled": full_orchestrator_payload.get("skip_cleanup_enabled", False),
                    "initial_continued_video_path": full_orchestrator_payload.get("continue_from_video_resolved_path")
                }
                sm_add_task_to_db(stitch_payload, db_path_next_add, "travel_stitch", stitch_task_id)
                dprint(f"Seg {segment_idx}: Last. Enqueued stitch task {stitch_task_id}.")
        else:
            dprint(f"Seg {segment_idx}: WGP sub-task failed/skipped. No next task enqueued.")

        # The return for _handle_travel_segment_task is about THIS segment's WGP sub-task outcome
        return generation_success, final_segment_video_output_path_str

    except Exception as e:
        print(f"ERROR Task {segment_task_id_str}: Unexpected error during segment processing: {e}")
        traceback.print_exc()
        return False, f"Unexpected error: {str(e)[:200]}"

def _handle_travel_stitch_task(task_params_from_db: dict, main_output_dir_base: Path, stitch_task_id_str: str):
    """Handles the final 'travel_stitch' task."""
    dprint(f"_handle_travel_stitch_task: Starting for {stitch_task_id_str}")
    stitch_params = task_params_from_db # This now contains full_orchestrator_payload
    stitch_success = False
    final_video_location_for_db = None
    
    try:
        # --- 1. Initialization & Parameter Extraction --- 
        orchestrator_task_id_ref = stitch_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = stitch_params.get("orchestrator_run_id")
        full_orchestrator_payload = stitch_params.get("full_orchestrator_payload")

        if not all([orchestrator_task_id_ref, orchestrator_run_id, full_orchestrator_payload]):
            msg = f"Stitch task {stitch_task_id_str} missing critical orchestrator refs or full_orchestrator_payload."
            print(f"[ERROR Task {stitch_task_id_str}]: {msg}")
            return False, msg

        current_run_base_output_dir_str = stitch_params.get("current_run_base_output_dir", 
                                                            full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve())))
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        # If current_run_base_output_dir was the generic one, ensure it includes the run_id subfolder.
        if not str(current_run_base_output_dir.name).endswith(orchestrator_run_id):
            current_run_base_output_dir = current_run_base_output_dir / f"travel_run_{orchestrator_run_id}"
        
        stitch_processing_dir = current_run_base_output_dir / f"stitch_final_output_{stitch_task_id_str[:8]}"
        stitch_processing_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Stitch Task {stitch_task_id_str}: Processing in {stitch_processing_dir.resolve()}")

        num_expected_new_segments = full_orchestrator_payload["num_new_segments_to_generate"]
        parsed_res_wh = full_orchestrator_payload["parsed_resolution_wh"]
        final_fps = full_orchestrator_payload["fps_helpers"]
        expanded_frame_overlaps = full_orchestrator_payload["frame_overlap_expanded"]
        crossfade_sharp_amt = full_orchestrator_payload.get("crossfade_sharp_amt", 0.3)
        initial_continued_video_path_str = full_orchestrator_payload.get("continue_from_video_resolved_path")

        # --- 2. Collect Paths to All Segment Videos --- 
        segment_video_paths_for_stitch = []
        if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists():
            dprint(f"Stitch: Prepending initial continued video: {initial_continued_video_path_str}")
            segment_video_paths_for_stitch.append(str(Path(initial_continued_video_path_str).resolve()))
        
        # Query DB for all completed travel_segment tasks for this run_id
        completed_segment_outputs_from_db = []
        if DB_TYPE == "sqlite":
            def _get_sqlite_segments_for_stitch(conn):
                cursor = conn.cursor()
                # Note: JSON extract for integer might need CAST in some SQL versions if directly comparing.
                # Sorting by segment_index is critical.
                sql_query = f"""SELECT params, output_location FROM tasks 
                               WHERE json_extract(params, '$.orchestrator_run_id') = ? 
                               AND task_type = 'travel_segment' AND status = ? 
                               ORDER BY CAST(json_extract(params, '$.segment_index') AS INTEGER) ASC"""
                cursor.execute(sql_query, (orchestrator_run_id, STATUS_COMPLETE))
                rows = cursor.fetchall()
                parsed_rows = []
                for params_json, out_loc in rows:
                    try: params_dict = json.loads(params_json); parsed_rows.append((params_dict.get("segment_index"), out_loc)) 
                    except: dprint(f"Stitch: Error parsing params for a segment: {params_json}")
                return parsed_rows
            completed_segment_outputs_from_db = execute_sqlite_with_retry(SQLITE_DB_PATH, _get_sqlite_segments_for_stitch)
        
        elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
            try:
                # Prefer RPC if available and correctly sorts by segment_index (integer)
                # Example: func_get_completed_travel_segments(p_run_id TEXT)
                # Ensure RPC sorts by CAST(params->>'segment_index' AS INTEGER)
                rpc_response = SUPABASE_CLIENT.rpc("func_get_completed_travel_segments", {"p_run_id": orchestrator_run_id}).execute()
                if rpc_response.data:
                    for item in rpc_response.data: # Assuming RPC returns list of dicts like {"segment_index": int, "output_location": str}
                        completed_segment_outputs_from_db.append((item.get("segment_index"), item.get("output_location")))
                elif rpc_response.error: 
                    dprint(f"Stitch Supabase: Error from RPC func_get_completed_travel_segments: {rpc_response.error}. Falling back to direct query.")
                    # Fallback direct query (sorting might be string-based if not careful with params->>'segment_index')
                    # Manual sorting after fetch will be needed for direct query fallback.
                    direct_query_response = SUPABASE_CLIENT.table(PG_TABLE_NAME)\
                        .select("params,output_location")\
                        .eq("params->>orchestrator_run_id", orchestrator_run_id)\
                        .eq("task_type", "travel_segment")\
                        .eq("status", STATUS_COMPLETE)\
                        .execute() # Add .order() if Supabase client allows direct ordering on JSON cast
                    if direct_query_response.data:
                        temp_list = []
                        for item_direct in direct_query_response.data:
                            try: s_idx = int(item_direct["params"].get("segment_index")); temp_list.append((s_idx, item_direct["output_location"]))
                            except: dprint(f"Stitch Supabase fallback: Error parsing segment_index for item {item_direct.get('task_id')}")
                        temp_list.sort(key=lambda x: x[0] if x[0] is not None else -1) # Sort manually
                        completed_segment_outputs_from_db.extend(temp_list)
                    elif direct_query_response.error:
                        dprint(f"Stitch Supabase fallback query error: {direct_query_response.error}")
            except Exception as e_supabase_fetch: 
                 dprint(f"Stitch Supabase: Exception during segment fetch: {e_supabase_fetch}. May result in incomplete stitch.")

        # Filter and add valid paths from DB query results
        for seg_idx, loc_str in completed_segment_outputs_from_db:
            if loc_str and Path(loc_str).exists():
                segment_video_paths_for_stitch.append(str(Path(loc_str).resolve()))
                dprint(f"Stitch: Adding segment {seg_idx} video: {loc_str}")
            else: 
                dprint(f"[WARNING] Stitch: Segment {seg_idx} output path '{loc_str}' is missing or invalid. It will be excluded.")

        total_videos_for_stitch = (1 if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists() else 0) + num_expected_new_segments
        if len(segment_video_paths_for_stitch) < total_videos_for_stitch:
            # This is a warning because some segments might have legitimately failed and been skipped by their handlers.
            # The stitcher should proceed with what it has, unless it has zero or one video when multiple were expected.
            dprint(f"[WARNING] Stitch: Expected {total_videos_for_stitch} videos for stitch, but found {len(segment_video_paths_for_stitch)}. Stitching with available videos.")
        
        if not segment_video_paths_for_stitch:
            raise ValueError("Stitch: No valid segment videos found to stitch.")
        if len(segment_video_paths_for_stitch) == 1 and total_videos_for_stitch > 1:
            dprint(f"Stitch: Only one video segment found ({segment_video_paths_for_stitch[0]}) but {total_videos_for_stitch} were expected. Using this single video as the 'stitched' output.")
            # No actual stitching needed, just move/copy this single video to final dest.

        # --- 3. Stitching (Crossfade or Concatenate) --- 
        temp_stitched_video_output_path = stitch_processing_dir / f"stitched_raw_{orchestrator_run_id}.mp4"
        current_stitched_video_path: Path | None = None

        if len(segment_video_paths_for_stitch) == 1:
            # If only one video, use it directly (copy to processing dir for consistency)
            shutil.copy2(segment_video_paths_for_stitch[0], temp_stitched_video_output_path)
            current_stitched_video_path = temp_stitched_video_output_path
        else: # More than one video, proceed with stitching logic
            # Determine if any actual overlap values require frame-based crossfade
            # expanded_frame_overlaps corresponds to NEW segments. If continuing, the overlap between continued video and 1st new segment is overlaps[0].
            # If not continuing, overlap between new_seg0 and new_seg1 is overlaps[0], etc.
            # The number of overlaps is num_new_segments if not continuing, or num_new_segments if continuing (as the first overlap is defined then).
            num_stitch_points = len(segment_video_paths_for_stitch) - 1
            actual_overlaps_for_stitching = []
            if initial_continued_video_path_str: # Continued video exists
                actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points] 
            else: # No continued video, overlaps are between the new segments themselves
                actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points]

            any_positive_overlap = any(o > 0 for o in actual_overlaps_for_stitching)

            if any_positive_overlap:
                dprint(f"Stitch: Using cross-fade due to overlap values: {actual_overlaps_for_stitching}")
                all_segment_frames_lists = [sm_extract_frames_from_video(p) for p in segment_video_paths_for_stitch]
                if not all(f_list is not None and len(f_list)>0 for f_list in all_segment_frames_lists):
                    raise ValueError("Stitch: Frame extraction failed for one or more segments during cross-fade prep.")
                
                final_stitched_frames = []
                # Add first segment up to its overlap point
                overlap_for_first_join = actual_overlaps_for_stitching[0] if actual_overlaps_for_stitching else 0
                if len(all_segment_frames_lists[0]) > overlap_for_first_join:
                    final_stitched_frames.extend(all_segment_frames_lists[0][:-overlap_for_first_join if overlap_for_first_join > 0 else len(all_segment_frames_lists[0])])
                else: # First segment is shorter than or equal to its overlap, take all of it (it will be fully part of the crossfade)
                    final_stitched_frames.extend(all_segment_frames_lists[0])

                for i in range(num_stitch_points): # Iterate through join points
                    frames_prev_segment = all_segment_frames_lists[i]
                    frames_curr_segment = all_segment_frames_lists[i+1]
                    current_overlap_val = actual_overlaps_for_stitching[i]

                    if current_overlap_val > 0:
                        faded_frames = sm_cross_fade_overlap_frames(frames_prev_segment, frames_curr_segment, current_overlap_val, "linear_sharp", crossfade_sharp_amt)
                        final_stitched_frames.extend(faded_frames)
                    else: # Zero overlap, simple concatenation of remainder of prev and all of curr (excluding its own next overlap)
                        # This case (zero overlap in a crossfade path) implies we should take the rest of frames_prev_segment if not already fully added
                        # The logic for adding first segment and then iterating joins should handle this correctly.
                        # If overlap is zero, the previous segment's tail (if any, beyond its own *next* overlap) needs to be added if not already.
                        # However, the current loop structure processes joins. If current_overlap_val is 0, it means no crossfade for *this* join.
                        # The remainder of frames_prev_segment (if it wasn't fully consumed by *its* previous crossfade) was added before this loop point.
                        # So, we just need to add frames_curr_segment up to *its* next overlap point.
                        pass # Handled by adding the tail of current segment below.
                    
                    # Add the tail of the current segment (frames_curr_segment) up to *its* next overlap point
                    if (i + 1) < num_stitch_points: # If this is NOT the join before the very last segment
                        overlap_for_next_join_of_curr = actual_overlaps_for_stitching[i+1]
                        start_index_for_curr_tail = current_overlap_val # Start after the current join's faded part
                        end_index_for_curr_tail = len(frames_curr_segment) - (overlap_for_next_join_of_curr if overlap_for_next_join_of_curr > 0 else 0)
                        if end_index_for_curr_tail > start_index_for_curr_tail:
                             final_stitched_frames.extend(frames_curr_segment[start_index_for_curr_tail : end_index_for_curr_tail])
                    else: # This IS the join before the very last segment, so add all remaining frames of last segment after its fade-in
                        start_index_for_last_segment_tail = current_overlap_val
                        if len(frames_curr_segment) > start_index_for_last_segment_tail:
                            final_stitched_frames.extend(frames_curr_segment[start_index_for_last_segment_tail:])
                
                if not final_stitched_frames: raise ValueError("Stitch: No frames produced after cross-fade logic.")
                current_stitched_video_path = sm_create_video_from_frames_list(final_stitched_frames, temp_stitched_video_output_path, final_fps, parsed_res_wh)
            else:
                dprint(f"Stitch: Using simple FFmpeg concatenation as no positive overlaps specified.")
                # Ensure common_utils.stitch_videos_ffmpeg is imported or accessible
                from Wan2GP.sm_functions.common_utils import stitch_videos_ffmpeg as sm_stitch_videos_ffmpeg
                if sm_stitch_videos_ffmpeg(segment_video_paths_for_stitch, str(temp_stitched_video_output_path)):
                    current_stitched_video_path = temp_stitched_video_output_path
                else: raise RuntimeError("Stitch: Simple FFmpeg concatenation failed.")

        if not current_stitched_video_path or not current_stitched_video_path.exists():
            raise RuntimeError(f"Stitch: Stitching process failed, output video not found at {current_stitched_video_path}")
        
        # --- 4. Optional Upscaling --- 
        upscale_factor = full_orchestrator_payload.get("upscale_factor", 0.0)
        upscale_model_name = full_orchestrator_payload.get("upscale_model_name")
        current_final_video_path_before_move = current_stitched_video_path # Start with stitched path

        if isinstance(upscale_factor, (float, int)) and upscale_factor > 1.0 and upscale_model_name:
            dprint(f"Stitch: Upscaling (x{upscale_factor}) video {current_stitched_video_path.name} using model {upscale_model_name}")
            upscaled_vid_basename = f"stitched_upscaled_{upscale_factor:.1f}x_{orchestrator_run_id}"
            # Upscaled video also goes into the stitch_processing_dir first
            temp_upscaled_video_path = sm_get_unique_target_path(stitch_processing_dir, upscaled_vid_basename, current_stitched_video_path.suffix)
            
            original_frames_count, _ = sm_get_video_frame_count_and_fps(str(current_stitched_video_path))
            if original_frames_count is None or original_frames_count == 0:
                raise ValueError(f"Stitch: Cannot get frame count or 0 frames for video {current_stitched_video_path} before upscaling.")

            target_width_upscaled = int(parsed_res_wh[0] * upscale_factor)
            target_height_upscaled = int(parsed_res_wh[1] * upscale_factor)
            
            upscale_sub_task_id = sm_generate_unique_task_id(f"upscale_stitch_{orchestrator_run_id}_")
            # Upscale sub-task outputs to its own folder under main_output_dir_base/travel_run_X/stitch_Y/upscale_Z for clarity
            upscale_sub_task_output_dir_relative = (stitch_processing_dir / f"upscale_assets_{upscale_sub_task_id[:8]}").relative_to(main_output_dir_base)

            upscale_payload = {
                "task_id": upscale_sub_task_id,
                "model": upscale_model_name,
                "video_source_path": str(current_stitched_video_path.resolve()), # Absolute path to the stitched video
                "resolution": f"{target_width_upscaled}x{target_height_upscaled}",
                "frames": original_frames_count, # Upscaler needs total frames
                "prompt": full_orchestrator_payload.get("original_task_args",{}).get("upscale_prompt", "cinematic, masterpiece, high detail, 4k"), 
                "seed": full_orchestrator_payload.get("seed_for_upscale", full_orchestrator_payload.get("seed_base", 12345) + 5000),
                "output_sub_dir": str(upscale_sub_task_output_dir_relative) # Relative path for where upscaler saves
            }
            # Add other relevant upscale params from original_task_args if present in full_orchestrator_payload
            # e.g., specific LoRAs, guidance for upscaler model if applicable

            db_path_for_upscale_add = SQLITE_DB_PATH if DB_TYPE == "sqlite" else None
            # Upscaler engine can be specified in orchestrator payload, defaults to main execution engine
            upscaler_engine_to_use = stitch_params.get("execution_engine_for_upscale", full_orchestrator_payload["execution_engine"])
            
            sm_add_task_to_db(upscale_payload, db_path_for_upscale_add, task_type=upscaler_engine_to_use, task_id_override=upscale_sub_task_id)
            print(f"Stitch Task {stitch_task_id_str}: Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...")
            
            from Wan2GP.sm_functions.common_utils import poll_task_status as sm_poll_status_direct # Ensure direct import
            poll_interval_ups = full_orchestrator_payload.get("poll_interval", 15)
            poll_timeout_ups = full_orchestrator_payload.get("poll_timeout_upscale", full_orchestrator_payload.get("poll_timeout", 30 * 60) * 2) # Longer timeout for upscale
            
            upscaled_video_location_from_db = sm_poll_status_direct(
                task_id_to_poll=upscale_sub_task_id, 
                db_path_str=db_path_for_upscale_add, 
                poll_interval_seconds=poll_interval_ups, 
                timeout_seconds=poll_timeout_ups
            )
            
            if upscaled_video_location_from_db and Path(upscaled_video_location_from_db).exists():
                dprint(f"Stitch: Upscale sub-task {upscale_sub_task_id} completed. Output: {upscaled_video_location_from_db}")
                # Copy the upscaled video from its sub-task output dir to the temp_upscaled_video_path in stitch_processing_dir
                shutil.copy2(upscaled_video_location_from_db, str(temp_upscaled_video_path))
                current_final_video_path_before_move = temp_upscaled_video_path
            else: 
                print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} failed or output missing. Using non-upscaled video.")
                # current_final_video_path_before_move remains the stitched, non-upscaled video path
        elif upscale_factor > 1.0 and not upscale_model_name:
            dprint(f"Stitch: Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.")

        # --- 5. Final Output Naming and Moving --- 
        final_video_name_base = f"travel_final_{orchestrator_run_id}"
        if upscale_factor > 1.0 and "upscaled" in str(current_final_video_path_before_move.name).lower(): 
            final_video_name_base = f"travel_final_upscaled_{upscale_factor:.1f}x_{orchestrator_run_id}"
        
        # Final video is placed in main_output_dir_base (e.g. ./steerable_motion_output/)
        # NOT under current_run_base_output_dir (which is ./steerable_motion_output/travel_run_XYZ/)
        final_output_destination_path = sm_get_unique_target_path(main_output_dir_base, final_video_name_base, current_final_video_path_before_move.suffix)
        shutil.move(str(current_final_video_path_before_move), str(final_output_destination_path))
        final_video_location_for_db = str(final_output_destination_path.resolve())
        print(f"Stitch Task {stitch_task_id_str}: Final Video produced at: {final_video_location_for_db}")
        stitch_success = True

    except Exception as e_stitch_main:
        msg = f"Stitch Task {stitch_task_id_str}: Main process failed: {e_stitch_main}"
        print(f"[ERROR Task {stitch_task_id_str}]: {msg}"); traceback.print_exc()
        stitch_success = False
        final_video_location_for_db = msg # Store truncated error for DB
    
    finally:
        # --- 6. Cleanup --- 
        # Cleanup logic depends on debug_mode_enabled and skip_cleanup_enabled from full_orchestrator_payload
        debug_mode = full_orchestrator_payload.get("debug_mode_enabled", False)
        skip_cleanup = full_orchestrator_payload.get("skip_cleanup_enabled", False)
        do_cleanup_processing_dir = not debug_mode and not skip_cleanup
        do_cleanup_run_dir = do_cleanup_processing_dir and stitch_success # Only clean run_dir if stitch fully succeeded and processing dir is cleaned

        if do_cleanup_processing_dir and stitch_processing_dir.exists():
            dprint(f"Stitch: Cleaning up stitch processing directory: {stitch_processing_dir}")
            try: shutil.rmtree(stitch_processing_dir)
            except Exception as e_c1: dprint(f"Stitch: Error cleaning up {stitch_processing_dir}: {e_c1}")
        else:
            dprint(f"Stitch: Skipping cleanup of stitch processing directory: {stitch_processing_dir} (debug:{debug_mode}, skip_cleanup:{skip_cleanup})")

        # Cleanup the entire run directory (containing all segment folders) only if stitch was successful AND not in debug/skip_cleanup mode.
        if do_cleanup_run_dir and current_run_base_output_dir.exists():
            dprint(f"Stitch: Full run successful. Cleaning up run directory: {current_run_base_output_dir}")
            try: shutil.rmtree(current_run_base_output_dir)
            except Exception as e_c2: dprint(f"Stitch: Error cleaning up run directory {current_run_base_output_dir}: {e_c2}")
        elif current_run_base_output_dir.exists(): # current_run_base_output_dir exists but conditions for cleanup not met
            dprint(f"Stitch: Skipping cleanup of run directory: {current_run_base_output_dir} (stitch_success:{stitch_success}, debug:{debug_mode}, skip_cleanup:{skip_cleanup})")
            
    return stitch_success, final_video_location_for_db

# --- End SM_RESTRUCTURE ---

# -----------------------------------------------------------------------------
# 7. Main server loop
# -----------------------------------------------------------------------------

def main():
    load_dotenv() # Load .env file variables into environment
    global DB_TYPE, PG_TABLE_NAME, SQLITE_DB_PATH, SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_VIDEO_BUCKET, SUPABASE_CLIENT

    # Determine DB type from environment variables
    env_db_type = os.getenv("DB_TYPE", "sqlite").lower()
    env_pg_table_name = os.getenv("POSTGRES_TABLE_NAME", "tasks")
    env_supabase_url = os.getenv("SUPABASE_URL")
    env_supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    env_supabase_bucket = os.getenv("SUPABASE_VIDEO_BUCKET", "videos")
    env_comfyui_output_path = os.getenv("COMFYUI_OUTPUT_PATH") # Load ComfyUI output path
    env_sqlite_db_path = os.getenv("SQLITE_DB_PATH_ENV") # Read SQLite DB path from .env

    cli_args = parse_args()

    # --- Configure DB Type and Connection Globals ---
    # This block sets DB_TYPE, SQLITE_DB_PATH, SUPABASE_CLIENT, PG_TABLE_NAME etc.
    if env_db_type == "supabase" and env_supabase_url and env_supabase_key:
        try:
            temp_supabase_client = create_client(env_supabase_url, env_supabase_key)
            if temp_supabase_client: 
                DB_TYPE = "supabase"
                PG_TABLE_NAME = env_pg_table_name 
                SUPABASE_URL = env_supabase_url
                SUPABASE_SERVICE_KEY = env_supabase_key
                SUPABASE_VIDEO_BUCKET = env_supabase_bucket
                SUPABASE_CLIENT = temp_supabase_client
                # Initial print about using Supabase will be done after migrations
            else:
                raise Exception("Supabase client creation returned None")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Supabase client: {e}. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.")
            print("Falling back to SQLite due to Supabase client initialization error.")
            DB_TYPE = "sqlite"
            # Determine SQLite path for fallback: .env, then CLI, then default
            SQLITE_DB_PATH = env_sqlite_db_path if env_sqlite_db_path else cli_args.db_file
    elif env_db_type == "sqlite":
        DB_TYPE = "sqlite"
        # Determine SQLite path: .env, then CLI, then default
        SQLITE_DB_PATH = env_sqlite_db_path if env_sqlite_db_path else cli_args.db_file
    else: # Default to sqlite if .env DB_TYPE is unrecognized or not set
        print(f"DB_TYPE '{env_db_type}' in .env is not recognized or not set. Defaulting to SQLite.")
        DB_TYPE = "sqlite"
        # Determine SQLite path: .env, then CLI, then default
        SQLITE_DB_PATH = env_sqlite_db_path if env_sqlite_db_path else cli_args.db_file
    # --- End DB Type Configuration ---

    # --- Run DB Migrations ---
    # Must be after DB type/config is determined but before DB schema is strictly enforced by init_db or heavy use.
    _run_db_migrations() 
    # --- End DB Migrations ---

    global COMFYUI_OUTPUT_PATH_CONFIG 
    if env_comfyui_output_path:
        COMFYUI_OUTPUT_PATH_CONFIG = Path(env_comfyui_output_path)
        if not COMFYUI_OUTPUT_PATH_CONFIG.is_dir():
            print(f"[WARNING] COMFYUI_OUTPUT_PATH '{COMFYUI_OUTPUT_PATH_CONFIG}' from .env is not a valid directory. ComfyUI tasks may fail to retrieve outputs.")
            # COMFYUI_OUTPUT_PATH_CONFIG = None # Or let it proceed and fail in the handler
        else:
            print(f"ComfyUI output path configured: {COMFYUI_OUTPUT_PATH_CONFIG}")
    else:
        print("[INFO] COMFYUI_OUTPUT_PATH not set in .env. ComfyUI tasks will require this to retrieve outputs.")

    main_output_dir = Path(cli_args.main_output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"WanGP Headless Server Started.")
    if DB_TYPE == "supabase":
        print(f"Monitoring Supabase (PostgreSQL backend) table: {PG_TABLE_NAME}")
    else: # SQLite
        print(f"Monitoring SQLite database: {SQLITE_DB_PATH}")
    print(f"Outputs will be saved under: {main_output_dir}")
    print(f"Polling interval: {cli_args.poll_interval} seconds.")

    # Initialize database
    if DB_TYPE == "supabase":
        init_db_supabase() # New call, uses globals
    else: # SQLite
        init_db(SQLITE_DB_PATH) # Existing SQLite init

    # Activate global debug switch early so that all subsequent code paths can use dprint()
    global debug_mode
    debug_mode = cli_args.debug
    dprint("Verbose debug logging enabled.")

    original_argv = sys.argv.copy()
    sys.argv = ["Wan2GP/wgp.py"]
    patch_gradio()
    
    # Add Wan2GP directory to Python path and import wgp directly
    wan2gp_path = Path(__file__).parent / "Wan2GP"
    if str(wan2gp_path) not in sys.path:
        sys.path.insert(0, str(wan2gp_path))
    
    import wgp as wgp_mod
    
    sys.argv = original_argv

    # Apply wgp.py global config overrides
    if cli_args.wgp_attention_mode is not None: wgp_mod.attention_mode = cli_args.wgp_attention_mode
    if cli_args.wgp_compile is not None: wgp_mod.compile = cli_args.wgp_compile
    # ... (all other wgp global config settings as before) ...
    if cli_args.wgp_profile is not None: wgp_mod.profile = cli_args.wgp_profile
    if cli_args.wgp_vae_config is not None: wgp_mod.vae_config = cli_args.wgp_vae_config
    if cli_args.wgp_boost is not None: wgp_mod.boost = cli_args.wgp_boost
    if cli_args.wgp_transformer_quantization is not None: wgp_mod.transformer_quantization = cli_args.wgp_transformer_quantization
    if cli_args.wgp_transformer_dtype_policy is not None: wgp_mod.transformer_dtype_policy = cli_args.wgp_transformer_dtype_policy
    if cli_args.wgp_text_encoder_quantization is not None: wgp_mod.text_encoder_quantization = cli_args.wgp_text_encoder_quantization
    if cli_args.wgp_vae_precision is not None: wgp_mod.server_config["vae_precision"] = cli_args.wgp_vae_precision
    if cli_args.wgp_mixed_precision is not None: wgp_mod.server_config["mixed_precision"] = cli_args.wgp_mixed_precision
    if cli_args.wgp_preload_policy is not None:
        wgp_mod.server_config["preload_model_policy"] = [flag.strip() for flag in cli_args.wgp_preload_policy.split(',')]

    # --- Ensure common support models from wgp.py are downloaded early --- 
    try:
        print("[INFO] Headless server: Ensuring common models are available via wgp.py's download logic...")
        if hasattr(wgp_mod, 'download_models') and hasattr(wgp_mod, 'transformer_filename'):
            # transformer_filename is initialized in wgp.py to a default.
            # Calling download_models for it will also fetch shared/support models like those in ckpts/pose/.
            wgp_mod.download_models(wgp_mod.transformer_filename) 
            print("[INFO] Headless server: Common model check/download complete.")
        else:
            print("[WARNING] Headless server: wgp_mod.download_models or wgp_mod.transformer_filename not found. \n                 Automatic download of support models might not occur if not triggered by a full model load.")
    except Exception as e_download_init:
        print(f"[WARNING] Headless server: Error during initial attempt to download common models via wgp.py: {e_download_init}")
        traceback.print_exc()
        print("             Ensure models are manually placed in 'ckpts' or a video generation task runs first to trigger downloads.")
    # --- End early model download ---

    db_path_str = str(SQLITE_DB_PATH) if DB_TYPE == "sqlite" else PG_TABLE_NAME # Use consistent string path for db functions

    # --- Ensure LoRA directories expected by wgp.py exist, especially for LTXV ---
    try:
        # Get the args namespace that wgp.py uses internally after its own parsing
        wgp_internal_args = wgp_mod.args 
        ltxv_lora_dir_path = Path(wgp_internal_args.lora_dir_ltxv) # Default is "loras_ltxv"
        if not ltxv_lora_dir_path.exists():
            print(f"[INFO] Headless: LTXV LoRA directory '{ltxv_lora_dir_path}' does not exist. Creating it.")
            ltxv_lora_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            dprint(f"[INFO] Headless: LTXV LoRA directory '{ltxv_lora_dir_path}' already exists.")
        
        # We could do this for other model types too if needed, e.g.:
        # wan_t2v_lora_dir_path = Path(wgp_internal_args.lora_dir)
        # if not wan_t2v_lora_dir_path.exists(): wan_t2v_lora_dir_path.mkdir(parents=True, exist_ok=True)
        # wan_i2v_lora_dir_path = Path(wgp_internal_args.lora_dir_i2v)
        # if not wan_i2v_lora_dir_path.exists(): wan_i2v_lora_dir_path.mkdir(parents=True, exist_ok=True)

    except AttributeError as e_lora_arg:
        print(f"[WARNING] Headless: Could not determine or create default LoRA directories expected by wgp.py (AttributeError: {e_lora_arg}). This might be an issue if wgp.py has changed its internal arg parsing.")
    except Exception as e_lora_dir:
        print(f"[WARNING] Headless: An error occurred while ensuring LoRA directories: {e_lora_dir}")
    # --- End LoRA directory check ---

    try:
        while True:
            task_info = None
            current_task_id_for_status_update = None # Used to hold the task_id for status updates

            if DB_TYPE == "supabase":
                dprint(f"Checking for queued tasks in Supabase (PostgreSQL backend) table {PG_TABLE_NAME} via Supabase RPC...")
                task_info = get_oldest_queued_task_supabase()
                if task_info:
                    current_task_id_for_status_update = task_info["task_id"]
                    # Status is already set to IN_PROGRESS by func_claim_task RPC
            else: # SQLite
                dprint(f"Checking for queued tasks in SQLite {SQLITE_DB_PATH}...")
                task_info = get_oldest_queued_task(SQLITE_DB_PATH)
                if task_info:
                    current_task_id_for_status_update = task_info["task_id"]
                    update_task_status(SQLITE_DB_PATH, current_task_id_for_status_update, STATUS_IN_PROGRESS)

            if not task_info:
                dprint("No queued tasks found. Sleeping...")
                time.sleep(cli_args.poll_interval)
                continue

            # current_task_data = task_info["params"] # Params are already a dict
            current_task_params = task_info["params"]
            current_task_type = task_info["task_type"] # Retrieve task_type

            print(f"Found task: {current_task_id_for_status_update} of type: {current_task_type}")
            # Status already set to IN_PROGRESS if task_info is not None

            task_succeeded, output_location = process_single_task(wgp_mod, current_task_params, main_output_dir, current_task_type)

            if task_succeeded:
                if DB_TYPE == "supabase":
                    update_task_status_supabase(current_task_id_for_status_update, STATUS_COMPLETE, output_location)
                else:
                    update_task_status(SQLITE_DB_PATH, current_task_id_for_status_update, STATUS_COMPLETE, output_location)
                print(f"Task {current_task_id_for_status_update} completed successfully. Output location: {output_location}")
            else:
                if DB_TYPE == "supabase":
                    update_task_status_supabase(current_task_id_for_status_update, STATUS_FAILED, output_location)
                else:
                    update_task_status(SQLITE_DB_PATH, current_task_id_for_status_update, STATUS_FAILED, output_location)
                print(f"Task {current_task_id_for_status_update} failed. Review logs for errors. Output location recorded: {output_location if output_location else 'N/A'}")
            
            time.sleep(1) # Brief pause before checking for the next task

    except KeyboardInterrupt:
        print("\nServer shutting down gracefully...")
    finally:
        if hasattr(wgp_mod, 'offloadobj') and wgp_mod.offloadobj is not None:
            try:
                print("Attempting to release wgp.py offload object...")
                wgp_mod.offloadobj.release()
            except Exception as e_release:
                print(f"Error during offloadobj release: {e_release}")
        print("Server stopped.")

# -----------------------------------------------------------------------------
# Supabase Storage Helper
# -----------------------------------------------------------------------------
def upload_to_supabase_storage(local_file_path: Path, object_name_in_bucket: str, bucket_name: str) -> str | None:
    """Uploads a file to Supabase storage and returns its public URL."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot upload.")
        return None

    try:

        with open(local_file_path, 'rb') as f:
            # The object name can include paths, e.g., "videos/task_123.mp4"
            res = SUPABASE_CLIENT.storage.from_(bucket_name).upload(
                path=object_name_in_bucket, 
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"} # Upsert to overwrite if exists
            )
        
        dprint(f"Supabase upload response data: {res.json() if hasattr(res, 'json') else res}")

        # Get public URL
        public_url_response = SUPABASE_CLIENT.storage.from_(bucket_name).get_public_url(object_name_in_bucket)
        
        # public_url_response is a string directly
        dprint(f"Supabase get_public_url response: {public_url_response}")
        if public_url_response:
            print(f"INFO: Successfully uploaded {local_file_path.name} to Supabase bucket '{bucket_name}' as '{object_name_in_bucket}'. URL: {public_url_response}")
            return public_url_response
        else:
            print(f"[ERROR] Failed to get public URL for {object_name_in_bucket} in Supabase bucket '{bucket_name}'. Upload may have succeeded.")
            return None # Or construct a presumed URL if your bucket is public and path is known

    except Exception as e:
        print(f"[ERROR] Failed to upload {local_file_path.name} to Supabase: {e}")
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# PostgreSQL Specific DB Functions (Now using Supabase RPC)
# Renamed to reflect Supabase usage more directly
# -----------------------------------------------------------------------------

def init_db_supabase(): # Renamed from init_db_postgres
    """Initializes the PostgreSQL tasks table via Supabase RPC if it doesn't exist."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot initialize database table.")
        sys.exit(1)
    try:
        # RPC call to the SQL function func_initialize_tasks_table
        # IMPORTANT: The func_initialize_tasks_table SQL function itself
        # must be updated to include "task_type TEXT NOT NULL" in its
        # CREATE TABLE statement for the specified p_table_name.
        SUPABASE_CLIENT.rpc("func_initialize_tasks_table", {"p_table_name": PG_TABLE_NAME}).execute()
        print(f"Supabase RPC: Table '{PG_TABLE_NAME}' initialization requested.")
        # Note: RPC for DDL might not return specific confirmation beyond successful execution.
        # You might need to add a SELECT to confirm table existence if strict feedback is needed.
    except Exception as e: # Broader exception for Supabase/PostgREST errors
        print(f"[ERROR] Supabase RPC for table initialization failed: {e}")
        traceback.print_exc()
        sys.exit(1)

def get_oldest_queued_task_supabase(): # Renamed from get_oldest_queued_task_postgres
    """Fetches the oldest task via Supabase RPC using func_claim_task."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot get task.")
        return None
    try:
        worker_id = f"worker_{os.getpid()}" # Example worker ID
        dprint(f"DEBUG get_oldest_queued_task_supabase: About to call RPC func_claim_task.")
        dprint(f"DEBUG get_oldest_queued_task_supabase: PG_TABLE_NAME = '{PG_TABLE_NAME}' (type: {type(PG_TABLE_NAME)})")
        dprint(f"DEBUG get_oldest_queued_task_supabase: worker_id = '{worker_id}' (type: {type(worker_id)})")
        
        response = SUPABASE_CLIENT.rpc(
            "func_claim_task", 
            {"p_table_name": PG_TABLE_NAME, "p_worker_id": worker_id}
        ).execute()
        
        dprint(f"Supabase RPC func_claim_task response data: {response.data}")

        if response.data and len(response.data) > 0:
            task_data = response.data[0] # RPC should return a single row or empty
            # Ensure the RPC returns task_id_out, params_out, and now task_type_out
            if task_data.get("task_id_out") and task_data.get("params_out") is not None and task_data.get("task_type_out") is not None:
                dprint(f"Supabase RPC: Claimed task {task_data['task_id_out']} of type {task_data['task_type_out']}")
                return {"task_id": task_data["task_id_out"], "params": task_data["params_out"], "task_type": task_data["task_type_out"]}
            else:
                dprint("Supabase RPC: func_claim_task returned but no task was claimed or required fields (task_id_out, params_out, task_type_out) are missing.")
                return None
        else:
            dprint("Supabase RPC: No task claimed or empty response from func_claim_task.")
            return None
    except Exception as e:
        print(f"[ERROR] Supabase RPC func_claim_task failed: {e}")
        traceback.print_exc()
        return None

def update_task_status_supabase(task_id_str, status_str, output_location_val=None): # Renamed from update_task_status_postgres
    """Updates a task's status via Supabase RPC using func_update_task_status."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot update task status.")
        return
    try:
        params = {
            "p_table_name": PG_TABLE_NAME,
            "p_task_id": task_id_str,
            "p_status": status_str
        }
        if output_location_val is not None:
            params["p_output_location"] = output_location_val
        
        SUPABASE_CLIENT.rpc("func_update_task_status", params).execute()
        dprint(f"Supabase RPC: Updated status of task {task_id_str} to {status_str}. Output: {output_location_val if output_location_val else 'N/A'}")
    except Exception as e:
        print(f"[ERROR] Supabase RPC func_update_task_status for {task_id_str} failed: {e}")

def _migrate_supabase_schema():
    """Applies necessary schema migrations to an existing Supabase/PostgreSQL database via RPC."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase Migration: Supabase client not initialized. Cannot run migration.")
        return

    dprint(f"Supabase Migration: Requesting schema migration via RPC 'func_migrate_tasks_for_task_type' for table {PG_TABLE_NAME}...")
    try:
        # IMPORTANT: The user must create the SQL function 'func_migrate_tasks_for_task_type'
        # in their Supabase/PostgreSQL database.
        # This function should take p_table_name as TEXT argument and:
        # 1. Check if the 'task_type' column exists in the p_table_name.
        # 2. If not, execute: ALTER TABLE {p_table_name} ADD COLUMN task_type TEXT;
        #    (Add as nullable first to allow adding to existing tables with data).
        # 3. Populate the new 'task_type' column for existing rows where it's NULL,
        #    by extracting the type from the 'params' JSONB column. Example:
        #    UPDATE {p_table_name}
        #    SET task_type = params->>'task_type' -- Or other logic if task type was stored differently
        #    WHERE task_type IS NULL AND params IS NOT NULL AND params->>'task_type' IS NOT NULL;
        #    (Adjust the params->>'task_type' according to how task type was previously stored in params).
        # 4. The func_initialize_tasks_table RPC should already ensure task_type is NOT NULL for new tables.
        
        response = SUPABASE_CLIENT.rpc("func_migrate_tasks_for_task_type", {"p_table_name": PG_TABLE_NAME}).execute()
        
        # Improved response handling based on Supabase Python client v2+ structure
        if response.error:
            print(f"[ERROR] Supabase Migration: RPC 'func_migrate_tasks_for_task_type' returned an error: {response.error.message} (Code: {response.error.code}, Details: {response.error.details})")
        elif response.data:
            dprint(f"Supabase Migration: RPC 'func_migrate_tasks_for_task_type' executed. Response data: {response.data}")
        else:
            dprint("Supabase Migration: RPC 'func_migrate_tasks_for_task_type' executed. (No specific data or error in response, check RPC logs if issues)")
            
    except Exception as e:
        print(f"[ERROR] Supabase Migration: Failed to execute RPC 'func_migrate_tasks_for_task_type': {e}")
        traceback.print_exc()

def _run_db_migrations():
    """Runs database migrations based on the configured DB_TYPE."""
    dprint(f"DB Migrations: Running for DB_TYPE: {DB_TYPE}")
    if DB_TYPE == "sqlite":
        if SQLITE_DB_PATH:
            _migrate_sqlite_schema(SQLITE_DB_PATH)
        else:
            print("[ERROR] DB Migration: SQLITE_DB_PATH not set. Skipping SQLite migration.")
    elif DB_TYPE == "supabase":
        if SUPABASE_CLIENT and PG_TABLE_NAME:
            _migrate_supabase_schema()
        else:
            print("[ERROR] DB Migration: Supabase client or PG_TABLE_NAME not configured. Skipping Supabase migration.")
    else:
        dprint(f"DB Migrations: No migration logic for DB_TYPE '{DB_TYPE}'. Skipping migrations.")

# Helper to query DB for a specific task's output (needed by segment handler)
def get_task_output_location_from_db(task_id_to_find: str) -> str | None:
    dprint(f"Querying DB for output location of task: {task_id_to_find}")
    if DB_TYPE == "sqlite":
        def _get_op(conn):
            cursor = conn.cursor()
            # Ensure we only get tasks that are actually complete with an output
            cursor.execute("SELECT output_location FROM tasks WHERE task_id = ? AND status = ? AND output_location IS NOT NULL", 
                           (task_id_to_find, STATUS_COMPLETE))
            row = cursor.fetchone()
            return row[0] if row else None
        try:
            return execute_sqlite_with_retry(SQLITE_DB_PATH, _get_op)
        except Exception as e:
            print(f"Error querying SQLite for task output {task_id_to_find}: {e}")
            return None
    elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
        try:
            # This assumes an RPC function `func_get_task_details` exists or similar direct query access.
            # The RPC should return at least `output_location` and `status`.
            # Example: response = SUPABASE_CLIENT.rpc("func_get_task_details", {"p_task_id": task_id_to_find}).execute()
            # If direct query:
            response = SUPABASE_CLIENT.table(PG_TABLE_NAME)\
                .select("output_location, status")\
                .eq("task_id", task_id_to_find)\
                .execute()

            if response.data and len(response.data) > 0:
                task_details = response.data[0]
                if task_details.get("status") == STATUS_COMPLETE and task_details.get("output_location"):
                    return task_details.get("output_location")
                else:
                    dprint(f"Task {task_id_to_find} found but not complete or no output_location. Status: {task_details.get('status')}")
                    return None
            else:
                dprint(f"Task {task_id_to_find} not found in Supabase.")
                return None
        except Exception as e:
            print(f"Error querying Supabase for task output {task_id_to_find}: {e}")
            traceback.print_exc()
            return None
    dprint(f"DB type {DB_TYPE} not supported or client not init for get_task_output_location_from_db")
    return None

# Function to find the task_id of a previous segment by its index and run_id
# This is crucial for _handle_travel_segment_task to find its dependency.
def get_previous_segment_task_id(orchestrator_run_id: str, target_segment_index: int) -> str | None:
    dprint(f"Querying DB for task_id of segment_index {target_segment_index} for run_id {orchestrator_run_id}")
    if DB_TYPE == "sqlite":
        def _get_op(conn):
            cursor = conn.cursor()
            # Assumes segment_task_id is stored in `params` or is the main `task_id`
            # And that `segment_index` and `orchestrator_run_id` are in `params`
            query = f"SELECT task_id FROM tasks WHERE json_extract(params, '$.orchestrator_run_id') = ? AND json_extract(params, '$.segment_index') = ? AND task_type = 'travel_segment' AND status = ?"
            cursor.execute(query, (orchestrator_run_id, target_segment_index, STATUS_COMPLETE))
            row = cursor.fetchone()
            return row[0] if row else None
        try:
            return execute_sqlite_with_retry(SQLITE_DB_PATH, _get_op)
        except Exception as e:
            print(f"Error querying SQLite for previous segment task_id: {e}")
            return None
    elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
        try:
            response = SUPABASE_CLIENT.table(PG_TABLE_NAME)\
                .select("task_id")\
                .eq("params->>orchestrator_run_id", orchestrator_run_id)\
                .eq("params->>segment_index", target_segment_index)\
                .eq("task_type", "travel_segment")\
                .eq("status", STATUS_COMPLETE)\
                .limit(1)\
                .execute()
            if response.data and len(response.data) > 0:
                return response.data[0]["task_id"]
            return None
        except Exception as e:
            print(f"Error querying Supabase for previous segment task_id: {e}")
            return None
    return None

if __name__ == "__main__":
    main() 