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

# --- Add imports for OpenPose generation ---
import numpy as np
try:
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
# Database Helper Functions
# -----------------------------------------------------------------------------
def init_db(db_path_str: str):
    """Initializes the SQLite database and creates the tasks table if it doesn't exist.
       If the table is newly created and empty, it attempts to populate it from default_tasks.json."""
    conn = sqlite3.connect(db_path_str)
    cursor = conn.cursor()
    # Ensure STATUS_QUEUED is correctly embedded in the DEFAULT clause
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            params TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '{STATUS_QUEUED}',
            output_location TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    cursor.execute(create_table_sql)
    conn.commit() # Commit table creation first

    # Check if the table is empty
    cursor.execute("SELECT COUNT(*) FROM tasks")
    count = cursor.fetchone()[0]
    dprint(f"Found {count} existing tasks in the database.")

    if count == 0:
        print("INFO: Tasks table is empty. Attempting to populate from default_tasks.json...")
        try:
            script_dir = Path(os.path.dirname(__file__)).resolve()
            default_tasks_path = script_dir / "default_tasks.json"
            print(f"INFO: Looking for default_tasks.json at: {default_tasks_path}")

            if default_tasks_path.is_file():
                print(f"INFO: Found default_tasks.json at {default_tasks_path}.")
                with open(default_tasks_path, 'r') as f:
                    content = f.read()
                    dprint(f"Content of default_tasks.json: {content[:500]}...") # Print first 500 chars for debugging
                    default_tasks = json.loads(content) # Use content for json.loads
                
                if isinstance(default_tasks, list):
                    print(f"INFO: Successfully parsed default_tasks.json as a list. Found {len(default_tasks)} potential tasks.")
                    tasks_added_count = 0
                    for task_data in default_tasks:
                        task_id = task_data.get("task_id")
                        if not task_id:
                            print(f"WARNING: Skipping task in default_tasks.json due to missing 'task_id': {task_data}")
                            continue
                        
                        params_json = json.dumps(task_data)
                        
                        try:
                            cursor.execute("INSERT INTO tasks (task_id, params, status) VALUES (?, ?, ?)",
                                           (task_id, params_json, STATUS_QUEUED))
                            tasks_added_count += 1
                            dprint(f"Inserted task {task_id} from default_tasks.json")
                        except sqlite3.IntegrityError:
                            print(f"WARNING: Task ID {task_id} from default_tasks.json already exists or another integrity error occurred. Skipping.")
                    conn.commit()
                    if tasks_added_count > 0:
                        print(f"INFO: Successfully populated database with {tasks_added_count} tasks from {default_tasks_path}")
                    else:
                        print("INFO: No new tasks were added from default_tasks.json (possibly all skipped or file was empty).")
                else:
                    print(f"WARNING: {default_tasks_path} does not contain a JSON list. No default tasks loaded.")
            else:
                print(f"WARNING: Default tasks file not found at {default_tasks_path}. No default tasks loaded.")
        except json.JSONDecodeError as jde:
            print(f"ERROR: Could not decode JSON from {default_tasks_path}. Error: {jde}. No default tasks loaded.")
        except FileNotFoundError:
             print(f"ERROR: File not found at {default_tasks_path} (This should be caught by is_file(), but as a fallback). No default tasks loaded.")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while trying to load default tasks: {e}. No default tasks loaded.")
            traceback.print_exc() # Print full traceback for unexpected errors

    # Add an index on status and created_at for faster querying of queued tasks (if not exists)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_created_at ON tasks (status, created_at)")
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path_str}")

def get_oldest_queued_task(db_path_str: str):
    """Fetches the oldest task with 'Queued' status."""
    conn = sqlite3.connect(db_path_str)
    conn.row_factory = sqlite3.Row # Access columns by name
    cursor = conn.cursor()
    cursor.execute("SELECT task_id, params FROM tasks WHERE status = ? ORDER BY created_at ASC LIMIT 1", (STATUS_QUEUED,))
    task_row = cursor.fetchone()
    conn.close()
    if task_row:
        return {"task_id": task_row["task_id"], "params": json.loads(task_row["params"])}
    return None

def update_task_status(db_path_str: str, task_id: str, status: str, output_location_val: str | None = None):
    """Updates a task's status and updated_at timestamp.
       Optionally updates output_location if provided (typically on COMPLETED status)."""
    conn = sqlite3.connect(db_path_str)
    cursor = conn.cursor()
    if status == STATUS_COMPLETE and output_location_val is not None:
        cursor.execute("UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP, output_location = ? WHERE task_id = ?", 
                       (status, output_location_val, task_id))
    else:
        # For other statuses or if output_location is None, don't update output_location
        cursor.execute("UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?", (status, task_id))
    conn.commit()
    conn.close()
    dprint(f"SQLite: Updated status of task {task_id} to {status}. Output: {output_location_val if output_location_val else 'N/A'}")

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
        "gen": {"queue": [], "file_list": [], "prompt_no": 1, "prompts_max": 1},
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

def process_single_task(wgp_mod, task_params_dict, main_output_dir_base: Path):
    dprint(f"PROCESS_SINGLE_TASK received task_params_dict: {json.dumps(task_params_dict)}") # DEBUG ADDED
    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))
    print(f"--- Processing task ID: {task_id} ---")
    output_location_to_db = None # Will store the final path/URL for the DB

    # --- Check for new task type ---
    task_type = task_params_dict.get("task_type")
    if task_type == "generate_openpose":
        if PoseBodyFaceVideoAnnotator is None:
            print(f"[ERROR Task ID: {task_id}] PoseBodyFaceVideoAnnotator not imported. Cannot process 'generate_openpose' task.")
            return False, "PoseBodyFaceVideoAnnotator module not available."
        
        print(f"[Task ID: {task_id}] Identified as 'generate_openpose' task.")
        return _handle_generate_openpose_task(task_params_dict, main_output_dir_base, task_id)
    elif task_type == "rife_interpolate_images":
        print(f"[Task ID: {task_id}] Identified as 'rife_interpolate_images' task.")
        return _handle_rife_interpolate_task(wgp_mod, task_params_dict, main_output_dir_base, task_id)
    # --- End check for new task type ---

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
            ltxv_conditioning_frames_spec=ui_params.get("ltxv_conditioning_frames_spec", ""),
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
            remove_background_image_ref=ui_params.get("remove_background_image_ref", 1),
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
            "gen": {"queue": [], "file_list": [], "prompt_no": 1, "prompts_max": 1} # Added missing 'gen' key
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
# 7. Main server loop
# -----------------------------------------------------------------------------

def main():
    load_dotenv() # Load .env file variables into environment
    global DB_TYPE, PG_TABLE_NAME, SQLITE_DB_PATH, SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_VIDEO_BUCKET, SUPABASE_CLIENT

    # Determine DB type from environment variables
    env_db_type = os.getenv("DB_TYPE", "sqlite").lower()
    # env_pg_dsn = os.getenv("POSTGRES_DSN") # REMOVED - DSN no longer used by Python
    env_pg_table_name = os.getenv("POSTGRES_TABLE_NAME", "tasks")
    env_supabase_url = os.getenv("SUPABASE_URL")
    env_supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    env_supabase_bucket = os.getenv("SUPABASE_VIDEO_BUCKET", "videos")

    cli_args = parse_args()

    if env_db_type == "supabase" and env_supabase_url and env_supabase_key:
        # Attempt to configure for Supabase
        try:
            # Test client initialization early
            temp_supabase_client = create_client(env_supabase_url, env_supabase_key)
            if temp_supabase_client: # Basic check, doesn't guarantee server is reachable yet
                DB_TYPE = "supabase" # Set internal DB_TYPE to "supabase"
                PG_TABLE_NAME = env_pg_table_name # Used for RPC calls to specify table
                SUPABASE_URL = env_supabase_url
                SUPABASE_SERVICE_KEY = env_supabase_key
                SUPABASE_VIDEO_BUCKET = env_supabase_bucket
                global SUPABASE_CLIENT
                SUPABASE_CLIENT = temp_supabase_client # Assign the successfully created client
                print(f"Using Supabase (PostgreSQL backend) for tasks. Table: {PG_TABLE_NAME}")
                print(f"Supabase storage configured. Bucket: {SUPABASE_VIDEO_BUCKET}")
                print("Supabase client initialized successfully.")
            else: # Should not happen if create_client returns None without exception
                raise Exception("Supabase client creation returned None")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Supabase client: {e}. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.")
            traceback.print_exc()
            print("Falling back to SQLite due to Supabase client initialization error.")
            DB_TYPE = "sqlite" # Explicitly fall back to SQLite
    
    # If DB_TYPE is still "sqlite" (either by default, or because .env didn't specify postgres, or due to Supabase init failure)
    if DB_TYPE == "sqlite":
        SQLITE_DB_PATH = cli_args.db_file # Get SQLite path from args
        print(f"Using SQLite database for tasks: {SQLITE_DB_PATH}")
        # Add specific warning if user intended Supabase (DB_TYPE=supabase in .env) but it failed
        if env_db_type == "supabase" and not (SUPABASE_URL and SUPABASE_SERVICE_KEY and SUPABASE_CLIENT):
            print("[WARNING] DB_TYPE was 'supabase' in .env but Supabase client initialization failed or SUPABASE_URL/SUPABASE_SERVICE_KEY were missing. Falling back to SQLite.")

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
    from Wan2GP import wgp as wgp_mod
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

            print(f"Found task: {current_task_id_for_status_update}")
            # Status already set to IN_PROGRESS if task_info is not None

            task_succeeded, output_location = process_single_task(wgp_mod, current_task_params, main_output_dir)

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
            if task_data.get("task_id_out") and task_data.get("params_out") is not None: # Check if task was actually claimed
                dprint(f"Supabase RPC: Claimed task {task_data['task_id_out']}")
                return {"task_id": task_data["task_id_out"], "params": task_data["params_out"]}
            else:
                dprint("Supabase RPC: func_claim_task returned but no task was claimed (e.g., empty data or null task_id_out).")
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

if __name__ == "__main__":
    main() 