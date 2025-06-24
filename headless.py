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
import json
import time
import traceback
import urllib.parse
import tempfile
import shutil

from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient

# Add the current directory to Python path so Wan2GP can be imported as a module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add the Wan2GP subdirectory to the path for its internal imports
wan2gp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2GP")
if wan2gp_path not in sys.path:
    sys.path.append(wan2gp_path)

# --- SM_RESTRUCTURE: Import moved/new utilities ---
from source import db_operations as db_ops
from source.specialized_handlers import (
    handle_generate_openpose_task,
    handle_rife_interpolate_task
)
from source.common_utils import (
    sm_get_unique_target_path,
    download_image_if_url as sm_download_image_if_url,
    download_file,
    load_pil_images as sm_load_pil_images
)
from source.sm_functions import travel_between_images as tbi
from source.sm_functions import different_pose as dp
# --- End SM_RESTRUCTURE imports ---


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
    pgroup_server.add_argument("--save-logging", type=str, nargs='?', const='logs/headless.log', default=None,
                               help="Save all logging output to a file (in addition to console output). Optionally specify path, defaults to 'logs/headless.log'")
    pgroup_server.add_argument("--delete-db", action="store_true",
                               help="Delete existing database files before starting (fresh start)")
    pgroup_server.add_argument("--migrate-only", action="store_true",
                               help="Run database migrations and then exit.")
    pgroup_server.add_argument("--apply-reward-lora", action="store_true",
                               help="Apply the reward LoRA with a fixed strength of 0.5.")
    pgroup_server.add_argument("--colour-match-videos", action="store_true",
                               help="Apply colour matching to travel videos.")
    # --- New flag: automatically generate and pass a video mask marking active/inactive frames ---
    pgroup_server.add_argument("--mask-active-frames", dest="mask_active_frames", action="store_true", default=True,
                               help="Generate and pass a mask video where frames that are re-used remain unmasked while new frames are masked (enabled by default).")
    pgroup_server.add_argument("--no-mask-active-frames", dest="mask_active_frames", action="store_false",
                               help="Disable automatic mask video generation.")

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
        
    return _send

# -----------------------------------------------------------------------------
# 4. State builder for a single task (same as before)
# -----------------------------------------------------------------------------
def build_task_state(wgp_mod, model_filename, task_params_dict, all_loras_for_model, image_download_dir: Path | str | None = None, apply_reward_lora: bool = False):
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

    current_task_id_for_log = task_params_dict.get('task_id', 'build_task_state_unknown')

    if task_params_dict.get("image_start_paths"):
        loaded = sm_load_pil_images(
            task_params_dict["image_start_paths"],
            wgp_mod.convert_image,
            image_download_dir,
            current_task_id_for_log,
            dprint
        )
        if loaded: ui_defaults["image_start"] = loaded

    if task_params_dict.get("image_end_paths"):
        loaded = sm_load_pil_images(
            task_params_dict["image_end_paths"],
            wgp_mod.convert_image,
            image_download_dir,
            current_task_id_for_log,
            dprint
        )
        if loaded: ui_defaults["image_end"] = loaded

    if task_params_dict.get("image_refs_paths"):
        loaded = sm_load_pil_images(
            task_params_dict["image_refs_paths"],
            wgp_mod.convert_image,
            image_download_dir,
            current_task_id_for_log,
            dprint
        )
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
        
        causvid_lora_basename = "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
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
            final_multipliers.append("1.0")

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
            ui_defaults["loras_multipliers"] = "1.0"
            ui_defaults["activated_loras"] = [causvid_lora_basename] # ensure only causvid if no others

    if apply_reward_lora:
        print(f"[Task ID: {task_params_dict.get('task_id')}] Applying Reward LoRA settings.")

        reward_lora = {"filename": "Wan2.1-Fun-14B-InP-MPS_reward_lora_wgp.safetensors", "strength": "0.5"}

        # Get current activated LoRAs
        current_activated = ui_defaults.get("activated_loras", [])
        if not isinstance(current_activated, list):
            try:
                current_activated = [str(item).strip() for item in str(current_activated).split(',') if item.strip()]
            except:
                current_activated = []

        # Get current multipliers
        current_multipliers_str = ui_defaults.get("loras_multipliers", "")
        if isinstance(current_multipliers_str, (list, tuple)):
            current_multipliers_list = [str(m).strip() for m in current_multipliers_str if str(m).strip()]
        elif isinstance(current_multipliers_str, str):
            current_multipliers_list = [m.strip() for m in current_multipliers_str.split(" ") if m.strip()]
        else:
            current_multipliers_list = []

        # Pad multipliers to match activated LoRAs before creating map
        while len(current_multipliers_list) < len(current_activated):
            current_multipliers_list.append("1.0")

        # Create a dictionary to map lora to multiplier for easy update (preserves order in Python 3.7+)
        lora_mult_map = dict(zip(current_activated, current_multipliers_list))

        # Add/update reward lora
        lora_mult_map[reward_lora['filename']] = reward_lora['strength']

        ui_defaults["activated_loras"] = list(lora_mult_map.keys())
        ui_defaults["loras_multipliers"] = " ".join(list(lora_mult_map.values()))
        dprint(f"Reward LoRA applied. Activated: {ui_defaults['activated_loras']}, Multipliers: {ui_defaults['loras_multipliers']}")

    # Apply additional LoRAs that may have been passed via task params (e.g. from travel orchestrator)
    processed_additional_loras = task_params_dict.get("processed_additional_loras", {})
    if processed_additional_loras:
        dprint(f"[Task ID: {task_id_for_dprint}] Applying processed additional LoRAs: {processed_additional_loras}")
        
        # Get current activated LoRAs and multipliers again, as they may have been modified by other logic.
        current_activated = ui_defaults.get("activated_loras", [])
        if not isinstance(current_activated, list):
            try:
                current_activated = [str(item).strip() for item in str(current_activated).split(',') if item.strip()]
            except:
                current_activated = []

        current_multipliers_str = ui_defaults.get("loras_multipliers", "")
        if isinstance(current_multipliers_str, (list, tuple)):
            current_multipliers_list = [str(m).strip() for m in current_multipliers_str if str(m).strip()]
        elif isinstance(current_multipliers_str, str):
            current_multipliers_list = [m.strip() for m in current_multipliers_str.split(" ") if m.strip()]
        else:
            current_multipliers_list = []

        # Pad multipliers to match activated LoRAs before creating map
        while len(current_multipliers_list) < len(current_activated):
            current_multipliers_list.append("1.0")

        lora_mult_map = dict(zip(current_activated, current_multipliers_list))
        
        # Add/update additional loras - this will overwrite strength if lora was already present
        lora_mult_map.update(processed_additional_loras)

        ui_defaults["activated_loras"] = list(lora_mult_map.keys())
        ui_defaults["loras_multipliers"] = " ".join(list(lora_mult_map.values()))
        dprint(f"Additional LoRAs applied. Final Activated: {ui_defaults['activated_loras']}, Final Multipliers: {ui_defaults['loras_multipliers']}")

    state[model_type_key] = ui_defaults
    return state, ui_defaults


# -----------------------------------------------------------------------------
# 6. Process a single task dictionary from the tasks.json list
# -----------------------------------------------------------------------------

def process_single_task(wgp_mod, task_params_dict, main_output_dir_base: Path, task_type: str, project_id_for_task: str | None, image_download_dir: Path | str | None = None, apply_reward_lora: bool = False, colour_match_videos: bool = False, mask_active_frames: bool = True):
    dprint(f"--- Entering process_single_task ---")
    dprint(f"Task Type: {task_type}")
    dprint(f"Project ID for task: {project_id_for_task}") # Added dprint for project_id
    dprint(f"Task Params (first 1000 chars): {json.dumps(task_params_dict, default=str, indent=2)[:1000]}...")
    # dprint(f"PROCESS_SINGLE_TASK received task_params_dict: {json.dumps(task_params_dict)}") # DEBUG ADDED - Now covered by above
    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))
    print(f"--- Processing task ID: {task_id} of type: {task_type} ---")
    output_location_to_db = None # Will store the final path/URL for the DB

    # --- Check for new task type ---
    if task_type == "generate_openpose":
        print(f"[Task ID: {task_id}] Identified as 'generate_openpose' task.")
        # This handler might need access to wgp_mod for some utilities, so pass it along.
        # It's better to pass dependencies explicitly than rely on globals.
        return handle_generate_openpose_task(task_params_dict, main_output_dir_base, task_id, dprint)
    elif task_type == "rife_interpolate_images":
        print(f"[Task ID: {task_id}] Identified as 'rife_interpolate_images' task.")
        return handle_rife_interpolate_task(wgp_mod, task_params_dict, main_output_dir_base, task_id, dprint)

    # --- SM_RESTRUCTURE: Add new travel task handlers ---
    elif task_type == "travel_orchestrator":
        print(f"[Task ID: {task_id}] Identified as 'travel_orchestrator' task.")
        return tbi._handle_travel_orchestrator_task(task_params_from_db=task_params_dict, main_output_dir_base=main_output_dir_base, orchestrator_task_id_str=task_id, orchestrator_project_id=project_id_for_task, dprint=dprint)
    elif task_type == "travel_segment":
        print(f"[Task ID: {task_id}] Identified as 'travel_segment' task.")
        # This will call wgp_mod like a standard task but might have pre/post processing
        # based on orchestrator details passed in its params.
        return tbi._handle_travel_segment_task(wgp_mod, task_params_dict, main_output_dir_base, task_id, apply_reward_lora, colour_match_videos, mask_active_frames, process_single_task=process_single_task, dprint=dprint)
    elif task_type == "travel_stitch":
        print(f"[Task ID: {task_id}] Identified as 'travel_stitch' task.")
        return tbi._handle_travel_stitch_task(task_params_from_db=task_params_dict, main_output_dir_base=main_output_dir_base, stitch_task_id_str=task_id, dprint=dprint)
    # --- End SM_RESTRUCTURE ---
    
    # --- Different Pose Orchestrator ---
    elif task_type == "different_pose_orchestrator":
        print(f"[Task ID: {task_id}] Identified as 'different_pose_orchestrator' task.")
        return dp._handle_different_pose_orchestrator_task(
            task_params_from_db=task_params_dict,
            main_output_dir_base=main_output_dir_base,
            orchestrator_task_id_str=task_id,
            dprint=dprint
        )
    # --- End Different Pose ---

    # Default handling for standard wgp tasks (original logic)
    task_model_type_logical = task_params_dict.get("model", "t2v")
    # Determine the actual model filename before checking/downloading LoRA, as LoRA path depends on it
    model_filename_for_task = wgp_mod.get_model_filename(task_model_type_logical,
                                                         wgp_mod.transformer_quantization,
                                                         wgp_mod.transformer_dtype_policy)
    
    if apply_reward_lora:
        print(f"[Task ID: {task_id}] --apply-reward-lora flag is active. Checking and downloading reward LoRA.")
        reward_lora_data = {
            "url": "https://huggingface.co/peteromallet/Wan2.1-Fun-14B-InP-MPS_reward_lora_diffusers/resolve/main/Wan2.1-Fun-14B-InP-MPS_reward_lora_wgp.safetensors",
            "filename": "Wan2.1-Fun-14B-InP-MPS_reward_lora_wgp.safetensors"
        }
        
        base_lora_dir_for_model = Path(wgp_mod.get_lora_dir(model_filename_for_task))
        download_file(reward_lora_data['url'], base_lora_dir_for_model, reward_lora_data['filename'])

    effective_image_download_dir = image_download_dir # Use passed-in dir if available

    if effective_image_download_dir is None: # Not passed, so determine for this standard/individual task
        if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH: # SQLITE_DB_PATH is global
            try:
                sqlite_db_path_obj = Path(db_ops.SQLITE_DB_PATH).resolve()
                if sqlite_db_path_obj.is_file():
                    sqlite_db_parent_dir = sqlite_db_path_obj.parent
                    candidate_download_dir = sqlite_db_parent_dir / "public" / "data" / "image_downloads" / task_id
                    candidate_download_dir.mkdir(parents=True, exist_ok=True)
                    effective_image_download_dir = str(candidate_download_dir.resolve())
                    dprint(f"Task {task_id}: Determined SQLite-based image_download_dir for standard task: {effective_image_download_dir}")
                else:
                    dprint(f"Task {task_id}: SQLITE_DB_PATH '{db_ops.SQLITE_DB_PATH}' is not a file. Cannot determine parent for image_download_dir.")        
            except Exception as e_idir_sqlite:
                dprint(f"Task {task_id}: Could not create SQLite-based image_download_dir for standard task: {e_idir_sqlite}.")
        # Add similar logic for Supabase if a writable shared path convention exists.

    use_causvid = task_params_dict.get("use_causvid_lora", False)
    causvid_lora_basename = "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
    causvid_lora_url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"

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

    # Handle additional LoRAs from task params, which may be passed by travel orchestrator
    additional_loras = task_params_dict.get("additional_loras", {})
    if additional_loras:
        dprint(f"[Task ID: {task_id}] Processing additional LoRAs: {additional_loras}")
        base_lora_dir_for_model = Path(wgp_mod.get_lora_dir(model_filename_for_task))
        processed_loras = {}
        for lora_path, lora_strength in additional_loras.items():
            try:
                lora_filename = Path(urllib.parse.urlparse(lora_path).path).name
                
                if lora_path.startswith("http://") or lora_path.startswith("https://"):
                    # URL case: download the LoRA to central Wan2GP/loras directory
                    wan2gp_loras_dir = Path("Wan2GP/loras")
                    wan2gp_loras_dir.mkdir(parents=True, exist_ok=True)
                    dprint(f"Task {task_id}: Downloading LoRA from {lora_path} to {wan2gp_loras_dir}")
                    download_file(lora_path, wan2gp_loras_dir, lora_filename)
                    
                    # Now copy from Wan2GP/loras to model-specific directory if needed
                    downloaded_lora_path = wan2gp_loras_dir / lora_filename
                    target_path = base_lora_dir_for_model / lora_filename
                    if not target_path.exists() or downloaded_lora_path.resolve() != target_path.resolve():
                        shutil.copy(str(downloaded_lora_path), str(target_path))
                        dprint(f"Copied downloaded LoRA from {downloaded_lora_path} to {target_path}")
                    else:
                        dprint(f"Downloaded LoRA already exists at target: {target_path}")
                else:
                    # Local file case: check multiple locations
                    source_path = Path(lora_path)
                    target_path = base_lora_dir_for_model / lora_filename
                    
                    if source_path.is_absolute() and source_path.exists():
                        # Full absolute path that exists
                        if not target_path.exists() or source_path.resolve() != target_path.resolve():
                            shutil.copy(str(source_path), str(target_path))
                            dprint(f"Copied local LoRA from {source_path} to {target_path}")
                        else:
                            dprint(f"LoRA already exists at target: {target_path}")
                    elif (base_lora_dir_for_model / lora_path).exists():
                        # Relative path within LoRA directory (e.g., "my_lora.safetensors" or "subfolder/my_lora.safetensors")
                        existing_lora_path = base_lora_dir_for_model / lora_path
                        lora_filename = existing_lora_path.name  # Update filename to actual file
                        dprint(f"Found existing LoRA in directory: {existing_lora_path}")
                        # No need to copy, it's already in the right place
                    elif (Path("Wan2GP/loras") / lora_path).exists():
                        # Check standard Wan2GP/loras directory
                        wan2gp_lora_path = Path("Wan2GP/loras") / lora_path
                        target_path = base_lora_dir_for_model / wan2gp_lora_path.name
                        if not target_path.exists() or wan2gp_lora_path.resolve() != target_path.resolve():
                            shutil.copy(str(wan2gp_lora_path), str(target_path))
                            dprint(f"Copied LoRA from Wan2GP/loras: {wan2gp_lora_path} to {target_path}")
                        else:
                            dprint(f"LoRA from Wan2GP/loras already exists at target: {target_path}")
                        lora_filename = wan2gp_lora_path.name
                    elif source_path.exists():
                        # Relative path from current working directory
                        if not target_path.exists() or source_path.resolve() != target_path.resolve():
                            shutil.copy(str(source_path), str(target_path))
                            dprint(f"Copied local LoRA from {source_path} to {target_path}")
                        else:
                            dprint(f"LoRA already exists at target: {target_path}")
                    else:
                        dprint(f"[WARNING Task ID: {task_id}] LoRA not found at any location: '{lora_path}'. Checked: full path, LoRA directory, and relative path. Skipping.")
                        continue # Skip this lora
                
                processed_loras[lora_filename] = str(lora_strength)
            except Exception as e_lora:
                print(f"[ERROR Task ID: {task_id}] Failed to process additional LoRA {lora_path}: {e_lora}")
        
        task_params_dict["processed_additional_loras"] = processed_loras

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

    state, ui_params = build_task_state(wgp_mod, model_filename_for_task, task_params_dict, all_loras_for_active_model, image_download_dir, apply_reward_lora=apply_reward_lora)
    
    gen_task_placeholder = {"id": 1, "prompt": ui_params.get("prompt"), "params": {"model_filename_from_gui_state": model_filename_for_task, "model": task_model_type_logical}}
    send_cmd = make_send_cmd(task_id)

    # Adjust resolution in ui_params to be multiples of 16 for consistency with image processing
    if "resolution" in ui_params:
        try:
            width, height = map(int, ui_params["resolution"].split("x"))
            new_width, new_height = (width // 16) * 16, (height // 16) * 16
            ui_params["resolution"] = f"{new_width}x{new_height}"
            dprint(f"Adjusted resolution in ui_params to {ui_params['resolution']}")
        except Exception as err:
            dprint(f"Error adjusting resolution: {err}")
    
    tea_cache_value = ui_params.get("tea_cache_setting", ui_params.get("tea_cache", 0.0))

    print(f"[Task ID: {task_id}] Starting generation with effective params: {json.dumps(ui_params, default=lambda o: 'Unserializable' if isinstance(o, Image.Image) else o.__dict__ if hasattr(o, '__dict__') else str(o), indent=2)}")
    generation_success = False

    # --- Determine frame_num for wgp.py --- 
    requested_frames_from_task = ui_params.get("video_length", 81)
    frame_num_for_wgp = requested_frames_from_task
    dprint(f"[Task ID: {task_id}] Using requested frame count: {frame_num_for_wgp}")
    # --- End frame_num determination ---

    ui_params["video_length"] = frame_num_for_wgp

    try:
        dprint(f"[Task ID: {task_id}] Calling wgp_mod.generate_video with effective ui_params (first 1000 chars): {json.dumps(ui_params, default=lambda o: 'Unserializable' if isinstance(o, Image.Image) else o.__dict__ if hasattr(o, '__dict__') else str(o), indent=2)[:1000]}...")
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
            remove_background_images_ref=ui_params.get("remove_background_images_ref", False),
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
        # Find the generated video file(s)
        generated_video_files = sorted([
            item for item in Path(temp_output_dir).iterdir()
            if item.is_file() and item.suffix.lower() == ".mp4"
        ])
        
        generated_video_file = None
        if not generated_video_files:
            print(f"[WARNING Task ID: {task_id}] Generation reported success, but no .mp4 files found in {temp_output_dir}")
            generation_success = False
        elif len(generated_video_files) == 1:
            generated_video_file = generated_video_files[0]
            dprint(f"[Task ID: {task_id}] Found a single generated video: {generated_video_file}")
        else:
            dprint(f"[Task ID: {task_id}] Found {len(generated_video_files)} video segments to stitch: {generated_video_files}")
            # Stitch the videos together
            stitched_video_path = Path(temp_output_dir) / f"{task_id}_stitched.mp4"
            
            # Ensure the stitching utility is available
            if 'sm_stitch_videos_ffmpeg' not in globals() and 'sm_stitch_videos_ffmpeg' not in locals():
                try:
                    from sm_functions.common_utils import stitch_videos_ffmpeg as sm_stitch_videos_ffmpeg
                except ImportError:
                    print(f"[CRITICAL ERROR Task ID: {task_id}] Failed to import 'stitch_videos_ffmpeg'. Cannot proceed with stitching.")
                    generation_success = False # Cannot proceed
            
            if generation_success: # Check if import was successful before proceeding
                try:
                    # The function expects a list of strings
                    video_paths_str = [str(p.resolve()) for p in generated_video_files]
                    sm_stitch_videos_ffmpeg(video_paths_str, str(stitched_video_path.resolve()))
                    
                    if stitched_video_path.exists() and stitched_video_path.stat().st_size > 0:
                        generated_video_file = stitched_video_path
                        dprint(f"[Task ID: {task_id}] Successfully stitched video segments to: {generated_video_file}")
                        # Optional: clean up the original segments now
                        for segment_file in generated_video_files:
                            try:
                                segment_file.unlink()
                            except OSError as e_clean:
                                print(f"[WARNING Task ID: {task_id}] Could not delete segment file {segment_file}: {e_clean}")
                    else:
                        print(f"[ERROR Task ID: {task_id}] Stitching failed. Output file '{stitched_video_path}' not created or is empty.")
                        generation_success = False
                except Exception as e_stitch:
                    print(f"[ERROR Task ID: {task_id}] An exception occurred during video stitching: {e_stitch}")
                    traceback.print_exc()
                    generation_success = False

        if generated_video_file and generation_success:
            dprint(f"[Task ID: {task_id}] Processing final video: {generated_video_file}")
            if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
                sqlite_db_file_path = Path(db_ops.SQLITE_DB_PATH).resolve()
                target_files_dir = sqlite_db_file_path.parent / "public" / "files"
                target_files_dir.mkdir(parents=True, exist_ok=True)
                
                final_video_path = sm_get_unique_target_path(
                    target_files_dir, 
                    task_id, # name_stem
                    generated_video_file.suffix # suffix (e.g., ".mp4")
                )
                output_location_to_db = f"files/{final_video_path.name}"
                try:
                    shutil.move(str(generated_video_file), str(final_video_path))
                    print(f"[Task ID: {task_id}] Output video saved to: {final_video_path.resolve()} (DB location: {output_location_to_db})")
    
                    if task_params_dict.get("output_path"):
                        default_dir_to_clean = (main_output_dir_base / task_id)
                        if default_dir_to_clean.exists() and default_dir_to_clean.is_dir() and default_dir_to_clean != final_video_path.parent:
                            try:
                                shutil.rmtree(default_dir_to_clean)
                                dprint(f"[Task ID: {task_id}] Cleaned up auxiliary task directory: {default_dir_to_clean}")
                            except Exception as e_cleanup_aux:
                                print(f"[WARNING Task ID: {task_id}] Could not clean up auxiliary task directory {default_dir_to_clean}: {e_cleanup_aux}")
                except Exception as e_move:
                    print(f"[ERROR Task ID: {task_id}] Failed to move video to final local destination: {e_move}")
                    generation_success = False
            elif db_ops.DB_TYPE == "supabase" and db_ops.SUPABASE_CLIENT:
                encoded_file_name = urllib.parse.quote(generated_video_file.name)
                object_name = f"{task_id}/{encoded_file_name}"
                dprint(f"[Task ID: {task_id}] Original filename: {generated_video_file.name}")
                dprint(f"[Task ID: {task_id}] Encoded filename for Supabase object: {encoded_file_name}")
                dprint(f"[Task ID: {task_id}] Final Supabase object_name: {object_name}")
                
                public_url = db_ops.upload_to_supabase_storage(generated_video_file, object_name, db_ops.SUPABASE_VIDEO_BUCKET)
                if public_url:
                    output_location_to_db = public_url
                else:
                    print(f"[WARNING Task ID: {task_id}] Supabase upload failed or no URL returned. No output location will be saved.")
                    generation_success = False
            else:
                print(f"[WARNING Task ID: {task_id}] Output generated but DB_TYPE ({db_ops.DB_TYPE}) is not sqlite or Supabase client is not configured for upload.")
                generation_success = False
        else:
            print(f"[WARNING Task ID: {task_id}] Generation reported success, but no .mp4 file found in {temp_output_dir}")
            generation_success = False
    
    try:
        shutil.rmtree(temp_output_dir)
        dprint(f"[Task ID: {task_id}] Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e_clean:
        print(f"[WARNING Task ID: {task_id}] Failed to clean up temporary directory {temp_output_dir}: {e_clean}")

    if generation_success:
        chaining_result_path_override = None

        if task_params_dict.get("travel_chain_details"):
            dprint(f"WGP Task {task_id} is part of a travel sequence. Attempting to chain.")
            chain_success, chain_message, final_path_from_chaining = tbi._handle_travel_chaining_after_wgp(
                wgp_task_params=task_params_dict, 
                actual_wgp_output_video_path=output_location_to_db,
                wgp_mod=wgp_mod,
                image_download_dir=image_download_dir,
                dprint=dprint
            )
            if chain_success:
                chaining_result_path_override = final_path_from_chaining
                dprint(f"Task {task_id}: Travel chaining successful. Message: {chain_message}")
            else:
                print(f"[ERROR Task ID: {task_id}] Travel sequence chaining failed after WGP completion: {chain_message}. The raw WGP output '{output_location_to_db}' will be used for this task's DB record.")
        
        elif task_params_dict.get("different_pose_chain_details"):
            dprint(f"Task {task_id} is part of a different_pose sequence. Attempting to chain.")
            
            # This call will enqueue the next task in the sequence.
            # It may return a new path if post-processing occurred on the current task's output.
            chain_success, chain_message, final_path_from_chaining = dp._handle_different_pose_chaining(
                completed_task_params=task_params_dict, 
                task_output_path=output_location_to_db,
                dprint=dprint
            )
            if chain_success:
                # The final path of the *entire* sequence is what matters for the orchestrator task,
                # but for this intermediate step, we just need to know if the output path was modified.
                chaining_result_path_override = final_path_from_chaining
                dprint(f"Task {task_id}: Different Pose chaining successful. Message: {chain_message}")
            else:
                print(f"[ERROR Task ID: {task_id}] Different Pose sequence chaining failed: {chain_message}. This may halt the sequence.")


        if chaining_result_path_override:
            # This logic handles if the chaining function (travel or other) returned a new, modified path for the output.
            # E.g., a path to a color-corrected video.
            path_to_check_existence: Path | None = None
            if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and isinstance(chaining_result_path_override, str) and chaining_result_path_override.startswith("files/"):
                sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                path_to_check_existence = (sqlite_db_parent / "public" / chaining_result_path_override).resolve()
                dprint(f"Task {task_id}: Chaining returned SQLite relative path '{chaining_result_path_override}'. Resolved to '{path_to_check_existence}' for existence check.")
            elif chaining_result_path_override:
                # It could be an absolute path from Supabase or other non-SQLite setups
                path_to_check_existence = Path(chaining_result_path_override).resolve()
                dprint(f"Task {task_id}: Chaining returned absolute-like path '{chaining_result_path_override}'. Resolved to '{path_to_check_existence}' for existence check.")

            if path_to_check_existence and path_to_check_existence.exists() and path_to_check_existence.is_file():
                # A crude check to see if the path changed, for logging purposes.
                is_output_path_different = str(chaining_result_path_override) != str(output_location_to_db)
                if is_output_path_different:
                    dprint(f"Task {task_id}: Chaining modified output path for DB. Original: {output_location_to_db}, New: {chaining_result_path_override} (Checked file: {path_to_check_existence})")
                output_location_to_db = chaining_result_path_override
            elif chaining_result_path_override is not None:
                # A path was returned but it's not valid. This is a warning. The original output will be used.
                print(f"[WARNING Task ID: {task_id}] Chaining reported success, but final path '{chaining_result_path_override}' (checked as '{path_to_check_existence}') is invalid or not a file. Using original WGP output '{output_location_to_db}' for DB.")


    print(f"--- Finished task ID: {task_id} (Success: {generation_success}) ---")
    return generation_success, output_location_to_db


# -----------------------------------------------------------------------------
# 7. Main server loop
# -----------------------------------------------------------------------------

def main():
    load_dotenv() # Load .env file variables into environment
    global DB_TYPE, SQLITE_DB_PATH, SUPABASE_CLIENT, SUPABASE_VIDEO_BUCKET

    # Determine DB type from environment variables
    env_db_type = os.getenv("DB_TYPE", "sqlite").lower()
    env_pg_table_name = os.getenv("POSTGRES_TABLE_NAME", "tasks")
    env_supabase_url = os.getenv("SUPABASE_URL")
    env_supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    env_supabase_bucket = os.getenv("SUPABASE_VIDEO_BUCKET", "videos")
    env_sqlite_db_path = os.getenv("SQLITE_DB_PATH_ENV") # Read SQLite DB path from .env

    cli_args = parse_args()

    # --- Handle --delete-db flag ---
    if cli_args.delete_db:
        db_file_to_delete = cli_args.db_file
        env_sqlite_db_path = os.getenv("SQLITE_DB_PATH_ENV")
        if env_sqlite_db_path:
            db_file_to_delete = env_sqlite_db_path
        
        db_files_to_remove = [
            db_file_to_delete,
            f"{db_file_to_delete}-wal",
            f"{db_file_to_delete}-shm"
        ]
        
        for db_file in db_files_to_remove:
            if Path(db_file).exists():
                try:
                    Path(db_file).unlink()
                    print(f"[DELETE-DB] Removed: {db_file}")
                except Exception as e:
                    print(f"[DELETE-DB ERROR] Could not remove {db_file}: {e}")
        
        print("[DELETE-DB] Database cleanup complete. Starting fresh.")
    # --- End delete-db handling ---

    # --- Setup logging to file if requested ---
    log_file = None
    if cli_args.save_logging:
        import logging
        log_file_path = Path(cli_args.save_logging)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup file handler for logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
                logging.StreamHandler()  # Also keep console output
            ]
        )
        
        # Redirect print statements to also log to file
        original_print = print
        def enhanced_print(*args, **kwargs):
            # Call original print for console output
            original_print(*args, **kwargs)
            # Also log to file
            message = ' '.join(str(arg) for arg in args)
            logging.info(message)
        
        # Override print globally
        import builtins
        builtins.print = enhanced_print
        
        print(f"[LOGGING] All output will be saved to: {log_file_path.resolve()}")
    # --- End logging setup ---

    # --- Configure DB Type and Connection Globals ---
    # This block sets DB_TYPE, SQLITE_DB_PATH, SUPABASE_CLIENT, etc. in the db_ops module
    if env_db_type == "supabase" and env_supabase_url and env_supabase_key:
        try:
            temp_supabase_client = create_client(env_supabase_url, env_supabase_key)
            if temp_supabase_client:
                db_ops.DB_TYPE = "supabase"
                db_ops.PG_TABLE_NAME = env_pg_table_name
                db_ops.SUPABASE_URL = env_supabase_url
                db_ops.SUPABASE_SERVICE_KEY = env_supabase_key
                db_ops.SUPABASE_VIDEO_BUCKET = env_supabase_bucket
                db_ops.SUPABASE_CLIENT = temp_supabase_client
                # Also set local globals for non-db logic if needed
                DB_TYPE = "supabase"
                SUPABASE_CLIENT = temp_supabase_client
                SUPABASE_VIDEO_BUCKET = env_supabase_bucket
            else:
                raise Exception("Supabase client creation returned None")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Supabase client: {e}. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.")
            print("Falling back to SQLite due to Supabase client initialization error.")
            db_ops.DB_TYPE = "sqlite"
            db_ops.SQLITE_DB_PATH = env_sqlite_db_path if env_sqlite_db_path else cli_args.db_file
            DB_TYPE = "sqlite"
            SQLITE_DB_PATH = db_ops.SQLITE_DB_PATH
    else: # Default to sqlite if .env DB_TYPE is unrecognized or not set, or if it's explicitly "sqlite"
        if env_db_type != "sqlite":
            print(f"DB_TYPE '{env_db_type}' in .env is not recognized. Defaulting to SQLite.")
        db_ops.DB_TYPE = "sqlite"
        db_ops.SQLITE_DB_PATH = env_sqlite_db_path if env_sqlite_db_path else cli_args.db_file
        DB_TYPE = "sqlite"
        SQLITE_DB_PATH = db_ops.SQLITE_DB_PATH
    # --- End DB Type Configuration ---

    # --- Run DB Migrations ---
    # Must be after DB type/config is determined but before DB schema is strictly enforced by init_db or heavy use.
    db_ops._run_db_migrations()
    # --- End DB Migrations ---

    # --- Handle --migrate-only flag --- (Section 6)
    if cli_args.migrate_only:
        print("Database migrations complete (called with --migrate-only). Exiting.")
        sys.exit(0)
    # --- End --migrate-only handler ---


    main_output_dir = Path(cli_args.main_output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"WanGP Headless Server Started.")
    if db_ops.DB_TYPE == "supabase":
        print(f"Monitoring Supabase (PostgreSQL backend) table: {db_ops.PG_TABLE_NAME}")
    else: # SQLite
        print(f"Monitoring SQLite database: {db_ops.SQLITE_DB_PATH}")
    print(f"Outputs will be saved under: {main_output_dir}")
    print(f"Polling interval: {cli_args.poll_interval} seconds.")

    # Initialize database
    if db_ops.DB_TYPE == "supabase":
        db_ops.init_db_supabase() # New call, uses globals in db_ops
    else: # SQLite
        db_ops.init_db() # Existing SQLite init

    # Activate global debug switch early so that all subsequent code paths can use dprint()
    global debug_mode
    debug_mode = cli_args.debug
    db_ops.debug_mode = cli_args.debug # Also set it in the db_ops module
    dprint("Verbose debug logging enabled.")

    # original_argv = sys.argv.copy() # No longer needed to copy if not restoring fully
    sys.argv = ["Wan2GP/wgp.py"] # Set for wgp.py import and its potential internal arg parsing
    patch_gradio()
    
    # Import wgp from the Wan2GP sub-package
    from Wan2GP import wgp as wgp_mod
    
    # sys.argv = original_argv # DO NOT RESTORE: This was causing wgp.py to parse headless.py args

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

    db_path_str = str(db_ops.SQLITE_DB_PATH) if db_ops.DB_TYPE == "sqlite" else db_ops.PG_TABLE_NAME # Use consistent string path for db functions

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
        # --- Add a one-time diagnostic log for task counts ---
        if db_ops.DB_TYPE == "sqlite" and debug_mode:
            try:
                counts = db_ops.get_initial_task_counts()
                if counts:
                    total_tasks, queued_tasks = counts
                    dprint(f"SQLite Initial State: Total tasks in '{db_ops.PG_TABLE_NAME}': {total_tasks}. Tasks with status '{db_ops.STATUS_QUEUED}': {queued_tasks}.")
            except Exception as e_diag:
                dprint(f"SQLite Diagnostic Error: Could not get initial task counts: {e_diag}")
        # --- End one-time diagnostic log ---

        while True:
            task_info = None
            current_task_id_for_status_update = None # Used to hold the task_id for status updates
            current_project_id = None # To hold the project_id for the current task

            if db_ops.DB_TYPE == "supabase":
                dprint(f"Checking for queued tasks in Supabase (PostgreSQL backend) table {db_ops.PG_TABLE_NAME} via Supabase RPC...")
                task_info = db_ops.get_oldest_queued_task_supabase()
                dprint(f"Supabase task_info: {task_info}") # ADDED DPRINT
                if task_info:
                    current_task_id_for_status_update = task_info["task_id"]
                    # Status is already set to IN_PROGRESS by func_claim_task RPC
            else: # SQLite
                dprint(f"Checking for queued tasks in SQLite {db_ops.SQLITE_DB_PATH}...")
                task_info = db_ops.get_oldest_queued_task()
                dprint(f"SQLite task_info: {task_info}") # ADDED DPRINT
                if task_info:
                    current_task_id_for_status_update = task_info["task_id"]
                    db_ops.update_task_status(current_task_id_for_status_update, db_ops.STATUS_IN_PROGRESS)

            if not task_info:
                dprint("No queued tasks found. Sleeping...")
                if db_ops.DB_TYPE == "sqlite":
                    # Wait until either the WAL/db file changes or the normal
                    # poll interval elapses.  This reduces perceived latency
                    # without hammering the database.
                    _wait_for_sqlite_change(db_ops.SQLITE_DB_PATH, cli_args.poll_interval)
                else:
                    time.sleep(cli_args.poll_interval)
                continue

            # current_task_data = task_info["params"] # Params are already a dict
            current_task_params = task_info["params"]
            current_task_type = task_info["task_type"] # Retrieve task_type
            current_project_id = task_info.get("project_id") # Get project_id, might be None if not returned
            
            # This fallback logic remains, but it's less likely to be needed
            # if get_oldest_queued_task and its supabase equivalent are reliable.
            if current_project_id is None and current_task_id_for_status_update:
                dprint(f"Project ID not directly available for task {current_task_id_for_status_update}. Attempting to fetch manually...")
                if db_ops.DB_TYPE == "supabase" and db_ops.SUPABASE_CLIENT:
                    try:
                        # Using 'id' as the column name for task_id based on Supabase schema conventions seen elsewhere (e.g. init_db)
                        response = db_ops.SUPABASE_CLIENT.table(db_ops.PG_TABLE_NAME)\
                            .select("project_id")\
                            .eq("id", current_task_id_for_status_update)\
                            .single()\
                            .execute()
                        if response.data and response.data.get("project_id"):
                            current_project_id = response.data["project_id"]
                            dprint(f"Successfully fetched project_id '{current_project_id}' for task {current_task_id_for_status_update} from Supabase.")
                        else:
                            dprint(f"Could not fetch project_id for task {current_task_id_for_status_update} from Supabase. Response data: {response.data}, error: {response.error}")
                    except Exception as e_fetch_proj_id:
                        dprint(f"Exception while fetching project_id for {current_task_id_for_status_update} from Supabase: {e_fetch_proj_id}")
                elif db_ops.DB_TYPE == "sqlite": # Should have been fetched, but as a fallback
                    # This fallback no longer needs its own db connection logic.
                    # A new helper could be added to db_ops if this is truly needed,
                    # but for now, we assume the primary fetch works.
                    dprint(f"Project_id was not fetched for {current_task_id_for_status_update} from SQLite. This is unexpected.")

            
            # Critical check: project_id is NOT NULL for sub-tasks created by orchestrator
            if current_project_id is None and current_task_type == "travel_orchestrator":
                print(f"[CRITICAL ERROR] Task {current_task_id_for_status_update} (travel_orchestrator) has no project_id. Sub-tasks cannot be created. Skipping task.")
                # Update status to FAILED to prevent re-processing this broken state
                error_message_for_db = "Failed: Orchestrator task missing project_id, cannot create sub-tasks."
                if db_ops.DB_TYPE == "supabase":
                    db_ops.update_task_status_supabase(current_task_id_for_status_update, db_ops.STATUS_FAILED, error_message_for_db)
                else:
                    db_ops.update_task_status(current_task_id_for_status_update, db_ops.STATUS_FAILED, error_message_for_db)
                time.sleep(1) # Brief pause
                continue # Skip to next polling cycle

            print(f"Found task: {current_task_id_for_status_update} of type: {current_task_type}, Project ID: {current_project_id}")
            # Status already set to IN_PROGRESS if task_info is not None

            # Inserted: define segment_image_download_dir from task params if available
            segment_image_download_dir = current_task_params.get("segment_image_download_dir")
            
            task_succeeded, output_location = process_single_task(
                wgp_mod, current_task_params, main_output_dir, current_task_type, current_project_id,
                image_download_dir=segment_image_download_dir,
                apply_reward_lora=cli_args.apply_reward_lora,
                colour_match_videos=cli_args.colour_match_videos,
                mask_active_frames=cli_args.mask_active_frames
            )

            if task_succeeded:
                if db_ops.DB_TYPE == "supabase":
                    db_ops.update_task_status_supabase(current_task_id_for_status_update, db_ops.STATUS_COMPLETE, output_location)
                else:
                    db_ops.update_task_status(current_task_id_for_status_update, db_ops.STATUS_COMPLETE, output_location)
                print(f"Task {current_task_id_for_status_update} completed successfully. Output location: {output_location}")
            else:
                if db_ops.DB_TYPE == "supabase":
                    db_ops.update_task_status_supabase(current_task_id_for_status_update, db_ops.STATUS_FAILED, output_location)
                else:
                    db_ops.update_task_status(current_task_id_for_status_update, db_ops.STATUS_FAILED, output_location)
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



def _wait_for_sqlite_change(db_path_str: str, timeout_seconds: int):
    """Block up to timeout_seconds waiting for mtime change on db / -wal / -shm.

    This lets the headless server react almost immediately when another process
    commits a transaction that only touches the WAL file.  We fall back to a
    normal sleep when the auxiliary files do not exist (e.g. before first
    write).
    """
    related_paths = [
        Path(db_path_str),
        Path(f"{db_path_str}-wal"),
        Path(f"{db_path_str}-shm"),
    ]

    # Snapshot the most-recent modification time we can observe now.
    last_mtime = 0.0
    for p in related_paths:
        try:
            last_mtime = max(last_mtime, p.stat().st_mtime)
        except FileNotFoundError:
            # Aux file not created yet – ignore.
            pass

    # Poll in small increments until something changes or timeout expires.
    poll_step = 0.25  # seconds
    waited = 0.0
    while waited < timeout_seconds:
        time.sleep(poll_step)
        waited += poll_step
        for p in related_paths:
            try:
                if p.stat().st_mtime > last_mtime:
                    return  # Change detected – return immediately
            except FileNotFoundError:
                # File still missing – keep waiting
                pass
    # Timed out – return control to caller
    return

if __name__ == "__main__":
    main() 