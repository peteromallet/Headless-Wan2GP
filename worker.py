"""Wan2GP Worker Server.

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

Wan 2.2 Support:
• Default task_type "vace" now uses Wan 2.2 (vace_14B_cocktail_2_2) for 2.5x speed improvement
• Use task_type "vace_21" to explicitly request Wan 2.1 compatibility
• Use task_type "vace_22" to explicitly request Wan 2.2 optimizations
• Wan 2.2 models automatically apply optimized settings (10 steps, guidance_scale=1.0, etc.)
• Built-in acceleration LoRAs (CausVid, DetailEnhancer) are auto-enabled for Wan 2.2
• All optimizations can be overridden by explicit task parameters
"""

import argparse
import sys
import os
import json
import time
import datetime
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
    handle_rife_interpolate_task,
    handle_extract_frame_task
)
from source.common_utils import (
    sm_get_unique_target_path,
    download_image_if_url as sm_download_image_if_url,
    download_file,
    load_pil_images as sm_load_pil_images,
    build_task_state,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    get_lora_dir_from_filename
)
from source.sm_functions import travel_between_images as tbi
from source.sm_functions import different_perspective as dp
from source.sm_functions import single_image as si
from source.sm_functions import magic_edit as me
# --- New Queue-based Architecture Imports ---
from headless_model_management import HeadlessTaskQueue, GenerationTask
# --- Structured Logging ---
from source.logging_utils import headless_logger, enable_debug_mode, disable_debug_mode
# --- End SM_RESTRUCTURE imports ---


# -----------------------------------------------------------------------------
# Debug / Verbose Logging Helpers (Legacy)
# -----------------------------------------------------------------------------

debug_mode = False  # This will be toggled on via the --debug CLI flag in main()

def dprint(msg: str):
    """Print a debug message if --debug flag is enabled. (Legacy - use logging_utils instead)"""
    if debug_mode:
        # Prefix with timestamp for easier tracing
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] {msg}")

# -----------------------------------------------------------------------------
# Queue-based Task Processing Functions
# -----------------------------------------------------------------------------

def db_task_to_generation_task(db_task_params: dict, task_id: str, task_type: str) -> GenerationTask:
    """
    Convert a database task row to a GenerationTask object for the queue system.
    
    Args:
        db_task_params: Task parameters from the database
        task_id: Task ID from the database
        task_type: Task type (used to determine model if not explicitly set)
        
    Returns:
        GenerationTask object ready for queue processing
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    dprint(f"Converting DB task {task_id} to GenerationTask")
    
    # Extract basic parameters
    prompt = db_task_params.get("prompt", "")
    if not prompt:
        raise ValueError(f"Task {task_id}: prompt is required")
    
    # Determine model - prefer explicit model param, otherwise infer from task_type
    model = db_task_params.get("model")
    if not model:
        # Map task types to model keys
        task_type_to_model = {
            "generate_video": "t2v",  # Default T2V
            "vace": "vace_14B_cocktail_2_2",  # Default to Wan 2.2 for better performance
            "vace_21": "vace_14B",  # Explicit Wan 2.1 VACE
            "vace_22": "vace_14B_cocktail_2_2",  # Explicit Wan 2.2 VACE
            "flux": "flux",
            "t2v": "t2v",
            "t2v_22": "t2v_2_2",  # Wan 2.2 T2V
            "i2v": "i2v_14B",
            "i2v_22": "i2v_2_2",  # Wan 2.2 I2V
            "hunyuan": "hunyuan",
            "ltxv": "ltxv_13B"
        }
        model = task_type_to_model.get(task_type, "t2v")  # Default to T2V
    
    # Create clean parameters dict for generation
    generation_params = {}
    
    # Core generation parameters
    param_whitelist = {
        "negative_prompt", "resolution", "video_length", "num_inference_steps", 
        "guidance_scale", "seed", "embedded_guidance_scale", "flow_shift",
        "audio_guidance_scale", "repeat_generation", "multi_images_gen_type",
        
        # VACE parameters
        "video_guide", "video_mask", "video_guide2", "video_mask2", 
        "video_prompt_type", "control_net_weight", "control_net_weight2",
        "keep_frames_video_guide", "video_guide_outpainting", "mask_expand",
        
        # Image parameters
        "image_prompt_type", "image_start", "image_end", "image_refs", 
        "frames_positions", "image_guide", "image_mask",
        
        # Video source parameters  
        "model_mode", "video_source", "keep_frames_video_source",
        
        # Audio parameters
        "audio_guide", "audio_guide2", "audio_source", "audio_prompt_type", "speakers_locations",
        
        # LoRA parameters
        "activated_loras", "loras_multipliers", "additional_loras", "use_causvid_lora", "use_lighti2x_lora",
        
        # Advanced parameters
        "tea_cache_setting", "tea_cache_start_step_perc", "RIFLEx_setting", 
        "slg_switch", "slg_layers", "slg_start_perc", "slg_end_perc",
        "cfg_star_switch", "cfg_zero_step", "prompt_enhancer",
        
        # Sliding window parameters
        "sliding_window_size", "sliding_window_overlap", "sliding_window_overlap_noise",
        "sliding_window_discard_last_frames",
        
        # Post-processing parameters
        "remove_background_images_ref", "temporal_upsampling", "spatial_upsampling",
        "film_grain_intensity", "film_grain_saturation",
        
        # Output parameters
        "output_dir", "custom_output_dir",
        
        # Special flags
        "apply_reward_lora"
    }
    
    # Copy whitelisted parameters
    for param in param_whitelist:
        if param in db_task_params:
            generation_params[param] = db_task_params[param]
    
    # Handle LoRA parameter format conversion
    if "activated_loras" in generation_params:
        loras = generation_params["activated_loras"]
        if isinstance(loras, str):
            # Convert comma-separated string to list
            generation_params["lora_names"] = [lora.strip() for lora in loras.split(",") if lora.strip()]
        elif isinstance(loras, list):
            generation_params["lora_names"] = loras
        del generation_params["activated_loras"]  # Remove old format
    
    if "loras_multipliers" in generation_params:
        multipliers = generation_params["loras_multipliers"]
        if isinstance(multipliers, str):
            # Convert comma-separated string to list of floats
            generation_params["lora_multipliers"] = [float(x.strip()) for x in multipliers.split(",") if x.strip()]
        # Keep as-is if already a list
    
    # Set default values for common parameters if not specified
    defaults = {
        "resolution": "1280x720",
        "video_length": 49,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "seed": -1,  # Random seed
        "negative_prompt": "",
    }
    
    for param, default_value in defaults.items():
        if param not in generation_params:
            generation_params[param] = default_value
    
    # Apply Wan 2.2 optimizations automatically (can be overridden by explicit task parameters)
    if "2_2" in model or "cocktail_2_2" in model:
        dprint(f"Task {task_id}: Applying Wan 2.2 optimizations for model '{model}'")
        
        # Wan 2.2 optimized defaults (only set if not explicitly provided)
        wan22_optimizations = {
            "num_inference_steps": 10,    # 2.5x faster than Wan 2.1's 25 steps
            "guidance_scale": 1.0,        # Optimized for Wan 2.2 architecture
            "flow_shift": 2.0,            # Better quality with Wan 2.2
            "switch_threshold": 875,      # Dual-phase switching point
        }
        
        for param, optimized_value in wan22_optimizations.items():
            if param not in db_task_params:  # Only apply if not explicitly set in task
                generation_params[param] = optimized_value
                dprint(f"Task {task_id}: Applied Wan 2.2 optimization {param}={optimized_value}")
        
        # Auto-enable built-in acceleration LoRAs if no LoRAs explicitly specified
        if "lora_names" not in generation_params and "activated_loras" not in db_task_params:
            generation_params["lora_names"] = ["CausVid", "DetailEnhancerV1"]
            generation_params["lora_multipliers"] = [1.0, 0.2]  # DetailEnhancer at reduced strength
            dprint(f"Task {task_id}: Auto-enabled Wan 2.2 acceleration LoRAs")
    
    # Determine task priority (orchestrator tasks get higher priority)
    priority = db_task_params.get("priority", 0)
    if task_type.endswith("_orchestrator"):
        priority = max(priority, 10)  # Boost orchestrator priority
    
    # Create and return GenerationTask
    generation_task = GenerationTask(
        id=task_id,
        model=model,
        prompt=prompt,
        parameters=generation_params,
        priority=priority
    )
    
    dprint(f"Created GenerationTask for {task_id}: model={model}, priority={priority}")
    return generation_task

# -----------------------------------------------------------------------------
# 1. Parse arguments for the server
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("WanGP Worker Server")

    pgroup_server = parser.add_argument_group("Server Settings")
    pgroup_server.add_argument("--db-file", type=str, default="tasks.db",
                               help="Path to the SQLite database file (if not using PostgreSQL via .env).")
    pgroup_server.add_argument("--main-output-dir", type=str, default="./outputs",
                               help="Base directory where outputs for each task will be saved (in subdirectories)")
    pgroup_server.add_argument("--poll-interval", type=int, default=10,
                               help="How often (in seconds) to check tasks.json for new tasks.")
    pgroup_server.add_argument("--debug", action="store_true",
                               help="Enable verbose debug logging (prints additional diagnostics)")
    pgroup_server.add_argument("--worker", type=str, default=None,
                               help="Worker name/ID - creates a log file named {worker}.log in the logs folder")
    pgroup_server.add_argument("--save-logging", type=str, nargs='?', const='logs/worker.log', default=None,
                               help="Save all logging output to a file (in addition to console output). Optionally specify path, defaults to 'logs/worker.log'")
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
    pgroup_server.add_argument("--use-legacy", action="store_true", default=False,
                               help="Use legacy task processing (DEPRECATED). By default, the new queue-based task processing system is used.")
    pgroup_server.add_argument("--queue-workers", type=int, default=1,
                               help="Number of queue workers for task processing (default: 1, recommended for GPU systems)")
    
    # --- New Supabase-related arguments ---
    pgroup_server.add_argument("--db-type", type=str, choices=["sqlite", "supabase"], default="sqlite",
                               help="Database type to use (default: sqlite)")
    pgroup_server.add_argument("--supabase-url", type=str, default=None,
                               help="Supabase project URL (required if db_type = supabase)")
    pgroup_server.add_argument("--supabase-access-token", type=str, default=None,
                               help="Supabase access token (JWT) for authentication (required if db_type = supabase)")
    pgroup_server.add_argument("--supabase-anon-key", type=str, default=None,
                               help="Supabase anon (public) API key used to create the client when authenticating with a user JWT. If omitted, falls back to SUPABASE_ANON_KEY env var or service key.")

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
                if isinstance(prog, tuple) and len(prog) == 2:
                    step, total = prog
                    print(f"{prefix}[Progress] {step}/{total} – {txt}")
                else:
                    print(f"{prefix}[Progress] {txt}")
        elif cmd == "status":
            print(f"{prefix}[Status] {data}")
        elif cmd == "info":
            print(f"{prefix}[INFO] {data}")
        elif cmd == "error":
            print(f"{prefix}[ERROR] {data}")
            raise RuntimeError(f"wgp.py error for {task_id}: {data}")
        elif cmd == "output":
            print(f"{prefix}[Output] video written.")
    return _send

def _ensure_lora_downloaded(task_id: str, lora_name: str, lora_url: str, target_dir: Path, 
                           task_params_dict: dict, flag_key: str, model_filename: str = None,
                           model_requirements: list = None) -> bool:
    """
    Shared helper to download a LoRA if it doesn't exist.
    
    Args:
        task_id: Task identifier for logging
        lora_name: Filename of the LoRA to download
        lora_url: URL to download fromCan you add
        target_dir: Directory to save the LoRA
        task_params_dict: Task parameters dictionary to modify if download fails
        flag_key: Key in task_params_dict to set to False if download fails
        model_filename: Optional model filename for requirement checking
        model_requirements: Optional list of strings that must be in model_filename
    
    Returns:
        True if LoRA exists or was successfully downloaded, False otherwise
    """
    target_path = target_dir / lora_name
    
    # Check model requirements if specified
    if model_filename and model_requirements:
        requirements_met = all(req in model_filename.lower() for req in model_requirements)
        if not requirements_met:
            print(f"[WARNING Task ID: {task_id}] {lora_name} is intended for models with {model_requirements}. Current model is {model_filename}. Results may vary.")
    
    # Download if not present
    if not target_path.exists():
        print(f"[Task ID: {task_id}] {lora_name} not found. Attempting download...")
        if not download_file(lora_url, target_dir, lora_name):
            print(f"[WARNING Task ID: {task_id}] Failed to download {lora_name}. Proceeding without it.")
            task_params_dict[flag_key] = False
            return False
    
    return True

# -----------------------------------------------------------------------------
# 4. State builder for a single task (same as before)
# -----------------------------------------------------------------------------
# --- SM_RESTRUCTURE: This function has been moved to source/common_utils.py ---


# -----------------------------------------------------------------------------
# 6. Process a single task dictionary from the tasks.json list
# -----------------------------------------------------------------------------

def process_single_task(wgp_mod, task_params_dict, main_output_dir_base: Path, task_type: str, project_id_for_task: str | None, image_download_dir: Path | str | None = None, apply_reward_lora: bool = False, colour_match_videos: bool = False, mask_active_frames: bool = True, task_queue: HeadlessTaskQueue = None):
    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))
    
    headless_logger.debug(f"Entering process_single_task", task_id=task_id)
    headless_logger.debug(f"Task Type: {task_type}", task_id=task_id)
    headless_logger.debug(f"Project ID: {project_id_for_task}", task_id=task_id)
    headless_logger.debug(f"Task Params: {json.dumps(task_params_dict, default=str, indent=2)[:1000]}...", task_id=task_id)
    
    headless_logger.essential(f"Processing {task_type} task", task_id=task_id)
    output_location_to_db = None # Will store the final path/URL for the DB
    generation_success = False

    # --- Orchestrator & Self-Contained Task Handlers ---
    # These tasks manage their own sub-task queuing and can return directly, as they
    # are either the start of a chain or a self-contained unit.
    if task_type == "travel_orchestrator":
        headless_logger.debug("Delegating to travel orchestrator handler", task_id=task_id)
        # Ensure the orchestrator uses the DB row ID as its canonical task_id
        task_params_dict["task_id"] = task_id
        if "orchestrator_details" in task_params_dict:
            task_params_dict["orchestrator_details"]["orchestrator_task_id"] = task_id
        return tbi._handle_travel_orchestrator_task(task_params_from_db=task_params_dict, main_output_dir_base=main_output_dir_base, orchestrator_task_id_str=task_id, orchestrator_project_id=project_id_for_task, dprint=dprint)
    elif task_type == "travel_segment":
        headless_logger.debug("Delegating to travel segment handler", task_id=task_id)
        return tbi._handle_travel_segment_task(wgp_mod, task_params_dict, main_output_dir_base, task_id, apply_reward_lora, colour_match_videos, mask_active_frames, process_single_task=process_single_task, dprint=dprint, task_queue=task_queue)
    elif task_type == "travel_stitch":
        headless_logger.debug("Delegating to travel stitch handler", task_id=task_id)
        return tbi._handle_travel_stitch_task(task_params_from_db=task_params_dict, main_output_dir_base=main_output_dir_base, stitch_task_id_str=task_id, dprint=dprint)
    elif task_type == "different_perspective_orchestrator":
        headless_logger.debug("Delegating to different perspective orchestrator handler", task_id=task_id)
        return dp._handle_different_perspective_orchestrator_task(
            task_params_from_db=task_params_dict,
            main_output_dir_base=main_output_dir_base,
            orchestrator_task_id_str=task_id,
            dprint=dprint
        )
    elif task_type == "dp_final_gen":
        headless_logger.debug("Delegating to different perspective final generation handler", task_id=task_id)
        return dp._handle_dp_final_gen_task(
            wgp_mod=wgp_mod,
            main_output_dir_base=main_output_dir_base,
            process_single_task=process_single_task,
            task_params_from_db=task_params_dict,
            dprint=dprint
        )
    elif task_type == "single_image":
        headless_logger.debug("Delegating to single image handler", task_id=task_id)
        return si._handle_single_image_task(
            wgp_mod=wgp_mod,
            task_params_from_db=task_params_dict,
            main_output_dir_base=main_output_dir_base,
            task_id=task_id,
            image_download_dir=image_download_dir,
            apply_reward_lora=apply_reward_lora,
            dprint=dprint
        )
    elif task_type == "magic_edit":
        headless_logger.debug("Delegating to magic edit handler", task_id=task_id)
        return me._handle_magic_edit_task(
            task_params_from_db=task_params_dict,
            main_output_dir_base=main_output_dir_base,
            task_id=task_id,
            dprint=dprint
        )

    # --- Primitive Task Execution Block ---
    # These tasks (openpose, rife, wgp) might be part of a chain.
    # They set generation_success and output_location_to_db, then execution
    # falls through to the chaining logic at the end of this function.
    if task_type == "generate_openpose":
        headless_logger.debug("Processing OpenPose generation task", task_id=task_id)
        generation_success, output_location_to_db = handle_generate_openpose_task(task_params_dict, main_output_dir_base, task_id, dprint)

    elif task_type == "extract_frame":
        headless_logger.debug("Processing frame extraction task", task_id=task_id)
        generation_success, output_location_to_db = handle_extract_frame_task(task_params_dict, main_output_dir_base, task_id, dprint)

    elif task_type == "rife_interpolate_images":
        headless_logger.debug("Processing RIFE interpolation task", task_id=task_id)
        generation_success, output_location_to_db = handle_rife_interpolate_task(wgp_mod, task_params_dict, main_output_dir_base, task_id, dprint)

    # Default handling for standard wgp tasks
    else:
        # NEW QUEUE-BASED PROCESSING: Delegate to task queue if available
        if task_queue is not None:
            headless_logger.debug("Using queue-based processing system", task_id=task_id)
            
            try:
                # Create GenerationTask object from DB parameters
                generation_task = db_task_to_generation_task(task_params_dict, task_id, task_type)
                
                # Apply global flags to task parameters
                if apply_reward_lora:
                    generation_task.parameters["apply_reward_lora"] = True
                if colour_match_videos:
                    generation_task.parameters["colour_match_videos"] = True
                if mask_active_frames:
                    generation_task.parameters["mask_active_frames"] = True
                
                # Submit task to queue
                submitted_task_id = task_queue.submit_task(generation_task)
                headless_logger.essential(f"Submitted to generation queue as {submitted_task_id}", task_id=task_id)
                
                # Block until task completion (simple synchronous approach for now)
                max_wait_time = 3600  # 1 hour max wait
                wait_interval = 2  # Check every 2 seconds
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    status = task_queue.get_task_status(task_id)
                    if status is None:
                        print(f"[ERROR Task ID: {task_id}] Task status became None, assuming failure")
                        generation_success = False
                        output_location_to_db = "Error: Task status became None during processing"
                        break
                        
                    if status.status == "completed":
                        generation_success = True
                        output_location_to_db = status.result_path
                        processing_time = status.processing_time or 0
                        print(f"[Task ID: {task_id}] Queue processing completed in {processing_time:.1f}s")
                        print(f"[Task ID: {task_id}] Output: {output_location_to_db}")
                        break
                    elif status.status == "failed":
                        generation_success = False
                        output_location_to_db = status.error_message or "Generation failed without specific error message"
                        print(f"[ERROR Task ID: {task_id}] Queue processing failed: {output_location_to_db}")
                        break
                    else:
                        # Still processing
                        dprint(f"[Task ID: {task_id}] Queue status: {status.status}, waiting...")
                        time.sleep(wait_interval)
                        elapsed_time += wait_interval
                else:
                    # Timeout reached
                    print(f"[ERROR Task ID: {task_id}] Queue processing timeout after {max_wait_time}s")
                    generation_success = False
                    output_location_to_db = f"Error: Processing timeout after {max_wait_time} seconds"
                
            except Exception as e_queue:
                print(f"[ERROR Task ID: {task_id}] Queue processing error: {e_queue}")
                traceback.print_exc()
                generation_success = False
                output_location_to_db = f"Error: Queue processing failed - {str(e_queue)}"
                
        # LEGACY PROCESSING: Original wgp.py integration (fallback when queue not available)
        if task_queue is None:
            dprint(f"[Task ID: {task_id}] Using legacy processing system")
            task_model_type_logical = task_params_dict.get("model", "t2v")
            print(f"[DEBUG] Task {task_id}: task_model_type_logical = {task_model_type_logical}")
            print(f"[DEBUG] Task {task_id}: transformer_quantization = {wgp_mod.transformer_quantization}")
            print(f"[DEBUG] Task {task_id}: transformer_dtype_policy = {wgp_mod.transformer_dtype_policy}")
            
            model_filename_for_task = wgp_mod.get_model_filename(task_model_type_logical,
                                                                 wgp_mod.transformer_quantization,
                                                                 wgp_mod.transformer_dtype_policy)
            print(f"[DEBUG] Task {task_id}: model_filename_for_task = '{model_filename_for_task}'")
            custom_output_dir = task_params_dict.get("output_dir")
            
            if apply_reward_lora:
                print(f"[Task ID: {task_id}] --apply-reward-lora flag is active. Checking and downloading reward LoRA.")
                # Use shared helper (reward LoRA download never fails)
                reward_url = "https://huggingface.co/peteromallet/Wan2.1-Fun-14B-InP-MPS_reward_lora_diffusers/resolve/main/Wan2.1-Fun-14B-InP-MPS_reward_lora_wgp.safetensors"
                _ensure_lora_downloaded(task_id, "Wan2.1-Fun-14B-InP-MPS_reward_lora_wgp.safetensors", 
                                      reward_url, Path(get_lora_dir_from_filename(wgp_mod, model_filename_for_task)), {}, "dummy_key")

            effective_image_download_dir = image_download_dir

            if effective_image_download_dir is None:
                if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
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

            # Handle special LoRA downloads using shared helper
            base_lora_dir_for_model = Path(get_lora_dir_from_filename(wgp_mod, model_filename_for_task))
            
            if task_params_dict.get("use_causvid_lora", False):
                _ensure_lora_downloaded(
                    task_id, "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
                    "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
                    base_lora_dir_for_model, task_params_dict, "use_causvid_lora",
                    model_filename_for_task, ["14b", "t2v"]
                )

            if task_params_dict.get("use_lighti2x_lora", False):
                _ensure_lora_downloaded(
                    task_id, "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
                    "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
                    base_lora_dir_for_model, task_params_dict, "use_lighti2x_lora"
                )

            additional_loras = task_params_dict.get("additional_loras", {})
        if additional_loras:
            dprint(f"[Task ID: {task_id}] Processing additional LoRAs: {additional_loras}")
            processed_loras = {}
            for lora_path, lora_strength in additional_loras.items():
                try:
                    lora_filename = Path(urllib.parse.urlparse(lora_path).path).name
                    
                    if lora_path.startswith("http://") or lora_path.startswith("https://"):
                        wan2gp_loras_dir = Path("Wan2GP/loras")
                        wan2gp_loras_dir.mkdir(parents=True, exist_ok=True)
                        dprint(f"Task {task_id}: Downloading LoRA from {lora_path} to {wan2gp_loras_dir}")
                        download_file(lora_path, wan2gp_loras_dir, lora_filename)
                        
                        downloaded_lora_path = wan2gp_loras_dir / lora_filename
                        target_path = base_lora_dir_for_model / lora_filename
                        if not target_path.exists() or downloaded_lora_path.resolve() != target_path.resolve():
                            shutil.copy(str(downloaded_lora_path), str(target_path))
                            dprint(f"Copied downloaded LoRA from {downloaded_lora_path} to {target_path}")
                        else:
                            dprint(f"Downloaded LoRA already exists at target: {target_path}")
                    else:
                        source_path = Path(lora_path)
                        target_path = base_lora_dir_for_model / lora_filename
                        
                        if source_path.is_absolute() and source_path.exists():
                            if not target_path.exists() or source_path.resolve() != target_path.resolve():
                                shutil.copy(str(source_path), str(target_path))
                                dprint(f"Copied local LoRA from {source_path} to {target_path}")
                            else:
                                dprint(f"LoRA already exists at target: {target_path}")
                        elif (base_lora_dir_for_model / lora_path).exists():
                            existing_lora_path = base_lora_dir_for_model / lora_path
                            lora_filename = existing_lora_path.name
                            dprint(f"Found existing LoRA in directory: {existing_lora_path}")
                        elif (Path("Wan2GP/loras") / lora_path).exists():
                            wan2gp_lora_path = Path("Wan2GP/loras") / lora_path
                            target_path = base_lora_dir_for_model / wan2gp_lora_path.name
                            if not target_path.exists() or wan2gp_lora_path.resolve() != target_path.resolve():
                                shutil.copy(str(wan2gp_lora_path), str(target_path))
                                dprint(f"Copied LoRA from Wan2GP/loras: {wan2gp_lora_path} to {target_path}")
                            else:
                                dprint(f"LoRA from Wan2GP/loras already exists at target: {target_path}")
                            lora_filename = wan2gp_lora_path.name
                        elif source_path.exists():
                            if not target_path.exists() or source_path.resolve() != target_path.resolve():
                                shutil.copy(str(source_path), str(target_path))
                                dprint(f"Copied local LoRA from {source_path} to {target_path}")
                            else:
                                dprint(f"LoRA already exists at target: {target_path}")
                        else:
                            dprint(f"[WARNING Task ID: {task_id}] LoRA not found at any location: '{lora_path}'. Checked: full path, LoRA directory, and relative path. Skipping.")
                            continue
                    
                    processed_loras[lora_filename] = str(lora_strength)
                except Exception as e_lora:
                    print(f"[ERROR Task ID: {task_id}] Failed to process additional LoRA {lora_path}: {e_lora}")
            
            task_params_dict["processed_additional_loras"] = processed_loras

        print(f"[Task ID: {task_id}] Using model file: {model_filename_for_task}")

        temp_output_dir = tempfile.mkdtemp(prefix=f"wgp_headless_{task_id}_")
        dprint(f"[Task ID: {task_id}] Using temporary output directory: {temp_output_dir}")

        original_wgp_save_path = wgp_mod.save_path
        wgp_mod.save_path = str(temp_output_dir)

        lora_dir_for_active_model = get_lora_dir_from_filename(wgp_mod, model_filename_for_task)
        # setup_loras needs model_type, not model_filename
        model_type_for_task = wgp_mod.get_model_type(model_filename_for_task) if model_filename_for_task else "t2v"
        all_loras_for_active_model, _, _, _, _, _, _ = wgp_mod.setup_loras(
            model_type_for_task, None, lora_dir_for_active_model, "", None
        )

        state, ui_params = build_task_state(wgp_mod, model_filename_for_task, task_params_dict, all_loras_for_active_model, image_download_dir, apply_reward_lora=apply_reward_lora)
        
        gen_task_placeholder = {"id": 1, "prompt": ui_params.get("prompt"), "params": {"model_filename_from_gui_state": model_filename_for_task, "model": task_model_type_logical}}
        send_cmd = make_send_cmd(task_id)

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

        requested_frames_from_task = ui_params.get("video_length", 81)
        frame_num_for_wgp = requested_frames_from_task
        print(f"[HEADLESS_DEBUG] Task {task_id}: FRAME COUNT ANALYSIS")
        print(f"[HEADLESS_DEBUG]   requested_frames_from_task: {requested_frames_from_task}")
        print(f"[HEADLESS_DEBUG]   frame_num_for_wgp: {frame_num_for_wgp}")
        print(f"[HEADLESS_DEBUG]   ui_params video_length: {ui_params.get('video_length')}")
        dprint(f"[Task ID: {task_id}] Using requested frame count: {frame_num_for_wgp}")

        ui_params["video_length"] = frame_num_for_wgp

        try:
            print(f"[HEADLESS_DEBUG] Task {task_id}: CALLING WGP GENERATION")
            print(f"[HEADLESS_DEBUG]   Final video_length parameter: {ui_params.get('video_length')}")
            print(f"[HEADLESS_DEBUG]   Resolution: {ui_params.get('resolution')}")
            print(f"[HEADLESS_DEBUG]   Seed: {ui_params.get('seed')}")
            print(f"[HEADLESS_DEBUG]   Steps: {ui_params.get('num_inference_steps')}")
            print(f"[HEADLESS_DEBUG]   Model: {model_filename_for_task}")
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
                control_net_weight=ui_params.get("control_net_weight", 1.0),
                control_net_weight2=ui_params.get("control_net_weight2", 1.0),
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
            print(f"[HEADLESS_DEBUG] Task {task_id}: WGP GENERATION COMPLETED")
            print(f"[HEADLESS_DEBUG]   Temporary output directory: {temp_output_dir}")
            
            # List all files in temp directory for debugging
            temp_dir_contents = list(Path(temp_output_dir).iterdir())
            print(f"[HEADLESS_DEBUG]   Files in temp directory: {len(temp_dir_contents)}")
            for item in temp_dir_contents:
                if item.is_file():
                    print(f"[HEADLESS_DEBUG]     {item.name} ({item.stat().st_size} bytes)")
            
            generation_success = True
        except Exception as e:
            print(f"[ERROR] Task ID {task_id} failed during generation: {e}")
            traceback.print_exc()
        finally:
            wgp_mod.save_path = original_wgp_save_path

        if generation_success:
            generated_video_files = sorted([
                item for item in Path(temp_output_dir).iterdir()
                if item.is_file() and item.suffix.lower() == ".mp4"
            ])
            
            print(f"[HEADLESS_DEBUG] Task {task_id}: ANALYZING GENERATED FILES")
            print(f"[HEADLESS_DEBUG]   Found {len(generated_video_files)} .mp4 files")
            
            # Analyze each video file found
            for i, video_file in enumerate(generated_video_files):
                try:
                    from source.common_utils import get_video_frame_count_and_fps
                    frame_count, fps = get_video_frame_count_and_fps(str(video_file))
                    file_size = video_file.stat().st_size
                    duration = frame_count / fps if fps and fps > 0 else 0
                    print(f"[HEADLESS_DEBUG]   Video {i}: {video_file.name}")
                    print(f"[HEADLESS_DEBUG]     Frames: {frame_count}")
                    print(f"[HEADLESS_DEBUG]     FPS: {fps}")
                    print(f"[HEADLESS_DEBUG]     Duration: {duration:.2f}s")
                    print(f"[HEADLESS_DEBUG]     Size: {file_size / (1024*1024):.2f} MB")
                    print(f"[HEADLESS_DEBUG]     Expected frames: {frame_num_for_wgp}")
                    if frame_count != frame_num_for_wgp:
                        print(f"[HEADLESS_DEBUG]     ⚠️  FRAME COUNT MISMATCH! Expected {frame_num_for_wgp}, got {frame_count}")
                except Exception as e_analysis:
                    print(f"[HEADLESS_DEBUG]     ERROR analyzing {video_file.name}: {e_analysis}")
            
            generated_video_file = None
            if not generated_video_files:
                print(f"[WARNING Task ID: {task_id}] Generation reported success, but no .mp4 files found in {temp_output_dir}")
                generation_success = False
            elif len(generated_video_files) == 1:
                generated_video_file = generated_video_files[0]
                dprint(f"[Task ID: {task_id}] Found a single generated video: {generated_video_file}")
            else:
                dprint(f"[Task ID: {task_id}] Found {len(generated_video_files)} video segments to stitch: {generated_video_files}")
                stitched_video_path = Path(temp_output_dir) / f"{task_id}_stitched.mp4"
                
                if 'sm_stitch_videos_ffmpeg' not in globals() and 'sm_stitch_videos_ffmpeg' not in locals():
                    try:
                        from source.common_utils import stitch_videos_ffmpeg as sm_stitch_videos_ffmpeg
                    except ImportError:
                        print(f"[CRITICAL ERROR Task ID: {task_id}] Failed to import 'stitch_videos_ffmpeg'. Cannot proceed with stitching.")
                        generation_success = False
                
                if generation_success:
                    try:
                        video_paths_str = [str(p.resolve()) for p in generated_video_files]
                        sm_stitch_videos_ffmpeg(video_paths_str, str(stitched_video_path.resolve()))
                        
                        if stitched_video_path.exists() and stitched_video_path.stat().st_size > 0:
                            generated_video_file = stitched_video_path
                            dprint(f"[Task ID: {task_id}] Successfully stitched video segments to: {generated_video_file}")
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
                
                # Use custom output directory if provided, otherwise use default logic
                if custom_output_dir:
                    target_dir = Path(custom_output_dir)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    final_video_path = sm_get_unique_target_path(
                        target_dir, 
                        task_id,
                        generated_video_file.suffix
                    )
                    
                    # For custom output dir, create a relative path for DB storage
                    try:
                        output_location_to_db = str(final_video_path.relative_to(Path.cwd()))
                    except ValueError:
                        output_location_to_db = str(final_video_path.resolve())
                    
                    try:
                        shutil.move(str(generated_video_file), str(final_video_path))
                        print(f"[Task ID: {task_id}] Output video saved to: {final_video_path.resolve()} (DB location: {output_location_to_db})")
                    except Exception as e_move:
                        print(f"[ERROR Task ID: {task_id}] Failed to move video to custom output directory: {e_move}")
                        generation_success = False
                        
                else:
                    # Use the generalized upload-aware output handling for all other cases
                    final_video_path, initial_db_location = prepare_output_path_with_upload(
                        task_id=task_id,
                        filename=generated_video_file.name,
                        main_output_dir_base=main_output_dir_base,
                        dprint=dprint
                    )
                    
                    try:
                        shutil.move(str(generated_video_file), str(final_video_path))
                        dprint(f"[Task ID: {task_id}] Moved generated video to: {final_video_path}")
                        
                        # Handle Supabase upload (if configured) and get final location for DB
                        output_location_to_db = upload_and_get_final_output_location(
                            final_video_path,
                            task_id,
                            initial_db_location,
                            dprint=dprint
                        )
                        
                        print(f"[Task ID: {task_id}] Output video saved to: {final_video_path.resolve()} (DB location: {output_location_to_db})")
                        
                    except Exception as e_move:
                        print(f"[ERROR Task ID: {task_id}] Failed to move video to final destination: {e_move}")
                        generation_success = False
            else:
                print(f"[WARNING Task ID: {task_id}] Generation reported success, but no .mp4 file found in {temp_output_dir}")
                generation_success = False
        
        try:
            shutil.rmtree(temp_output_dir)
            dprint(f"[Task ID: {task_id}] Cleaned up temporary directory: {temp_output_dir}")
        except Exception as e_clean:
            print(f"[WARNING Task ID: {task_id}] Failed to clean up temporary directory {temp_output_dir}: {e_clean}")

    # --- Chaining Logic ---
    # This block is now executed for any successful primitive task that doesn't return early.
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
        
        elif task_params_dict.get("different_perspective_chain_details"):
            # SM_RESTRUCTURE_FIX: Prevent double-chaining. This is now handled in the 'generate_openpose' block.
            # The only other task type that can have these details is 'wgp', which is the intended target for this block.
            if task_type != 'generate_openpose':
                dprint(f"Task {task_id} is part of a different_perspective sequence. Attempting to chain.")
                
                chain_success, chain_message, final_path_from_chaining = dp._handle_different_perspective_chaining(
                    completed_task_params=task_params_dict, 
                    task_output_path=output_location_to_db,
                    dprint=dprint
                )
                if chain_success:
                    chaining_result_path_override = final_path_from_chaining
                    dprint(f"Task {task_id}: Different Perspective chaining successful. Message: {chain_message}")
                else:
                    print(f"[ERROR Task ID: {task_id}] Different Perspective sequence chaining failed: {chain_message}. This may halt the sequence.")


        if chaining_result_path_override:
            path_to_check_existence: Path | None = None
            if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and isinstance(chaining_result_path_override, str) and chaining_result_path_override.startswith("files/"):
                sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                path_to_check_existence = (sqlite_db_parent / "public" / chaining_result_path_override).resolve()
                dprint(f"Task {task_id}: Chaining returned SQLite relative path '{chaining_result_path_override}'. Resolved to '{path_to_check_existence}' for existence check.")
            elif chaining_result_path_override:
                path_to_check_existence = Path(chaining_result_path_override).resolve()
                dprint(f"Task {task_id}: Chaining returned absolute-like path '{chaining_result_path_override}'. Resolved to '{path_to_check_existence}' for existence check.")

            if path_to_check_existence and path_to_check_existence.exists() and path_to_check_existence.is_file():
                is_output_path_different = str(chaining_result_path_override) != str(output_location_to_db)
                if is_output_path_different:
                    dprint(f"Task {task_id}: Chaining modified output path for DB. Original: {output_location_to_db}, New: {chaining_result_path_override} (Checked file: {path_to_check_existence})")
                output_location_to_db = chaining_result_path_override
            elif chaining_result_path_override is not None:
                print(f"[WARNING Task ID: {task_id}] Chaining reported success, but final path '{chaining_result_path_override}' (checked as '{path_to_check_existence}') is invalid or not a file. Using original WGP output '{output_location_to_db}' for DB.")


    # Ensure orchestrator tasks use their DB row ID as task_id so that
    # downstream sub-tasks reference the right row when updating status.
    if task_type in {"travel_orchestrator", "different_perspective_orchestrator"}:
        # Overwrite/insert the canonical task_id inside params to the DB row's ID
        task_params_dict["task_id"] = task_id

    print(f"--- Finished task ID: {task_id} (Success: {generation_success}) ---")
    return generation_success, output_location_to_db


# -----------------------------------------------------------------------------
# 7. Main server loop
# -----------------------------------------------------------------------------

def main():
    load_dotenv() # Load .env file variables into environment
    global DB_TYPE, SQLITE_DB_PATH, SUPABASE_CLIENT, SUPABASE_VIDEO_BUCKET

    # Parse CLI arguments first to determine debug mode
    cli_args = parse_args()
    
    # Set up logging early based on debug flag
    global debug_mode
    debug_mode = cli_args.debug
    if debug_mode:
        enable_debug_mode()
        headless_logger.debug("Debug mode enabled")
    else:
        disable_debug_mode()

    # Determine DB type from environment variables
    env_db_type = os.getenv("DB_TYPE", "sqlite").lower()
    env_pg_table_name = os.getenv("POSTGRES_TABLE_NAME", "tasks")
    env_supabase_url = os.getenv("SUPABASE_URL")
    env_supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")  # Support both names
    env_supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
    env_supabase_bucket = os.getenv("SUPABASE_VIDEO_BUCKET", "image_uploads")
    env_sqlite_db_path = os.getenv("SQLITE_DB_PATH_ENV") # Read SQLite DB path from .env

    # ------------------------------------------------------------------
    # Auto-enable file logging when --debug flag is present
    # ------------------------------------------------------------------
    if cli_args.debug and not cli_args.save_logging:
        from datetime import datetime
        default_logs_dir = Path("logs")
        default_logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cli_args.save_logging = str(default_logs_dir / f"debug_{timestamp}.log")
        headless_logger.debug(f"Auto-enabled file logging: {cli_args.save_logging}")
    # ------------------------------------------------------------------

    # Handle --worker parameter for worker-specific logging
    if cli_args.worker and not cli_args.save_logging:
        default_logs_dir = Path("logs")
        default_logs_dir.mkdir(parents=True, exist_ok=True)
        cli_args.save_logging = str(default_logs_dir / f"{cli_args.worker}.log")
        headless_logger.debug(f"Worker-specific logging enabled: {cli_args.save_logging}")
    # ------------------------------------------------------------------

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
        
        headless_logger.essential("Cleaning up database files")
        for db_file in db_files_to_remove:
            if Path(db_file).exists():
                try:
                    Path(db_file).unlink()
                    headless_logger.success(f"Removed database file: {db_file}")
                except Exception as e:
                    headless_logger.error(f"Could not remove database file {db_file}: {e}")
        
        headless_logger.success("Database cleanup complete. Starting fresh.")
    # --- End delete-db handling ---

    # --- Setup logging to file if requested ---
    if cli_args.save_logging:
        import logging
        
        log_file_path = Path(cli_args.save_logging)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a custom stream that writes to both console and file
        class DualWriter:
            def __init__(self, log_file_path):
                self.terminal = sys.stdout
                self.log_file = open(log_file_path, 'a', encoding='utf-8')
                
            def write(self, message):
                self.terminal.write(message)
                self.log_file.write(message)
                self.log_file.flush()  # Ensure immediate write
                
            def flush(self):
                self.terminal.flush()
                self.log_file.flush()
                
            def close(self):
                if hasattr(self, 'log_file'):
                    self.log_file.close()
        
        # Redirect stdout to our dual writer
        sys.stdout = DualWriter(log_file_path)
        
        if cli_args.worker:
            headless_logger.essential(f"Worker '{cli_args.worker}' output will be saved to: {log_file_path.resolve()}")
        else:
            headless_logger.essential(f"All output will be saved to: {log_file_path.resolve()}")
        
        # Ensure cleanup on exit
        import atexit
        atexit.register(lambda: hasattr(sys.stdout, 'close') and sys.stdout.close())
    # --- End logging setup ---

    # --- Configure DB Type and Connection Globals ---
    # This block sets DB_TYPE, SQLITE_DB_PATH, SUPABASE_CLIENT, etc. in the db_ops module
    if cli_args.db_type == "supabase" and cli_args.supabase_url and cli_args.supabase_access_token:
        try:
            # --- New PAT/JWT-based Supabase client initialization ---
            # For PATs and user JWTs, we primarily use edge functions and avoid direct 
            # Supabase client authentication which expects specific JWT formats.
            # We'll create a client with the service key for internal operations,
            # but use the access token in headers for edge function calls.
            
            # Use service key for admin operations if available, otherwise anon key
            client_key = env_supabase_key or cli_args.supabase_anon_key or env_supabase_anon_key
            
            if not client_key:
                raise ValueError("Need either service key or anon key for Supabase client initialization.")

            headless_logger.debug(f"Initializing Supabase client for {cli_args.supabase_url}")
            temp_supabase_client = create_client(cli_args.supabase_url, client_key)
            
            # For PATs and user tokens, we'll primarily rely on edge functions
            # The access token will be passed in Authorization headers
            headless_logger.debug("Supabase client initialized. Access token will be used in edge function calls.")

            # --- Assign to db_ops globals on success ---
            db_ops.DB_TYPE = "supabase"
            db_ops.PG_TABLE_NAME = env_pg_table_name
            db_ops.SUPABASE_URL = cli_args.supabase_url
            db_ops.SUPABASE_SERVICE_KEY = env_supabase_key # Keep service key if present
            db_ops.SUPABASE_VIDEO_BUCKET = env_supabase_bucket
            db_ops.SUPABASE_CLIENT = temp_supabase_client
            # Store the access token for use in Edge Function calls
            db_ops.SUPABASE_ACCESS_TOKEN = cli_args.supabase_access_token

            # Local globals for convenience
            DB_TYPE = "supabase"
            SUPABASE_CLIENT = temp_supabase_client
            SUPABASE_VIDEO_BUCKET = env_supabase_bucket
            
            headless_logger.success("Supabase client initialized successfully")

        except Exception as e:
            headless_logger.error(f"Failed to initialize Supabase client: {e}")
            headless_logger.debug(traceback.format_exc())
            headless_logger.warning("Falling back to SQLite due to Supabase client initialization error")
            db_ops.DB_TYPE = "sqlite"
            db_ops.SQLITE_DB_PATH = env_sqlite_db_path if env_sqlite_db_path else cli_args.db_file
    else: # Default to sqlite if .env DB_TYPE is unrecognized or not set, or if it's explicitly "sqlite"
        if cli_args.db_type != "sqlite":
            headless_logger.warning(f"DB_TYPE '{cli_args.db_type}' in CLI args is not recognized. Defaulting to SQLite.")
        db_ops.DB_TYPE = "sqlite"
        db_ops.SQLITE_DB_PATH = env_sqlite_db_path if env_sqlite_db_path else cli_args.db_file
        DB_TYPE = "sqlite"
        SQLITE_DB_PATH = db_ops.SQLITE_DB_PATH
    # --- End DB Type Configuration ---

    # --- Run DB Migrations ---
    # Must be after DB type/config is determined but before DB schema is strictly enforced by init_db or heavy use.
    # Note: Migrations completed - now using Edge Functions exclusively
    # db_ops._run_db_migrations()  # Commented out - migration to Edge Functions complete
    # --- End DB Migrations ---

    # --- Handle --migrate-only flag --- (Section 6)
    if cli_args.migrate_only:
        print("Database migrations complete (called with --migrate-only). Exiting.")
        sys.exit(0)
    # --- End --migrate-only handler ---


    main_output_dir = Path(cli_args.main_output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"WanGP Headless Server Started.")
    if cli_args.worker:
        print(f"Worker ID: {cli_args.worker}")
    if db_ops.DB_TYPE == "supabase":
        print(f"Monitoring Supabase (PostgreSQL backend) table: {db_ops.PG_TABLE_NAME}")
    else: # SQLite
        print(f"Monitoring SQLite database: {db_ops.SQLITE_DB_PATH}")
    print(f"Outputs will be saved under: {main_output_dir}")
    print(f"Polling interval: {cli_args.poll_interval} seconds.")

    # Initialize database
    # Supabase table/schema assumed to exist; skip initialization RPC
    if db_ops.DB_TYPE == "supabase":
        dprint("Supabase: Skipping init_db_supabase – table assumed present.")
    else: # SQLite
        db_ops.init_db() # Existing SQLite init

    # Activate global debug switch early so that all subsequent code paths can use dprint()
    debug_mode = cli_args.debug
    db_ops.debug_mode = cli_args.debug # Also set it in the db_ops module
    dprint("Verbose debug logging enabled.")

    # Before importing Wan2GP.wgp we need to ensure that a CPU-only PyTorch build
    # does not crash when third-party code unconditionally calls CUDA helpers.
    try:
        import torch  # Local import to limit overhead when torch is missing

        if not torch.cuda.is_available():
            # Monkey-patch *early* to stub out problematic functions that trigger
            # torch.cuda initialisation on CPU-only builds (which raises
            # "Torch not compiled with CUDA enabled").  This is safe because the
            # downstream Wan2GP code only checks the return value tuple.
            def _dummy_get_device_capability(device=None):
                return (0, 0)

            torch.cuda.get_device_capability = _dummy_get_device_capability  # type: ignore[attr-defined]

            # Some libraries check for the attribute rather than calling it, so
            # we also advertise zero GPUs.
            torch.cuda.device_count = lambda: 0  # type: ignore[attr-defined]
    except ImportError:
        # torch not installed – Wan2GP import will fail later anyway.
        pass

    # Ensure Wan2GP sees a clean argv list and Gradio functions are stubbed
    sys.argv = ["Wan2GP/wgp.py"]  # Prevent wgp.py from parsing worker.py CLI args
    patch_gradio()

    # Change to WGP directory so it can find its defaults/*.json files
    original_cwd = os.getcwd()
    wgp_dir = os.path.join(original_cwd, "Wan2GP")
    print(f"[DEBUG] Changing to WGP directory: {wgp_dir}")
    os.chdir(wgp_dir)
    
    try:
        # Check if defaults directory exists and has JSON files
        defaults_dir = "defaults"
        if os.path.exists(defaults_dir):
            json_files = [f for f in os.listdir(defaults_dir) if f.endswith('.json')]
            print(f"[DEBUG] Found {len(json_files)} JSON files in {defaults_dir}/")
        else:
            print(f"[DEBUG] WARNING: {defaults_dir}/ directory not found!")
        
        # Import wgp from the current directory (now Wan2GP)
        import wgp as wgp_mod
        print(f"[DEBUG] WGP imported successfully. Found {len(wgp_mod.models_def)} model definitions.")
        
        # CRITICAL: Force correct config values immediately after import to override any bad existing config
        if not isinstance(wgp_mod.server_config.get("transformer_types"), list):
            print(f"[DEBUG] Fixing transformer_types: was {type(wgp_mod.server_config.get('transformer_types'))}, setting to []")
            wgp_mod.server_config["transformer_types"] = []
        if not isinstance(wgp_mod.server_config.get("preload_model_policy"), list):
            print(f"[DEBUG] Fixing preload_model_policy: was {type(wgp_mod.server_config.get('preload_model_policy'))}, setting to []")
            wgp_mod.server_config["preload_model_policy"] = []
        
        # Debug: Show what model types are available
        if wgp_mod.models_def:
            model_types = list(wgp_mod.models_def.keys())[:5]  # Show first 5
            print(f"[DEBUG] Available model types: {model_types}...")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

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
    else:
        # Ensure preload_model_policy is always a list, never None or int
        if "preload_model_policy" not in wgp_mod.server_config or not isinstance(wgp_mod.server_config["preload_model_policy"], list):
            wgp_mod.server_config["preload_model_policy"] = []
    
    # Ensure transformer_types is always a list to prevent character iteration
    if "transformer_types" not in wgp_mod.server_config or not isinstance(wgp_mod.server_config["transformer_types"], list):
        wgp_mod.server_config["transformer_types"] = []

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

    # --- Initialize Task Queue System (default) ---
    task_queue = None
    if not cli_args.use_legacy:
        print(f"[INFO] Initializing queue-based task processing system (default)...")
        wan_dir = str(Path(__file__).parent / "Wan2GP")
        if not Path(wan_dir).exists():
            wan_dir = str(Path(__file__).parent)  # Fallback if Wan2GP is in current dir
        
        try:
            task_queue = HeadlessTaskQueue(wan_dir=wan_dir, max_workers=cli_args.queue_workers)
            task_queue.start()
            print(f"[INFO] Task queue initialized with {cli_args.queue_workers} workers")
            print(f"[INFO] Queue system will handle generation tasks efficiently with model reuse")
        except Exception as e_queue_init:
            print(f"[ERROR] Failed to initialize task queue: {e_queue_init}")
            traceback.print_exc()
            print("[WARNING] Falling back to legacy task processing")
            task_queue = None
            cli_args.use_legacy = True
    else:
        print(f"[INFO] Using legacy task processing (--use-legacy specified - DEPRECATED)")
    # --- End Task Queue Initialization ---

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
                task_info = db_ops.get_oldest_queued_task_supabase(worker_id=cli_args.worker)
                dprint(f"Supabase task_info: {task_info}") # ADDED DPRINT
                if task_info:
                    current_task_id_for_status_update = task_info["task_id"]
                    # Status is already set to IN_PROGRESS by claim-next-task Edge Function
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
            
            # Ensure orchestrator tasks propagate the DB row ID as their canonical task_id *before* processing
            if current_task_type in {"travel_orchestrator", "different_perspective_orchestrator"}:
                current_task_params["task_id"] = current_task_id_for_status_update
                if "orchestrator_details" in current_task_params:
                    current_task_params["orchestrator_details"]["orchestrator_task_id"] = current_task_id_for_status_update

            task_succeeded, output_location = process_single_task(
                wgp_mod, current_task_params, main_output_dir, current_task_type, current_project_id,
                image_download_dir=segment_image_download_dir,
                apply_reward_lora=cli_args.apply_reward_lora,
                colour_match_videos=cli_args.colour_match_videos,
                mask_active_frames=cli_args.mask_active_frames,
                task_queue=task_queue
            )

            if task_succeeded:
                # Orchestrator tasks stay "In Progress" until their children report back.
                orchestrator_types_waiting = {"travel_orchestrator", "different_perspective_orchestrator"}

                if current_task_type in orchestrator_types_waiting:
                    # Keep status as IN_PROGRESS (already set when we claimed the task).
                    # We still store the output message (if any) so operators can see it.
                    db_ops.update_task_status(
                        current_task_id_for_status_update,
                        db_ops.STATUS_IN_PROGRESS,
                        output_location,
                    )
                    headless_logger.status(
                        f"Orchestrator task queued child tasks; awaiting completion", 
                        task_id=current_task_id_for_status_update
                    )
                else:
                    if db_ops.DB_TYPE == "supabase":
                        db_ops.update_task_status_supabase(
                            current_task_id_for_status_update,
                            db_ops.STATUS_COMPLETE,
                            output_location,
                        )
                    else:
                        db_ops.update_task_status(
                            current_task_id_for_status_update,
                            db_ops.STATUS_COMPLETE,
                            output_location,
                        )
                    headless_logger.success(
                        f"Task completed successfully: {output_location}", 
                        task_id=current_task_id_for_status_update
                    )
            else:
                if db_ops.DB_TYPE == "supabase":
                    db_ops.update_task_status_supabase(current_task_id_for_status_update, db_ops.STATUS_FAILED, output_location)
                else:
                    db_ops.update_task_status(current_task_id_for_status_update, db_ops.STATUS_FAILED, output_location)
                headless_logger.error(
                    f"Task failed. Output: {output_location if output_location else 'N/A'}", 
                    task_id=current_task_id_for_status_update
                )
            
            time.sleep(1) # Brief pause before checking for the next task

    except KeyboardInterrupt:
        headless_logger.essential("Server shutting down gracefully...")
    finally:
        # Shutdown task queue first (if it was initialized)
        if task_queue is not None:
            try:
                headless_logger.essential("Shutting down task queue...")
                task_queue.stop(timeout=30.0)
                headless_logger.success("Task queue shutdown complete")
            except Exception as e_queue_shutdown:
                headless_logger.error(f"Error during task queue shutdown: {e_queue_shutdown}")
        
        # Legacy wgp cleanup
        if hasattr(wgp_mod, 'offloadobj') and wgp_mod.offloadobj is not None:
            try:
                headless_logger.debug("Attempting to release wgp.py offload object...")
                wgp_mod.offloadobj.release()
            except Exception as e_release:
                headless_logger.error(f"Error during offloadobj release: {e_release}")
        headless_logger.essential("Server stopped")



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