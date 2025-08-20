"""Wan2GP Worker Server.

This long-running process polls the `tasks` table (SQLite by default, or
Supabase-backed Postgres when configured), claims queued tasks, and executes
them using the HeadlessTaskQueue system. Besides standard generation
tasks it also contains specialised handlers for:

• `generate_openpose` – creates OpenPose skeleton images using dwpose.
• `rife_interpolate_images` – does frame interpolation between two stills.

The server uses a queue-based architecture for efficient task processing with
model persistence and memory management. It moves or uploads finished artefacts,
and updates task status in the database before looping again. It serves as the
runtime backend that clients rely upon to perform heavy generation work.

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

from pathlib import Path
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
    load_pil_images as sm_load_pil_images,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location
)
from source.sm_functions import travel_between_images as tbi
from source.sm_functions import different_perspective as dp
from source.sm_functions import single_image as si
from source.sm_functions import magic_edit as me
# --- New Queue-based Architecture Imports ---
# Protect sys.argv before importing queue management which imports wgp.py
_original_argv = sys.argv[:]
sys.argv = ["worker.py"]  # Prevent wgp.py from parsing our CLI args
from headless_model_management import HeadlessTaskQueue, GenerationTask
sys.argv = _original_argv  # Restore original arguments
# --- Structured Logging ---
from source.logging_utils import headless_logger, enable_debug_mode, disable_debug_mode
# --- End SM_RESTRUCTURE imports ---


# -----------------------------------------------------------------------------
# Debug / Verbose Logging Helpers
# -----------------------------------------------------------------------------

debug_mode = False  # This will be toggled on via the --debug CLI flag in main()

def dprint(msg: str):
    """Print a debug message if --debug flag is enabled."""
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
    
    # Copy whitelisted parameters with field name mapping
    for param in param_whitelist:
        if param in db_task_params:
            generation_params[param] = db_task_params[param]
    
    # [DEEP_DEBUG] Log LoRA parameter transfer for debugging
    print(f"[WORKER_DEBUG] Worker {task_id}: CONVERTING DB TASK TO GENERATION TASK")
    print(f"[WORKER_DEBUG]   task_type: {task_type}")
    print(f"[WORKER_DEBUG]   db_task_params keys: {list(db_task_params.keys())}")
    print(f"[WORKER_DEBUG]   'use_lighti2x_lora' in db_task_params: {'use_lighti2x_lora' in db_task_params}")
    if 'use_lighti2x_lora' in db_task_params:
        print(f"[WORKER_DEBUG]   db_task_params['use_lighti2x_lora']: {db_task_params['use_lighti2x_lora']}")
    print(f"[WORKER_DEBUG]   generation_params keys after copy: {list(generation_params.keys())}")
    
    if task_type == "travel_segment":
        dprint(f"[DEEP_DEBUG] Worker {task_id}: DB TASK TO GENERATION TASK CONVERSION")
        dprint(f"[DEEP_DEBUG]   db_task_params.get('use_causvid_lora'): {db_task_params.get('use_causvid_lora')}")
        dprint(f"[DEEP_DEBUG]   db_task_params.get('use_lighti2x_lora'): {db_task_params.get('use_lighti2x_lora')}")
        dprint(f"[DEEP_DEBUG]   generation_params.get('use_causvid_lora'): {generation_params.get('use_causvid_lora')}")
        dprint(f"[DEEP_DEBUG]   generation_params.get('use_lighti2x_lora'): {generation_params.get('use_lighti2x_lora')}")
    
    # Handle field name variations (orchestrator uses different names than WGP)
    # Map 'steps' to 'num_inference_steps' for compatibility
    if "steps" in db_task_params and "num_inference_steps" not in generation_params:
        generation_params["num_inference_steps"] = db_task_params["steps"]
        dprint(f"Task {task_id}: Mapped 'steps' ({db_task_params['steps']}) to 'num_inference_steps'")
    
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
    
    # Handle additional_loras dict format conversion
    if "additional_loras" in generation_params:
        additional_loras = generation_params["additional_loras"]
        if isinstance(additional_loras, dict) and additional_loras:
            # Convert dict format to lists format
            # Dict format: {"url1": multiplier1, "url2": multiplier2}
            # List format: additional_lora_names=["url1", "url2"], additional_lora_multipliers=[multiplier1, multiplier2]
            lora_names = list(additional_loras.keys())
            lora_multipliers = list(additional_loras.values())
            
            generation_params["additional_lora_names"] = lora_names
            generation_params["additional_lora_multipliers"] = lora_multipliers
            
            dprint(f"Task {task_id}: Converted additional_loras dict to lists: {len(lora_names)} LoRAs")
            for i, (name, mult) in enumerate(zip(lora_names, lora_multipliers)):
                dprint(f"  {i+1}. {name} (multiplier: {mult})")
        
        # Remove the old format to avoid confusion
        del generation_params["additional_loras"]
    
    # Set default values for essential parameters only (orchestrator handles generation defaults)
    # Only set defaults for parameters that are required for basic task functionality
    essential_defaults = {
        "seed": -1,  # Random seed (essential for reproducibility)
        "negative_prompt": "",  # Empty negative prompt (essential to prevent errors)
    }
    
    for param, default_value in essential_defaults.items():
        if param not in generation_params:
            generation_params[param] = default_value
    
    # NOTE: resolution, video_length, num_inference_steps, guidance_scale are now handled 
    # by WanOrchestrator._resolve_parameters() with proper model config precedence
    
    # REMOVED: Wan 2.2 optimizations that interfere with model preset precedence
    # 
    # PROPER PRECEDENCE CHAIN:
    # 1. User explicit parameters (highest priority) - handled by WanOrchestrator._resolve_parameters()
    # 2. Model preset JSON config (medium priority) - handled by WanOrchestrator._resolve_parameters()  
    # 3. System defaults (lowest priority) - handled by WanOrchestrator._resolve_parameters()
    #
    # All parameter resolution is now centralized in WanOrchestrator._resolve_parameters()
    # to ensure consistent precedence without conflicts
    
    # Auto-enable built-in acceleration LoRAs only if no LoRAs explicitly specified
    # This doesn't interfere with parameter precedence
    if ("2_2" in model or "cocktail_2_2" in model) and "lora_names" not in generation_params and "activated_loras" not in db_task_params:
        generation_params["lora_names"] = ["CausVid", "DetailEnhancerV1"]
        generation_params["lora_multipliers"] = [1.0, 0.2]  # DetailEnhancer at reduced strength
        dprint(f"Task {task_id}: Auto-enabled Wan 2.2 acceleration LoRAs (no parameter overrides)")
    
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
# Travel Segment Queue Integration
# -----------------------------------------------------------------------------

def _handle_travel_segment_via_queue(task_params_dict, main_output_dir_base: Path, task_id: str, apply_reward_lora: bool, colour_match_videos: bool, mask_active_frames: bool, task_queue: HeadlessTaskQueue, dprint):
    """
    Handle travel segment tasks via direct queue integration to eliminate blocking waits.
    
    This replaces the complex blocking logic in travel_between_images.py with direct
    queue submission, maintaining model persistence and eliminating triple-queue inefficiency.
    """
    dprint(f"_handle_travel_segment_via_queue: Starting for {task_id}")
    
    try:
        # Import required functions from travel_between_images
        from source.sm_functions.travel_between_images import (
            debug_video_analysis, _handle_travel_chaining_after_wgp
        )
        from source.common_utils import (
            parse_resolution as sm_parse_resolution,
            snap_resolution_to_model_grid,
            ensure_valid_prompt,
            ensure_valid_negative_prompt,
            create_mask_video_from_inactive_indices
        )
        from source.video_utils import (
            prepare_vace_ref_for_segment as sm_prepare_vace_ref_for_segment,
            create_guide_video_for_travel_segment as sm_create_guide_video_for_travel_segment
        )
        from source import db_operations as db_ops
        
        # Extract core parameters needed for generation
        segment_params = task_params_dict
        orchestrator_task_id_ref = segment_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = segment_params.get("orchestrator_run_id")
        segment_idx = segment_params.get("segment_index")
        
        if None in [orchestrator_task_id_ref, orchestrator_run_id, segment_idx]:
            return False, f"Travel segment {task_id} missing critical orchestrator references"
        
        # Get full orchestrator payload for processing
        full_orchestrator_payload = segment_params.get("full_orchestrator_payload")
        if not full_orchestrator_payload:
            # Fetch from orchestrator task if not included
            orchestrator_task_raw_params_json = db_ops.get_task_params(orchestrator_task_id_ref)
            if orchestrator_task_raw_params_json:
                fetched_params = json.loads(orchestrator_task_raw_params_json) if isinstance(orchestrator_task_raw_params_json, str) else orchestrator_task_raw_params_json
                full_orchestrator_payload = fetched_params.get("orchestrator_details")
            
            if not full_orchestrator_payload:
                return False, f"Travel segment {task_id}: Could not retrieve orchestrator payload"
        
        # Extract generation parameters
        model_name = full_orchestrator_payload["model_name"]
        prompt_for_wgp = ensure_valid_prompt(segment_params.get("base_prompt", " "))
        negative_prompt_for_wgp = ensure_valid_negative_prompt(segment_params.get("negative_prompt", " "))
        
        # Parse and snap resolution
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        parsed_res_raw = sm_parse_resolution(parsed_res_wh_str)
        if parsed_res_raw is None:
            return False, f"Travel segment {task_id}: Invalid resolution format {parsed_res_wh_str}"
        parsed_res_wh = snap_resolution_to_model_grid(parsed_res_raw)
        
        # Get frame parameters
        total_frames_for_segment = segment_params.get("segment_frames_target", 
                                                    full_orchestrator_payload["segment_frames_expanded"][segment_idx])
        
        # Set up processing directory
        current_run_base_output_dir_str = segment_params.get("current_run_base_output_dir")
        if not current_run_base_output_dir_str:
            current_run_base_output_dir_str = full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
        
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        segment_processing_dir = current_run_base_output_dir
        segment_processing_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the shared TravelSegmentProcessor to eliminate code duplication
        try:
            from source.travel_segment_processor import TravelSegmentProcessor, TravelSegmentContext
            
            # Get debug mode from either segment params or orchestrator payload
            debug_enabled = segment_params.get("debug_mode_enabled", full_orchestrator_payload.get("debug_mode_enabled", False))
            
            # Create context for the shared processor
            processor_context = TravelSegmentContext(
                task_id=task_id,
                segment_idx=segment_idx,
                model_name=model_name,
                total_frames_for_segment=total_frames_for_segment,
                parsed_res_wh=parsed_res_wh,
                segment_processing_dir=segment_processing_dir,
                full_orchestrator_payload=full_orchestrator_payload,
                segment_params=segment_params,
                mask_active_frames=mask_active_frames,
                debug_enabled=debug_enabled,
                dprint=dprint
            )
            
            # Create and use the shared processor
            processor = TravelSegmentProcessor(processor_context)
            segment_outputs = processor.process_segment()
            
            # Extract outputs for queue handler variables
            guide_video_path = segment_outputs.get("video_guide")
            mask_video_path_for_wgp = Path(segment_outputs["video_mask"]) if segment_outputs.get("video_mask") else None
            video_prompt_type_str = segment_outputs["video_prompt_type"]
            
            dprint(f"[SHARED_PROCESSOR_QUEUE] Seg {segment_idx}: Guide video: {guide_video_path}")
            dprint(f"[SHARED_PROCESSOR_QUEUE] Seg {segment_idx}: Mask video: {mask_video_path_for_wgp}")
            dprint(f"[SHARED_PROCESSOR_QUEUE] Seg {segment_idx}: Video prompt type: {video_prompt_type_str}")
            
        except Exception as e_shared_processor:
            dprint(f"[ERROR] Seg {segment_idx}: Shared processor failed in queue handler: {e_shared_processor}")
            traceback.print_exc()
            return False, f"Shared processor failed: {e_shared_processor}"
        
        # Create generation task parameters optimized for queue processing
        # Let WanOrchestrator handle parameter resolution with proper model preset precedence
        generation_params = {
            "negative_prompt": negative_prompt_for_wgp,
            "resolution": f"{parsed_res_wh[0]}x{parsed_res_wh[1]}",
            "video_length": total_frames_for_segment,
            "seed": segment_params.get("seed_to_use", 12345),
            "use_causvid_lora": full_orchestrator_payload.get("apply_causvid", False),
            "use_lighti2x_lora": full_orchestrator_payload.get("use_lighti2x_lora", False),
            "apply_reward_lora": apply_reward_lora,
        }
        
        # Only add explicit parameters if they're provided (let model preset handle defaults)
        # Check both 'steps' and 'num_inference_steps' from orchestrator payload
        # Priority: segment_params > orchestrator_payload (user explicit > orchestrator defaults)
        explicit_steps = (
            segment_params.get("num_inference_steps") or
            segment_params.get("steps") or
            full_orchestrator_payload.get("num_inference_steps") or 
            full_orchestrator_payload.get("steps")
        )
        if explicit_steps:
            generation_params["num_inference_steps"] = explicit_steps
            dprint(f"[QUEUE_PARAMS] Using explicit steps: {explicit_steps} (user override)")
        else:
            dprint(f"[QUEUE_PARAMS] No explicit steps provided - model preset will handle defaults")
        
        # Only add guidance_scale if explicitly provided
        explicit_guidance = (
            full_orchestrator_payload.get("guidance_scale") or
            segment_params.get("guidance_scale")
        )
        if explicit_guidance:
            generation_params["guidance_scale"] = explicit_guidance
            dprint(f"[QUEUE_PARAMS] Using explicit guidance_scale: {explicit_guidance}")
        
        # Only add flow_shift if explicitly provided
        explicit_flow_shift = (
            full_orchestrator_payload.get("flow_shift") or
            segment_params.get("flow_shift")
        )
        if explicit_flow_shift:
            generation_params["flow_shift"] = explicit_flow_shift
            dprint(f"[QUEUE_PARAMS] Using explicit flow_shift: {explicit_flow_shift}")
        
        # Add video guide if available (ESSENTIAL for VACE models)
        if guide_video_path:
            generation_params["video_guide"] = str(guide_video_path)
            dprint(f"[QUEUE_GUIDE_DEBUG] Added video_guide to generation_params: {guide_video_path}")
        else:
            dprint(f"[QUEUE_GUIDE_DEBUG] No guide video available for generation_params")
        
        # Add video mask if available (SAME as original travel_between_images.py)
        if mask_video_path_for_wgp:
            generation_params["video_mask"] = str(mask_video_path_for_wgp.resolve())
            dprint(f"[QUEUE_MASK_DEBUG] Added video_mask to generation_params: {mask_video_path_for_wgp}")
        else:
            dprint(f"[QUEUE_MASK_DEBUG] No mask video available for generation_params")
        
        # Add video_prompt_type to generation params (ESSENTIAL for VACE)
        generation_params["video_prompt_type"] = video_prompt_type_str
        
        # Create and submit generation task
        from headless_model_management import GenerationTask
        generation_task = GenerationTask(
            id=f"travel_seg_{task_id}",
            model=model_name,
            prompt=prompt_for_wgp,
            parameters=generation_params
        )
        
        # Submit to queue
        submitted_task_id = task_queue.submit_task(generation_task)
        dprint(f"Travel segment {task_id}: Submitted to queue as {submitted_task_id}")
        
        # Wait for completion with timeout
        max_wait_time = 1800  # 30 minutes for travel segments
        wait_interval = 2
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            status = task_queue.get_task_status(f"travel_seg_{task_id}")
            if status is None:
                return False, f"Travel segment {task_id}: Task status became None"
            
            if status.status == "completed":
                dprint(f"Travel segment {task_id}: Generation completed: {status.result_path}")
                
                # Apply post-processing chain (saturation, brightness, color matching)
                chain_success, chain_message, final_chained_path = _handle_travel_chaining_after_wgp(
                    wgp_task_params={"travel_chain_details": {
                        "orchestrator_task_id_ref": orchestrator_task_id_ref,
                        "orchestrator_run_id": orchestrator_run_id,
                        "segment_index_completed": segment_idx,
                        "full_orchestrator_payload": full_orchestrator_payload,
                        "segment_processing_dir_for_saturation": str(segment_processing_dir),
                        "is_first_new_segment_after_continue": segment_params.get("is_first_segment", False) and full_orchestrator_payload.get("continue_from_video_resolved_path"),
                        "is_subsequent_segment": not segment_params.get("is_first_segment", True),
                        "colour_match_videos": colour_match_videos,
                        "cm_start_ref_path": None,  # Simplified for now
                        "cm_end_ref_path": None,    # Simplified for now
                        "show_input_images": False, # Simplified for now
                        "start_image_path": None,   # Simplified for now
                        "end_image_path": None,     # Simplified for now
                    }},
                    actual_wgp_output_video_path=status.result_path,
                    image_download_dir=segment_processing_dir,
                    dprint=dprint
                )
                
                if chain_success and final_chained_path:
                    return True, final_chained_path
                else:
                    return True, status.result_path  # Use raw output if chaining failed
                    
            elif status.status == "failed":
                return False, f"Travel segment {task_id}: Generation failed: {status.error_message}"
            
            # Still processing
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        # Timeout
        return False, f"Travel segment {task_id}: Generation timeout after {max_wait_time}s"
        
    except Exception as e:
        dprint(f"Travel segment {task_id}: Exception: {e}")
        traceback.print_exc()
        return False, f"Travel segment {task_id}: Exception: {str(e)}"

# -----------------------------------------------------------------------------
# Process a single task using queue-based architecture
# -----------------------------------------------------------------------------

def process_single_task(task_params_dict, main_output_dir_base: Path, task_type: str, project_id_for_task: str | None, image_download_dir: Path | str | None = None, apply_reward_lora: bool = False, colour_match_videos: bool = False, mask_active_frames: bool = True, task_queue: HeadlessTaskQueue = None):
    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))
    
    headless_logger.debug(f"Entering process_single_task", task_id=task_id)
    headless_logger.debug(f"Task Type: {task_type}", task_id=task_id)
    headless_logger.debug(f"Project ID: {project_id_for_task}", task_id=task_id)
    headless_logger.debug(f"Task Params: {json.dumps(task_params_dict, default=str, indent=2)[:1000]}...", task_id=task_id)
    
    headless_logger.essential(f"Processing {task_type} task", task_id=task_id)
    output_location_to_db = None # Will store the final path/URL for the DB
    generation_success = False

    # --- NEW: Direct Queue Integration for Simple Generation Tasks ---
    # Route simple generation tasks directly to HeadlessTaskQueue to eliminate blocking waits
    direct_queue_task_types = {
        "single_image", "vace", "vace_21", "vace_22", "flux", "t2v", "t2v_22", 
        "i2v", "i2v_22", "hunyuan", "ltxv", "generate_video"
    }
    
    if task_type in direct_queue_task_types and task_queue is not None:
        headless_logger.debug(f"Using direct queue integration for task type: {task_type}", task_id=task_id)
        
        try:
            # Create GenerationTask object from DB parameters
            generation_task = db_task_to_generation_task(task_params_dict, task_id, task_type)
            
            # For single_image tasks, ensure video_length=1 for PNG conversion
            if task_type == "single_image":
                generation_task.parameters["video_length"] = 1
                headless_logger.debug(f"Overriding video_length=1 for single_image task", task_id=task_id)
            
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
            
            # Return early for direct queue tasks
            print(f"--- Finished task ID: {task_id} (Success: {generation_success}) ---")
            return generation_success, output_location_to_db
            
        except Exception as e_queue:
            print(f"[ERROR Task ID: {task_id}] Queue processing error: {e_queue}")
            traceback.print_exc()
            generation_success = False
            output_location_to_db = f"Error: Queue processing failed - {str(e_queue)}"
            print(f"--- Finished task ID: {task_id} (Success: {generation_success}) ---")
            return generation_success, output_location_to_db

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
        headless_logger.debug("Using direct queue integration for travel segment", task_id=task_id)
        # NEW: Route travel segments directly to queue to eliminate blocking wait
        return _handle_travel_segment_via_queue(task_params_dict, main_output_dir_base, task_id, apply_reward_lora, colour_match_videos, mask_active_frames, task_queue=task_queue, dprint=dprint)
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
            main_output_dir_base=main_output_dir_base,
            process_single_task=process_single_task,
            task_params_from_db=task_params_dict,
            dprint=dprint,
            task_queue=task_queue
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
        generation_success, output_location_to_db = handle_rife_interpolate_task(task_params_dict, main_output_dir_base, task_id, dprint, task_queue=task_queue)

    # Default handling for standard wgp tasks
    else:
        # NEW QUEUE-BASED PROCESSING: Delegate to task queue (always required now)
        if task_queue is None:
            raise RuntimeError(f"Task {task_id}: Queue-based processing is required but task_queue is None")
            
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

    # --- Chaining Logic ---
    # This block is now executed for any successful primitive task that doesn't return early.
    if generation_success:
        chaining_result_path_override = None

        if task_params_dict.get("travel_chain_details"):
            dprint(f"WGP Task {task_id} is part of a travel sequence. Attempting to chain.")
            chain_success, chain_message, final_path_from_chaining = tbi._handle_travel_chaining_after_wgp(
                wgp_task_params=task_params_dict, 
                actual_wgp_output_video_path=output_location_to_db,
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
# Main server loop
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

    db_path_str = str(db_ops.SQLITE_DB_PATH) if db_ops.DB_TYPE == "sqlite" else db_ops.PG_TABLE_NAME # Use consistent string path for db functions

    # --- Initialize Task Queue System (required) ---
    print(f"[INFO] Initializing queue-based task processing system...")
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
        print("[FATAL] Queue initialization failed - cannot continue without task queue")
        sys.exit(1)
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
                current_task_params, main_output_dir, current_task_type, current_project_id,
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