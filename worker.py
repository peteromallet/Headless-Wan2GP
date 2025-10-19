"""Wan2GP Worker Server.

This long-running process polls the Supabase-backed Postgres `tasks` table,
claims queued tasks, and executes them using the HeadlessTaskQueue system. Besides standard generation
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
• Parameter resolution is centralized in WanOrchestrator; model presets apply unless overridden
• Built-in acceleration LoRAs (CausVid, DetailEnhancer) are auto-enabled for Wan 2.2 when no explicit LoRAs are provided
• All optimizations can be overridden by explicit task parameters
• All generation is routed through HeadlessTaskQueue (no legacy blocking pipelines)
"""

import argparse
import sys
import os
import json
import time
import datetime
import traceback
import threading
from multiprocessing import Process, Queue

from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient

# RAM monitoring
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    print("[WORKER] psutil not available - RAM monitoring disabled")

# Add the current directory to Python path so Wan2GP can be imported as a module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add the Wan2GP subdirectory to the path for its internal imports
wan2gp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2GP")
if wan2gp_path not in sys.path:
    sys.path.append(wan2gp_path)

# --- SM_RESTRUCTURE: Import moved/new utilities ---
from source import db_operations as db_ops
from source.fatal_error_handler import FatalWorkerError, safe_handle_fatal_error, reset_fatal_error_counter
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
# single_image tasks now use direct queue integration (wan_2_2_t2i)
from source.sm_functions import magic_edit as me
from source.sm_functions.join_clips import _handle_join_clips_task
from source.sm_functions.inpaint_frames import _handle_inpaint_frames_task
from source.sm_functions.create_visualization import _handle_create_visualization_task
# --- New Queue-based Architecture Imports ---
# Protect sys.argv before importing queue management which imports wgp.py
_original_argv = sys.argv[:]
sys.argv = ["worker.py"]  # Prevent wgp.py from parsing our CLI args
from headless_model_management import HeadlessTaskQueue, GenerationTask
sys.argv = _original_argv  # Restore original arguments
# --- Structured Logging ---
from source.logging_utils import (
    headless_logger,
    enable_debug_mode,
    disable_debug_mode,
    LogBuffer,
    CustomLogInterceptor,
    set_log_interceptor,
    # Safe logging utilities for large data structures
    safe_repr,
    safe_dict_repr,
    safe_json_repr,
    safe_log_params,
    safe_log_change
)
# --- End SM_RESTRUCTURE imports ---


# -----------------------------------------------------------------------------
# Debug / Verbose Logging Helpers
# -----------------------------------------------------------------------------

debug_mode = False  # This will be toggled on via the --debug CLI flag in main()

def dprint(msg: str, task_id: str = None):
    """Print a debug message if --debug flag is enabled."""
    if debug_mode:
        # Prefix with timestamp for easier tracing
        if task_id:
            print(f"[DEBUG {datetime.datetime.now().isoformat()}] [Task {task_id}] {msg}")
        else:
            print(f"[DEBUG {datetime.datetime.now().isoformat()}] {msg}")

def make_task_dprint(task_id: str):
    """Create a task-aware dprint function that automatically includes task_id."""
    def task_dprint(msg: str):
        dprint(msg, task_id=task_id)
    return task_dprint

def log_ram_usage(label: str, task_id: str = "unknown") -> dict:
    """
    Log current RAM usage with a descriptive label.
    Returns dict with RAM metrics for programmatic use.
    """
    if not _PSUTIL_AVAILABLE:
        return {"available": False}

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024**2
        rss_gb = rss_mb / 1024

        # Get system-wide memory stats
        sys_mem = psutil.virtual_memory()
        sys_total_gb = sys_mem.total / 1024**3
        sys_available_gb = sys_mem.available / 1024**3
        sys_used_percent = sys_mem.percent

        headless_logger.info(
            f"[RAM] {label}: Process={rss_mb:.0f}MB ({rss_gb:.2f}GB) | "
            f"System={sys_used_percent:.1f}% used, {sys_available_gb:.1f}GB/{sys_total_gb:.1f}GB available",
            task_id=task_id
        )

        return {
            "available": True,
            "process_rss_mb": rss_mb,
            "process_rss_gb": rss_gb,
            "system_total_gb": sys_total_gb,
            "system_available_gb": sys_available_gb,
            "system_used_percent": sys_used_percent
        }

    except Exception as e:
        headless_logger.warning(f"[RAM] Failed to get RAM usage: {e}", task_id=task_id)
        return {"available": False, "error": str(e)}

def cleanup_generated_files(output_location: str, task_id: str = "unknown") -> None:
    """
    Delete generated files after successful task completion unless in debug mode.
    This includes the main output file/directory and any temporary files that may have been created.

    Args:
        output_location: Path to the generated file or directory to clean up
        task_id: Task ID for logging purposes
    """
    if debug_mode:
        headless_logger.debug(f"Debug mode enabled - skipping file cleanup for {output_location}", task_id=task_id)
        return

    if not output_location:
        return

    try:
        file_path = Path(output_location)

        # Clean up main output file/directory
        if file_path.exists() and file_path.is_file():
            file_size = file_path.stat().st_size
            file_path.unlink()
            headless_logger.debug(f"Cleaned up generated file: {output_location} ({file_size:,} bytes)", task_id=task_id)
        elif file_path.exists() and file_path.is_dir():
            import shutil
            dir_size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
            shutil.rmtree(file_path)
            headless_logger.debug(f"Cleaned up generated directory: {output_location} ({dir_size:,} bytes)", task_id=task_id)
        else:
            headless_logger.debug(f"File cleanup skipped - path not found: {output_location}", task_id=task_id)

        # Clean up temporary files that may have been created during processing
        _cleanup_temporary_files(task_id)

    except Exception as e:
        headless_logger.warning(f"Failed to cleanup generated file {output_location}: {e}", task_id=task_id)


def _cleanup_temporary_files(task_id: str = "unknown") -> None:
    """
    Clean up temporary files that were specifically created during this task's execution.
    This function is intentionally conservative - it only targets files that we can
    be reasonably certain were created by our task processing.

    Args:
        task_id: Task ID for logging purposes
    """
    if debug_mode:
        return  # Skip temp file cleanup in debug mode

    # Note: Most temporary files are already cleaned up by their respective functions:
    # - db_operations.py cleans up temporary frame files
    # - audio_video.py has cleanup_temp_audio_files() function
    # - TemporaryDirectory contexts auto-cleanup
    #
    # This function is mainly a safety net for any edge cases where cleanup
    # might not have occurred properly during normal execution.

    # Clean up phase_config in-memory model patches
    # Note: The cleanup is now handled by restore_model_patches() which should be
    # called at the end of task processing. This is just a safety net.
    try:
        import sys
        wan_dir_path = str(Path(__file__).parent / "Wan2GP")
        if wan_dir_path in sys.path:
            # Check if we have any stored patch info to restore
            # (This would only happen if the task didn't clean up properly)
            pass  # Actual restoration happens in the task processing flow
    except Exception as e:
        headless_logger.warning(f"Failed during model patch cleanup check: {e}", task_id=task_id)

    headless_logger.debug(f"Temporary file cleanup completed (most temp files auto-cleaned by their creators)", task_id=task_id)

# -----------------------------------------------------------------------------
# Phase Config Model Patching Functions
# -----------------------------------------------------------------------------

def apply_phase_config_patch(parsed_phase_config: dict, model_name: str, task_id: str):
    """
    Apply phase_config patches to the model definition in WGP.
    This must be called AFTER WGP is initialized (right before generation).

    Args:
        parsed_phase_config: The result dict from parse_phase_config() containing patch info
        model_name: The model name to patch
        task_id: Task ID for logging
    """
    if not parsed_phase_config.get("_patch_config"):
        return  # No patch needed

    try:
        import sys
        import json
        import os
        from pathlib import Path
        import copy

        wan_dir = Path(__file__).parent / "Wan2GP"
        defaults_dir = wan_dir / "defaults"

        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        # Change to Wan2GP directory so WGP can find model definitions
        _saved_cwd = os.getcwd()
        os.chdir(str(wan_dir))

        # Protect sys.argv before importing WGP to avoid argparse errors
        _saved_argv = sys.argv[:]
        sys.argv = ["apply_phase_config_patch.py"]
        try:
            import wgp

            # FORCE-load the model definition if not already loaded
            # This ensures wgp.models_def is populated before patching
            if model_name not in wgp.models_def:
                headless_logger.debug(
                    f"Model definition '{model_name}' not in wgp.models_def, force-loading it now",
                    task_id=task_id
                )
                model_def = wgp.get_model_def(model_name)
                if not model_def:
                    headless_logger.warning(
                        f"Model '{model_name}' not found, cannot patch for phase_config",
                        task_id=task_id
                    )
                    return

                # Ensure the model definition is now in wgp.models_def
                # get_model_def should have added it, but verify
                if model_name not in wgp.models_def:
                    # Manually add it if get_model_def didn't
                    wgp.models_def[model_name] = wgp.init_model_def(model_name, model_def)
                    headless_logger.debug(
                        f"Manually added '{model_name}' to wgp.models_def",
                        task_id=task_id
                    )

            # Check if model exists in models_def (should be loaded now)
            if model_name in wgp.models_def:
                # Store original for restoration later
                parsed_phase_config["_original_model_def"] = copy.deepcopy(wgp.models_def[model_name])
                parsed_phase_config["_model_was_patched"] = True

                # Get the patch config we saved earlier
                patch_config = parsed_phase_config["_patch_config"]

                # Overwrite the ORIGINAL model entry in models_def
                temp_model_def = copy.deepcopy(patch_config["model"])

                # Store settings (top-level config minus model) in model_def
                temp_settings = copy.deepcopy(patch_config)
                del temp_settings["model"]
                temp_model_def["settings"] = temp_settings

                # DIRECTLY PATCH the original model name's entry
                # First set the partial def, then run init_model_def to add family-handler defaults like vace_class
                wgp.models_def[model_name] = temp_model_def
                temp_model_def = wgp.init_model_def(model_name, temp_model_def)
                wgp.models_def[model_name] = temp_model_def

                headless_logger.info(
                    f"✅ Patched wgp.models_def['{model_name}'] in memory: {patch_config.get('guidance_phases')} phases, "
                    f"cleared built-in LoRAs, flow_shift={patch_config.get('flow_shift')}, "
                    f"solver={patch_config.get('sample_solver')}",
                    task_id=task_id
                )
            else:
                headless_logger.warning(
                    f"Model '{model_name}' not found in wgp.models_def after load attempt",
                    task_id=task_id
                )
        finally:
            sys.argv = _saved_argv
            os.chdir(_saved_cwd)
    except Exception as e:
        headless_logger.warning(f"Failed to apply phase_config patch: {e}", task_id=task_id)
        import traceback
        headless_logger.debug(f"Patch error traceback: {traceback.format_exc()}", task_id=task_id)


def restore_model_patches(parsed_phase_config: dict, model_name: str, task_id: str):
    """
    Restore the original model definition after phase_config patching.

    Args:
        parsed_phase_config: The result dict from parse_phase_config()
        model_name: The original model name that was patched
        task_id: Task ID for logging
    """
    if not parsed_phase_config.get("_model_was_patched"):
        return  # Nothing to restore

    try:
        import sys
        wan_dir_path = str(Path(__file__).parent / "Wan2GP")
        if wan_dir_path in sys.path:
            import wgp

            if "_original_model_def" in parsed_phase_config and model_name in wgp.models_def:
                # Restore the original model definition
                wgp.models_def[model_name] = parsed_phase_config["_original_model_def"]
                headless_logger.info(
                    f"✅ Restored original wgp.models_def['{model_name}']",
                    task_id=task_id
                )
    except Exception as e:
        headless_logger.warning(f"Failed to restore model patches: {e}", task_id=task_id)


# -----------------------------------------------------------------------------
# Queue-based Task Processing Functions
# -----------------------------------------------------------------------------

def parse_phase_config(phase_config: dict, num_inference_steps: int, task_id: str = "unknown", model_name: str = None) -> dict:
    """
    Parse phase_config override block and return computed parameters.

    Args:
        phase_config: Phase configuration dict with structure:
            {
                "num_phases": 3,
                "steps_per_phase": [2, 2, 2],
                "flow_shift": 5.0,
                "sample_solver": "euler",
                "model_switch_phase": 2,
                "phases": [
                    {
                        "phase": 1,
                        "guidance_scale": 3.0,
                        "loras": [
                            {"url": "...", "multiplier": "1.0"} or {"url": "...", "multiplier": "0.5,0.7,1.0"}
                        ]
                    },
                    ...
                ]
            }
        num_inference_steps: Total number of inference steps
        task_id: Task ID for logging

    Returns:
        dict with computed parameters ready for generation:
            - guidance_phases: int
            - switch_threshold: float
            - switch_threshold2: float (if 3 phases)
            - guidance_scale, guidance2_scale, guidance3_scale: float
            - flow_shift: float
            - sample_solver: str
            - model_switch_phase: int
            - lora_names: list[str]
            - lora_multipliers: list[str] (phase-formatted)
            - additional_loras: dict[str, float]
    """
    import numpy as np

    # Import the timestep_transform function from phase_distribution_simulator
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'testing_tools'))
    try:
        from phase_distribution_simulator import timestep_transform
    except ImportError:
        # Fallback: inline implementation
        def timestep_transform(t: float, shift: float = 5.0, num_timesteps: int = 1000) -> float:
            t = t / num_timesteps
            new_t = shift * t / (1 + (shift - 1) * t)
            new_t = new_t * num_timesteps
            return new_t

    # Validation with auto-correction
    num_phases = phase_config.get("num_phases", 3)
    steps_per_phase = phase_config.get("steps_per_phase", [2, 2, 2])
    flow_shift = phase_config.get("flow_shift", 5.0)
    phases_config = phase_config.get("phases", [])

    # Auto-correct num_phases if it doesn't match steps_per_phase or phases array
    if len(steps_per_phase) != num_phases or len(phases_config) != num_phases:
        inferred_phases = len(steps_per_phase)
        if len(phases_config) != inferred_phases:
            # Mismatch between all three - use phases array as source of truth
            inferred_phases = len(phases_config)
            headless_logger.warning(
                f"phase_config mismatch: num_phases={num_phases}, steps_per_phase has {len(steps_per_phase)} values, "
                f"phases array has {len(phases_config)} entries. Using phases array length: {inferred_phases}",
                task_id=task_id
            )
        else:
            headless_logger.warning(
                f"phase_config mismatch: num_phases={num_phases} but steps_per_phase has {len(steps_per_phase)} values. "
                f"Auto-correcting num_phases to {inferred_phases}",
                task_id=task_id
            )
        num_phases = inferred_phases

    if num_phases not in [2, 3]:
        raise ValueError(f"Task {task_id}: num_phases must be 2 or 3, got {num_phases}")

    total_steps = sum(steps_per_phase)
    if total_steps != num_inference_steps:
        raise ValueError(f"Task {task_id}: steps_per_phase {steps_per_phase} sum to {total_steps}, but num_inference_steps is {num_inference_steps}")

    phases_config = phase_config.get("phases", [])
    if len(phases_config) != num_phases:
        raise ValueError(f"Task {task_id}: Expected {num_phases} phase configs, got {len(phases_config)}")

    # Generate timesteps using the same logic as WGP's actual scheduler
    sample_solver = phase_config.get("sample_solver", "euler")

    # Validate sample_solver compatibility
    if sample_solver == "causvid":
        headless_logger.warning(f"phase_config with causvid solver is not recommended - causvid uses hardcoded timesteps", task_id=task_id)

    # Replicate the EXACT logic from the actual WGP schedulers
    # See: Wan2GP/shared/utils/fm_solvers_unipc.py and fm_solvers.py
    if sample_solver == "unipc" or sample_solver == "":
        # UniPC: linspace(1.0, 0.001, n+1)[:-1] → shift → * 1000
        # See: fm_solvers_unipc.py lines 182-205
        sigma_max = 1.0
        sigma_min = 0.001

        sigmas = list(np.linspace(sigma_max, sigma_min, num_inference_steps + 1, dtype=np.float32))[:-1]
        sigmas = [flow_shift * s / (1 + (flow_shift - 1) * s) for s in sigmas]
        timesteps = [s * 1000 for s in sigmas]

        headless_logger.debug(f"Generated UniPC timesteps with shift={flow_shift}", task_id=task_id)

    elif sample_solver in ["dpm++", "dpm++_sde"]:
        # DPM++: linspace(1, 0, n+1)[:n] → shift → * 1000
        # See: get_sampling_sigmas() in fm_solvers.py lines 22-26
        sigmas = list(np.linspace(1, 0, num_inference_steps + 1, dtype=np.float32))[:num_inference_steps]
        sigmas = [flow_shift * s / (1 + (flow_shift - 1) * s) for s in sigmas]
        timesteps = [s * 1000 for s in sigmas]

        headless_logger.debug(f"Generated DPM++ timesteps with shift={flow_shift} for {sample_solver}", task_id=task_id)
    elif sample_solver == "euler":
        # Euler uses direct timestep transform
        timesteps = list(np.linspace(1000, 1, num_inference_steps, dtype=np.float32))
        timesteps.append(0.)
        timesteps = [timestep_transform(t, shift=flow_shift, num_timesteps=1000) for t in timesteps][:-1]
        headless_logger.debug(f"Generated Euler-style timesteps with shift={flow_shift}", task_id=task_id)
    else:
        # Fallback for unknown solvers
        timesteps = list(np.linspace(1000, 1, num_inference_steps, dtype=np.float32))
        headless_logger.debug(f"Generated linear timesteps for unknown solver {sample_solver}", task_id=task_id)

    headless_logger.debug(f"Generated timesteps for phase_config: {[f'{t:.1f}' for t in timesteps]}", task_id=task_id)

    # Calculate switch thresholds based on steps_per_phase
    # steps_per_phase=[5,2,2] means: phase1=steps 0-4, phase2=steps 5-6, phase3=steps 7-8
    # The switch must trigger BEFORE entering the new phase's first step
    # Since the check is "t <= threshold", we set threshold between the last step of old phase and first step of new phase

    switch_step_1 = steps_per_phase[0]  # First step of phase 2
    switch_threshold = None
    switch_threshold2 = None

    if switch_step_1 < num_inference_steps:
        # Set threshold between last step of phase 1 (switch_step_1-1) and first step of phase 2 (switch_step_1)
        # Use midpoint so the switch triggers when entering phase 2's first step
        last_phase1_t = timesteps[switch_step_1 - 1]
        first_phase2_t = timesteps[switch_step_1]
        switch_threshold = float((last_phase1_t + first_phase2_t) / 2)
        headless_logger.debug(f"Calculated switch_threshold (phase 1→2): {switch_threshold:.1f} (between step {switch_step_1-1} t={last_phase1_t:.1f} and step {switch_step_1} t={first_phase2_t:.1f})", task_id=task_id)

    if num_phases >= 3:
        switch_step_2 = steps_per_phase[0] + steps_per_phase[1]  # First step of phase 3
        if switch_step_2 < num_inference_steps:
            last_phase2_t = timesteps[switch_step_2 - 1]
            first_phase3_t = timesteps[switch_step_2]
            switch_threshold2 = float((last_phase2_t + first_phase3_t) / 2)
            headless_logger.debug(f"Calculated switch_threshold2 (phase 2→3): {switch_threshold2:.1f} (between step {switch_step_2-1} t={last_phase2_t:.1f} and step {switch_step_2} t={first_phase3_t:.1f})", task_id=task_id)

    # Build result dict
    result = {
        "guidance_phases": num_phases,
        "switch_threshold": switch_threshold,
        "switch_threshold2": switch_threshold2,
        "flow_shift": flow_shift,
        "sample_solver": sample_solver,
        "model_switch_phase": phase_config.get("model_switch_phase", 2),
    }

    # Extract guidance scales per phase
    if num_phases >= 1:
        result["guidance_scale"] = phases_config[0].get("guidance_scale", 3.0)
    if num_phases >= 2:
        result["guidance2_scale"] = phases_config[1].get("guidance_scale", 1.0)
    if num_phases >= 3:
        result["guidance3_scale"] = phases_config[2].get("guidance_scale", 1.0)

    # Process LoRAs: collect all unique LoRA URLs and build phase-formatted multipliers
    # IMPORTANT: Preserve the order LoRAs appear in phases (first occurrence order)
    # Do NOT sort alphabetically - this would break the multiplier correspondence
    all_lora_urls = []
    seen_urls = set()
    for phase_cfg in phases_config:
        for lora in phase_cfg.get("loras", []):
            url = lora["url"]
            # Skip empty URLs (phases without LoRAs)
            if not url or not url.strip():
                headless_logger.debug(f"Skipping empty LoRA URL in phase_config", task_id=task_id)
                continue
            if url not in seen_urls:
                all_lora_urls.append(url)
                seen_urls.add(url)

    lora_multipliers = []
    additional_loras = {}

    for lora_url in all_lora_urls:
        # Build phase-formatted multiplier string: "phase1;phase2;phase3"
        phase_mults = []

        for phase_idx, phase_cfg in enumerate(phases_config):
            # Find this LoRA in this phase
            lora_in_phase = None
            for lora in phase_cfg.get("loras", []):
                if lora["url"] == lora_url:
                    lora_in_phase = lora
                    break

            if lora_in_phase:
                multiplier_str = str(lora_in_phase["multiplier"])

                # Validate multiplier format
                if "," in multiplier_str:
                    # Ramp format: "0.5,0.7,1.0"
                    values = multiplier_str.split(",")
                    expected_count = steps_per_phase[phase_idx]
                    if len(values) != expected_count:
                        raise ValueError(
                            f"Task {task_id}: Phase {phase_idx+1} LoRA multiplier has {len(values)} values, "
                            f"but phase has {expected_count} steps"
                        )
                    # Validate each value
                    for val in values:
                        try:
                            num = float(val)
                            if num < 0 or num > 2.0:
                                raise ValueError(f"Multiplier {val} out of range [0.0-2.0]")
                        except ValueError as e:
                            raise ValueError(f"Task {task_id}: Invalid multiplier value '{val}': {e}")

                    phase_mults.append(multiplier_str)
                else:
                    # Single value
                    try:
                        num = float(multiplier_str)
                        if num < 0 or num > 2.0:
                            raise ValueError(f"Multiplier {multiplier_str} out of range [0.0-2.0]")
                    except ValueError as e:
                        raise ValueError(f"Task {task_id}: Invalid multiplier value '{multiplier_str}': {e}")

                    phase_mults.append(multiplier_str)
            else:
                # LoRA not in this phase, use "0"
                phase_mults.append("0")

        # Combine phases with semicolons
        if num_phases == 2:
            multiplier_string = f"{phase_mults[0]};{phase_mults[1]}"
        else:  # 3 phases
            multiplier_string = f"{phase_mults[0]};{phase_mults[1]};{phase_mults[2]}"

        lora_multipliers.append(multiplier_string)

        # For additional_loras, use the first phase's value for download handling
        try:
            first_val = float(phase_mults[0].split(",")[0])
            additional_loras[lora_url] = first_val
        except (ValueError, IndexError):
            additional_loras[lora_url] = 1.0

    # PREPARE patch config (will be applied later when WGP is initialized)
    # We don't apply the patch here because WGP might not be loaded yet.
    # The patch will be applied by apply_phase_config_patch() right before generation.
    headless_logger.debug(f"[PATCH_CHECK] model_name parameter = {repr(model_name)}", task_id=task_id)
    if model_name:
        try:
            import json
            from pathlib import Path
            wan_dir = Path(__file__).parent / "Wan2GP"
            defaults_dir = wan_dir / "defaults"

            # Load the original model config file to prepare the patch
            original_config_path = defaults_dir / f"{model_name}.json"
            if original_config_path.exists():
                with open(original_config_path, 'r') as f:
                    original_config = json.load(f)

                # Create modified config with phase_config LoRAs instead of built-in ones
                import copy
                temp_config = copy.deepcopy(original_config)

                # Override guidance_phases and scales
                temp_config["guidance_phases"] = num_phases

                # Set guidance scales from phase_config
                guidance_scales = [p.get("guidance_scale", 1.0) for p in phases_config]
                if len(guidance_scales) >= 1:
                    temp_config["guidance_scale"] = guidance_scales[0]
                if len(guidance_scales) >= 2:
                    temp_config["guidance2_scale"] = guidance_scales[1]
                if len(guidance_scales) >= 3:
                    temp_config["guidance3_scale"] = guidance_scales[2]

                # Set flow_shift and sample_solver from phase_config
                temp_config["flow_shift"] = phase_config.get("flow_shift", temp_config.get("flow_shift", 5.0))
                temp_config["sample_solver"] = phase_config.get("sample_solver", temp_config.get("sample_solver", "euler"))

                # Clear built-in LoRAs - phase_config LoRAs will be passed separately
                if "model" in temp_config:
                    temp_config["model"]["loras"] = []
                    temp_config["model"]["loras_multipliers"] = []

                # Store the patch config to be applied later
                result["_patch_config"] = temp_config
                result["_patch_model_name"] = model_name

                headless_logger.info(
                    f"📦 Prepared phase_config patch for '{model_name}': {num_phases} phases, "
                    f"cleared built-in LoRAs, flow_shift={temp_config['flow_shift']}, "
                    f"solver={temp_config['sample_solver']} (will apply before generation)",
                    task_id=task_id
                )
            else:
                headless_logger.warning(
                    f"Model config file not found: {original_config_path}",
                    task_id=task_id
                )
        except Exception as e:
            headless_logger.warning(
                f"Could not prepare phase_config patch: {e}",
                task_id=task_id
            )
            import traceback
            headless_logger.debug(f"Patch prep error traceback: {traceback.format_exc()}", task_id=task_id)

    result["lora_names"] = all_lora_urls
    result["lora_multipliers"] = lora_multipliers
    result["additional_loras"] = additional_loras

    dprint(f"[PHASE_CONFIG_DEBUG] Task {task_id}: parse_phase_config returning lora_multipliers={lora_multipliers}")

    headless_logger.info(
        f"phase_config parsed: {num_phases} phases, "
        f"steps={steps_per_phase}, "
        f"thresholds=[{switch_threshold}, {switch_threshold2}], "
        f"{len(all_lora_urls)} LoRAs, "
        f"lora_multipliers={lora_multipliers}",
        task_id=task_id
    )

    return result


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
    headless_logger.debug(f"Converting DB task to GenerationTask", task_id=task_id)
    
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
            "wan_2_2_t2i": "t2v_2_2",  # Wan 2.2 T2I
            "flux": "flux",
            "t2v": "t2v",
            "t2v_22": "t2v_2_2",  # Wan 2.2 T2V
            "i2v": "i2v_14B",
            "i2v_22": "i2v_2_2",  # Wan 2.2 I2V
            "hunyuan": "hunyuan",
            "ltxv": "ltxv_13B",
            "join_clips": "lightning_baseline_2_2_2",  # Join clips uses Lightning baseline for fast generation
            "inpaint_frames": "lightning_baseline_2_2_2"  # Inpaint frames uses Lightning baseline for fast generation
        }
        # Custom mapping for Qwen image style tasks
        if task_type == "qwen_image_style":
            # Use Qwen Image Edit base for style transfer via image guide
            model = "qwen_image_edit_20B"
        else:
            model = task_type_to_model.get(task_type, "t2v")  # Default to T2V
    
    # Create clean parameters dict for generation
    generation_params = {}
    
    # Core generation parameters
    param_whitelist = {
        "negative_prompt", "resolution", "video_length", "num_inference_steps", 
        "guidance_scale", "seed", "embedded_guidance_scale", "flow_shift",
        "audio_guidance_scale", "repeat_generation", "multi_images_gen_type",
        # Multi-phase guidance controls
        "guidance2_scale", "guidance3_scale", "guidance_phases", "switch_threshold", "switch_threshold2", "model_switch_phase",
        
        # VACE parameters
        "video_guide", "video_mask",
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
        "image_refs_relative_size",
        
        # Output parameters
        "output_dir", "custom_output_dir",

        # Special flags
        "apply_reward_lora",
        "override_profile",
    }
    
    # Copy whitelisted parameters with field name mapping
    for param in param_whitelist:
        if param in db_task_params:
            generation_params[param] = db_task_params[param]
    
    # Use centralized extraction function for orchestrator_details
    import sys
    source_dir = Path(__file__).parent / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from common_utils import extract_orchestrator_parameters  # type: ignore
    
    # Extract parameters from orchestrator_details using centralized function
    extracted_params = extract_orchestrator_parameters(db_task_params, task_id, dprint)

    # Update db_task_params with extracted phase_config (checked later at line ~768)
    if "phase_config" in extracted_params:
        db_task_params["phase_config"] = extracted_params["phase_config"]
        headless_logger.info(f"[PHASE_CONFIG_DEBUG] Extracted phase_config from orchestrator_details", task_id=task_id)
    else:
        headless_logger.info(f"[PHASE_CONFIG_DEBUG] No phase_config in extracted_params. Keys: {list(extracted_params.keys())}", task_id=task_id)
        if "orchestrator_details" in db_task_params:
            orch_keys = list(db_task_params["orchestrator_details"].keys())
            headless_logger.info(f"[PHASE_CONFIG_DEBUG] orchestrator_details keys: {orch_keys}", task_id=task_id)
            headless_logger.info(f"[PHASE_CONFIG_DEBUG] phase_config in orchestrator_details: {'phase_config' in db_task_params['orchestrator_details']}", task_id=task_id)

    # Update generation_params with extracted parameters
    for param in ["additional_loras", "orchestrator_payload"]:
        if param in extracted_params and param not in generation_params:
            generation_params[param] = extracted_params[param]
    
    # [DEEP_DEBUG] Log LoRA parameter transfer for debugging
    dprint(f"[WORKER_DEBUG] Worker {task_id}: CONVERTING DB TASK TO GENERATION TASK")
    dprint(f"[WORKER_DEBUG]   task_type: {task_type}")
    dprint(f"[WORKER_DEBUG]   db_task_params keys: {list(db_task_params.keys())}")
    dprint(f"[WORKER_DEBUG]   'use_lighti2x_lora' in db_task_params: {'use_lighti2x_lora' in db_task_params}")
    if 'use_lighti2x_lora' in db_task_params:
        dprint(f"[WORKER_DEBUG]   db_task_params['use_lighti2x_lora']: {db_task_params['use_lighti2x_lora']}")
    dprint(f"[WORKER_DEBUG]   generation_params keys after copy: {list(generation_params.keys())}")
    
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
        headless_logger.debug(f"Mapped 'steps' ({db_task_params['steps']}) to 'num_inference_steps'", task_id=task_id)
    
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
            # Check if this is phase-config format (contains semicolons)
            if ";" in multipliers:
                # Phase-config format: space-separated strings like "1.0;0 0;1.0"
                # Keep as strings, don't convert to floats
                generation_params["lora_multipliers"] = [x.strip() for x in multipliers.split(" ") if x.strip()]
            else:
                # Regular format: comma-separated floats like "1.0,0.8"
                generation_params["lora_multipliers"] = [float(x.strip()) for x in multipliers.split(",") if x.strip()]
        # Keep as-is if already a list

    # Specialized handling for Qwen Image Style tasks
    if task_type == "qwen_image_style":
        # Ensure model is Qwen Image Edit
        model = "qwen_image_edit_20B"

        # Build the prompt with style and subject modifications
        original_prompt = prompt
        modified_prompt = original_prompt

        # Get style and subject parameters
        style_strength = db_task_params.get("style_reference_strength", 0.0)
        subject_strength = db_task_params.get("subject_strength", 0.0)
        subject_description = db_task_params.get("subject_description", "")
        in_this_scene = db_task_params.get("in_this_scene", False)

        try:
            style_strength = float(style_strength) if style_strength is not None else 0.0
        except Exception:
            style_strength = 0.0

        try:
            subject_strength = float(subject_strength) if subject_strength is not None else 0.0
        except Exception:
            subject_strength = 0.0

        # Build prompt modifications
        prompt_parts = []
        has_style_prefix = False

        # Add style prefix if style_strength > 0
        if style_strength > 0.0:
            prompt_parts.append("In the style of this image,")
            has_style_prefix = True

        # Add subject prefix if subject_strength > 0
        if subject_strength > 0.0 and subject_description:
            # Use lowercase 'make' if style prefix is already present
            make_word = "make" if has_style_prefix else "Make"
            if in_this_scene:
                prompt_parts.append(f"{make_word} an image of this {subject_description} in this scene:")
            else:
                prompt_parts.append(f"{make_word} an image of this {subject_description}:")

        # Combine prompt parts with original prompt
        if prompt_parts:
            modified_prompt = " ".join(prompt_parts) + " " + original_prompt
            headless_logger.info(f"[QWEN_STYLE] Modified prompt from '{original_prompt}' to '{modified_prompt}'", task_id=task_id)

        # Use the modified prompt
        prompt = modified_prompt

        # Map reference image (style or subject) to image_guide
        # Use style_reference_image first, fallback to subject_reference_image
        reference_image = db_task_params.get("style_reference_image") or db_task_params.get("subject_reference_image")
        if reference_image:
            try:
                downloads_dir = Path("outputs/style_refs")
                local_ref_path = sm_download_image_if_url(reference_image, downloads_dir, task_id_for_logging=task_id, debug_mode=debug_mode, descriptive_name="reference_image")
                generation_params["image_guide"] = str(local_ref_path)
                headless_logger.info(f"[QWEN_STYLE] Using reference image: {reference_image}", task_id=task_id)
            except Exception as e:
                dprint(f"[QWEN_STYLE] Failed to process reference image: {e}")

        # Default settings tuned for Qwen style transfer
        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("sample_solver", "lightning")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", 12)

        # Resolve LoRA directory for Qwen using absolute path (CWD may be changed to Wan2GP)
        try:
            base_wan2gp_dir = Path(wan2gp_path)
        except Exception:
            base_wan2gp_dir = Path(__file__).parent / "Wan2GP"
        qwen_lora_dir = base_wan2gp_dir / "loras_qwen"
        qwen_lora_dir.mkdir(parents=True, exist_ok=True)
        headless_logger.info(f"[QWEN_STYLE] Using Qwen LoRA directory: {qwen_lora_dir}")

        # Ensure Lightning LoRA (8-step) is attached at 1.0
        # Prefer an existing local file from loras_qwen; fall back to a downloadable default name.
        lightning_repo = "lightx2v/Qwen-Image-Lightning"
        # Prefer the user-requested Edit V1.0 bf16 first
        lightning_candidates = [
            "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
            # Other common Edit/non-Edit variants as fallbacks
            "Qwen-Image-Edit-Lightning-8steps-V2.0-bf16.safetensors",
            "Qwen-Image-Edit-Lightning-8steps-V1.1-bf16.safetensors",
            "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
            "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors",
            "Qwen-Image-Lightning-8steps-V1.0-bf16.safetensors",
        ]

        selected_lightning = None
        for cand in lightning_candidates:
            if (qwen_lora_dir / cand).exists():
                selected_lightning = cand
                break

        # If none found locally, try to fetch a known-good filename from HF (best effort)
        if selected_lightning is None:
            # Default to the user-requested Edit V1.0 bf16 if nothing was found locally
            selected_lightning = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
            target = qwen_lora_dir / selected_lightning
            if not target.exists():
                try:
                    from huggingface_hub import hf_hub_download  # type: ignore
                    dprint(f"[QWEN_STYLE] Fetching {selected_lightning} from {lightning_repo} into {qwen_lora_dir}")
                    dl_path = hf_hub_download(repo_id=lightning_repo, filename=selected_lightning, revision="main", local_dir=str(qwen_lora_dir))
                    # Ensure file is located exactly at target
                    try:
                        from pathlib import Path as _P
                        _p = _P(dl_path)
                        if _p.exists() and _p.resolve() != target.resolve():
                            _p.replace(target)
                    except Exception:
                        pass
                except Exception as e:
                    dprint(f"[QWEN_STYLE] Lightning LoRA download failed: {e}")
        else:
            dprint(f"[QWEN_STYLE] Using local Lightning LoRA: {selected_lightning}")

        # Ensure InStyle LoRA present in loras_qwen and attach at provided strength
        instyle_repo = "peteromallet/Qwen-Image-Edit-InStyle"
        instyle_fname = "InStyle-0.5.safetensors"
        instyle_target = qwen_lora_dir / instyle_fname
        if not instyle_target.exists():
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
                dprint(f"[QWEN_STYLE] Fetching {instyle_fname} from {instyle_repo} into {qwen_lora_dir}")
                dl_path2 = hf_hub_download(repo_id=instyle_repo, filename=instyle_fname, revision="main", local_dir=str(qwen_lora_dir))
                try:
                    from pathlib import Path as _P
                    _p2 = _P(dl_path2)
                    if _p2.exists() and _p2.resolve() != instyle_target.resolve():
                        _p2.replace(instyle_target)
                except Exception:
                    pass
            except Exception as e:
                dprint(f"[QWEN_STYLE] InStyle LoRA download failed: {e}")
        else:
            dprint(f"[QWEN_STYLE] InStyle LoRA already present at {instyle_target}")

        # Build LoRA lists
        lora_names = generation_params.get("lora_names", []) or []
        lora_mults = generation_params.get("lora_multipliers", []) or []
        if not isinstance(lora_names, list):
            lora_names = [lora_names]
        if not isinstance(lora_mults, list):
            lora_mults = [lora_mults]

        # Attach Lightning LoRA first at 1.0 if available
        if selected_lightning and (qwen_lora_dir / selected_lightning).exists() and selected_lightning not in lora_names:
            lora_names.append(selected_lightning)
            lora_mults.append(1.0)
            headless_logger.info(f"[QWEN_STYLE] Added Lightning LoRA: {selected_lightning} with strength 1.0", task_id=task_id)

        # Add style transfer LoRA if style_strength > 0
        if style_strength > 0.0:
            # Use InStyle LoRA for style transfer
            if instyle_fname not in lora_names:
                lora_names.append(instyle_fname)
                lora_mults.append(style_strength)
                headless_logger.info(f"[QWEN_STYLE] Added style transfer LoRA: {instyle_fname} with strength {style_strength}", task_id=task_id)

        # Add subject LoRA if subject_strength > 0
        if subject_strength > 0.0:
            # Download and add subject LoRA
            subject_lora_fname = "in_subject_qwen_edit_2_000006750.safetensors"
            subject_lora_target = qwen_lora_dir / subject_lora_fname

            if not subject_lora_target.exists():
                try:
                    from huggingface_hub import hf_hub_download  # type: ignore
                    subject_repo = "peteromallet/mystery_models"
                    dprint(f"[QWEN_STYLE] Fetching {subject_lora_fname} from {subject_repo} into {qwen_lora_dir}")
                    dl_path = hf_hub_download(repo_id=subject_repo, filename=subject_lora_fname, revision="main", local_dir=str(qwen_lora_dir))
                    try:
                        from pathlib import Path as _P
                        _p = _P(dl_path)
                        if _p.exists() and _p.resolve() != subject_lora_target.resolve():
                            _p.replace(subject_lora_target)
                    except Exception:
                        pass
                except Exception as e:
                    dprint(f"[QWEN_STYLE] Subject LoRA download failed: {e}")
            else:
                dprint(f"[QWEN_STYLE] Subject LoRA already present at {subject_lora_target}")

            # Add subject LoRA to the list
            if subject_lora_fname not in lora_names:
                lora_names.append(subject_lora_fname)
                lora_mults.append(subject_strength)
                headless_logger.info(f"[QWEN_STYLE] Added subject LoRA: {subject_lora_fname} with strength {subject_strength}", task_id=task_id)

        # Add any additional LoRAs from params
        additional_loras = db_task_params.get("loras", [])
        if additional_loras:
            for lora in additional_loras:
                if isinstance(lora, dict) and "path" in lora and "scale" in lora:
                    lora_filename = Path(lora["path"]).name
                    if lora_filename not in lora_names:
                        lora_names.append(lora_filename)
                        lora_mults.append(float(lora["scale"]))
            headless_logger.info(f"[QWEN_STYLE] Added {len(additional_loras)} additional LoRAs", task_id=task_id)

        generation_params["lora_names"] = lora_names
        generation_params["lora_multipliers"] = lora_mults
        headless_logger.info(f"[QWEN_STYLE] Final LoRA configuration: {lora_names} with multipliers {lora_mults}", task_id=task_id)
    
    # NOTE: additional_loras conversion is now handled centrally in HeadlessTaskQueue
    # Both dict format ({"url": multiplier}) and list format are supported
    
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
    
    # ========================================================================
    # PHASE_CONFIG OVERRIDE - Complete phase control system
    # ========================================================================
    # If phase_config is present, it overrides ALL phase-related parameters:
    # - guidance_phases, switch_threshold, switch_threshold2
    # - guidance_scale, guidance2_scale, guidance3_scale
    # - flow_shift, sample_solver, model_switch_phase
    # - lora_names, lora_multipliers, additional_loras
    #
    # This provides complete per-phase control when needed, while preserving
    # existing behavior when phase_config is not provided.

    if "phase_config" in db_task_params:
        headless_logger.info("phase_config detected - parsing comprehensive phase override", task_id=task_id)

        try:
            # Calculate num_inference_steps from phase_config (it overrides everything)
            steps_per_phase = db_task_params["phase_config"].get("steps_per_phase", [2, 2, 2])
            phase_config_steps = sum(steps_per_phase)

            headless_logger.info(f"phase_config overriding num_inference_steps: {phase_config_steps} (from steps_per_phase {steps_per_phase})", task_id=task_id)

            # Parse phase_config and get all computed parameters
            parsed_phase_config = parse_phase_config(
                phase_config=db_task_params["phase_config"],
                num_inference_steps=phase_config_steps,
                task_id=task_id,
                model_name=model
            )

            # Override ALL phase-related parameters INCLUDING num_inference_steps
            generation_params["num_inference_steps"] = phase_config_steps

            for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                       "guidance_scale", "guidance2_scale", "guidance3_scale",
                       "flow_shift", "sample_solver", "model_switch_phase",
                       "lora_names", "lora_multipliers", "additional_loras"]:
                if key in parsed_phase_config and parsed_phase_config[key] is not None:
                    generation_params[key] = parsed_phase_config[key]

            # CRITICAL: Also set old-style WGP parameters for compatibility
            # WGP internally uses activated_loras (list) and loras_multipliers (space-separated string)
            if "lora_names" in parsed_phase_config:
                generation_params["activated_loras"] = parsed_phase_config["lora_names"]
            if "lora_multipliers" in parsed_phase_config:
                # Convert list of phase-config strings to space-separated string
                # e.g., ['1.0;0', '0;1.0'] → '1.0;0 0;1.0'
                generation_params["loras_multipliers"] = " ".join(str(m) for m in parsed_phase_config["lora_multipliers"])
                headless_logger.debug(
                    f"Set loras_multipliers string: {generation_params['loras_multipliers']}",
                    task_id=task_id
                )

            # PHASE_CONFIG IN-MEMORY PATCH:
            # Store the parsed_phase_config in task parameters so the GenerationWorker can apply it
            # The patch MUST be applied in the worker thread where wgp is actually imported
            if "_patch_config" in parsed_phase_config:
                generation_params["_parsed_phase_config"] = parsed_phase_config
                generation_params["_phase_config_model_name"] = model
                headless_logger.info(
                    f"[PHASE_CONFIG] Prepared phase_config patch for model '{model}': "
                    f"cleared built-in LoRAs, guidance_phases={parsed_phase_config['guidance_phases']} "
                    f"(will be applied in GenerationWorker)",
                    task_id=task_id
                )
            else:
                headless_logger.info(
                    f"[HOT_SWAP] Using base model '{model}' with default parameters",
                    task_id=task_id
                )

            headless_logger.info(
                f"phase_config applied: {parsed_phase_config['guidance_phases']} phases, "
                f"steps={phase_config_steps}, "
                f"guidance=[{parsed_phase_config.get('guidance_scale')}, "
                f"{parsed_phase_config.get('guidance2_scale')}, "
                f"{parsed_phase_config.get('guidance3_scale')}], "
                f"flow_shift={parsed_phase_config['flow_shift']}, "
                f"{len(parsed_phase_config['lora_names'])} LoRAs",
                task_id=task_id
            )

        except Exception as e:
            headless_logger.error(f"Failed to parse phase_config: {e}", task_id=task_id)
            raise ValueError(f"Task {task_id}: Invalid phase_config: {e}")

    # ========================================================================
    # END PHASE_CONFIG OVERRIDE
    # ========================================================================

    # Auto-enable built-in acceleration LoRAs only if no LoRAs explicitly specified
    # This doesn't interfere with parameter precedence
    # SKIP if phase_config was used (it already set LoRAs)
    if ("phase_config" not in db_task_params and
        ("2_2" in model or "cocktail_2_2" in model) and
        "lora_names" not in generation_params and
        "activated_loras" not in db_task_params):
        generation_params["lora_names"] = ["CausVid", "DetailEnhancerV1"]
        generation_params["lora_multipliers"] = [1.0, 0.2]  # DetailEnhancer at reduced strength
        headless_logger.debug(f"Auto-enabled Wan 2.2 acceleration LoRAs (no parameter overrides)", task_id=task_id)

    # Auto-add/update Lightning LoRA based on amount_of_motion parameter
    # SKIP if phase_config was used (it already controls LoRAs completely)
    amount_of_motion = db_task_params.get("amount_of_motion", None) if "phase_config" not in db_task_params else None

    if amount_of_motion is not None and ("2_2" in model or "cocktail_2_2" in model):
        try:
            amount_of_motion = float(amount_of_motion)
        except (ValueError, TypeError):
            amount_of_motion = None

    if amount_of_motion is not None and ("2_2" in model or "cocktail_2_2" in model):
        # Calculate strength: input 0.0→0.5, input 1.0→1.0 (linear scaling from 0.5 to 1.0)
        first_phase_strength = 0.5 + (amount_of_motion * 0.5)

        lightning_lora_name = "high_noise_model.safetensors"

        # Get current LoRA lists or initialize
        current_lora_names = generation_params.get("lora_names", [])
        current_lora_mults = generation_params.get("lora_multipliers", [])

        # Ensure lists
        if not isinstance(current_lora_names, list):
            current_lora_names = [current_lora_names] if current_lora_names else []
        if not isinstance(current_lora_mults, list):
            current_lora_mults = [current_lora_mults] if current_lora_mults else []

        # Make copies to avoid modifying references
        current_lora_names = list(current_lora_names)
        current_lora_mults = list(current_lora_mults)

        # Check if Lightning LoRA already exists and update/add it
        lightning_index = -1
        for i, lora_name in enumerate(current_lora_names):
            if "Lightning" in str(lora_name) or "lightning" in str(lora_name).lower():
                lightning_index = i
                break

        if lightning_index >= 0:
            # Update existing Lightning LoRA - preserve phases 2 and 3 if they exist
            current_lora_names[lightning_index] = lightning_lora_name
            if lightning_index < len(current_lora_mults):
                existing_mult = str(current_lora_mults[lightning_index])
                existing_phases = existing_mult.split(";")
                # Keep existing phase 2 and 3 values, or default to 0.0
                phase2 = existing_phases[1] if len(existing_phases) > 1 else "0.0"
                phase3 = existing_phases[2] if len(existing_phases) > 2 else "0.0"
                lightning_strength = f"{first_phase_strength:.2f};{phase2};{phase3}"
                current_lora_mults[lightning_index] = lightning_strength
            else:
                lightning_strength = f"{first_phase_strength:.2f};0.0;0.0"
                current_lora_mults.append(lightning_strength)
            headless_logger.debug(f"Updated Lightning LoRA strength to {lightning_strength} based on amount_of_motion={amount_of_motion}", task_id=task_id)
        else:
            # Add new Lightning LoRA with default phase 2/3 values
            lightning_strength = f"{first_phase_strength:.2f};0.0;0.0"
            current_lora_names.append(lightning_lora_name)
            current_lora_mults.append(lightning_strength)
            headless_logger.debug(f"Added Lightning LoRA with strength {lightning_strength} based on amount_of_motion={amount_of_motion}", task_id=task_id)

        # Ensure multipliers list matches names list length
        while len(current_lora_mults) < len(current_lora_names):
            current_lora_mults.append("1.0")

        generation_params["lora_names"] = current_lora_names
        generation_params["lora_multipliers"] = current_lora_mults

        # Add to additional_loras for download handling
        if "additional_loras" not in generation_params:
            generation_params["additional_loras"] = {}

        lightning_url = "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors"
        generation_params["additional_loras"][lightning_url] = first_phase_strength

        headless_logger.info(f"Lightning LoRA configured: strength={lightning_strength}, amount_of_motion={amount_of_motion}", task_id=task_id)
    
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
    
    headless_logger.debug(f"Created GenerationTask: model={model}, priority={priority}", task_id=task_id)
    return generation_task

# -----------------------------------------------------------------------------
# 1. Parse arguments for the server
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("WanGP Worker Server")

    pgroup_server = parser.add_argument_group("Server Settings")
    pgroup_server.add_argument("--main-output-dir", type=str, default="./outputs",
                               help="Base directory where outputs for each task will be saved (in subdirectories)")
    pgroup_server.add_argument("--poll-interval", type=int, default=10,
                               help="How often (in seconds) to check tasks.json for new tasks.")
    pgroup_server.add_argument("--debug", action="store_true",
                               help="Enable verbose debug logging (prints additional diagnostics) and preserve generated files (skips automatic cleanup)")
    pgroup_server.add_argument("--worker", type=str, default=None,
                               help="Worker name/ID - creates a log file named {worker}.log in the logs folder")
    pgroup_server.add_argument("--save-logging", type=str, nargs='?', const='logs/worker.log', default=None,
                               help="Save all logging output to a file (in addition to console output). Optionally specify path, defaults to 'logs/worker.log'")
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
    pgroup_server.add_argument("--preload-model", type=str, default="lightning_baseline_2_2_2",
                               help="Model to pre-load on worker startup for faster first task (default: lightning_baseline_2_2_2, set to empty string to disable)")
    pgroup_server.add_argument("--db-type", type=str, default="supabase",
                               help="Database type (accepted but not used, kept for compatibility)")

    # --- Supabase-related arguments (REQUIRED) ---
    pgroup_server.add_argument("--supabase-url", type=str, required=True,
                               help="Supabase project URL (required)")
    pgroup_server.add_argument("--supabase-access-token", type=str, required=True,
                               help="Supabase access token (JWT) for authentication (required)")
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
    pgroup_wgp_globals.add_argument("--wgp-preload", type=int, default=None,
                                help="VRAM budget (MB) for text_encoder/transformer preloading. Higher values keep more in VRAM. 0=minimal (default), 3000+=keep in VRAM")

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
    print(f"[SEGMENT_ROUTING_DEBUG] _handle_travel_segment_via_queue called for task_id={task_id}")
    headless_logger.debug(f"Starting travel segment queue processing", task_id=task_id)
    log_ram_usage("Segment via queue - start", task_id=task_id)
    
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
        # IMPORTANT: We use the CANONICAL model name (e.g., lightning_baseline_2_2_2) to avoid
        # model reloads, and pass all phase_config parameters explicitly to hot-swap them.
        generation_params = {
            "model_name": model_name,  # CRITICAL: Use canonical model name, NOT temp config name
            "negative_prompt": negative_prompt_for_wgp,
            "resolution": f"{parsed_res_wh[0]}x{parsed_res_wh[1]}",
            "video_length": total_frames_for_segment,
            "seed": segment_params.get("seed_to_use", 12345),
            "use_causvid_lora": full_orchestrator_payload.get("apply_causvid", False),
            "use_lighti2x_lora": full_orchestrator_payload.get("use_lighti2x_lora", False),
            "apply_reward_lora": apply_reward_lora,
        }
        
        # Add additional LoRAs from orchestrator payload if present
        additional_loras = full_orchestrator_payload.get("additional_loras", {})
        if additional_loras:
            generation_params["additional_loras"] = additional_loras
            dprint(f"[QUEUE_PARAMS] Added {len(additional_loras)} additional LoRAs from orchestrator payload")

        # Auto-add Lightning LoRA based on amount_of_motion from orchestrator
        # SKIP if phase_config is present (it will handle all LoRAs)
        amount_of_motion = full_orchestrator_payload.get("amount_of_motion", None)

        if amount_of_motion is not None and ("2_2" in model_name or "lightning" in model_name.lower()) and "phase_config" not in full_orchestrator_payload:
            try:
                amount_of_motion = float(amount_of_motion)
            except (ValueError, TypeError):
                amount_of_motion = None

            # Only proceed if amount_of_motion is still valid after conversion
            if amount_of_motion is not None:
                # Calculate strength: input 0.0→0.5, input 1.0→1.0 (linear scaling from 0.5 to 1.0)
                first_phase_strength = 0.5 + (amount_of_motion * 0.5)

                lightning_lora_name = "high_noise_model.safetensors"

                # Get or initialize LoRA lists
                current_lora_names = generation_params.get("lora_names", [])
                current_lora_mults = generation_params.get("lora_multipliers", [])

                if not isinstance(current_lora_names, list):
                    current_lora_names = [current_lora_names] if current_lora_names else []
                if not isinstance(current_lora_mults, list):
                    current_lora_mults = [current_lora_mults] if current_lora_mults else []

                current_lora_names = list(current_lora_names)
                current_lora_mults = list(current_lora_mults)

                # Check if Lightning LoRA already exists and update/add it
                lightning_index = -1
                for i, lora_name in enumerate(current_lora_names):
                    if "Lightning" in str(lora_name) or "lightning" in str(lora_name).lower():
                        lightning_index = i
                        break

                if lightning_index >= 0:
                    # Update existing Lightning LoRA - preserve phases 2 and 3 if they exist
                    current_lora_names[lightning_index] = lightning_lora_name
                    if lightning_index < len(current_lora_mults):
                        existing_mult = str(current_lora_mults[lightning_index])
                        existing_phases = existing_mult.split(";")
                        # Keep existing phase 2 and 3 values, or default to 0.0
                        phase2 = existing_phases[1] if len(existing_phases) > 1 else "0.0"
                        phase3 = existing_phases[2] if len(existing_phases) > 2 else "0.0"
                        lightning_strength = f"{first_phase_strength:.2f};{phase2};{phase3}"
                        current_lora_mults[lightning_index] = lightning_strength
                    else:
                        lightning_strength = f"{first_phase_strength:.2f};0.0;0.0"
                        current_lora_mults.append(lightning_strength)
                    dprint(f"[LIGHTNING_LORA] Travel segment {task_id}: Updated Lightning LoRA strength to {lightning_strength} based on amount_of_motion={amount_of_motion}")
                else:
                    # Add new Lightning LoRA with default phase 2/3 values
                    lightning_strength = f"{first_phase_strength:.2f};0.0;0.0"
                    current_lora_names.append(lightning_lora_name)
                    current_lora_mults.append(lightning_strength)
                    dprint(f"[LIGHTNING_LORA] Travel segment {task_id}: Added Lightning LoRA with strength {lightning_strength} based on amount_of_motion={amount_of_motion}")

                # Ensure multipliers list matches names list length
                while len(current_lora_mults) < len(current_lora_names):
                    current_lora_mults.append("1.0")

                generation_params["lora_names"] = current_lora_names
                generation_params["lora_multipliers"] = current_lora_mults

                # Add to additional_loras for download handling
                if "additional_loras" not in generation_params:
                    generation_params["additional_loras"] = {}

                lightning_url = "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors"
                generation_params["additional_loras"][lightning_url] = first_phase_strength

                headless_logger.info(f"Travel segment Lightning LoRA configured: strength={lightning_strength}, amount_of_motion={amount_of_motion}", task_id=task_id)
        
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

        # ========================================================================
        # PHASE_CONFIG OVERRIDE for Travel Segments
        # ========================================================================
        headless_logger.info(f"[DEBUG] Checking for phase_config. Keys in full_orchestrator_payload: {list(full_orchestrator_payload.keys())[:10]}", task_id=task_id)
        if "phase_config" in full_orchestrator_payload:
            headless_logger.info("phase_config detected in orchestrator payload - parsing comprehensive phase override", task_id=task_id)

            try:
                # Calculate num_inference_steps from phase_config (it overrides everything)
                steps_per_phase = full_orchestrator_payload["phase_config"].get("steps_per_phase", [2, 2, 2])
                phase_config_steps = sum(steps_per_phase)

                headless_logger.info(f"phase_config overriding num_inference_steps: {phase_config_steps} (from steps_per_phase {steps_per_phase})", task_id=task_id)

                # Parse phase_config and get all computed parameters
                parsed_phase_config = parse_phase_config(
                    phase_config=full_orchestrator_payload["phase_config"],
                    num_inference_steps=phase_config_steps,
                    task_id=task_id,
                    model_name=generation_params.get("model_name")
                )

                # Override ALL phase-related parameters INCLUDING num_inference_steps
                generation_params["num_inference_steps"] = phase_config_steps

                for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                           "guidance_scale", "guidance2_scale", "guidance3_scale",
                           "flow_shift", "sample_solver", "model_switch_phase",
                           "lora_names", "lora_multipliers", "additional_loras"]:
                    if key in parsed_phase_config and parsed_phase_config[key] is not None:
                        generation_params[key] = parsed_phase_config[key]

                # CRITICAL: Also set old-style WGP parameters for compatibility
                # WGP internally uses activated_loras (list) and loras_multipliers (space-separated string)
                if "lora_names" in parsed_phase_config:
                    generation_params["activated_loras"] = parsed_phase_config["lora_names"]
                if "lora_multipliers" in parsed_phase_config:
                    # Convert list of phase-config strings to space-separated string
                    # e.g., ['1.0;0', '0;1.0'] → '1.0;0 0;1.0'
                    generation_params["loras_multipliers"] = " ".join(str(m) for m in parsed_phase_config["lora_multipliers"])
                    headless_logger.debug(
                        f"Set loras_multipliers string: {generation_params['loras_multipliers']}",
                        task_id=task_id
                    )

                # HOT-SWAP OPTIMIZATION:
                # Instead of creating a unique temp model and reloading, we:
                # 1. Keep using the base model (e.g., lightning_baseline_2_2_2)
                # 2. Patch wgp.models_def in memory (gets wiped on first orchestrator init, but persists after)
                # 3. Pass phase_config patch through to GenerationWorker for restoration after each task

                # Apply patch here in main thread (will persist after first orchestrator init)
                if "_patch_config" in parsed_phase_config:
                    apply_phase_config_patch(parsed_phase_config, model_name, task_id)
                    headless_logger.info(
                        f"[PHASE_CONFIG] Patched wgp.models_def for model '{model_name}': "
                        f"cleared built-in LoRAs, guidance_phases={parsed_phase_config['guidance_phases']} "
                        f"(will also be re-applied in GenerationWorker after orchestrator init if needed)",
                        task_id=task_id
                    )
                else:
                    headless_logger.info(
                        f"[HOT_SWAP] Using base model '{model_name}' with default parameters",
                        task_id=task_id
                    )

                dprint(f"[PHASE_CONFIG_DEBUG] Task {task_id}: After applying parse_phase_config, generation_params['lora_multipliers']={generation_params.get('lora_multipliers')}")

                headless_logger.info(
                    f"phase_config applied: {parsed_phase_config['guidance_phases']} phases, "
                    f"steps={phase_config_steps}, "
                    f"guidance=[{parsed_phase_config.get('guidance_scale')}, "
                    f"{parsed_phase_config.get('guidance2_scale')}, "
                    f"{parsed_phase_config.get('guidance3_scale')}], "
                    f"flow_shift={parsed_phase_config['flow_shift']}, "
                    f"{len(parsed_phase_config['lora_names'])} LoRAs, "
                    f"lora_multipliers={generation_params.get('lora_multipliers')}",
                    task_id=task_id
                )

            except Exception as e:
                headless_logger.error(f"Failed to parse phase_config: {e}", task_id=task_id)
                raise ValueError(f"Task {task_id}: Invalid phase_config: {e}")

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
        log_ram_usage("Before queue submission", task_id=task_id)
        submitted_task_id = task_queue.submit_task(generation_task)
        headless_logger.debug(f"Travel segment submitted to queue as {submitted_task_id}", task_id=task_id)
        
        # Wait for completion with timeout
        max_wait_time = 1800  # 30 minutes for travel segments
        wait_interval = 2
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            status = task_queue.get_task_status(f"travel_seg_{task_id}")
            if status is None:
                return False, f"Travel segment {task_id}: Task status became None"
            
            if status.status == "completed":
                headless_logger.success(f"Travel segment generation completed: {status.result_path}", task_id=task_id)
                
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
                    log_ram_usage("Segment via queue - end (success with chain)", task_id=task_id)
                    return True, final_chained_path
                else:
                    log_ram_usage("Segment via queue - end (success, chain failed)", task_id=task_id)
                    return True, status.result_path  # Use raw output if chaining failed

            elif status.status == "failed":
                log_ram_usage("Segment via queue - end (generation failed)", task_id=task_id)
                return False, f"Travel segment {task_id}: Generation failed: {status.error_message}"
            
            # Still processing
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        # Timeout
        log_ram_usage("Segment via queue - end (timeout)", task_id=task_id)
        return False, f"Travel segment {task_id}: Generation timeout after {max_wait_time}s"

    except Exception as e:
        headless_logger.error(f"Travel segment exception: {e}", task_id=task_id)
        traceback.print_exc()
        log_ram_usage("Segment via queue - end (exception)", task_id=task_id)
        return False, f"Travel segment {task_id}: Exception: {str(e)}"

# -----------------------------------------------------------------------------
# Process a single task using queue-based architecture
# -----------------------------------------------------------------------------

def process_single_task(task_params_dict, main_output_dir_base: Path, task_type: str, project_id_for_task: str | None, image_download_dir: Path | str | None = None, apply_reward_lora: bool = False, colour_match_videos: bool = False, mask_active_frames: bool = True, task_queue: HeadlessTaskQueue = None):
    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))

    print(f"[PROCESS_TASK_DEBUG] process_single_task called: task_type='{task_type}', task_id={task_id}")
    headless_logger.debug(f"Entering process_single_task", task_id=task_id)
    headless_logger.debug(f"Task Type: {task_type}", task_id=task_id)
    headless_logger.debug(f"Project ID: {project_id_for_task}", task_id=task_id)
    # Safe logging: Use safe_json_repr to prevent hangs on large nested structures
    headless_logger.debug(f"Task Params: {safe_json_repr(task_params_dict)}", task_id=task_id)
    
    headless_logger.essential(f"Processing {task_type} task", task_id=task_id)
    output_location_to_db = None # Will store the final path/URL for the DB
    generation_success = False

    # --- NEW: Direct Queue Integration for Simple Generation Tasks ---
    # Route simple generation tasks directly to HeadlessTaskQueue to eliminate blocking waits
    direct_queue_task_types = {
        "wan_2_2_t2i", "vace", "vace_21", "vace_22", "flux", "t2v", "t2v_22", 
        "i2v", "i2v_22", "hunyuan", "ltxv", "generate_video"
    }
    
    if task_type in direct_queue_task_types and task_queue is not None:
        headless_logger.debug(f"Using direct queue integration for task type: {task_type}", task_id=task_id)
        
        try:
            # Create GenerationTask object from DB parameters
            generation_task = db_task_to_generation_task(task_params_dict, task_id, task_type)
            
            # For wan_2_2_t2i tasks, ensure video_length=1 for PNG conversion
            if task_type == "wan_2_2_t2i":
                generation_task.parameters["video_length"] = 1
                headless_logger.debug(f"Overriding video_length=1 for wan_2_2_t2i task", task_id=task_id)
            
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
                    headless_logger.error("Task status became None, assuming failure", task_id=task_id)
                    generation_success = False
                    output_location_to_db = "Error: Task status became None during processing"
                    break
                    
                if status.status == "completed":
                    generation_success = True
                    output_location_to_db = status.result_path
                    processing_time = status.processing_time or 0
                    headless_logger.success(f"Queue processing completed in {processing_time:.1f}s", task_id=task_id)
                    headless_logger.essential(f"Output: {output_location_to_db}", task_id=task_id)
                    break
                elif status.status == "failed":
                    generation_success = False
                    output_location_to_db = status.error_message or "Generation failed without specific error message"
                    headless_logger.error(f"Queue processing failed: {output_location_to_db}", task_id=task_id)
                    break
                else:
                    # Still processing
                    headless_logger.debug(f"Queue status: {status.status}, waiting...", task_id=task_id)
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
            else:
                # Timeout reached
                headless_logger.error(f"Queue processing timeout after {max_wait_time}s", task_id=task_id)
                generation_success = False
                output_location_to_db = f"Error: Processing timeout after {max_wait_time} seconds"
            
            # Return early for direct queue tasks
            headless_logger.essential(f"Finished task (Success: {generation_success})", task_id=task_id)
            return generation_success, output_location_to_db
            
        except Exception as e_queue:
            headless_logger.error(f"Queue processing error: {e_queue}", task_id=task_id)
            traceback.print_exc()
            generation_success = False
            output_location_to_db = f"Error: Queue processing failed - {str(e_queue)}"
            headless_logger.essential(f"Finished task (Success: {generation_success})", task_id=task_id)
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
        # Create task-aware dprint wrapper
        task_dprint = make_task_dprint(task_id)
        return tbi._handle_travel_orchestrator_task(task_params_from_db=task_params_dict, main_output_dir_base=main_output_dir_base, orchestrator_task_id_str=task_id, orchestrator_project_id=project_id_for_task, dprint=task_dprint)
    elif task_type == "travel_segment":
        print(f"[ROUTING_DEBUG] Matched travel_segment for task_id={task_id}")
        headless_logger.debug("Using direct queue integration for travel segment", task_id=task_id)
        # NEW: Route travel segments directly to queue to eliminate blocking wait
        # Create task-aware dprint wrapper
        task_dprint = make_task_dprint(task_id)
        return _handle_travel_segment_via_queue(task_params_dict, main_output_dir_base, task_id, apply_reward_lora, colour_match_videos, mask_active_frames, task_queue=task_queue, dprint=task_dprint)
    elif task_type == "travel_stitch":
        headless_logger.debug("Delegating to travel stitch handler", task_id=task_id)
        # Create task-aware dprint wrapper
        task_dprint = make_task_dprint(task_id)
        return tbi._handle_travel_stitch_task(task_params_from_db=task_params_dict, main_output_dir_base=main_output_dir_base, stitch_task_id_str=task_id, dprint=task_dprint)
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
    elif task_type == "join_clips":
        headless_logger.debug("Delegating to join clips handler", task_id=task_id)
        return _handle_join_clips_task(
            task_params_from_db=task_params_dict,
            main_output_dir_base=main_output_dir_base,
            task_id=task_id,
            task_queue=task_queue,
            dprint=dprint
        )
    elif task_type == "inpaint_frames":
        headless_logger.debug("Delegating to inpaint frames handler", task_id=task_id)
        return _handle_inpaint_frames_task(
            task_params_from_db=task_params_dict,
            main_output_dir_base=main_output_dir_base,
            task_id=task_id,
            task_queue=task_queue,
            dprint=dprint
        )
    elif task_type == "create_visualization":
        headless_logger.debug("Delegating to create visualization handler", task_id=task_id)
        return _handle_create_visualization_task(
            task_params_from_db=task_params_dict,
            main_output_dir_base=main_output_dir_base,
            viz_task_id_str=task_id,
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
                    headless_logger.error("Task status became None, assuming failure", task_id=task_id)
                    generation_success = False
                    output_location_to_db = "Error: Task status became None during processing"
                    break
                    
                if status.status == "completed":
                    generation_success = True
                    output_location_to_db = status.result_path
                    processing_time = status.processing_time or 0
                    headless_logger.success(f"Queue processing completed in {processing_time:.1f}s", task_id=task_id)
                    headless_logger.essential(f"Output: {output_location_to_db}", task_id=task_id)
                    break
                elif status.status == "failed":
                    generation_success = False
                    output_location_to_db = status.error_message or "Generation failed without specific error message"
                    headless_logger.error(f"Queue processing failed: {output_location_to_db}", task_id=task_id)
                    break
                else:
                    # Still processing
                    headless_logger.debug(f"Queue status: {status.status}, waiting...", task_id=task_id)
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
            else:
                # Timeout reached
                headless_logger.error(f"Queue processing timeout after {max_wait_time}s", task_id=task_id)
                generation_success = False
                output_location_to_db = f"Error: Processing timeout after {max_wait_time} seconds"
            
        except Exception as e_queue:
            headless_logger.error(f"Queue processing error: {e_queue}", task_id=task_id)
            traceback.print_exc()
            generation_success = False
            output_location_to_db = f"Error: Queue processing failed - {str(e_queue)}"

    # --- Chaining Logic ---
    # This block is now executed for any successful primitive task that doesn't return early.
    if generation_success:
        chaining_result_path_override = None

        if task_params_dict.get("travel_chain_details"):
            headless_logger.debug(f"Task is part of a travel sequence. Attempting to chain.", task_id=task_id)
            chain_success, chain_message, final_path_from_chaining = tbi._handle_travel_chaining_after_wgp(
                wgp_task_params=task_params_dict, 
                actual_wgp_output_video_path=output_location_to_db,
                image_download_dir=image_download_dir,
                dprint=dprint
            )
            if chain_success:
                chaining_result_path_override = final_path_from_chaining
                headless_logger.debug(f"Travel chaining successful. Message: {chain_message}", task_id=task_id)
            else:
                headless_logger.error(f"Travel sequence chaining failed after WGP completion: {chain_message}. The raw WGP output '{output_location_to_db}' will be used for this task's DB record.", task_id=task_id)
        
        elif task_params_dict.get("different_perspective_chain_details"):
            # SM_RESTRUCTURE_FIX: Prevent double-chaining. This is now handled in the 'generate_openpose' block.
            # The only other task type that can have these details is 'wgp', which is the intended target for this block.
            if task_type != 'generate_openpose':
                headless_logger.debug(f"Task is part of a different_perspective sequence. Attempting to chain.", task_id=task_id)
                
                chain_success, chain_message, final_path_from_chaining = dp._handle_different_perspective_chaining(
                    completed_task_params=task_params_dict, 
                    task_output_path=output_location_to_db,
                    dprint=dprint
                )
                if chain_success:
                    chaining_result_path_override = final_path_from_chaining
                    headless_logger.debug(f"Different Perspective chaining successful. Message: {chain_message}", task_id=task_id)
                else:
                    headless_logger.error(f"Different Perspective sequence chaining failed: {chain_message}. This may halt the sequence.", task_id=task_id)


        if chaining_result_path_override:
            path_to_check_existence = Path(chaining_result_path_override).resolve()
            headless_logger.debug(f"Chaining returned path '{chaining_result_path_override}'. Resolved to '{path_to_check_existence}' for existence check.", task_id=task_id)

            if path_to_check_existence.exists() and path_to_check_existence.is_file():
                is_output_path_different = str(chaining_result_path_override) != str(output_location_to_db)
                if is_output_path_different:
                    headless_logger.debug(f"Chaining modified output path for DB. Original: {output_location_to_db}, New: {chaining_result_path_override} (Checked file: {path_to_check_existence})", task_id=task_id)
                output_location_to_db = chaining_result_path_override
            else:
                headless_logger.warning(f"Chaining reported success, but final path '{chaining_result_path_override}' (checked as '{path_to_check_existence}') is invalid or not a file. Using original WGP output '{output_location_to_db}' for DB.", task_id=task_id)


    # Ensure orchestrator tasks use their DB row ID as task_id so that
    # downstream sub-tasks reference the right row when updating status.
    if task_type in {"travel_orchestrator", "different_perspective_orchestrator"}:
        # Overwrite/insert the canonical task_id inside params to the DB row's ID
        task_params_dict["task_id"] = task_id

    headless_logger.essential(f"Finished task (Success: {generation_success})", task_id=task_id)
    return generation_success, output_location_to_db


# -----------------------------------------------------------------------------
# Heartbeat System
# -----------------------------------------------------------------------------

# Global log buffer for centralized logging
_global_log_buffer: LogBuffer = None
_current_task_id: str = None

def get_gpu_memory_usage():
    """
    Get GPU memory usage in MB.
    
    Returns:
        Tuple of (total_mb, used_mb) or (None, None) if unavailable
    """
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            return int(total), int(allocated)
    except Exception:
        pass
    
    return None, None


def start_heartbeat_guardian_process(worker_id: str, supabase_url: str, supabase_key: str):
    """
    Start bulletproof heartbeat guardian as a separate process.

    The guardian process:
    - Runs completely independently with its own GIL
    - Cannot be blocked by worker downloads, model loading, or I/O
    - Receives logs from worker via multiprocessing.Queue
    - Sends heartbeats using curl (no Python HTTP library conflicts)
    - Only stops if machine crashes or worker process terminates

    Args:
        worker_id: Worker's unique ID
        supabase_url: Supabase project URL
        supabase_key: Supabase API key

    Returns:
        tuple: (guardian_process, log_queue) - Process handle and queue for sending logs
    """
    from heartbeat_guardian import guardian_main

    # Create shared queue for logs (sized to handle bursts)
    log_queue = Queue(maxsize=1000)

    # Prepare config for guardian
    config = {
        'worker_id': worker_id,
        'worker_pid': os.getpid(),
        'db_url': supabase_url,
        'api_key': supabase_key
    }

    # Start guardian as daemon process
    guardian = Process(
        target=guardian_main,
        args=(worker_id, os.getpid(), log_queue, config),
        name=f'guardian-{worker_id}',
        daemon=True  # Dies when parent dies
    )
    guardian.start()

    headless_logger.essential(f"✅ Heartbeat guardian started: PID {guardian.pid} monitoring worker PID {os.getpid()}")
    headless_logger.essential(f"   Guardian uses separate process - immune to GIL, downloads, and blocking I/O")

    return guardian, log_queue


# -----------------------------------------------------------------------------
# Main server loop
# -----------------------------------------------------------------------------

def main():
    load_dotenv() # Load .env file variables into environment
    global SUPABASE_CLIENT, SUPABASE_VIDEO_BUCKET

    # Parse CLI arguments first to determine debug mode
    cli_args = parse_args()
    
    # Set worker ID environment variable for fatal error tracking
    if cli_args.worker:
        os.environ["WORKER_ID"] = cli_args.worker
        os.environ["WAN2GP_WORKER_MODE"] = "true"

    # Global heartbeat control
    global heartbeat_thread, heartbeat_stop_event
    heartbeat_thread = None
    heartbeat_stop_event = threading.Event()
    
    # Set up logging early based on debug flag
    global debug_mode
    debug_mode = cli_args.debug
    if debug_mode:
        enable_debug_mode()
        headless_logger.debug("Debug mode enabled")
    else:
        disable_debug_mode()

    # Install global exception hook to capture uncaught exceptions
    # This must be done AFTER logging is initialized but BEFORE guardian starts
    def exception_hook(exc_type, exc_value, exc_traceback):
        """Capture uncaught exceptions and log them through the logging system."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't capture keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        headless_logger.critical(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
        headless_logger.critical(f"Traceback:\n{error_msg}")

        # Also print to stderr for startup script logs
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = exception_hook

    # Read Supabase configuration from environment variables
    env_pg_table_name = os.getenv("POSTGRES_TABLE_NAME", "tasks")
    env_supabase_url = os.getenv("SUPABASE_URL")
    env_supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")  # Support both names
    env_supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
    env_supabase_bucket = os.getenv("SUPABASE_VIDEO_BUCKET", "image_uploads")

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

    # --- Configure Supabase Connection ---
    try:
        client_key = env_supabase_key or cli_args.supabase_anon_key or env_supabase_anon_key

        if not client_key:
            raise ValueError("Need either service key or anon key for Supabase client initialization.")

        headless_logger.debug(f"Initializing Supabase client for {cli_args.supabase_url}")
        temp_supabase_client = create_client(cli_args.supabase_url, client_key)

        headless_logger.debug("Supabase client initialized. Access token will be used in edge function calls.")

        # --- Assign to db_ops globals ---
        db_ops.DB_TYPE = "supabase"
        db_ops.PG_TABLE_NAME = env_pg_table_name
        db_ops.SUPABASE_URL = cli_args.supabase_url
        db_ops.SUPABASE_SERVICE_KEY = env_supabase_key
        db_ops.SUPABASE_VIDEO_BUCKET = env_supabase_bucket
        db_ops.SUPABASE_CLIENT = temp_supabase_client
        db_ops.SUPABASE_ACCESS_TOKEN = cli_args.supabase_access_token

        # Local globals for convenience
        SUPABASE_CLIENT = temp_supabase_client
        SUPABASE_VIDEO_BUCKET = env_supabase_bucket

        headless_logger.success("Supabase client initialized successfully")

    except Exception as e:
        headless_logger.error(f"Failed to initialize Supabase client: {e}")
        headless_logger.debug(traceback.format_exc())
        headless_logger.critical("Cannot continue without Supabase connection. Exiting.")
        sys.exit(1)
    # --- End Supabase Configuration ---

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


    main_output_dir = Path(cli_args.main_output_dir).resolve()  # Resolve to absolute path BEFORE chdir
    main_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize Centralized Logging (Worker → Orchestrator) ---
    global _global_log_buffer
    guardian_process = None
    log_queue = None

    if cli_args.worker:
        # Start heartbeat guardian FIRST (before creating log buffer)
        # This ensures guardian is running before any heavy operations
        print(f"🔒 Starting bulletproof heartbeat guardian for worker: {cli_args.worker}")

        # CRITICAL: Guardian MUST use service role key to bypass RLS policies
        # Service role key bypasses Row-Level Security on workers table
        guardian_key = env_supabase_key  # This is the service role key from environment

        if not guardian_key:
            print(f"⚠️ WARNING: No service role key found in environment!")
            print(f"⚠️ Guardian will use client_key but may fail due to RLS policies")
            guardian_key = client_key

        guardian_process, log_queue = start_heartbeat_guardian_process(
            worker_id=cli_args.worker,
            supabase_url=cli_args.supabase_url,
            supabase_key=guardian_key  # Use service role key to bypass RLS
        )

        # Quick check if guardian started (should be immediate)
        if not guardian_process.is_alive():
            print(f"❌ ERROR: Guardian process failed to start!")
            print(f"   Check /tmp/guardian_crash_{cli_args.worker}.log for details")
            sys.exit(1)

        # Create log buffer with shared queue to guardian
        _global_log_buffer = LogBuffer(max_size=100, shared_queue=log_queue)

        # Create interceptor to capture custom logging function calls
        log_interceptor = CustomLogInterceptor(_global_log_buffer)
        set_log_interceptor(log_interceptor)

        headless_logger.essential(f"✅ Centralized logging enabled for worker: {cli_args.worker}")
        headless_logger.debug(f"Log buffer initialized (max_size=100, logs sent to guardian process)")
        headless_logger.debug(f"Guardian process heartbeats every 20s - immune to GIL/download blocking")
    else:
        headless_logger.debug("No worker ID provided - centralized logging disabled")
    # --- End Centralized Logging Initialization ---

    print(f"WanGP Headless Server Started.")
    if cli_args.worker:
        print(f"Worker ID: {cli_args.worker}")
    print(f"Monitoring Supabase (PostgreSQL backend) table: {db_ops.PG_TABLE_NAME}")
    print(f"Outputs will be saved under: {main_output_dir}")
    print(f"Polling interval: {cli_args.poll_interval} seconds.")

    # Initialize database
    # Supabase table/schema assumed to exist; skip initialization RPC
    dprint("Supabase: Skipping init_db – table assumed present.")

    # Activate global debug switch early so that all subsequent code paths can use dprint()
    debug_mode = cli_args.debug
    db_ops.debug_mode = cli_args.debug # Also set it in the db_ops module
    dprint("Verbose debug logging enabled.")

    # --- Apply WGP Global Config Overrides (before task queue initialization) ---
    # Import wgp.py and apply CLI overrides to its global config variables
    # ALWAYS use the local Wan2GP under this directory
    wan_dir = str((Path(__file__).parent / "Wan2GP").resolve())
    original_cwd = os.getcwd()
    original_argv = sys.argv[:]  # Save original argv
    try:
        os.chdir(wan_dir)
        sys.path.insert(0, wan_dir)

        # Protect sys.argv from wgp.py's argparse
        sys.argv = ["worker.py"]
        import wgp as wgp_mod
        sys.argv = original_argv  # Restore immediately after import

        # Apply wgp.py global config overrides from CLI arguments
        if cli_args.wgp_attention_mode is not None: wgp_mod.attention_mode = cli_args.wgp_attention_mode
        if cli_args.wgp_compile is not None: wgp_mod.compile = cli_args.wgp_compile
        if cli_args.wgp_profile is not None:
            wgp_mod.force_profile_no = cli_args.wgp_profile
            wgp_mod.default_profile = cli_args.wgp_profile
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
            if "preload_model_policy" not in wgp_mod.server_config or not isinstance(wgp_mod.server_config.get("preload_model_policy"), list):
                wgp_mod.server_config["preload_model_policy"] = []

        if cli_args.wgp_preload is not None:
            wgp_mod.server_config["preload_in_VRAM"] = cli_args.wgp_preload
            headless_logger.essential(f"Set text_encoder/transformer VRAM preload budget to {cli_args.wgp_preload}MB")

        # Ensure transformer_types is always a list to prevent character iteration
        if "transformer_types" not in wgp_mod.server_config or not isinstance(wgp_mod.server_config.get("transformer_types"), list):
            wgp_mod.server_config["transformer_types"] = []

        headless_logger.essential("WGP global config overrides applied successfully")
    except Exception as e_wgp_import:
        headless_logger.error(f"Failed to import wgp module: {e_wgp_import}")
        headless_logger.error(f"Error type: {type(e_wgp_import).__name__}")
        headless_logger.error(traceback.format_exc())
        headless_logger.critical("Cannot continue without wgp module. Exiting.")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)
    # --- End WGP Config Overrides ---

    # --- Initialize Task Queue System (required) ---
    headless_logger.essential("Initializing queue-based task processing system...")

    try:
        task_queue = HeadlessTaskQueue(wan_dir=wan_dir, max_workers=cli_args.queue_workers)

        # Pre-load model for faster first task (if specified)
        preload_model = cli_args.preload_model if cli_args.preload_model else None
        task_queue.start(preload_model=preload_model)

        headless_logger.success(f"Task queue initialized with {cli_args.queue_workers} workers")
        if preload_model:
            headless_logger.essential(f"Queue will pre-load {preload_model} model for faster first task")
        headless_logger.essential("Queue system will handle generation tasks efficiently with model reuse")
    except Exception as e_queue_init:
        headless_logger.error(f"Failed to initialize task queue: {e_queue_init}")
        traceback.print_exc()
        headless_logger.error("Queue initialization failed - cannot continue without task queue")
        sys.exit(1)
    # --- End Task Queue Initialization ---

    try:
        while True:
            task_info = None
            current_task_id_for_status_update = None # Used to hold the task_id for status updates
            current_project_id = None # To hold the project_id for the current task

            dprint(f"Checking for queued tasks in Supabase (PostgreSQL backend) table {db_ops.PG_TABLE_NAME} via Supabase RPC...")
            task_info = db_ops.get_oldest_queued_task_supabase(worker_id=cli_args.worker)
            dprint(f"Supabase task_info: {task_info}")
            if task_info:
                current_task_id_for_status_update = task_info["task_id"]
                # Status is already set to IN_PROGRESS by claim-next-task Edge Function
                
                # Set current task ID for centralized logging
                global _current_task_id
                _current_task_id = current_task_id_for_status_update

            if not task_info:
                dprint("No queued tasks found. Sleeping...")
                time.sleep(cli_args.poll_interval)
                continue

            # current_task_data = task_info["params"] # Params are already a dict
            current_task_params = task_info["params"]
            current_task_type = task_info["task_type"] # Retrieve task_type
            current_project_id = task_info.get("project_id") # Get project_id, might be None if not returned
            
            # This fallback logic remains, but it's less likely to be needed
            # if get_oldest_queued_task_supabase is reliable.
            if current_project_id is None and current_task_id_for_status_update:
                headless_logger.debug(f"Project ID not directly available. Attempting to fetch manually...", task_id=current_task_id_for_status_update)
                try:
                    # Using 'id' as the column name for task_id based on Supabase schema conventions
                    response = db_ops.SUPABASE_CLIENT.table(db_ops.PG_TABLE_NAME)\
                        .select("project_id")\
                        .eq("id", current_task_id_for_status_update)\
                        .single()\
                        .execute()
                    if response.data and response.data.get("project_id"):
                        current_project_id = response.data["project_id"]
                        headless_logger.debug(f"Successfully fetched project_id '{current_project_id}' from Supabase", task_id=current_task_id_for_status_update)
                    else:
                        headless_logger.debug(f"Could not fetch project_id from Supabase. Response data: {response.data}, error: {response.error}", task_id=current_task_id_for_status_update)
                except Exception as e_fetch_proj_id:
                    headless_logger.debug(f"Exception while fetching project_id from Supabase: {e_fetch_proj_id}", task_id=current_task_id_for_status_update)

            
            # Critical check: project_id is NOT NULL for sub-tasks created by orchestrator
            if current_project_id is None and current_task_type == "travel_orchestrator":
                headless_logger.error(f"Orchestrator task has no project_id. Sub-tasks cannot be created. Skipping task.", task_id=current_task_id_for_status_update)
                # Update status to FAILED to prevent re-processing this broken state
                error_message_for_db = "Failed: Orchestrator task missing project_id, cannot create sub-tasks."
                db_ops.update_task_status_supabase(current_task_id_for_status_update, db_ops.STATUS_FAILED, error_message_for_db)
                time.sleep(1) # Brief pause
                continue # Skip to next polling cycle

            headless_logger.essential(f"Found task of type: {current_task_type}, Project ID: {current_project_id}", task_id=current_task_id_for_status_update)
            # Status already set to IN_PROGRESS if task_info is not None

            # Inserted: define segment_image_download_dir from task params if available
            segment_image_download_dir = current_task_params.get("segment_image_download_dir")
            
            # Ensure orchestrator tasks and travel segments propagate the DB row ID as their canonical task_id *before* processing
            if current_task_type in {"travel_orchestrator", "different_perspective_orchestrator", "travel_segment"}:
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
                # Reset fatal error counter on successful task completion
                reset_fatal_error_counter()

                # Orchestrator tasks stay "In Progress" until their children report back.
                orchestrator_types_waiting = {"travel_orchestrator", "different_perspective_orchestrator"}

                if current_task_type in orchestrator_types_waiting:
                    # Check if orchestrator is signaling that all children are complete
                    if output_location and output_location.startswith("[ORCHESTRATOR_COMPLETE]"):
                        # All children complete! Mark orchestrator as COMPLETE
                        actual_output = output_location.replace("[ORCHESTRATOR_COMPLETE]", "")
                        db_ops.update_task_status_supabase(
                            current_task_id_for_status_update,
                            db_ops.STATUS_COMPLETE,
                            actual_output,
                        )
                        headless_logger.success(
                            f"Orchestrator completed: All child tasks finished. Output: {actual_output}",
                            task_id=current_task_id_for_status_update
                        )
                    else:
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
                    db_ops.update_task_status_supabase(
                        current_task_id_for_status_update,
                        db_ops.STATUS_COMPLETE,
                        output_location,
                    )
                    headless_logger.success(
                        f"Task completed successfully: {output_location}",
                        task_id=current_task_id_for_status_update
                    )

                    # Clean up generated files unless debug mode is enabled
                    cleanup_generated_files(output_location, current_task_id_for_status_update)
            else:
                db_ops.update_task_status_supabase(current_task_id_for_status_update, db_ops.STATUS_FAILED, output_location)
                headless_logger.error(
                    f"Task failed. Output: {output_location if output_location else 'N/A'}",
                    task_id=current_task_id_for_status_update
                )
            
            # Clear current task ID for centralized logging
            _current_task_id = None
            
            time.sleep(1) # Brief pause before checking for the next task

    except FatalWorkerError as fatal_error:
        # Fatal error detected - worker must terminate
        headless_logger.critical(
            f"🚨 FATAL ERROR - WORKER TERMINATING 🚨\n"
            f"Category: {fatal_error.error_category}\n"
            f"Error: {fatal_error}\n"
            f"Worker ID: {cli_args.worker if cli_args.worker else 'unknown'}\n"
            f"The worker has encountered a fatal error and will shut down."
        )
        
        # Note: Worker and task status already updated by fatal_error_handler
        # For Supabase workers: marked as 'error', task reset to 'Queued'
        # Orchestrator will detect error status and terminate the RunPod pod
        
        if cli_args.worker:
            headless_logger.info(
                f"Worker {cli_args.worker} marked for termination. "
                f"Orchestrator will terminate the pod within grace period."
            )
        
        # Exit with error code to signal fatal error
        # For RunPod: Orchestrator detects error status and terminates pod
        # For local: Process exits and can be restarted by supervisor
        headless_logger.critical("Exiting process due to fatal error...")
        sys.exit(1)
    
    except KeyboardInterrupt:
        headless_logger.essential("Server shutting down gracefully...")

        # Reset current task back to Queued if it was in progress
        if current_task_id_for_status_update:
            try:
                headless_logger.essential(f"Resetting task {current_task_id_for_status_update} back to Queued...")
                db_ops.update_task_status_supabase(
                    current_task_id_for_status_update,
                    db_ops.STATUS_QUEUED,
                    "Reset to Queued due to worker shutdown"
                )
                headless_logger.success(f"Task {current_task_id_for_status_update} reset to Queued")
            except Exception as e_reset:
                headless_logger.error(f"Failed to reset task status: {e_reset}")
    finally:
        # Stop heartbeat thread
        if cli_args.worker:
            print("Stopping heartbeat thread...")
            heartbeat_stop_event.set()
            if heartbeat_thread and heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=5.0)
                if heartbeat_thread.is_alive():
                    print("⚠️  Heartbeat thread did not stop cleanly")
                else:
                    print("✅ Heartbeat thread stopped")

        # Shutdown task queue first (if it was initialized)
        if task_queue is not None:
            try:
                headless_logger.essential("Shutting down task queue...")
                task_queue.stop(timeout=30.0)
                headless_logger.success("Task queue shutdown complete")
            except Exception as e_queue_shutdown:
                headless_logger.error(f"Error during task queue shutdown: {e_queue_shutdown}")
        
        headless_logger.essential("Server stopped")


if __name__ == "__main__":
    main() 
