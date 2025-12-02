"""
Task Registry Module

This module defines the TaskRegistry class and the TASK_HANDLERS dictionary.
It allows for a cleaner way to route tasks to their appropriate handlers
instead of using a massive if/elif block.
"""
from typing import Dict, Any, Callable, Optional, Tuple
from pathlib import Path
import time
import traceback
import json

from source.logging_utils import headless_logger
from source.worker_utils import make_task_dprint, log_ram_usage, cleanup_generated_files, dprint
from source.task_conversion import db_task_to_generation_task, parse_phase_config
from source.phase_config import apply_phase_config_patch

# Import task handlers
# These imports should be available from the environment where this module is used
from source.specialized_handlers import (
    handle_rife_interpolate_task,
    handle_extract_frame_task
)
from source.sm_functions import travel_between_images as tbi
from source.sm_functions import magic_edit as me
from source.sm_functions.join_clips import _handle_join_clips_task
from source.sm_functions.join_clips_orchestrator import _handle_join_clips_orchestrator_task
from source.sm_functions.inpaint_frames import _handle_inpaint_frames_task
from source.sm_functions.create_visualization import _handle_create_visualization_task
from source.travel_segment_processor import TravelSegmentProcessor, TravelSegmentContext
from source.common_utils import (
    parse_resolution as sm_parse_resolution,
    snap_resolution_to_model_grid,
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    download_image_if_url as sm_download_image_if_url
)
from source.video_utils import (
    prepare_vace_ref_for_segment as sm_prepare_vace_ref_for_segment,
    create_guide_video_for_travel_segment as sm_create_guide_video_for_travel_segment,
    extract_last_frame_as_image as sm_extract_last_frame_as_image
)
from source import db_operations as db_ops
from headless_model_management import HeadlessTaskQueue, GenerationTask

# Define Direct Queue Task Types
DIRECT_QUEUE_TASK_TYPES = {
    "wan_2_2_t2i", "vace", "vace_21", "vace_22", "flux", "t2v", "t2v_22", 
    "i2v", "i2v_22", "hunyuan", "ltxv", "generate_video",
    "qwen_image_edit", "qwen_image_style", "image_inpaint", "annotated_image_edit"
}


def _handle_travel_segment_via_queue(task_params_dict, main_output_dir_base: Path, task_id: str, colour_match_videos: bool, mask_active_frames: bool, task_queue: HeadlessTaskQueue, dprint_func, is_standalone: bool = False):
    """
    Handle travel segment tasks via direct queue integration to eliminate blocking waits.
    This is moved from worker.py.
    
    Supports two modes:
    1. Orchestrator mode (is_standalone=False): Requires orchestrator_task_id_ref, orchestrator_run_id, segment_index
    2. Standalone mode (is_standalone=True): full_orchestrator_payload provided directly in params
    """
    headless_logger.debug(f"Starting travel segment queue processing (standalone={is_standalone})", task_id=task_id)
    log_ram_usage("Segment via queue - start", task_id=task_id)
    
    try:
        from source.sm_functions.travel_between_images import _handle_travel_chaining_after_wgp
        
        segment_params = task_params_dict
        orchestrator_task_id_ref = segment_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = segment_params.get("orchestrator_run_id")
        segment_idx = segment_params.get("segment_index")
        
        full_orchestrator_payload = segment_params.get("full_orchestrator_payload")
        
        if is_standalone:
            # Standalone mode: use provided payload, default segment_idx to 0
            if segment_idx is None:
                segment_idx = 0
            if not full_orchestrator_payload:
                return False, f"Individual travel segment {task_id} missing full_orchestrator_payload"
            headless_logger.debug(f"Running in standalone mode (individual_travel_segment)", task_id=task_id)
        else:
            # Orchestrator mode: require references and fetch payload
            if None in [orchestrator_task_id_ref, orchestrator_run_id, segment_idx]:
                return False, f"Travel segment {task_id} missing critical orchestrator references"
            
            if not full_orchestrator_payload:
                orchestrator_task_raw_params_json = db_ops.get_task_params(orchestrator_task_id_ref)
                if orchestrator_task_raw_params_json:
                    fetched_params = json.loads(orchestrator_task_raw_params_json) if isinstance(orchestrator_task_raw_params_json, str) else orchestrator_task_raw_params_json
                    full_orchestrator_payload = fetched_params.get("orchestrator_details")
            
            if not full_orchestrator_payload:
                return False, f"Travel segment {task_id}: Could not retrieve orchestrator payload"
        
        model_name = full_orchestrator_payload["model_name"]
        prompt_for_wgp = ensure_valid_prompt(segment_params.get("base_prompt", " "))
        negative_prompt_for_wgp = ensure_valid_negative_prompt(segment_params.get("negative_prompt", " "))
        
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        parsed_res_raw = sm_parse_resolution(parsed_res_wh_str)
        if parsed_res_raw is None:
            return False, f"Travel segment {task_id}: Invalid resolution format {parsed_res_wh_str}"
        parsed_res_wh = snap_resolution_to_model_grid(parsed_res_raw)
        
        total_frames_for_segment = segment_params.get("segment_frames_target", 
                                                    full_orchestrator_payload["segment_frames_expanded"][segment_idx])
        
        current_run_base_output_dir_str = segment_params.get("current_run_base_output_dir")
        if not current_run_base_output_dir_str:
            current_run_base_output_dir_str = full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
        
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        segment_processing_dir = current_run_base_output_dir
        segment_processing_dir.mkdir(parents=True, exist_ok=True)
        
        debug_enabled = segment_params.get("debug_mode_enabled", full_orchestrator_payload.get("debug_mode_enabled", False))

        travel_mode = full_orchestrator_payload.get("model_type", "vace")

        start_ref_path = None
        end_ref_path = None
        
        input_images_resolved = full_orchestrator_payload.get("input_image_paths_resolved", [])
        print(f"!!! DEBUG TASK REGISTRY !!! segment_idx: {segment_idx}, input_images_resolved len: {len(input_images_resolved)}")
        print(f"!!! DEBUG TASK REGISTRY !!! input_images: {input_images_resolved}")
        is_continuing = full_orchestrator_payload.get("continue_from_video_resolved_path") is not None
        
        if is_continuing:
            if segment_idx == 0:
                continued_video_path = full_orchestrator_payload.get("continue_from_video_resolved_path")
                if continued_video_path and Path(continued_video_path).exists():
                    start_ref_path = sm_extract_last_frame_as_image(continued_video_path, segment_processing_dir, task_id)
                if input_images_resolved:
                    end_ref_path = input_images_resolved[0]
            else:
                if len(input_images_resolved) > segment_idx:
                    start_ref_path = input_images_resolved[segment_idx - 1]
                    end_ref_path = input_images_resolved[segment_idx]
        else:
            print("!!! DEBUG TASK REGISTRY !!! Entering ELSE block (from scratch)")
            if len(input_images_resolved) > segment_idx:
                start_ref_path = input_images_resolved[segment_idx]
            
            if len(input_images_resolved) > segment_idx + 1:
                end_ref_path = input_images_resolved[segment_idx + 1]
        print(f"!!! DEBUG TASK REGISTRY !!! start_ref_path after logic: {start_ref_path}")
        
        if start_ref_path:
            start_ref_path = sm_download_image_if_url(start_ref_path, segment_processing_dir, task_id, debug_mode=debug_enabled)
            print(f"!!! DEBUG TASK REGISTRY !!! start_ref_path AFTER DOWNLOAD: {start_ref_path}")
        if end_ref_path:
            end_ref_path = sm_download_image_if_url(end_ref_path, segment_processing_dir, task_id, debug_mode=debug_enabled)

        guide_video_path = None
        mask_video_path_for_wgp = None
        video_prompt_type_str = None

        if travel_mode == "vace":
            try:
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
                    dprint=dprint_func
                )
                processor = TravelSegmentProcessor(processor_context)
                segment_outputs = processor.process_segment()
                
                guide_video_path = segment_outputs.get("video_guide")
                mask_video_path_for_wgp = Path(segment_outputs["video_mask"]) if segment_outputs.get("video_mask") else None
                video_prompt_type_str = segment_outputs["video_prompt_type"]
                
            except Exception as e_shared_processor:
                traceback.print_exc()
                return False, f"Shared processor failed: {e_shared_processor}"
        
        generation_params = {
            "model_name": model_name,
            "negative_prompt": negative_prompt_for_wgp,
            "resolution": f"{parsed_res_wh[0]}x{parsed_res_wh[1]}",
            "video_length": total_frames_for_segment,
            "seed": segment_params.get("seed_to_use", 12345),
        }
        
        # Always pass images if available, regardless of specific travel_mode string (hybrid models need them)
        if start_ref_path: 
            generation_params["image_start"] = str(Path(start_ref_path).resolve())
        if end_ref_path: 
            generation_params["image_end"] = str(Path(end_ref_path).resolve())
            
        print(f"!!! DEBUG TASK REGISTRY !!! generation_params image_start: {generation_params.get('image_start')}")
        
        additional_loras = full_orchestrator_payload.get("additional_loras", {})
        if additional_loras:
            generation_params["additional_loras"] = additional_loras

        explicit_steps = (segment_params.get("num_inference_steps") or segment_params.get("steps") or 
                          full_orchestrator_payload.get("num_inference_steps") or full_orchestrator_payload.get("steps"))
        if explicit_steps: generation_params["num_inference_steps"] = explicit_steps
        
        explicit_guidance = (full_orchestrator_payload.get("guidance_scale") or segment_params.get("guidance_scale"))
        if explicit_guidance: generation_params["guidance_scale"] = explicit_guidance
        
        explicit_flow_shift = (full_orchestrator_payload.get("flow_shift") or segment_params.get("flow_shift"))
        if explicit_flow_shift: generation_params["flow_shift"] = explicit_flow_shift

        # Phase Config Override
        if "phase_config" in full_orchestrator_payload:
            try:
                steps_per_phase = full_orchestrator_payload["phase_config"].get("steps_per_phase", [2, 2, 2])
                phase_config_steps = sum(steps_per_phase)
                
                parsed_phase_config = parse_phase_config(
                    phase_config=full_orchestrator_payload["phase_config"],
                    num_inference_steps=phase_config_steps,
                    task_id=task_id,
                    model_name=generation_params.get("model_name"),
                    debug_mode=debug_enabled
                )
                
                generation_params["num_inference_steps"] = phase_config_steps
                
                for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                           "guidance_scale", "guidance2_scale", "guidance3_scale",
                           "flow_shift", "sample_solver", "model_switch_phase",
                           "lora_names", "lora_multipliers", "additional_loras"]:
                    if key in parsed_phase_config and parsed_phase_config[key] is not None:
                        generation_params[key] = parsed_phase_config[key]
                
                if "lora_names" in parsed_phase_config:
                    generation_params["activated_loras"] = parsed_phase_config["lora_names"]
                if "lora_multipliers" in parsed_phase_config:
                    generation_params["loras_multipliers"] = " ".join(str(m) for m in parsed_phase_config["lora_multipliers"])
                
                if "_patch_config" in parsed_phase_config:
                    apply_phase_config_patch(parsed_phase_config, model_name, task_id)

            except Exception as e:
                raise ValueError(f"Task {task_id}: Invalid phase_config: {e}")

        if guide_video_path: generation_params["video_guide"] = str(guide_video_path)
        if mask_video_path_for_wgp: generation_params["video_mask"] = str(mask_video_path_for_wgp.resolve())
        generation_params["video_prompt_type"] = video_prompt_type_str
        
        generation_task = GenerationTask(
            id=f"travel_seg_{task_id}",
            model=model_name,
            prompt=prompt_for_wgp,
            parameters=generation_params
        )
        
        submitted_task_id = task_queue.submit_task(generation_task)
        
        max_wait_time = 1800
        wait_interval = 2
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            status = task_queue.get_task_status(f"travel_seg_{task_id}")
            if status is None: return False, f"Travel segment {task_id}: Task status became None"
            
            if status.status == "completed":
                # In standalone mode, skip chaining (no orchestrator to coordinate with)
                if is_standalone:
                    headless_logger.debug(f"Standalone segment completed, skipping chaining", task_id=task_id)
                    return True, status.result_path
                
                # Orchestrator mode: run chaining
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
                        "cm_start_ref_path": None, "cm_end_ref_path": None, "show_input_images": False, "start_image_path": None, "end_image_path": None,
                    }},
                    actual_wgp_output_video_path=status.result_path,
                    image_download_dir=segment_processing_dir,
                    dprint=dprint_func
                )
                if chain_success and final_chained_path:
                    return True, final_chained_path
                else:
                    return True, status.result_path
            elif status.status == "failed":
                return False, f"Travel segment {task_id}: Generation failed: {status.error_message}"
            
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        return False, f"Travel segment {task_id}: Generation timeout"

    except Exception as e:
        traceback.print_exc()
        return False, f"Travel segment {task_id}: Exception: {str(e)}"


class TaskRegistry:
    """Registry for task handlers."""
    
    @staticmethod
    def dispatch(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Dispatch a task to the appropriate handler.
        
        Args:
            task_type: The type of task to execute.
            context: Dictionary containing all necessary context variables:
                - task_params_dict
                - main_output_dir_base
                - task_id
                - project_id
                - task_queue
                - colour_match_videos
                - mask_active_frames
                - debug_mode
                - wan2gp_path
        
        Returns:
            Tuple (success, output_location)
        """
        task_id = context["task_id"]
        params = context["task_params_dict"]
        dprint_func = make_task_dprint(task_id, context["debug_mode"])

        # 1. Direct Queue Tasks
        if task_type in DIRECT_QUEUE_TASK_TYPES and context["task_queue"]:
            return TaskRegistry._handle_direct_queue_task(task_type, context)

        # 2. Orchestrator & Specialized Handlers
        handlers = {
            "travel_orchestrator": lambda: tbi._handle_travel_orchestrator_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                orchestrator_task_id_str=task_id,
                orchestrator_project_id=context["project_id"],
                dprint=dprint_func
            ),
            "travel_segment": lambda: _handle_travel_segment_via_queue(
                task_params_dict=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                colour_match_videos=context["colour_match_videos"],
                mask_active_frames=context["mask_active_frames"],
                task_queue=context["task_queue"],
                dprint_func=dprint_func,
                is_standalone=False
            ),
            "individual_travel_segment": lambda: _handle_travel_segment_via_queue(
                task_params_dict=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                colour_match_videos=context["colour_match_videos"],
                mask_active_frames=context["mask_active_frames"],
                task_queue=context["task_queue"],
                dprint_func=dprint_func,
                is_standalone=True
            ),
            "travel_stitch": lambda: tbi._handle_travel_stitch_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                stitch_task_id_str=task_id,
                dprint=dprint_func
            ),
            "magic_edit": lambda: me._handle_magic_edit_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                dprint=dprint_func
            ),
            "join_clips_orchestrator": lambda: _handle_join_clips_orchestrator_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                orchestrator_task_id_str=task_id,
                orchestrator_project_id=context["project_id"],
                dprint=dprint_func
            ),
            "join_clips_segment": lambda: _handle_join_clips_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                task_queue=context["task_queue"],
                dprint=dprint_func
            ),
            "inpaint_frames": lambda: _handle_inpaint_frames_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                task_queue=context["task_queue"],
                dprint=dprint_func
            ),
            "create_visualization": lambda: _handle_create_visualization_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                viz_task_id_str=task_id,
                dprint=dprint_func
            ),
            "extract_frame": lambda: handle_extract_frame_task(
                params, context["main_output_dir_base"], task_id, dprint_func
            ),
            "rife_interpolate_images": lambda: handle_rife_interpolate_task(
                params, context["main_output_dir_base"], task_id, dprint_func, task_queue=context["task_queue"]
            )
        }

        if task_type in handlers:
            # Orchestrator setup
            if task_type in ["travel_orchestrator", "join_clips_orchestrator"]:
                params["task_id"] = task_id
                if "orchestrator_details" in params:
                    params["orchestrator_details"]["orchestrator_task_id"] = task_id
            
            return handlers[task_type]()

        # Default fallthrough to queue
        if context["task_queue"]:
             return TaskRegistry._handle_direct_queue_task(task_type, context)
        
        raise ValueError(f"Unknown task type {task_type} and no queue available")

    @staticmethod
    def _handle_direct_queue_task(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        task_id = context["task_id"]
        task_queue = context["task_queue"]
        params = context["task_params_dict"]
        
        try:
            generation_task = db_task_to_generation_task(
                params, task_id, task_type, context["wan2gp_path"], context["debug_mode"]
            )
            
            if task_type == "wan_2_2_t2i":
                generation_task.parameters["video_length"] = 1
            
            if context["colour_match_videos"]:
                generation_task.parameters["colour_match_videos"] = True
            if context["mask_active_frames"]:
                generation_task.parameters["mask_active_frames"] = True
            
            task_queue.submit_task(generation_task)
            
            # Wait for completion
            max_wait_time = 3600
            elapsed = 0
            while elapsed < max_wait_time:
                status = task_queue.get_task_status(task_id)
                if not status: return False, "Task status became None"
                
                if status.status == "completed":
                    return True, status.result_path
                elif status.status == "failed":
                    return False, status.error_message or "Failed without message"
                
                time.sleep(2)
                elapsed += 2
            
            return False, "Timeout"
            
        except Exception as e:
            traceback.print_exc()
            return False, f"Queue error: {e}"

