"""Wan2GP Worker Server.

This long-running process polls the Supabase-backed Postgres `tasks` table,
claims queued tasks, and executes them using the HeadlessTaskQueue system.
"""

import argparse
import sys
import os
import time
import datetime
import traceback
import threading
import logging
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
wan2gp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2GP")
if wan2gp_path not in sys.path:
    sys.path.append(wan2gp_path)

from source import db_operations as db_ops
from source.fatal_error_handler import FatalWorkerError, reset_fatal_error_counter
from headless_model_management import HeadlessTaskQueue

from source.logging_utils import (
    headless_logger, enable_debug_mode, disable_debug_mode,
    LogBuffer, CustomLogInterceptor, set_log_interceptor, set_log_file
)
from source.worker_utils import (
    dprint, log_ram_usage, cleanup_generated_files
)
from source.heartbeat_utils import start_heartbeat_guardian_process
from source.task_registry import TaskRegistry
from source.sm_functions import travel_between_images as tbi
from source.lora_utils import cleanup_legacy_lora_collisions

# Global heartbeat control
heartbeat_thread = None
heartbeat_stop_event = threading.Event()
debug_mode = False

def process_single_task(task_params_dict, main_output_dir_base: Path, task_type: str, project_id_for_task: str | None, image_download_dir: Path | str | None = None, colour_match_videos: bool = False, mask_active_frames: bool = True, task_queue: HeadlessTaskQueue = None):
    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))
    headless_logger.essential(f"Processing {task_type} task", task_id=task_id)
    
    context = {
        "task_params_dict": task_params_dict,
        "main_output_dir_base": main_output_dir_base,
        "task_id": task_id,
        "project_id": project_id_for_task,
        "task_queue": task_queue,
        "colour_match_videos": colour_match_videos,
        "mask_active_frames": mask_active_frames,
        "debug_mode": debug_mode,
        "wan2gp_path": wan2gp_path,
    }

    generation_success, output_location_to_db = TaskRegistry.dispatch(task_type, context)

    # Chaining Logic
    if generation_success:
        chaining_result_path_override = None

        if task_params_dict.get("travel_chain_details"):
            chain_success, chain_message, final_path_from_chaining = tbi._handle_travel_chaining_after_wgp(
                wgp_task_params=task_params_dict, 
                actual_wgp_output_video_path=output_location_to_db,
                image_download_dir=image_download_dir,
                dprint=lambda msg: dprint(msg, task_id=task_id, debug_mode=debug_mode)
            )
            if chain_success:
                chaining_result_path_override = final_path_from_chaining
            else:
                headless_logger.error(f"Travel chaining failed: {chain_message}", task_id=task_id)

        if chaining_result_path_override:
                output_location_to_db = chaining_result_path_override

    # Ensure orchestrator tasks use their DB row ID
    if task_type in {"travel_orchestrator"}:
        task_params_dict["task_id"] = task_id

    headless_logger.essential(f"Finished task (Success: {generation_success})", task_id=task_id)
    return generation_success, output_location_to_db


def parse_args():
    parser = argparse.ArgumentParser("WanGP Worker Server")
    parser.add_argument("--main-output-dir", type=str, default="./outputs")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--save-logging", type=str, nargs='?', const='logs/worker.log', default=None)
    parser.add_argument("--migrate-only", action="store_true")
    parser.add_argument("--colour-match-videos", action="store_true")
    parser.add_argument("--mask-active-frames", dest="mask_active_frames", action="store_true", default=True)
    parser.add_argument("--no-mask-active-frames", dest="mask_active_frames", action="store_false")
    parser.add_argument("--queue-workers", type=int, default=1)
    parser.add_argument("--preload-model", type=str, default="lightning_baseline_2_2_2")
    parser.add_argument("--db-type", type=str, default="supabase")
    parser.add_argument("--supabase-url", type=str, required=True)
    parser.add_argument("--supabase-access-token", type=str, required=True)
    parser.add_argument("--supabase-anon-key", type=str, default=None)
    
    # WGP Globals
    parser.add_argument("--wgp-attention-mode", type=str, default=None)
    parser.add_argument("--wgp-compile", type=str, default=None)
    parser.add_argument("--wgp-profile", type=int, default=None)
    parser.add_argument("--wgp-vae-config", type=int, default=None)
    parser.add_argument("--wgp-boost", type=int, default=None)
    parser.add_argument("--wgp-transformer-quantization", type=str, default=None)
    parser.add_argument("--wgp-transformer-dtype-policy", type=str, default=None)
    parser.add_argument("--wgp-text-encoder-quantization", type=str, default=None)
    parser.add_argument("--wgp-vae-precision", type=str, default=None)
    parser.add_argument("--wgp-mixed-precision", type=str, default=None)
    parser.add_argument("--wgp-preload-policy", type=str, default=None)
    parser.add_argument("--wgp-preload", type=int, default=None)

    return parser.parse_args()

def main():
    load_dotenv()

    cli_args = parse_args()
    
    if cli_args.worker:
        os.environ["WORKER_ID"] = cli_args.worker
        os.environ["WAN2GP_WORKER_MODE"] = "true"

    global debug_mode
    debug_mode = cli_args.debug
    if debug_mode:
        enable_debug_mode()
        if not cli_args.save_logging:
            # Automatically save logs to debug/ directory if debug mode is on
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"debug_{timestamp}.log")
            set_log_file(log_file)
            headless_logger.essential(f"Debug logging enabled. Saving to {log_file}")
    else:
        disable_debug_mode()

    if cli_args.save_logging:
        set_log_file(cli_args.save_logging)

    # Supabase Setup
    try:
        client_key = cli_args.supabase_anon_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if not client_key: raise ValueError("No Supabase key found")
        
        db_ops.DB_TYPE = "supabase"
        db_ops.PG_TABLE_NAME = os.getenv("POSTGRES_TABLE_NAME", "tasks")
        db_ops.SUPABASE_URL = cli_args.supabase_url
        db_ops.SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        db_ops.SUPABASE_VIDEO_BUCKET = os.getenv("SUPABASE_VIDEO_BUCKET", "image_uploads")
        db_ops.SUPABASE_CLIENT = create_client(cli_args.supabase_url, client_key)
        db_ops.SUPABASE_ACCESS_TOKEN = cli_args.supabase_access_token
        db_ops.debug_mode = debug_mode

    except Exception as e:
        headless_logger.critical(f"Supabase init failed: {e}")
        sys.exit(1)

    if cli_args.migrate_only:
        sys.exit(0)

    main_output_dir = Path(cli_args.main_output_dir).resolve()
    main_output_dir.mkdir(parents=True, exist_ok=True)

    # Centralized Logging
    if cli_args.worker:
        guardian_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or client_key
        guardian_process, log_queue = start_heartbeat_guardian_process(cli_args.worker, cli_args.supabase_url, guardian_key)
        _global_log_buffer = LogBuffer(max_size=100, shared_queue=log_queue)
        set_log_interceptor(CustomLogInterceptor(_global_log_buffer))

    # Apply WGP Overrides
    wan_dir = str((Path(__file__).parent / "Wan2GP").resolve())
    original_cwd = os.getcwd()
    original_argv = sys.argv[:]
    try:
        os.chdir(wan_dir)
        sys.path.insert(0, wan_dir)
        sys.argv = ["worker.py"]
        import wgp as wgp_mod
        sys.argv = original_argv

        if cli_args.wgp_attention_mode: wgp_mod.attention_mode = cli_args.wgp_attention_mode
        if cli_args.wgp_compile: wgp_mod.compile = cli_args.wgp_compile
        if cli_args.wgp_profile: 
            wgp_mod.force_profile_no = cli_args.wgp_profile
            wgp_mod.default_profile = cli_args.wgp_profile
        if cli_args.wgp_vae_config: wgp_mod.vae_config = cli_args.wgp_vae_config
        if cli_args.wgp_boost: wgp_mod.boost = cli_args.wgp_boost
        if cli_args.wgp_transformer_quantization: wgp_mod.transformer_quantization = cli_args.wgp_transformer_quantization
        if cli_args.wgp_transformer_dtype_policy: wgp_mod.transformer_dtype_policy = cli_args.wgp_transformer_dtype_policy
        if cli_args.wgp_text_encoder_quantization: wgp_mod.text_encoder_quantization = cli_args.wgp_text_encoder_quantization
        if cli_args.wgp_vae_precision: wgp_mod.server_config["vae_precision"] = cli_args.wgp_vae_precision
        if cli_args.wgp_mixed_precision: wgp_mod.server_config["mixed_precision"] = cli_args.wgp_mixed_precision
        if cli_args.wgp_preload_policy: wgp_mod.server_config["preload_model_policy"] = [x.strip() for x in cli_args.wgp_preload_policy.split(',')]
        if cli_args.wgp_preload: wgp_mod.server_config["preload_in_VRAM"] = cli_args.wgp_preload
        if "transformer_types" not in wgp_mod.server_config: wgp_mod.server_config["transformer_types"] = []

    except Exception as e:
        headless_logger.critical(f"WGP import failed: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

    # Clean up legacy collision-prone LoRA files
    cleanup_legacy_lora_collisions()

    # Initialize Task Queue
    try:
        task_queue = HeadlessTaskQueue(wan_dir=wan_dir, max_workers=cli_args.queue_workers)
        preload_model = cli_args.preload_model if cli_args.preload_model else None
        task_queue.start(preload_model=preload_model)
    except Exception as e:
        headless_logger.critical(f"Queue init failed: {e}")
        sys.exit(1)

    headless_logger.essential(f"Worker {cli_args.worker or 'anonymous'} started. Polling every {cli_args.poll_interval}s.")

    try:
        while True:
            task_info = db_ops.get_oldest_queued_task_supabase(worker_id=cli_args.worker)

            if not task_info:
                time.sleep(cli_args.poll_interval)
                continue

            current_task_params = task_info["params"]
            current_task_type = task_info["task_type"]
            current_project_id = task_info.get("project_id")
            current_task_id = task_info["task_id"]

            if current_project_id is None and current_task_type == "travel_orchestrator":
                db_ops.update_task_status_supabase(current_task_id, db_ops.STATUS_FAILED, "Orchestrator missing project_id")
                continue

            # Ensure task_id in params
            if current_task_type in {"travel_orchestrator", "join_clips_orchestrator", "travel_segment", "individual_travel_segment", "join_clips_segment"}:
                current_task_params["task_id"] = current_task_id
                if "orchestrator_details" in current_task_params:
                    current_task_params["orchestrator_details"]["orchestrator_task_id"] = current_task_id

            task_succeeded, output_location = process_single_task(
                current_task_params, main_output_dir, current_task_type, current_project_id,
                image_download_dir=current_task_params.get("segment_image_download_dir"),
                colour_match_videos=cli_args.colour_match_videos,
                mask_active_frames=cli_args.mask_active_frames,
                task_queue=task_queue
            )

            if task_succeeded:
                reset_fatal_error_counter()

                orchestrator_types = {"travel_orchestrator", "join_clips_orchestrator"}

                if current_task_type in orchestrator_types:
                    if output_location and output_location.startswith("[ORCHESTRATOR_COMPLETE]"):
                        actual_output = output_location.replace("[ORCHESTRATOR_COMPLETE]", "")
                        thumbnail_url = None
                        try:
                            import json
                            data = json.loads(actual_output)
                            actual_output = data.get("output_location", actual_output)
                            thumbnail_url = data.get("thumbnail_url")
                        except: pass
                        
                        db_ops.update_task_status_supabase(current_task_id, db_ops.STATUS_COMPLETE, actual_output, thumbnail_url)
                    else:
                        db_ops.update_task_status(current_task_id, db_ops.STATUS_IN_PROGRESS, output_location)
                else:
                    db_ops.update_task_status_supabase(current_task_id, db_ops.STATUS_COMPLETE, output_location)
                    
                    # Note: Orchestrator completion is handled by the complete-task Edge Function
                    # based on checking if all child tasks are complete.
                    
                    cleanup_generated_files(output_location, current_task_id, debug_mode)
            else:
                db_ops.update_task_status_supabase(current_task_id, db_ops.STATUS_FAILED, output_location)
            
            time.sleep(1)

    except FatalWorkerError as e:
        headless_logger.critical(f"Fatal Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        headless_logger.essential("Shutting down...")
    finally:
        if cli_args.worker:
            heartbeat_stop_event.set()
        if task_queue:
            task_queue.stop()

if __name__ == "__main__":
    main() 
