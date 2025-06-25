"""Different-pose task handler."""

import json
import shutil
import traceback
from pathlib import Path

import cv2 # pip install opencv-python
from PIL import Image # pip install Pillow

# Import from the new common_utils and db_operations modules
from .. import db_operations as db_ops
from ..common_utils import (
    DEBUG_MODE, dprint, generate_unique_task_id, add_task_to_db, poll_task_status,
    save_frame_from_video, create_pose_interpolated_guide_video,
    generate_different_pose_debug_video_summary,
    parse_resolution as sm_parse_resolution
)

def _handle_different_pose_orchestrator_task(task_params_from_db: dict, main_output_dir_base: Path, orchestrator_task_id_str: str, dprint):
    """
    This is the entry point for a 'different_pose' job.
    It sets up the environment and enqueues all necessary child tasks with
    database-level dependencies.
    """
    print(f"--- Orchestrating Task: Different Pose (ID: {orchestrator_task_id_str}) ---")
    dprint(f"Orchestrator Task Params: {task_params_from_db}")

    try:
        public_files_dir = Path.cwd() / "public" / "files"
        public_files_dir.mkdir(parents=True, exist_ok=True)
        main_output_dir_base = public_files_dir

        run_id = generate_unique_task_id("dp_run_")
        work_dir = main_output_dir_base / f"different_pose_run_{run_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"Working directory for this 'different_pose' run: {work_dir.resolve()}")

        task_id_user_pose = generate_unique_task_id("dp_user_pose_")
        task_id_t2i = generate_unique_task_id("dp_t2i_")
        task_id_extract = generate_unique_task_id("dp_extract_")
        task_id_t2i_pose = generate_unique_task_id("dp_t2i_pose_")
        task_id_final_gen = generate_unique_task_id("dp_final_gen_")

        orchestrator_payload = {
            "run_id": run_id,
            "orchestrator_task_id": orchestrator_task_id_str,
            "work_dir": str(work_dir.resolve()),
            "main_output_dir": str(main_output_dir_base.resolve()),
            "original_params": task_params_from_db,
            "task_ids": {
                "user_pose": task_id_user_pose,
                "t2i": task_id_t2i,
                "extract": task_id_extract,
                "t2i_pose": task_id_t2i_pose,
                "final_gen": task_id_final_gen,
            },
            "debug_mode": task_params_from_db.get("debug_mode", False),
            "skip_cleanup": task_params_from_db.get("skip_cleanup", False),
        }

        previous_task_id = None

        payload_user_pose = {
            "task_id": task_id_user_pose,
            "input_image_path": task_params_from_db['input_image_path'],
            "dp_orchestrator_payload": orchestrator_payload,
            "output_dir": str(work_dir.resolve()),
        }
        db_ops.add_task_to_db(payload_user_pose, "generate_openpose", dependant_on=previous_task_id)
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued user_pose_gen ({task_id_user_pose})")
        previous_task_id = task_id_user_pose

        payload_t2i = {
            "task_id": task_id_t2i,
            "prompt": task_params_from_db.get("prompt"),
            "model": task_params_from_db.get("model_name"),
            "resolution": task_params_from_db.get("resolution"),
            "frames": 1,
            "seed": task_params_from_db.get("seed", -1),
            "use_causvid_lora": task_params_from_db.get("use_causvid_lora", False),
            "dp_orchestrator_payload": orchestrator_payload,
            "output_dir": str(work_dir.resolve()),
        }
        db_ops.add_task_to_db(payload_t2i, "wgp", dependant_on=previous_task_id)
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued t2i_gen ({task_id_t2i})")
        previous_task_id = task_id_t2i

        payload_extract = {
            "task_id": task_id_extract,
            "input_video_task_id": task_id_t2i,
            "frame_index": 0,
            "dp_orchestrator_payload": orchestrator_payload,
            "output_dir": str(work_dir.resolve()),
        }
        db_ops.add_task_to_db(payload_extract, "extract_frame", dependant_on=previous_task_id)
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued extract_frame ({task_id_extract})")
        previous_task_id = task_id_extract

        payload_t2i_pose = {
            "task_id": task_id_t2i_pose,
            "input_image_task_id": task_id_extract,
            "dp_orchestrator_payload": orchestrator_payload,
            "output_dir": str(work_dir.resolve()),
        }
        db_ops.add_task_to_db(payload_t2i_pose, "generate_openpose", dependant_on=previous_task_id)
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued t2i_pose_gen ({task_id_t2i_pose})")
        previous_task_id = task_id_t2i_pose

        payload_final_gen = {
            "task_id": task_id_final_gen,
            "dp_orchestrator_payload": orchestrator_payload,
        }
        db_ops.add_task_to_db(payload_final_gen, "dp_final_gen", dependant_on=previous_task_id)
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued dp_final_gen ({task_id_final_gen})")

        return True, f"Successfully enqueued different_pose job graph with run_id {run_id}."
    except Exception as e:
        error_msg = f"Different Pose orchestration failed: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return False, error_msg


def _handle_dp_final_gen_task(task_params_from_db: dict, dprint):
    """
    Handles the final step of the 'different_pose' process. It gathers all
    the required artifacts generated by previous tasks in the dependency graph,
    creates the final guide video, runs the last generation, extracts the
    final image, and performs cleanup.
    """
    payload = task_params_from_db
    orchestrator_payload = payload.get("dp_orchestrator_payload")
    if not orchestrator_payload:
        return False, "Final Gen failed: 'dp_orchestrator_payload' not found.", None

    task_ids = orchestrator_payload.get("task_ids", {})
    work_dir = Path(orchestrator_payload["work_dir"])
    original_params = orchestrator_payload["original_params"]
    final_path_for_db = None
    
    dprint("DP Final Gen: Starting final generation step.")

    try:
        user_pose_path_db = db_ops.get_task_output_location_from_db(task_ids["user_pose"])
        t2i_pose_path_db = db_ops.get_task_output_location_from_db(task_ids["t2i_pose"])
        t2i_image_path_db = db_ops.get_task_output_location_from_db(task_ids["extract"])

        user_pose_image_path = db_ops.get_abs_path_from_db_path(user_pose_path_db, dprint)
        t2i_pose_image_path = db_ops.get_abs_path_from_db_path(t2i_pose_path_db, dprint)
        t2i_image_path = db_ops.get_abs_path_from_db_path(t2i_image_path_db, dprint)

        if not all([user_pose_image_path, t2i_pose_image_path, t2i_image_path]):
            return False, "Could not resolve one or more required image paths from previous tasks.", None

        print("\nDP Final Gen: Creating custom guide video...")
        custom_guide_video_path = work_dir / f"{generate_unique_task_id('dp_custom_guide_')}.mp4"
        
        create_pose_interpolated_guide_video(
            output_video_path=custom_guide_video_path,
            resolution=sm_parse_resolution(original_params.get("resolution")),
            total_frames=original_params.get("output_video_frames", 16),
            start_image_path=Path(original_params['input_image_path']),
            end_image_path=t2i_image_path,
            fps=original_params.get("fps_helpers", 16),
        )
        print(f"DP Final Gen: Successfully created pose-interpolated guide video: {custom_guide_video_path}")

        final_video_task_id = task_ids["final_gen"]
        final_video_payload = {
            "task_id": final_video_task_id,
            "prompt": original_params.get("prompt"),
            "model": original_params.get("model_name"),
            "resolution": original_params.get("resolution"),
            "frames": original_params.get("output_video_frames", 16),
            "seed": original_params.get("seed", -1) + 1,
            "video_guide_path": str(custom_guide_video_path.resolve()),
            "image_refs_paths": [original_params['input_image_path']],
            "image_prompt_type": "IV",
            "use_causvid_lora": original_params.get("use_causvid_lora", False),
        }
        
        db_ops.add_task_to_db(final_video_payload, "wgp")
        
        final_video_output_db = poll_task_status(final_video_task_id, db_ops.SQLITE_DB_PATH)
        if not final_video_output_db:
             return False, "Polling for final video generation task failed or timed out.", None
        
        final_video_path = db_ops.get_abs_path_from_db_path(final_video_output_db, dprint)
        if not final_video_path:
             return False, f"Could not resolve final video path from '{final_video_output_db}'", None

        print("\nDP Final Gen: Extracting final posed image...")
        final_posed_image_output_path = Path(orchestrator_payload["main_output_dir"]) / f"final_posed_image_{orchestrator_payload['run_id']}.png"
        
        if not save_frame_from_video(final_video_path, -1, final_posed_image_output_path, sm_parse_resolution(original_params.get("resolution"))):
             return False, f"Failed to extract final posed image from {final_video_path}", None
        
        print(f"Successfully completed 'different_pose' task! Final image: {final_posed_image_output_path.resolve()}")
        final_path_for_db = str(final_posed_image_output_path.resolve())

        if not orchestrator_payload.get("skip_cleanup") and not orchestrator_payload.get("debug_mode"):
            print(f"DP Final Gen: Cleaning up intermediate files in {work_dir}...")
            try:
                shutil.rmtree(work_dir)
                print(f"Removed intermediate directory: {work_dir}")
            except OSError as e_clean:
                print(f"Error removing intermediate directory {work_dir}: {e_clean}")
        else:
            print(f"Skipping cleanup of intermediate files in {work_dir}.")
        
        db_ops.update_task_status(orchestrator_payload['orchestrator_task_id'], db_ops.STATUS_COMPLETE, final_path_for_db)
        dprint(f"DP Final Gen: Process complete. Final image at {final_path_for_db}")
        
        return True, final_path_for_db

    except Exception as e:
        error_msg = f"DP Final Gen failed: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        db_ops.update_task_status(orchestrator_payload['orchestrator_task_id'], db_ops.STATUS_FAILED, error_msg)
        return False, error_msg 