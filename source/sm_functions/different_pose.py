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

def _get_abs_path_from_db_path(db_path: str, dprint) -> Path | None:
    """Helper to resolve a path from the DB (which might be relative) to a usable absolute path."""
    if not db_path:
        return None
        
    resolved_path = None
    if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and isinstance(db_path, str) and db_path.startswith("files/"):
        sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
        resolved_path = (sqlite_db_parent / "public" / db_path).resolve()
        dprint(f"Resolved SQLite relative path '{db_path}' to '{resolved_path}'")
    else:
        # Path from DB is already absolute (Supabase) or a non-standard path
        resolved_path = Path(db_path).resolve()
    
    if resolved_path and resolved_path.exists():
        return resolved_path
    else:
        dprint(f"Warning: Resolved path '{resolved_path}' from DB path '{db_path}' does not exist.")
        return None

def _handle_different_pose_orchestrator_task(task_params_from_db: dict, main_output_dir_base: Path, orchestrator_task_id_str: str, dprint):
    """
    This is the entry point for a 'different_pose' job.
    It sets up the environment and enqueues the very first step of the sequence.
    The rest of the process is handled by the chaining function.
    """
    print(f"--- Orchestrating Task: Different Pose (ID: {orchestrator_task_id_str}) ---")
    dprint(f"Orchestrator Task Params: {task_params_from_db}")

    try:
        run_id = generate_unique_task_id("dp_run_")
        work_dir = main_output_dir_base / f"different_pose_run_{run_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"Working directory for this 'different_pose' run: {work_dir.resolve()}")

        # Prepare the state dictionary that will be passed through the entire chain
        chain_details = {
            "run_id": run_id,
            "orchestrator_task_id": orchestrator_task_id_str,
            "work_dir": str(work_dir.resolve()),
            "main_output_dir": str(main_output_dir_base.resolve()),
            "original_params": task_params_from_db,
            "current_step": "user_pose_gen", # This is the first step we are enqueuing
            "debug_mode": task_params_from_db.get("debug_mode", False),
            "skip_cleanup": task_params_from_db.get("skip_cleanup", False),
            "debug_video_stages": [], # For the final debug video collage
        }
        
        if chain_details["debug_mode"]:
            input_image_path = task_params_from_db.get('input_image_path')
            if input_image_path:
                chain_details["debug_video_stages"].append({
                    'label': f"0. User Input ({Path(input_image_path).name})", 
                    'type': 'image', 'path': input_image_path
                })

        # --- Enqueue Step 1: Generate OpenPose from User Input Image ---
        user_pose_task_id = generate_unique_task_id("dp_user_pose_")
        user_pose_output_path = work_dir / f"{user_pose_task_id}_openpose_from_user.png"

        payload = {
            "task_id": user_pose_task_id,
            "input_image_path": task_params_from_db['input_image_path'],
            "output_path": str(user_pose_output_path.resolve()),
            "different_pose_chain_details": chain_details, # Pass the state
        }
        
        db_ops.add_task_to_db(payload, "generate_openpose")
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued first step ('user_pose_gen') with task ID {user_pose_task_id}")

        return True, f"Successfully enqueued different_pose job with run_id {run_id}."
    except Exception as e:
        error_msg = f"Different Pose orchestration failed: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return False, error_msg

def _handle_different_pose_chaining(completed_task_params: dict, task_output_path: str, dprint):
    """
    The state machine for the 'different_pose' process.
    This function is called after each step completes. It analyzes the output
    and enqueues the next logical step in the sequence.
    """
    chain_details = completed_task_params.get("different_pose_chain_details")
    if not chain_details:
        return False, "Chaining failed: 'different_pose_chain_details' not found in completed task.", None

    current_step = chain_details.get("current_step")
    work_dir = Path(chain_details["work_dir"])
    original_params = chain_details["original_params"]
    final_path_for_db = None # Only populated on the very last step

    try:
        # --- Step 1 (USER_POSE_GEN) is complete -> Start Step 2 (T2I_GEN) ---
        if current_step == "user_pose_gen":
            dprint("DP Chain: Step 1 (user_pose_gen) complete. Starting Step 2 (t2i_gen).")
            user_pose_image_path = _get_abs_path_from_db_path(task_output_path, dprint)
            if not user_pose_image_path:
                return False, f"Could not resolve user pose image path from '{task_output_path}'", None
            
            chain_details["user_pose_image_path"] = str(user_pose_image_path)
            if chain_details["debug_mode"]:
                chain_details["debug_video_stages"].append({
                    'label': f"1. OpenPose from User Input ({user_pose_image_path.name})",
                    'type': 'image', 'path': str(user_pose_image_path)
                })

            # Prepare payload for T2I generation
            t2i_task_id = generate_unique_task_id("dp_t2i_")
            t2i_output_video_path = work_dir / f"{t2i_task_id}_video_raw.mp4"
            
            payload = {
                "task_id": t2i_task_id,
                "prompt": original_params.get("prompt"),
                "model": original_params.get("model_name"),
                "resolution": original_params.get("resolution"),
                "frames": 1,
                "seed": original_params.get("seed", -1),
                "output_path": str(t2i_output_video_path.resolve()),
                "use_causvid_lora": original_params.get("use_causvid_lora", False),
            }

            chain_details["current_step"] = "t2i_gen"
            payload["different_pose_chain_details"] = chain_details
            
            db_ops.add_task_to_db(payload, "wgp")
            dprint(f"DP Chain: Enqueued Step 2 (t2i_gen) with task ID {t2i_task_id}")

        # --- Step 2 (T2I_GEN) is complete -> Start Step 3 (T2I_POSE_GEN) ---
        elif current_step == "t2i_gen":
            dprint("DP Chain: Step 2 (t2i_gen) complete. Starting Step 3 (t2i_pose_gen).")
            t2i_video_path = _get_abs_path_from_db_path(task_output_path, dprint)
            if not t2i_video_path:
                return False, f"Could not resolve T2I video path from '{task_output_path}'", None
            
            parsed_res = sm_parse_resolution(original_params.get("resolution"))
            
            # Extract the single frame from the generated video
            t2i_image_path = work_dir / f"{completed_task_params['task_id']}_generated_from_prompt.png"
            if not save_frame_from_video(t2i_video_path, 0, t2i_image_path, parsed_res):
                return False, f"Failed to extract frame from T2I video {t2i_video_path}", None
            
            chain_details["t2i_image_path"] = str(t2i_image_path)
            if chain_details["debug_mode"]:
                chain_details["debug_video_stages"].append({
                    'label': f"2. Target T2I Image ({t2i_image_path.name})",
                    'type': 'image', 'path': str(t2i_image_path)
                })

            # Prepare payload for OpenPose generation on the T2I image
            t2i_pose_task_id = generate_unique_task_id("dp_t2i_pose_")
            t2i_pose_output_path = work_dir / f"{t2i_pose_task_id}_openpose_from_t2i.png"
            
            payload = {
                "task_id": t2i_pose_task_id,
                "input_image_path": str(t2i_image_path),
                "output_path": str(t2i_pose_output_path.resolve()),
            }

            chain_details["current_step"] = "t2i_pose_gen"
            payload["different_pose_chain_details"] = chain_details
            
            db_ops.add_task_to_db(payload, "generate_openpose")
            dprint(f"DP Chain: Enqueued Step 3 (t2i_pose_gen) with task ID {t2i_pose_task_id}")

        # --- Step 3 (T2I_POSE_GEN) is complete -> Start Step 4 (FINAL_VIDEO_GEN) ---
        elif current_step == "t2i_pose_gen":
            dprint("DP Chain: Step 3 (t2i_pose_gen) complete. Starting Step 4 (final_video_gen).")
            t2i_pose_image_path = _get_abs_path_from_db_path(task_output_path, dprint)
            if not t2i_pose_image_path:
                 return False, f"Could not resolve T2I pose image path from '{task_output_path}'", None
            
            chain_details["t2i_pose_image_path"] = str(t2i_pose_image_path)
            if chain_details["debug_mode"]:
                chain_details["debug_video_stages"].append({
                    'label': f"3. OpenPose from T2I ({t2i_pose_image_path.name})",
                    'type': 'image', 'path': str(t2i_pose_image_path)
                })
            
            # Create the final interpolated guide video directly, as it's a CPU-bound task
            print("\nDP Chain: Creating custom guide video...")
            custom_guide_video_path = work_dir / f"{generate_unique_task_id('dp_custom_guide_')}.mp4"
            
            create_pose_interpolated_guide_video(
                output_video_path=custom_guide_video_path,
                resolution=sm_parse_resolution(original_params.get("resolution")),
                total_frames=original_params.get("output_video_frames", 16),
                start_image_path=Path(original_params['input_image_path']),
                end_image_path=Path(chain_details['t2i_image_path']),
                fps=original_params.get("fps_helpers", 16),
                confidence_threshold=0.1,
                include_face=True,
                include_hands=True
            )
            print(f"DP Chain: Successfully created pose-interpolated guide video: {custom_guide_video_path}")
            chain_details['final_guide_video_path'] = str(custom_guide_video_path)
            if chain_details["debug_mode"]:
                chain_details["debug_video_stages"].append({
                    'label': f"4. Custom Guide Video ({custom_guide_video_path.name})",
                    'type': 'video', 'path': str(custom_guide_video_path)
                })

            # Prepare payload for final video generation
            final_video_task_id = generate_unique_task_id("dp_final_vid_")
            final_video_headless_output_path = work_dir / f"{final_video_task_id}_video_raw.mp4"
            
            payload = {
                "task_id": final_video_task_id,
                "prompt": original_params.get("prompt"),
                "model": original_params.get("model_name"),
                "resolution": original_params.get("resolution"),
                "frames": original_params.get("output_video_frames", 16),
                "seed": original_params.get("seed", -1) + 1,
                "video_guide_path": str(custom_guide_video_path.resolve()),
                "image_refs_paths": [original_params['input_image_path']],
                "image_prompt_type": "IV",
                "output_path": str(final_video_headless_output_path.resolve()),
                "use_causvid_lora": original_params.get("use_causvid_lora", False),
            }

            chain_details["current_step"] = "final_video_gen"
            payload["different_pose_chain_details"] = chain_details
            
            db_ops.add_task_to_db(payload, "wgp")
            dprint(f"DP Chain: Enqueued Step 4 (final_video_gen) with task ID {final_video_task_id}")

        # --- Step 4 (FINAL_VIDEO_GEN) is complete -> Do final extraction & cleanup ---
        elif current_step == "final_video_gen":
            dprint("DP Chain: Step 4 (final_video_gen) complete. Finalizing process.")
            final_video_path = _get_abs_path_from_db_path(task_output_path, dprint)
            if not final_video_path:
                 return False, f"Could not resolve final video path from '{task_output_path}'", None
            
            if chain_details["debug_mode"]:
                chain_details["debug_video_stages"].append({
                    'label': f"5. Final Video ({final_video_path.name})",
                    'type': 'video', 'path': str(final_video_path)
                })

            # Extract the final frame
            print("\nDP Chain: Extracting final posed image...")
            final_posed_image_output_path = Path(chain_details["main_output_dir"]) / f"final_posed_image_{chain_details['run_id']}.png"
            
            if not save_frame_from_video(final_video_path, -1, final_posed_image_output_path, sm_parse_resolution(original_params.get("resolution"))):
                 return False, f"Failed to extract final posed image from {final_video_path}", None
            
            print(f"Successfully completed 'different_pose' task!")
            print(f"Final posed image saved to: {final_posed_image_output_path.resolve()}")
            final_path_for_db = str(final_posed_image_output_path.resolve()) # This is the final output of the whole job

            if chain_details["debug_mode"]:
                chain_details["debug_video_stages"].append({
                    'label': f"6. Final Extracted Image ({final_posed_image_output_path.name})",
                    'type': 'image', 'path': str(final_posed_image_output_path)
                })
                video_collage_path = Path(chain_details["main_output_dir"]) / f"debug_summary_dp_{chain_details['run_id']}.mp4"
                print(f"DP Chain: Generating debug video summary to {video_collage_path}")
                # This function might need adjustment for the new data structure
                # generate_different_pose_debug_video_summary(...)
            
            # Cleanup intermediate files
            if not chain_details["skip_cleanup"] and not chain_details["debug_mode"]:
                print(f"DP Chain: Cleaning up intermediate files in {work_dir}...")
                try:
                    shutil.rmtree(work_dir)
                    print(f"Removed intermediate directory: {work_dir}")
                except OSError as e_clean:
                    print(f"Error removing intermediate directory {work_dir}: {e_clean}")
            else:
                print(f"Skipping cleanup of intermediate files in {work_dir}.")
            
            # The chain is complete. We return the final output path.
            # This will be saved to the original orchestrator task's DB record.
            db_ops.update_task_status(chain_details['orchestrator_task_id'], db_ops.STATUS_COMPLETE, final_path_for_db)
            dprint(f"DP Chain: Process complete. Final image at {final_path_for_db}")
        
        else:
            return False, f"Unknown step in different_pose chaining: {current_step}", None
            
    except Exception as e:
        error_msg = f"Different Pose chaining failed at step '{current_step}': {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        db_ops.update_task_status(chain_details['orchestrator_task_id'], db_ops.STATUS_FAILED, error_msg)
        return False, error_msg, None

    # For all intermediate steps, we don't have a final output for the orchestrator task yet.
    return True, f"Chaining successful for step {current_step}", final_path_for_db 