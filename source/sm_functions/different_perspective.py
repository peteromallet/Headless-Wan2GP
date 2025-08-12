"""Different-perspective task handler."""

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
    generate_different_perspective_debug_video_summary,
    parse_resolution as sm_parse_resolution,
    create_simple_first_frame_mask_video,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location
)
from ..video_utils import rife_interpolate_images_to_video  # For depth guide interpolation

def _handle_different_perspective_orchestrator_task(task_params_from_db: dict, main_output_dir_base: Path, orchestrator_task_id_str: str, dprint):
    """
    This is the entry point for a 'different_perspective' job.
    It sets up the environment and enqueues all necessary child tasks with
    database-level dependencies.
    """
    print(f"--- Orchestrating Task: Different Perspective (ID: {orchestrator_task_id_str}) ---")
    dprint(f"Orchestrator Task Params: {task_params_from_db}")

    try:
        public_files_dir = Path.cwd() / "public" / "files"
        public_files_dir.mkdir(parents=True, exist_ok=True)
        main_output_dir_base = public_files_dir

        run_id = generate_unique_task_id("dp_run_")
        work_dir = main_output_dir_base / f"different_perspective_run_{run_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"Working directory for this 'different_perspective' run: {work_dir.resolve()}")

        task_id_user_pose = generate_unique_task_id("dp_user_pose_")
        task_id_t2i = generate_unique_task_id("dp_t2i_")
        task_id_extract = generate_unique_task_id("dp_extract_")
        task_id_t2i_pose = generate_unique_task_id("dp_t2i_pose_")
        task_id_final_gen = generate_unique_task_id("dp_final_gen_")

        perspective_type = task_params_from_db.get("perspective_type", "pose").lower()

        orchestrator_payload = {
            "run_id": run_id,
            "orchestrator_task_id": orchestrator_task_id_str,
            "work_dir": str(work_dir.resolve()),
            "main_output_dir": str(main_output_dir_base.resolve()),
            "original_params": task_params_from_db,
            "perspective_type": perspective_type,
            "task_ids": {
                "user_persp": task_id_user_pose,
                "t2i": task_id_t2i,
                "extract": task_id_extract,
                "t2i_persp": task_id_t2i_pose,
                "final_gen": task_id_final_gen,
            },
            "debug_mode": task_params_from_db.get("debug_mode", False),
            "skip_cleanup": task_params_from_db.get("skip_cleanup", False),
        }

        previous_task_id = None

        payload_user_persp = {
            "task_id": task_id_user_pose,
            "input_image_path": task_params_from_db['input_image_path'],
            "dp_orchestrator_payload": orchestrator_payload,
            "output_dir": str(work_dir.resolve()),
        }
        user_gen_task_type = "generate_openpose" if perspective_type == "pose" else "generate_depth"
        db_ops.add_task_to_db(payload_user_persp, user_gen_task_type, dependant_on=previous_task_id)
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued {user_gen_task_type} ({task_id_user_pose})")
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

        payload_t2i_persp = {
            "task_id": task_id_t2i_pose,
            "input_image_task_id": task_id_extract,
            "dp_orchestrator_payload": orchestrator_payload,
            "output_dir": str(work_dir.resolve()),
        }
        db_ops.add_task_to_db(payload_t2i_persp, user_gen_task_type, dependant_on=previous_task_id)
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued {user_gen_task_type} ({task_id_t2i_pose})")
        previous_task_id = task_id_t2i_pose

        payload_final_gen = {
            "task_id": task_id_final_gen,
            "dp_orchestrator_payload": orchestrator_payload,
        }
        db_ops.add_task_to_db(payload_final_gen, "dp_final_gen", dependant_on=previous_task_id)
        dprint(f"Orchestrator {orchestrator_task_id_str} enqueued dp_final_gen ({task_id_final_gen})")

        return True, f"Successfully enqueued different_perspective job graph with run_id {run_id}."
    except Exception as e:
        error_msg = f"Different Perspective orchestration failed: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return False, error_msg


# -----------------------------------------------------------------------------
# Final generation step – run WGP inline (no DB queue / poll)
# -----------------------------------------------------------------------------
# We refactor this routine so that it behaves like the travel pipeline: the
# heavy WGP call is executed synchronously via ``process_single_task`` instead
# of being queued as a separate DB task and then polled for completion.  This
# avoids the single-worker dead-lock where the same process waits for a task it
# is supposed to execute itself.

# NOTE: The headless server will pass ``wgp_mod`` (imported Wan2GP module), the
# current ``main_output_dir_base`` and a reference to its own
# ``process_single_task`` so that we can leverage the existing wrapper.

def _handle_dp_final_gen_task(
    *,
    main_output_dir_base: Path,
    process_single_task,  # recursive call helper supplied by worker.py
    task_params_from_db: dict,
    dprint,
    task_queue=None,
):
    """
    Handles the final step of the 'different_perspective' process. It gathers all
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
        perspective_type = orchestrator_payload.get("perspective_type", "pose").lower()

        user_persp_id = task_ids["user_persp"]
        t2i_persp_id = task_ids["t2i_persp"]

        user_persp_path_db = db_ops.get_task_output_location_from_db(user_persp_id)
        t2i_persp_path_db = db_ops.get_task_output_location_from_db(t2i_persp_id)

        user_persp_image_path = db_ops.get_abs_path_from_db_path(user_persp_path_db, dprint)
        t2i_persp_image_path = db_ops.get_abs_path_from_db_path(t2i_persp_path_db, dprint)

        if not all([user_persp_image_path, t2i_persp_image_path, t2i_image_path]):
            return False, "Could not resolve one or more required image paths from previous tasks.", None

        print("\nDP Final Gen: Creating custom guide video…")
        custom_guide_video_path = work_dir / f"{generate_unique_task_id('dp_custom_guide_')}.mp4"
        
        if perspective_type == "pose":
            create_pose_interpolated_guide_video(
                output_video_path=custom_guide_video_path,
                resolution=sm_parse_resolution(original_params.get("resolution")),
                total_frames=original_params.get("output_video_frames", 16),
                start_image_path=Path(original_params['input_image_path']),
                end_image_path=t2i_image_path,
                fps=original_params.get("fps_helpers", 16),
            )
            print(f"DP Final Gen: Pose guide video created: {custom_guide_video_path}")
        else:  # depth
            # Use RIFE to interpolate between depth maps
            try:
                img_start = Image.open(user_persp_image_path).convert("RGB")
                img_end = Image.open(t2i_persp_image_path).convert("RGB")

                rife_success = rife_interpolate_images_to_video(
                    image1=img_start,
                    image2=img_end,
                    num_frames=original_params.get("output_video_frames", 16),
                    resolution_wh=sm_parse_resolution(original_params.get("resolution")),
                    output_path=custom_guide_video_path,
                    fps=original_params.get("fps_helpers", 16),
                    dprint_func=dprint,
                )
                if not rife_success:
                    return False, "Failed to create depth interpolated guide video via RIFE", None
                print(f"DP Final Gen: Depth guide video created: {custom_guide_video_path}")
            except Exception as e_gv:
                print(f"[ERROR] DP Final Gen: Depth guide video creation failed: {e_gv}")
                traceback.print_exc()
                return False, f"Guide video creation failed: {e_gv}", None

        # ------------------------------------------------------------------
        # 2. Prepare WGP inline generation payload
        # ------------------------------------------------------------------

        final_video_task_id = generate_unique_task_id("dp_final_video_")

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
            # ensure outputs stay inside the run work dir
            "output_dir": str(work_dir.resolve()),
        }

        # Optionally create a mask video to freeze first frame
        if original_params.get("use_mask_for_first_frame", True):
            dprint("DP Final Gen: Creating mask video to preserve first frame...")
            mask_video_path = work_dir / f"{generate_unique_task_id('dp_mask_')}.mp4"

            created_mask = create_simple_first_frame_mask_video(
                total_frames=original_params.get("output_video_frames", 16),
                resolution_wh=sm_parse_resolution(original_params.get("resolution")),
                output_path=mask_video_path,
                fps=original_params.get("fps_helpers", 16),
                task_id_for_logging=final_video_task_id,
                dprint=dprint
            )

            if created_mask:
                final_video_payload["video_mask_path"] = str(created_mask.resolve())
                final_video_payload["video_prompt_type"] = "IVM"  # Image + Video guide + Mask
                dprint(f"DP Final Gen: Mask video created at {created_mask}")
            else:
                dprint("DP Final Gen: Warning - Failed to create mask video, proceeding without mask")

        # ------------------------------------------------------------------
        # 3. Execute WGP synchronously (no DB queue / polling)
        # ------------------------------------------------------------------

        print("\nDP Final Gen: Launching inline WGP generation for final video…")

        generation_success, final_video_output_db = process_single_task(
            final_video_payload,
            main_output_dir_base,
            "wgp",
            project_id_for_task=original_params.get("project_id"),
            image_download_dir=None,
            apply_reward_lora=False,
            colour_match_videos=False,
            mask_active_frames=True,
            task_queue=task_queue
        )

        if not generation_success:
            return False, "Final video generation failed.", None

        final_video_path = db_ops.get_abs_path_from_db_path(final_video_output_db, dprint)
        if not final_video_path:
            return False, f"Could not resolve final video path from '{final_video_output_db}'", None

        print("\nDP Final Gen: Extracting final posed image...")
        
        # Use prepare_output_path_with_upload for Supabase-compatible output handling
        final_image_filename = f"final_posed_image_{orchestrator_payload['run_id']}.png"
        final_posed_image_output_path, initial_db_location = prepare_output_path_with_upload(
            task_id=payload.get("task_id", "dp_final_gen"),
            filename=final_image_filename,
            main_output_dir_base=Path(orchestrator_payload["main_output_dir"]),
            dprint=dprint
        )
        
        if not save_frame_from_video(final_video_path, -1, final_posed_image_output_path, sm_parse_resolution(original_params.get("resolution"))):
             return False, f"Failed to extract final posed image from {final_video_path}", None
        
        # Handle Supabase upload (if configured) and get final location for DB
        final_path_for_db = upload_and_get_final_output_location(
            final_posed_image_output_path,
            final_image_filename,  # Pass only the filename to avoid redundant subfolder
            initial_db_location,
            dprint=dprint
        )
        
        print(f"Successfully completed 'different_perspective' task! Final image: {final_posed_image_output_path.resolve()} (DB location: {final_path_for_db})")

        # Preserve intermediates when either the orchestrator payload says so *or*
        # the headless server is running with --debug (exposed via db_ops.debug_mode).
        if (not orchestrator_payload.get("skip_cleanup") and
            not orchestrator_payload.get("debug_mode") and
            not db_ops.debug_mode):
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