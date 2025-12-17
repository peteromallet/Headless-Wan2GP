"""Specialized task handlers for worker.py."""

import traceback
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from . import db_operations as db_ops
from .common_utils import sm_get_unique_target_path, parse_resolution as sm_parse_resolution, prepare_output_path, save_frame_from_video, report_orchestrator_failure, prepare_output_path_with_upload, upload_and_get_final_output_location
from .video_utils import rife_interpolate_images_to_video as sm_rife_interpolate_images_to_video
from .logging_utils import task_logger

def handle_extract_frame_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str, dprint: callable):
    """Handles the 'extract_frame' task."""
    task_logger.essential("Starting extract frame task", task_id=task_id)
    
    input_video_task_id = task_params_dict.get("input_video_task_id")
    frame_index = task_params_dict.get("frame_index", 0) # Default to first frame
    custom_output_dir = task_params_dict.get("output_dir")
    
    if not input_video_task_id:
        msg = f"Task {task_id}: Missing 'input_video_task_id' in payload."
        report_orchestrator_failure(task_params_dict, msg, dprint)
        return False, msg

    try:
        # Get the output path of the dependency task
        # Note: This is looking up the direct task output, not a dependency relationship
        video_path_from_db = db_ops.get_task_output_location_from_db(input_video_task_id)
        if not video_path_from_db:
            msg = f"Task {task_id}: Could not find output location for dependency task {input_video_task_id}."
            report_orchestrator_failure(task_params_dict, msg, dprint)
            return False, msg

        video_abs_path = db_ops.get_abs_path_from_db_path(video_path_from_db, dprint)
        if not video_abs_path:
            msg = f"Task {task_id}: Could not resolve or find video file from DB path '{video_path_from_db}'."
            report_orchestrator_failure(task_params_dict, msg, dprint)
            return False, msg

        # Use prepare_output_path_with_upload to determine the correct save location
        output_filename = f"{task_id}_frame_{frame_index}.png"
        final_save_path, initial_db_location = prepare_output_path_with_upload(
            task_id,
            output_filename,
            main_output_dir_base,
            task_type="extract_frame",
            dprint=dprint,
            custom_output_dir=custom_output_dir
        )

        # The resolution for save_frame_from_video can be inferred from the video itself
        # Or passed in the payload if a specific resize is needed. For now, we don't resize.
        cap = cv2.VideoCapture(str(video_abs_path))
        if not cap.isOpened():
            msg = f"Task {task_id}: Could not open video file {video_abs_path}"
            report_orchestrator_failure(task_params_dict, msg, dprint)
            return False, msg
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Now use the save_frame_from_video utility
        success = save_frame_from_video(
            video_path=video_abs_path,
            frame_index=frame_index,
            output_image_path=final_save_path,
            resolution=(width, height) # Use native resolution
        )
        
        if success:
            # Upload to Supabase if configured
            final_db_location = upload_and_get_final_output_location(
                final_save_path, task_id, initial_db_location, dprint=dprint
            )
            
            print(f"[Task ID: {task_id}] Successfully extracted frame {frame_index} to: {final_save_path}")
            return True, final_db_location
        else:
            msg = f"Task {task_id}: save_frame_from_video utility failed for video {video_abs_path}."
            report_orchestrator_failure(task_params_dict, msg, dprint)
            return False, msg
    
    except Exception as e:
        error_msg = f"Task {task_id}: Failed during frame extraction: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        report_orchestrator_failure(task_params_dict, error_msg, dprint)
        return False, str(e)

def handle_rife_interpolate_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str, dprint: callable, task_queue=None):
    """Handles the 'rife_interpolate_images' task."""
    task_logger.essential("Starting RIFE interpolation task", task_id=task_id)

    input_image_path1_str = task_params_dict.get("input_image_path1")
    input_image_path2_str = task_params_dict.get("input_image_path2")
    output_video_path_str = task_params_dict.get("output_path")
    num_rife_frames = task_params_dict.get("frames")
    resolution_str = task_params_dict.get("resolution")
    custom_output_dir = task_params_dict.get("output_dir")

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

    generation_success = False
    output_location_to_db = None

    final_save_path_for_video, initial_db_location_for_rife = prepare_output_path_with_upload(
        task_id,
        f"{task_id}_interpolated.mp4",
        main_output_dir_base,
        task_type="rife_interpolate_images",
        dprint=dprint,
        custom_output_dir=custom_output_dir
    )
    output_video_path = final_save_path_for_video
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    dprint(f"[Task ID: {task_id}] Checking input image paths.")
    if not input_image1_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image 1 not found: {input_image1_path}")
        return False, f"Input image 1 not found: {input_image1_path}"
    if not input_image2_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image 2 not found: {input_image2_path}")
        return False, f"Input image 2 not found: {input_image2_path}"
    dprint(f"[Task ID: {task_id}] Input images found.")

    temp_output_dir = tempfile.mkdtemp(prefix=f"wgp_rife_{task_id}_")
    # No longer using wgp_mod.save_path since we're using task_queue system

    try:
        pil_image_start = Image.open(input_image1_path).convert("RGB")
        pil_image_end = Image.open(input_image2_path).convert("RGB")

        print(f"[Task ID: {task_id}] Starting RIFE interpolation via video_utils.")
        dprint(f"  Input 1: {input_image1_path}")
        dprint(f"  Input 2: {input_image2_path}")

        rife_success = sm_rife_interpolate_images_to_video(
            image1=pil_image_start,
            image2=pil_image_end,
            num_frames=int(num_rife_frames),
            resolution_wh=sm_parse_resolution(resolution_str),
            output_path=final_save_path_for_video,
            fps=16,
            dprint_func=lambda msg: dprint(f"[Task ID: {task_id}] (rife_util) {msg}")
        )

        if rife_success:
            if final_save_path_for_video.exists() and final_save_path_for_video.stat().st_size > 0:
                generation_success = True
                # Upload to Supabase if configured
                output_location_to_db = upload_and_get_final_output_location(
                    final_save_path_for_video, task_id, initial_db_location_for_rife, dprint=dprint
                )
                print(f"[Task ID: {task_id}] RIFE video saved to: {final_save_path_for_video.resolve()} (DB: {output_location_to_db})")
            else:
                print(f"[ERROR Task ID: {task_id}] RIFE utility reported success, but output file is missing or empty: {final_save_path_for_video}")
                generation_success = False
        else:
            print(f"[ERROR Task ID: {task_id}] RIFE interpolation using video_utils failed.")
            generation_success = False

    except Exception as e:
        print(f"[ERROR Task ID: {task_id}] Overall _handle_rife_interpolate_task failed: {e}")
        traceback.print_exc()
        generation_success = False
    finally:
        # No longer need to restore wgp_mod.save_path since we're using task_queue system
        pass

    try:
        shutil.rmtree(temp_output_dir)
        dprint(f"[Task ID: {task_id}] Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e_clean:
        print(f"[WARNING Task ID: {task_id}] Failed to clean up temporary directory {temp_output_dir}: {e_clean}")


    return generation_success, output_location_to_db