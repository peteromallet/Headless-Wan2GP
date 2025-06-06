"""Specialized task handlers for headless.py."""

import traceback
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

# Add the parent directory to Python path to allow Wan2GP module import
import sys
wan2gp_path = Path(__file__).resolve().parent.parent / "Wan2GP"
if str(wan2gp_path) not in sys.path:
    sys.path.insert(0, str(wan2gp_path))

try:
    from preprocessing.dwpose.pose import PoseBodyFaceVideoAnnotator
except ImportError:
    PoseBodyFaceVideoAnnotator = None

from . import db_operations as db_ops
from .common_utils import sm_get_unique_target_path
from .common_utils import parse_resolution as sm_parse_resolution
from .video_utils import rife_interpolate_images_to_video as sm_rife_interpolate_images_to_video

def handle_generate_openpose_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str, dprint: callable):
    """Handles the 'generate_openpose' task."""
    print(f"[Task ID: {task_id}] Handling 'generate_openpose' task.")
    input_image_path_str = task_params_dict.get("input_image_path")
    suggested_output_image_path_str = task_params_dict.get("output_path")

    if PoseBodyFaceVideoAnnotator is None:
        msg = "PoseBodyFaceVideoAnnotator not imported. Cannot process 'generate_openpose' task."
        print(f"[ERROR Task ID: {task_id}] {msg}")
        return False, "PoseBodyFaceVideoAnnotator module not available."

    if not input_image_path_str:
        print(f"[ERROR Task ID: {task_id}] 'input_image_path' not specified for generate_openpose task.")
        return False, "Missing input_image_path"

    if not suggested_output_image_path_str:
        default_output_dir = main_output_dir_base / task_id
        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = default_output_dir / f"{task_id}_openpose.png"
        print(f"[WARNING Task ID: {task_id}] 'output_path' not specified. Defaulting to {output_image_path}")
    else:
        output_image_path = Path(suggested_output_image_path_str)

    input_image_path = Path(input_image_path_str)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_image_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image file not found: {input_image_path}")
        return False, f"Input image not found: {input_image_path}"

    final_save_path = output_image_path
    db_output_location = str(output_image_path.resolve())

    if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
        sqlite_db_file_path = Path(db_ops.SQLITE_DB_PATH).resolve()
        target_files_dir = sqlite_db_file_path.parent / "public" / "files"
        target_files_dir.mkdir(parents=True, exist_ok=True)

        file_stem = f"{task_id}_openpose"
        final_save_path = sm_get_unique_target_path(target_files_dir, file_stem, ".png")
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        db_output_location = f"files/{final_save_path.name}"
        dprint(f"[Task ID: {task_id}] SQLite mode: OpenPose will be saved to {final_save_path} (DB: {db_output_location})")

    try:
        pil_input_image = Image.open(input_image_path).convert("RGB")

        pose_cfg_dict = {
            "DETECTION_MODEL": "ckpts/pose/yolox_l.onnx",
            "POSE_MODEL": "ckpts/pose/dw-ll_ucoco_384.onnx",
            "RESIZE_SIZE": 1024
        }
        if PoseBodyFaceVideoAnnotator is None:
             raise ImportError("PoseBodyFaceVideoAnnotator could not be imported.")

        pose_annotator = PoseBodyFaceVideoAnnotator(pose_cfg_dict)

        openpose_np_frames_bgr = pose_annotator.forward([pil_input_image])

        if not openpose_np_frames_bgr or openpose_np_frames_bgr[0] is None:
            print(f"[ERROR Task ID: {task_id}] OpenPose generation failed or returned no frame.")
            return False, "OpenPose generation returned no data."

        openpose_np_frame_bgr = openpose_np_frames_bgr[0]

        openpose_pil_image = Image.fromarray(openpose_np_frame_bgr.astype(np.uint8))
        openpose_pil_image.save(final_save_path)

        print(f"[Task ID: {task_id}] Successfully generated OpenPose image to: {final_save_path.resolve()}")
        return True, db_output_location

    except ImportError as ie:
        print(f"[ERROR Task ID: {task_id}] Import error during OpenPose generation: {ie}. Ensure 'preprocessing' module is in PYTHONPATH and dependencies are installed.")
        traceback.print_exc()
        return False, f"Import error: {ie}"
    except FileNotFoundError as fnfe:
        print(f"[ERROR Task ID: {task_id}] ONNX model file not found for OpenPose: {fnfe}. Ensure 'ckpts/pose/*' models are present.")
        traceback.print_exc()
        return False, f"ONNX model not found: {fnfe}"
    except Exception as e:
        print(f"[ERROR Task ID: {task_id}] Failed during OpenPose image generation: {e}")
        traceback.print_exc()
        return False, f"OpenPose generation exception: {e}"


def handle_rife_interpolate_task(wgp_mod, task_params_dict: dict, main_output_dir_base: Path, task_id: str, dprint: callable):
    """Handles the 'rife_interpolate_images' task."""
    print(f"[Task ID: {task_id}] Handling 'rife_interpolate_images' task.")

    input_image_path1_str = task_params_dict.get("input_image_path1")
    input_image_path2_str = task_params_dict.get("input_image_path2")
    output_video_path_str = task_params_dict.get("output_path")
    num_rife_frames = task_params_dict.get("frames")
    resolution_str = task_params_dict.get("resolution")

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

    final_save_path_for_video = output_video_path
    db_output_location_for_rife = str(output_video_path.resolve())

    if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
        sqlite_db_file_path = Path(db_ops.SQLITE_DB_PATH).resolve()
        target_files_dir = sqlite_db_file_path.parent / "public" / "files"
        target_files_dir.mkdir(parents=True, exist_ok=True)

        file_stem = f"{task_id}_rife_interpolated"
        final_save_path_for_video = sm_get_unique_target_path(target_files_dir, file_stem, ".mp4")
        final_save_path_for_video.parent.mkdir(parents=True, exist_ok=True)
        db_output_location_for_rife = f"files/{final_save_path_for_video.name}"
        dprint(f"[Task ID: {task_id}] SQLite mode: RIFE video will be saved to {final_save_path_for_video} (DB: {db_output_location_for_rife})")
    else:
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
    original_wgp_save_path = wgp_mod.save_path
    wgp_mod.save_path = str(temp_output_dir)

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
                output_location_to_db = db_output_location_for_rife
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
        wgp_mod.save_path = original_wgp_save_path

    try:
        shutil.rmtree(temp_output_dir)
        dprint(f"[Task ID: {task_id}] Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e_clean:
        print(f"[WARNING Task ID: {task_id}] Failed to clean up temporary directory {temp_output_dir}: {e_clean}")


    return generation_success, output_location_to_db 