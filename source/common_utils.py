"""Common utility functions and constants for steerable_motion tasks."""

import json
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Generator

import cv2 # pip install opencv-python
import mediapipe as mp # pip install mediapipe
import numpy as np # pip install numpy
from PIL import Image, ImageDraw, ImageFont, ImageEnhance # pip install Pillow, ensure ImageEnhance is imported
import requests # Added for downloads
from urllib.parse import urlparse # Added for URL parsing
import urllib.parse

# --- Global Debug Mode ---
# This will be set by the main script (steerable_motion.py)
DEBUG_MODE = False

# --- Constants for DB interaction and defaults ---
STATUS_QUEUED = "Queued"
STATUS_IN_PROGRESS = "In Progress"
STATUS_COMPLETE = "Complete"
STATUS_FAILED = "Failed"
DEFAULT_DB_TABLE_NAME = "tasks"
# DEFAULT_MODEL_NAME = "vace_14B" # Defined in steerable_motion.py's argparser
# DEFAULT_SEGMENT_FRAMES = 81    # Defined in steerable_motion.py's argparser
# DEFAULT_FPS_HELPERS = 25       # Defined in steerable_motion.py's argparser
# DEFAULT_SEED = 12345           # Defined in steerable_motion.py's argparser

# --- Debug / Verbose Logging Helper ---
def dprint(msg: str):
    """Print a debug message if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(f"[DEBUG SM-COMMON {datetime.utcnow().isoformat()}Z] {msg}")

# --- Helper Functions ---

def get_lora_dir_from_filename(wgp_mod, model_filename: str) -> str:
    """
    Centralized helper to get LoRA directory from model filename.
    Properly converts model_filename to model_type before calling get_lora_dir.
    
    Args:
        wgp_mod: WGP module instance
        model_filename: Model filename or model type
        
    Returns:
        LoRA directory path
        
    Raises:
        Exception: If model type cannot be determined
    """
    # If it's already a model type (not a filename), use directly
    if model_filename and not model_filename.endswith('.safetensors'):
        model_type = model_filename
    else:
        # Convert filename to model type
        model_type = wgp_mod.get_model_type(model_filename) if model_filename else None
        
    if not model_type:
        print(f"[WARNING] Could not determine model type from: {model_filename}")
        model_type = "t2v"  # Default fallback - should match t2v.json
        
    print(f"[DEBUG] LoRA directory lookup: {model_filename} → {model_type}")
    return wgp_mod.get_lora_dir(model_type)

def snap_resolution_to_model_grid(parsed_res: tuple[int, int]) -> tuple[int, int]:
    """
    Snaps resolution to model grid requirements (multiples of 16).
    
    Args:
        parsed_res: (width, height) tuple
        
    Returns:
        (width, height) tuple snapped to nearest valid values
    """
    width, height = parsed_res
    # Ensure resolution is compatible with model requirements (multiples of 16)
    width = (width // 16) * 16
    height = (height // 16) * 16
    return width, height

def ensure_valid_prompt(prompt: str | None) -> str:
    """
    Ensures prompt is valid (not None or empty), returns space as default.
    
    Args:
        prompt: Input prompt string or None
        
    Returns:
        Valid prompt string (space if input was None/empty)
    """
    if not prompt or not prompt.strip():
        return " "
    return prompt.strip()

def ensure_valid_negative_prompt(negative_prompt: str | None) -> str:
    """
    Ensures negative prompt is valid (not None), returns space as default.
    
    Args:
        negative_prompt: Input negative prompt string or None
        
    Returns:
        Valid negative prompt string (space if input was None/empty)
    """
    if not negative_prompt or not negative_prompt.strip():
        return " "
    return negative_prompt.strip()

def process_additional_loras_shared(
    additional_loras: dict[str, float | str], 
    wgp_mod: Any, 
    model_filename_for_task: str, 
    task_id: str, 
    dprint: callable
) -> dict[str, str]:
    """
    Shared function to process additional LoRAs for any task handler.
    
    Args:
        additional_loras: Dict mapping LoRA paths to strength values
        wgp_mod: WGP module instance
        model_filename_for_task: Model filename for LoRA directory lookup
        task_id: Task ID for logging
        dprint: Debug print function
        
    Returns:
        Dict mapping LoRA filenames to strength strings
    """
    processed_additional_loras: dict[str, str] = {}
    
    if not additional_loras:
        return processed_additional_loras
    
    try:
        import urllib.parse
        base_lora_dir_for_model = Path(get_lora_dir_from_filename(wgp_mod, model_filename_for_task))
        base_lora_dir_for_model.mkdir(parents=True, exist_ok=True)
        wan2gp_lora_root = Path("Wan2GP/loras")
        wan2gp_lora_root.mkdir(parents=True, exist_ok=True)
        
        for lora_path, lora_strength in additional_loras.items():
            try:
                # Derive filename
                if lora_path.startswith("http://") or lora_path.startswith("https://"):
                    lora_filename = Path(urllib.parse.urlparse(lora_path).path).name
                else:
                    lora_filename = Path(lora_path).name
                
                target_path_in_model_dir = base_lora_dir_for_model / lora_filename
                
                # --------------------------------------------------
                # 1) Handle remote URL – download if not already cached
                # --------------------------------------------------
                if lora_path.startswith("http://") or lora_path.startswith("https://"):
                    dl_target_path = wan2gp_lora_root / lora_filename

                    # Some LoRA hubs publish weights under extremely generic
                    # names (e.g. "pytorch_model.bin", "model.safetensors").
                    # If we already have a file with that generic name cached
                    # in Wan2GP/loras **we reuse it** instead of downloading a
                    # potentially different LoRA into the same filename which
                    # would be ambiguous.
                    GENERIC_NAMES = {
                        "pytorch_model.bin",
                        "pytorch.bin",
                        "model.safetensors",
                        "pytorch_model.safetensors",
                        "diffusion_pytorch_model.safetensors",
                    }

                    if lora_filename in GENERIC_NAMES and dl_target_path.exists():
                        dprint(f"Task {task_id}: Generic-named LoRA '{lora_filename}' already present – using cached copy {dl_target_path} instead of re-downloading.")
                    else:
                        if not dl_target_path.exists():
                            dprint(f"Task {task_id}: Downloading LoRA from {lora_path} to {dl_target_path}")
                            try:
                                download_file(lora_path, wan2gp_lora_root, lora_filename)
                            except Exception as e_dl:
                                dprint(f"Task {task_id}: ERROR – failed to download LoRA {lora_path}: {e_dl}")
                                continue  # Skip this LoRA
                        else:
                            dprint(f"Task {task_id}: LoRA already downloaded at {dl_target_path}")
                    
                    # Copy into model-specific dir if needed
                    if not target_path_in_model_dir.exists():
                        try:
                            shutil.copy(str(dl_target_path), str(target_path_in_model_dir))
                            dprint(f"Task {task_id}: Copied LoRA to model dir {target_path_in_model_dir}")
                        except Exception as e_cp:
                            dprint(f"Task {task_id}: WARNING – could not copy LoRA to model dir: {e_cp}")
                else:
                    # --------------------------------------------------
                    # 2) Handle local / relative path
                    # --------------------------------------------------
                    src_path = Path(lora_path)
                    if not src_path.is_absolute():
                        # Try relative to CWD
                        src_path = Path.cwd() / src_path
                    if not src_path.exists():
                        dprint(f"Task {task_id}: WARNING – local LoRA path not found: {src_path}")
                        continue
                    # Ensure LoRA is available inside model dir
                    if not target_path_in_model_dir.exists() or src_path.resolve() != target_path_in_model_dir.resolve():
                        try:
                            shutil.copy(str(src_path), str(target_path_in_model_dir))
                            dprint(f"Task {task_id}: Copied local LoRA {src_path} to {target_path_in_model_dir}")
                        except Exception as e_cploc:
                            dprint(f"Task {task_id}: WARNING – could not copy local LoRA to model dir: {e_cploc}")
                
                # Register for WGP payload
                processed_additional_loras[lora_filename] = str(lora_strength)
            except Exception as e_lora_iter:
                dprint(f"Task {task_id}: Error processing LoRA entry '{lora_path}': {e_lora_iter}")
    except Exception as e_outer:
        dprint(f"Task {task_id}: Error during additional LoRA processing: {e_outer}")
    
    return processed_additional_loras

def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parses 'WIDTHxHEIGHT' string to (width, height) tuple."""
    try:
        w, h = map(int, res_str.split('x'))
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive.")
        return w, h
    except ValueError as e:
        raise ValueError(f"Resolution string must be in WIDTHxHEIGHT format with positive integers (e.g., '960x544'), got {res_str}. Error: {e}")

def generate_unique_task_id(prefix: str = "") -> str:
    """Generates a UUID4 string.

    The optional *prefix* parameter is now ignored so that the returned value
    is a bare RFC-4122 UUID which can be stored in a Postgres `uuid` column
    without casting errors.  The argument is kept in the signature to avoid
    breaking existing call-sites that still pass a prefix.
    """
    return str(uuid.uuid4())

def image_to_frame(image_path: str | Path, target_size: tuple[int, int]) -> np.ndarray | None:
    """Loads an image, resizes it, and converts to BGR NumPy array for OpenCV."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading or processing image {image_path}: {e}")
        return None

def create_color_frame(size: tuple[int, int], color_bgr: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Creates a single color BGR frame (default black)."""
    height, width = size[1], size[0] # size is (width, height)
    frame = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    return frame

def get_easing_function(name: str):
    """
    Returns an easing function by name.
    """
    if name == 'linear':
        return lambda t: t
    elif name == 'ease_in_quad':
        return lambda t: t * t
    elif name == 'ease_out_quad':
        return lambda t: t * (2 - t)
    elif name == 'ease_in_out_quad' or name == 'ease_in_out': # Added alias
        return lambda t: 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
    elif name == 'ease_in_cubic':
        return lambda t: t * t * t
    elif name == 'ease_out_cubic':
        return lambda t: 1 + ((t - 1) ** 3)
    elif name == 'ease_in_out_cubic':
        return lambda t: 4 * t * t * t if t < 0.5 else 1 + ((2 * t - 2) ** 3) / 2
    elif name == 'ease_in_quart':
        return lambda t: t * t * t * t
    elif name == 'ease_out_quart':
        return lambda t: 1 - ((t - 1) ** 4)
    elif name == 'ease_in_out_quart':
        return lambda t: 8 * t * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 4) / 2
    elif name == 'ease_in_quint':
        return lambda t: t * t * t * t * t
    elif name == 'ease_out_quint':
        return lambda t: 1 + ((t - 1) ** 5)
    elif name == 'ease_in_out_quint':
        return lambda t: 16 * t * t * t * t * t if t < 0.5 else 1 + ((-2 * t + 2) ** 5) / 2
    elif name == 'ease_in_sine':
        return lambda t: 1 - math.cos(t * math.pi / 2)
    elif name == 'ease_out_sine':
        return lambda t: math.sin(t * math.pi / 2)
    elif name == 'ease_in_out_sine':
        return lambda t: -(math.cos(math.pi * t) - 1) / 2
    elif name == 'ease_in_expo':
        return lambda t: 0 if t == 0 else 2 ** (10 * (t - 1))
    elif name == 'ease_out_expo':
        return lambda t: 1 if t == 1 else 1 - 2 ** (-10 * t)
    elif name == 'ease_in_out_expo':
        if t == 0: return 0
        if t == 1: return 1
        if t < 0.5:
            return (2 ** (20 * t - 10)) / 2
        else:
            return (2 - 2 ** (-20 * t + 10)) / 2
    else: # Default to ease_in_out
        return lambda t: 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

def create_video_from_frames_list(
    frames_list: list[np.ndarray],
    output_path: str | Path,
    fps: int,
    resolution: tuple[int, int] # width, height
):
    """Creates an MP4 video from a list of NumPy BGR frames."""
    output_path_obj = Path(output_path)
    output_path_mp4 = output_path_obj.with_suffix('.mp4')
    output_path_mp4.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{resolution[0]}x{resolution[1]}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-vf", "format=yuv420p,colorspace=bt709:iall=bt709:fast=1",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        str(output_path_mp4.resolve())
    ]

    processed_frames = []
    for frame_idx, frame_np in enumerate(frames_list):
        if frame_np is None or not isinstance(frame_np, np.ndarray):
            continue
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)
        if frame_np.shape[0] != resolution[1] or frame_np.shape[1] != resolution[0] or frame_np.shape[2] != 3:
            try:
                frame_np = cv2.resize(frame_np, resolution, interpolation=cv2.INTER_AREA)
            except Exception:
                continue
        processed_frames.append(frame_np)

    if not processed_frames:
        print(f"No valid frames to write for {output_path_mp4}")
        return None

    try:
        raw_video_data = b''.join(frame.tobytes() for frame in processed_frames)
    except Exception as e:
        print(f"Error preparing video data: {e}")
        return None

    if not raw_video_data:
        print(f"No video data to write for {output_path_mp4}")
        return None

    try:
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        process.communicate(input=raw_video_data)
        if process.returncode != 0:
            print(f"FFmpeg failed with return code {process.returncode}")
            return None
        return output_path_mp4
    except Exception as e:
        print(f"Error during FFmpeg encoding: {e}")
        return None

def add_task_to_db(task_payload: dict, db_path: str | Path, task_type_str: str, dependant_on: str | None = None):
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        # Ensure task_type is not in the params dict as it's now a separate column
        # This is important if the caller still includes it in task_payload by old habit.
        # The task_payload itself is still stored in the 'params' column.
        # The task_id is expected to be in task_payload.
        # The task_id for the DB record should be consistent.
        # If task_payload (which becomes params) already has a task_id, use that.
        # Otherwise, a task_id needs to be generated/provided.
        # The original line was: task_payload["task_id"]
        # This means the task_id to be inserted was expected to be part of the task_payload dict.
        
        # Let's assume the task_id passed to this function (if any) or generated by it
        # is the PRIMARY KEY for the DB.
        # The `task_payload` argument to this function is what gets stored in the `params` column.

        current_task_id = task_payload.get("task_id") # This is the task_id from the original payload structure.
        if not current_task_id:
            # This case should ideally not happen if `steerable_motion.py` prepares `task_payload` correctly.
            # For safety, one might generate one, but it's better to ensure the caller provides it.
            # Sticking to the original structure where task_payload["task_id"] was used:
            raise ValueError("task_id must be present in task_payload for add_task_to_db")

        headless_params_dict = task_payload.copy() # Work on a copy
        if "task_type" in headless_params_dict:
            del headless_params_dict["task_type"] # Remove if it exists, to avoid redundancy

        # The task_id for the DB record should be consistent.
        # If task_payload (which becomes params) already has a task_id, use that.
        # Otherwise, a task_id needs to be generated/provided.
        # The original line was: task_payload["task_id"]
        # This means the task_id to be inserted was expected to be part of the task_payload dict.
        
        # Let's assume the task_id passed to this function (if any) or generated by it
        # is the PRIMARY KEY for the DB.
        # The `task_payload` argument to this function is what gets stored in the `params` column.

        params_json_for_db = json.dumps(headless_params_dict)
        current_timestamp = datetime.utcnow().isoformat() + "Z"
        project_id = task_payload.get("project_id", "default_project_id") # Get project_id or use default

        cursor.execute(
            f"INSERT INTO {DEFAULT_DB_TABLE_NAME} (id, params, task_type, status, created_at, project_id, dependant_on) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (current_task_id, params_json_for_db, task_type_str, STATUS_QUEUED, current_timestamp, project_id, dependant_on)
        )
        conn.commit()
        print(f"Task {current_task_id} (Type: {task_type_str}) added to database {db_path}.")
    except sqlite3.Error as e:
        # Use current_task_id in error message if available
        task_id_for_error = task_payload.get("task_id", "UNKNOWN_TASK_ID")
        print(f"SQLite error when adding task {task_id_for_error} (Type: {task_type_str}): {e}")
        raise
    finally:
        conn.close()

def poll_task_status(task_id: str, db_path: str | Path, poll_interval_seconds: int = 10, timeout_seconds: int = 1800) -> str | None:
    """Polls the DB for task completion and returns the output_location."""
    print(f"Polling for completion of task {task_id} (timeout: {timeout_seconds}s)...")
    start_time = time.time()
    last_status_print_time = 0
    
    while True:
        current_time = time.time()
        if current_time - start_time > timeout_seconds:
            print(f"Error: Timeout polling for task {task_id} after {timeout_seconds} seconds.")
            return None

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT status, output_location FROM {DEFAULT_DB_TABLE_NAME} WHERE id = ?", (task_id,))
            row = cursor.fetchone()
        except sqlite3.Error as e:
            print(f"SQLite error while polling task {task_id}: {e}. Retrying...")
            conn.close()
            time.sleep(min(poll_interval_seconds, 5)) # Shorter sleep on DB error
            continue
        finally:
            conn.close()

        if row:
            status = row["status"]
            output_location = row["output_location"]
            
            if current_time - last_status_print_time > poll_interval_seconds * 2 : # Print status periodically
                 print(f"Task {task_id}: Status = {status} (Output: {output_location if output_location else 'N/A'})")
                 last_status_print_time = current_time

            if status == STATUS_COMPLETE:
                if output_location:
                    print(f"Task {task_id} completed successfully. Output: {output_location}")
                    return output_location
                else:
                    print(f"Error: Task {task_id} is COMPLETE but output_location is missing. Assuming failure.")
                    return None
            elif status == STATUS_FAILED:
                print(f"Error: Task {task_id} failed.")
                return None
            elif status not in [STATUS_QUEUED, STATUS_IN_PROGRESS]:
                print(f"Warning: Task {task_id} has unknown status '{status}'. Treating as error.")
                return None
        else:
            if current_time - last_status_print_time > poll_interval_seconds * 2 :
                print(f"Task {task_id}: Not found in DB yet or status pending...")
                last_status_print_time = current_time

        time.sleep(poll_interval_seconds)

def extract_video_segment_ffmpeg(
    input_video_path: str | Path,
    output_video_path: str | Path,
    start_frame_index: int, # 0-indexed
    num_frames_to_keep: int,
    input_fps: float, # FPS of the input video for accurate -ss calculation
    resolution: tuple[int,int] 
):
    """Extracts a video segment using FFmpeg with stream copy if possible."""
    dprint(f"EXTRACT_VIDEO_SEGMENT_FFMPEG: Called with input='{input_video_path}', output='{output_video_path}', start_idx={start_frame_index}, num_frames={num_frames_to_keep}, input_fps={input_fps}")
    if num_frames_to_keep <= 0:
        print(f"Warning: num_frames_to_keep is {num_frames_to_keep} for {output_video_path} (FFmpeg). Nothing to extract.")
        dprint("EXTRACT_VIDEO_SEGMENT_FFMPEG: num_frames_to_keep is 0 or less, returning.")
        Path(output_video_path).unlink(missing_ok=True)
        return

    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    start_time_seconds = start_frame_index / input_fps

    cmd = [
        'ffmpeg',
        '-y',  
        '-ss', str(start_time_seconds),
        '-i', str(input_video_path.resolve()),
        '-vframes', str(num_frames_to_keep),
        '-an', 
        str(output_video_path.resolve())
    ]

    dprint(f"EXTRACT_VIDEO_SEGMENT_FFMPEG: Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        dprint(f"EXTRACT_VIDEO_SEGMENT_FFMPEG: Successfully extracted segment to {output_video_path}")
        if process.stderr:
            dprint(f"FFmpeg stderr (for {output_video_path}):\n{process.stderr}")
        if not output_video_path.exists() or output_video_path.stat().st_size == 0:
            print(f"Error: FFmpeg command for {output_video_path} apparently succeeded but output file is missing or empty.")
            dprint(f"FFmpeg command for {output_video_path} produced no output. stdout:\n{process.stdout}\nstderr:\n{process.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg segment extraction for {output_video_path}:")
        print("FFmpeg command:", ' '.join(e.cmd))
        if e.stdout: print("FFmpeg stdout:\n", e.stdout)
        if e.stderr: print("FFmpeg stderr:\n", e.stderr)
        dprint(f"FFmpeg extraction failed for {output_video_path}. Error: {e}")
        Path(output_video_path).unlink(missing_ok=True)
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        dprint("FFmpeg command not found during segment extraction.")
        raise

def stitch_videos_ffmpeg(video_paths_list: list[str | Path], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_paths_list:
        print("No videos to stitch.")
        return

    valid_video_paths = []
    for p in video_paths_list:
        resolved_p = Path(p).resolve()
        if resolved_p.exists() and resolved_p.stat().st_size > 0:
            valid_video_paths.append(resolved_p)
        else:
            print(f"Warning: Video segment {resolved_p} is missing or empty. Skipping from stitch list.")
    
    if not valid_video_paths:
        print("No valid video segments found to stitch after checks.")
        return

    with tempfile.TemporaryDirectory(prefix="ffmpeg_concat_") as tmpdir:
        filelist_path = Path(tmpdir) / "ffmpeg_filelist.txt"
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for video_path in valid_video_paths:
                f.write(f"file '{video_path.as_posix()}'\n")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(filelist_path),
            '-c', 'copy', str(output_path)
        ]
        
        print(f"Running ffmpeg to stitch videos: {' '.join(cmd)}")
        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            print(f"Successfully stitched videos into: {output_path}")
            if process.stderr: print("FFmpeg log (stderr):\n", process.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error during ffmpeg stitching for {output_path}:")
            print("FFmpeg command:", ' '.join(e.cmd))
            if e.stdout: print("FFmpeg stdout:\n", e.stdout)
            if e.stderr: print("FFmpeg stderr:\n", e.stderr)
            raise 
        except FileNotFoundError:
            print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
            raise

def save_frame_from_video(video_path: Path, frame_index: int, output_image_path: Path, resolution: tuple[int, int]):
    """Extracts a specific frame from a video, resizes, and saves it as an image."""
    dprint(f"SAVE_FRAME_FROM_VIDEO: Input='{video_path}', Index={frame_index}, Output='{output_image_path}', Res={resolution}")
    if not video_path.exists() or video_path.stat().st_size == 0:
        print(f"Error: Video file for frame extraction not found or empty: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Allow Python‐style negative indexing (e.g. -1 for last frame)
    if frame_index < 0:
        frame_index = total_frames + frame_index  # Convert to positive index

    if frame_index < 0 or frame_index >= total_frames:
        print(f"Error: Frame index {frame_index} is out of bounds for video {video_path} (total frames: {total_frames}).")
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"Error: Could not read frame {frame_index} from {video_path}.")
        return False

    try:
        if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
            dprint(f"SAVE_FRAME_FROM_VIDEO: Resizing frame from {frame.shape[:2]} to {resolution[:2][::-1]}")
            frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
        
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), frame)
        print(f"Successfully saved frame {frame_index} from {video_path} to {output_image_path}")
        return True
    except Exception as e:
        print(f"Error saving frame to {output_image_path}: {e}")
        traceback.print_exc()
        return False

# --- FFMPEG-based specific frame extraction ---
def extract_specific_frame_ffmpeg(
    input_video_path: str | Path,
    frame_number: int, # 0-indexed
    output_image_path: str | Path,
    input_fps: float # Passed by caller, though not strictly needed for ffmpeg frame index selection using 'eq(n,frame_number)'
):
    """Extracts a specific frame from a video using FFmpeg and saves it as an image."""
    dprint(f"EXTRACT_SPECIFIC_FRAME_FFMPEG: Input='{input_video_path}', Frame={frame_number}, Output='{output_image_path}'")
    input_video_p = Path(input_video_path)
    output_image_p = Path(output_image_path)
    output_image_p.parent.mkdir(parents=True, exist_ok=True)

    if not input_video_p.exists() or input_video_p.stat().st_size == 0:
        print(f"Error: Input video for frame extraction not found or empty: {input_video_p}")
        dprint(f"EXTRACT_SPECIFIC_FRAME_FFMPEG: Input video {input_video_p} not found or empty. Returning False.")
        return False

    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output without asking
        '-i', str(input_video_p.resolve()),
        '-vf', f"select=eq(n\,{frame_number})", # Escaped comma for ffmpeg filter syntax
        '-vframes', '1',
        str(output_image_p.resolve())
    ]

    dprint(f"EXTRACT_SPECIFIC_FRAME_FFMPEG: Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        dprint(f"EXTRACT_SPECIFIC_FRAME_FFMPEG: Successfully extracted frame {frame_number} to {output_image_p}")
        if process.stderr:
            dprint(f"FFmpeg stderr (for frame extraction to {output_image_p}):\n{process.stderr}")
        if not output_image_p.exists() or output_image_p.stat().st_size == 0:
            print(f"Error: FFmpeg command for frame extraction to {output_image_p} apparently succeeded but output file is missing or empty.")
            dprint(f"FFmpeg command for {output_image_p} (frame extraction) produced no output. stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg frame extraction for {output_image_p}:")
        print("FFmpeg command:", ' '.join(e.cmd))
        if e.stdout: print("FFmpeg stdout:\n", e.stdout)
        if e.stderr: print("FFmpeg stderr:\n", e.stderr)
        dprint(f"FFmpeg frame extraction failed for {output_image_p}. Error: {e}")
        if output_image_p.exists(): output_image_p.unlink(missing_ok=True)
        return False
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        dprint("FFmpeg command not found during frame extraction.")
        raise

# --- FFMPEG-based video concatenation (alternative to stitch_videos_ffmpeg if caller manages temp dir) ---
def concatenate_videos_ffmpeg(
    video_paths: list[str | Path],
    output_path: str | Path,
    temp_dir_for_list: str | Path # Directory where the list file will be created
):
    """Concatenates multiple video files into one using FFmpeg, using a provided temp directory for the list file."""
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    temp_dir_p = Path(temp_dir_for_list)
    temp_dir_p.mkdir(parents=True, exist_ok=True)

    if not video_paths:
        print("No videos to concatenate.")
        dprint("CONCATENATE_VIDEOS_FFMPEG: No video paths provided. Returning.")
        if output_p.exists(): output_p.unlink(missing_ok=True)
        return

    valid_video_paths = []
    for p_item in video_paths:
        resolved_p_item = Path(p_item).resolve()
        if resolved_p_item.exists() and resolved_p_item.stat().st_size > 0:
            valid_video_paths.append(resolved_p_item)
        else:
            print(f"Warning: Video segment {resolved_p_item} for concatenation is missing or empty. Skipping.")
            dprint(f"CONCATENATE_VIDEOS_FFMPEG: Skipping invalid video segment {resolved_p_item}")
    
    if not valid_video_paths:
        print("No valid video segments found to concatenate after checks.")
        dprint("CONCATENATE_VIDEOS_FFMPEG: No valid video segments. Returning.")
        if output_p.exists(): output_p.unlink(missing_ok=True)
        return

    filelist_path = temp_dir_p / "ffmpeg_concat_filelist.txt"
    with open(filelist_path, 'w', encoding='utf-8') as f:
        for video_path_item in valid_video_paths:
            f.write(f"file '{video_path_item.as_posix()}'\n") # Use as_posix() for ffmpeg list file
    
    cmd = [
        'ffmpeg', '-y', 
        '-f', 'concat', 
        '-safe', '0', 
        '-i', str(filelist_path.resolve()),
        '-c', 'copy', 
        str(output_p.resolve())
    ]
    
    dprint(f"CONCATENATE_VIDEOS_FFMPEG: Running command: {' '.join(cmd)} with list file {filelist_path}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Successfully concatenated videos into: {output_p}")
        dprint(f"CONCATENATE_VIDEOS_FFMPEG: Success. Output: {output_p}")
        if process.stderr:
            dprint(f"FFmpeg stderr (for concatenation to {output_p}):\n{process.stderr}")
        if not output_p.exists() or output_p.stat().st_size == 0:
             print(f"Warning: FFmpeg concatenation to {output_p} apparently succeeded but output file is missing or empty.")
             dprint(f"FFmpeg command for {output_p} (concatenation) produced no output. stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg concatenation for {output_p}:")
        print("FFmpeg command:", ' '.join(e.cmd))
        if e.stdout: print("FFmpeg stdout:\n", e.stdout)
        if e.stderr: print("FFmpeg stderr:\n", e.stderr)
        dprint(f"FFmpeg concatenation failed for {output_p}. Error: {e}")
        if output_p.exists(): output_p.unlink(missing_ok=True)
        raise 
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        dprint("CONCATENATE_VIDEOS_FFMPEG: ffmpeg command not found.")
        raise

# --- OpenCV-based video properties extraction ---
def get_video_frame_count_and_fps(video_path: str | Path) -> tuple[int | None, float | None]:
    """Gets frame count and FPS of a video using OpenCV. Returns (None, None) on failure."""
    video_path_str = str(Path(video_path).resolve())
    cap = None
    try:
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            dprint(f"GET_VIDEO_FRAME_COUNT_FPS: Could not open video: {video_path_str}")
            return None, None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Validate frame_count and fps as they can sometimes be 0 or negative for problematic files/streams
        valid_frame_count = frame_count if frame_count > 0 else None
        valid_fps = fps if fps > 0 else None

        if valid_frame_count is None:
             dprint(f"GET_VIDEO_FRAME_COUNT_FPS: Video {video_path_str} reported non-positive frame count: {frame_count}. Treating as unknown.")
        if valid_fps is None:
             dprint(f"GET_VIDEO_FRAME_COUNT_FPS: Video {video_path_str} reported non-positive FPS: {fps}. Treating as unknown.")

        dprint(f"GET_VIDEO_FRAME_COUNT_FPS: Video {video_path_str} - Frames: {valid_frame_count}, FPS: {valid_fps}")
        return valid_frame_count, valid_fps
    except Exception as e:
        dprint(f"GET_VIDEO_FRAME_COUNT_FPS: Exception processing {video_path_str}: {e}")
        return None, None
    finally:
        if cap:
            cap.release()


body_colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]
face_color = [255, 255, 255]
hand_keypoint_color = [0, 0, 255]
hand_limb_colors = [
    [255,0,0],[255,60,0],[255,120,0],[255,180,0], [180,255,0],[120,255,0],[60,255,0],[0,255,0],
    [0,255,60],[0,255,120],[0,255,180],[0,180,255], [0,120,255],[0,60,255],[0,0,255],[60,0,255],
    [120,0,255],[180,0,255],[255,0,180],[255,0,120]
]

# MediaPipe Pose connections (33 landmarks, indices 0-32)
# Based on mp.solutions.pose.POSE_CONNECTIONS
body_skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Nose to Left Eye to Left Ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # Nose to Right Eye to Right Ear
    (9, 10),  # Mouth
    (11, 12), # Shoulders
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 19), (15, 21), # Left Arm and simplified Left Hand (wrist to fingers)
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 20), (16, 22), # Right Arm and simplified Right Hand (wrist to fingers)
    (11, 23), (12, 24), # Connect shoulders to Hips
    (23, 24), # Hip connection
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # Left Leg and Foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)  # Right Leg and Foot
]

face_skeleton = [] # Draw face dots only, no connections

# MediaPipe Hand connections (21 landmarks per hand, indices 0-20)
hand_skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
]

def draw_keypoints_and_skeleton(image, keypoints_data, skeleton_connections, colors_config, confidence_threshold=0.1, point_radius=3, line_thickness=2, is_face=False, is_hand=False):
    if not keypoints_data:
        return
    
    tri_tuples = []
    if isinstance(keypoints_data, list) and len(keypoints_data) > 0 and isinstance(keypoints_data[0], (int, float)) and len(keypoints_data) % 3 == 0:
        for i in range(0, len(keypoints_data), 3):
            tri_tuples.append(keypoints_data[i:i+3])
    else:
        dprint(f"draw_keypoints_and_skeleton: Unexpected keypoints_data format or length not divisible by 3. Data: {keypoints_data}")
        return

    if skeleton_connections:
        for i, (joint_idx_a, joint_idx_b) in enumerate(skeleton_connections):
            if joint_idx_a >= len(tri_tuples) or joint_idx_b >= len(tri_tuples):
                continue
            a_x, a_y, a_confidence = tri_tuples[joint_idx_a]
            b_x, b_y, b_confidence = tri_tuples[joint_idx_b]
            
            if a_confidence >= confidence_threshold and b_confidence >= confidence_threshold:
                limb_color = None
                if is_hand:
                    limb_color_list = colors_config['limbs']
                    limb_color = limb_color_list[i % len(limb_color_list)]
                else: 
                    limb_color_list = colors_config if isinstance(colors_config, list) else [colors_config]
                    limb_color = limb_color_list[i % len(limb_color_list)]
                if limb_color is not None:
                    cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), limb_color, line_thickness)

    for i, (x, y, confidence) in enumerate(tri_tuples):
        if confidence >= confidence_threshold:
            point_color = None
            current_radius = point_radius
            if is_hand:
                point_color = colors_config['points']
            elif is_face:
                point_color = colors_config 
                current_radius = 2
            else: 
                point_color_list = colors_config 
                point_color = point_color_list[i % len(point_color_list)]
            if point_color is not None:
                cv2.circle(image, (int(x), int(y)), current_radius, point_color, -1)

def gen_skeleton_with_face_hands(pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d,
                                 canvas_width, canvas_height, landmarkType, confidence_threshold=0.1):
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    def scale_keypoints(keypoints, target_w, target_h, input_is_normalized):
        if not keypoints: return []
        scaled = []
        if not isinstance(keypoints, list) or (keypoints and not isinstance(keypoints[0], (int, float))):
            dprint(f"scale_keypoints: Unexpected keypoints format: {type(keypoints)}. Expecting flat list of numbers.")
            return [] 

        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i+3]
            if input_is_normalized: scaled.extend([x * target_w, y * target_h, conf])
            else: scaled.extend([x, y, conf])
        return scaled
    
    input_is_normalized = (landmarkType == "OpenPose") # This might need adjustment based on actual landmarkType usage
    
    scaled_pose = scale_keypoints(pose_keypoints_2d, canvas_width, canvas_height, input_is_normalized)
    scaled_face = scale_keypoints(face_keypoints_2d, canvas_width, canvas_height, input_is_normalized)
    scaled_hand_left = scale_keypoints(hand_left_keypoints_2d, canvas_width, canvas_height, input_is_normalized)
    scaled_hand_right = scale_keypoints(hand_right_keypoints_2d, canvas_width, canvas_height, input_is_normalized)
    
    draw_keypoints_and_skeleton(image, scaled_pose, body_skeleton, body_colors, confidence_threshold, point_radius=6, line_thickness=4)
    if scaled_face:
        draw_keypoints_and_skeleton(image, scaled_face, face_skeleton, face_color, confidence_threshold, point_radius=2, line_thickness=1, is_face=True)
    hand_colors_config = {'limbs': hand_limb_colors, 'points': hand_keypoint_color}
    if scaled_hand_left:
        draw_keypoints_and_skeleton(image, scaled_hand_left, hand_skeleton, hand_colors_config, confidence_threshold, point_radius=3, line_thickness=2, is_hand=True)
    if scaled_hand_right:
        draw_keypoints_and_skeleton(image, scaled_hand_right, hand_skeleton, hand_colors_config, confidence_threshold, point_radius=3, line_thickness=2, is_hand=True)
    return image

def transform_all_keypoints(keypoints_1_dict, keypoints_2_dict, frames, interpolation="linear"):
    def interpolate_keypoint_set(kp1_list, kp2_list, num_frames, interp_method):
        if not kp1_list and not kp2_list: return [[] for _ in range(num_frames)]
        
        len1 = len(kp1_list) if kp1_list else 0
        len2 = len(kp2_list) if kp2_list else 0

        if not kp1_list: kp1_list = [0.0] * len2
        if not kp2_list: kp2_list = [0.0] * len1

        if len(kp1_list) != len(kp2_list) or not kp1_list or len(kp1_list) % 3 != 0:
             dprint(f"interpolate_keypoint_set: Mismatched, empty, or non-triplet keypoint lists after padding. KP1 len: {len(kp1_list)}, KP2 len: {len(kp2_list)}. Returning empty sequences.")
             return [[] for _ in range(num_frames)]
        
        tri_tuples_1 = [kp1_list[i:i + 3] for i in range(0, len(kp1_list), 3)]
        tri_tuples_2 = [kp2_list[i:i + 3] for i in range(0, len(kp2_list), 3)]
        
        keypoints_sequence = []
        for j in range(num_frames):
            interpolated_kps_for_frame = []
            t = j / float(num_frames - 1) if num_frames > 1 else 0.0
            
            interp_factor = t 
            if interp_method == "ease-in": interp_factor = t * t
            elif interp_method == "ease-out": interp_factor = 1 - (1 - t) * (1 - t)
            elif interp_method == "ease-in-out":
                if t < 0.5: interp_factor = 2 * t * t 
                else: interp_factor = 1 - pow(-2 * t + 2, 2) / 2
            
            for i in range(len(tri_tuples_1)):
                x1, y1, c1 = tri_tuples_1[i]
                x2, y2, c2 = tri_tuples_2[i]
                new_x, new_y, new_c = 0.0, 0.0, 0.0

                if c1 > 0.05 and c2 > 0.05:
                    new_x = x1 + (x2 - x1) * interp_factor
                    new_y = y1 + (y2 - y1) * interp_factor
                    new_c = c1 + (c2 - c1) * interp_factor
                elif c1 > 0.05 and c2 <= 0.05: 
                    new_x, new_y = x1, y1
                    new_c = c1 * (1.0 - interp_factor) 
                elif c1 <= 0.05 and c2 > 0.05: 
                    new_x, new_y = x2, y2
                    new_c = c2 * interp_factor 
                interpolated_kps_for_frame.extend([new_x, new_y, new_c])
            keypoints_sequence.append(interpolated_kps_for_frame)
        return keypoints_sequence

    pose_1 = keypoints_1_dict.get('pose_keypoints_2d', [])
    face_1 = keypoints_1_dict.get('face_keypoints_2d', [])
    hand_left_1 = keypoints_1_dict.get('hand_left_keypoints_2d', [])
    hand_right_1 = keypoints_1_dict.get('hand_right_keypoints_2d', [])
    
    pose_2 = keypoints_2_dict.get('pose_keypoints_2d', [])
    face_2 = keypoints_2_dict.get('face_keypoints_2d', [])
    hand_left_2 = keypoints_2_dict.get('hand_left_keypoints_2d', [])
    hand_right_2 = keypoints_2_dict.get('hand_right_keypoints_2d', [])
    
    pose_sequence = interpolate_keypoint_set(pose_1, pose_2, frames, interpolation)
    face_sequence = interpolate_keypoint_set(face_1, face_2, frames, interpolation)
    hand_left_sequence = interpolate_keypoint_set(hand_left_1, hand_left_2, frames, interpolation)
    hand_right_sequence = interpolate_keypoint_set(hand_right_1, hand_right_2, frames, interpolation)

    combined_sequence = []
    for i in range(frames):
        combined_frame_data = {
            'pose_keypoints_2d': pose_sequence[i] if i < len(pose_sequence) else [],
            'face_keypoints_2d': face_sequence[i] if i < len(face_sequence) else [],
            'hand_left_keypoints_2d': hand_left_sequence[i] if i < len(hand_left_sequence) else [],
            'hand_right_keypoints_2d': hand_right_sequence[i] if i < len(hand_right_sequence) else []
        }
        combined_sequence.append(combined_frame_data)
    return combined_sequence

def extract_pose_keypoints(image_path: str | Path, include_face=True, include_hands=True, resolution: tuple[int,int]=(640,480)) -> dict:
    # import mediapipe as mp # Already imported at top
    # import cv2 # Already imported at top
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize image before processing if its resolution differs significantly,
    # or ensure MediaPipe processes at a consistent internal resolution.
    # For now, assume MediaPipe handles various input sizes well, and keypoints are normalized.
    # The passed 'resolution' param will be used to scale normalized keypoints back to absolute.
    
    height, width = resolution[1], resolution[0] # For scaling output coords

    mp_holistic = mp.solutions.holistic
    # It's good practice to use a try-finally for resources like MediaPipe Holistic
    holistic_instance = mp_holistic.Holistic(static_image_mode=True, 
                                           min_detection_confidence=0.5, 
                                           min_tracking_confidence=0.5)
    try:
        # Convert BGR image to RGB for MediaPipe
        results = holistic_instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        holistic_instance.close()
        
    keypoints = {}
    pose_kps = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            # lm.x, lm.y are normalized coordinates; scale them to target resolution
            pose_kps.extend([lm.x * width, lm.y * height, lm.visibility])
    keypoints['pose_keypoints_2d'] = pose_kps
    
    face_kps = []
    if include_face and results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            face_kps.extend([lm.x * width, lm.y * height, lm.visibility if hasattr(lm, 'visibility') else 1.0]) # Some face landmarks might not have visibility
    keypoints['face_keypoints_2d'] = face_kps
    
    left_hand_kps = []
    if include_hands and results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            left_hand_kps.extend([lm.x * width, lm.y * height, lm.visibility])
    keypoints['hand_left_keypoints_2d'] = left_hand_kps
    
    right_hand_kps = []
    if include_hands and results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            right_hand_kps.extend([lm.x * width, lm.y * height, lm.visibility])
    keypoints['hand_right_keypoints_2d'] = right_hand_kps
    
    return keypoints

def create_pose_interpolated_guide_video(output_video_path: str | Path, resolution: tuple[int, int], total_frames: int,
                                           start_image_path: str | Path, end_image_path: str | Path,
                                           interpolation="linear", confidence_threshold=0.1,
                                           include_face=True, include_hands=True, fps=25):
    dprint(f"Creating pose interpolated guide: {output_video_path} from '{Path(start_image_path).name}' to '{Path(end_image_path).name}' ({total_frames} frames). First frame will be actual start image.")

    if total_frames <= 0:
        dprint(f"Video creation skipped for {output_video_path} as total_frames is {total_frames}.")
        return

    frames_list = []
    canvas_width, canvas_height = resolution

    first_visual_frame_np = image_to_frame(start_image_path, resolution)
    if first_visual_frame_np is None:
        print(f"Error loading start image {start_image_path} for guide video frame 0. Using black frame.")
        traceback.print_exc()
        first_visual_frame_np = create_color_frame(resolution, (0,0,0))
    frames_list.append(first_visual_frame_np)

    if total_frames > 1:
        try:
            # Pass the target resolution for keypoint scaling
            keypoints_from = extract_pose_keypoints(start_image_path, include_face, include_hands, resolution)
            keypoints_to = extract_pose_keypoints(end_image_path, include_face, include_hands, resolution)
        except Exception as e_extract:
            print(f"Error extracting keypoints for pose interpolation: {e_extract}. Filling remaining guide frames with black.")
            traceback.print_exc()
            black_frame = create_color_frame(resolution, (0,0,0))
            for _ in range(total_frames - 1):
                frames_list.append(black_frame)
            create_video_from_frames_list(frames_list, output_video_path, fps, resolution)
            return

        interpolated_sequence = transform_all_keypoints(keypoints_from, keypoints_to, total_frames, interpolation)
        
        # landmarkType for gen_skeleton_with_face_hands should indicate absolute coordinates
        # as extract_pose_keypoints now returns absolute coordinates scaled to 'resolution'
        landmark_type_for_gen = "AbsoluteCoords" 

        for i in range(1, total_frames):
            if i < len(interpolated_sequence):
                frame_data = interpolated_sequence[i]
                pose_kps = frame_data.get('pose_keypoints_2d', [])
                face_kps = frame_data.get('face_keypoints_2d', []) if include_face else []
                hand_left_kps = frame_data.get('hand_left_keypoints_2d', []) if include_hands else []
                hand_right_kps = frame_data.get('hand_right_keypoints_2d', []) if include_hands else []

                img = gen_skeleton_with_face_hands(
                    pose_kps, face_kps, hand_left_kps, hand_right_kps,
                    canvas_width, canvas_height, 
                    landmark_type_for_gen, # Keypoints are already absolute
                    confidence_threshold
                )
                frames_list.append(img)
            else:
                dprint(f"Warning: Interpolated sequence too short at index {i} for {output_video_path}. Appending black frame.")
                frames_list.append(create_color_frame(resolution, (0,0,0)))
    
    if len(frames_list) != total_frames:
        dprint(f"Warning: Generated {len(frames_list)} frames for {output_video_path}, expected {total_frames}. Adjusting.")
        if len(frames_list) < total_frames:
            last_frame = frames_list[-1] if frames_list else create_color_frame(resolution, (0,0,0))
            frames_list.extend([last_frame.copy() for _ in range(total_frames - len(frames_list))])
        else:
            frames_list = frames_list[:total_frames]

    if not frames_list:
        dprint(f"Error: No frames for video {output_video_path}. Skipping creation.")
        return

    create_video_from_frames_list(frames_list, output_video_path, fps, resolution)

# --- Debug Summary Video Helpers ---
def get_resized_frame(video_path_str: str, target_size: tuple[int, int], frame_ratio: float = 0.5) -> np.ndarray | None:
    """Extracts a frame (by ratio, e.g., 0.5 for middle) from a video and resizes it."""
    video_path = Path(video_path_str)
    if not video_path.exists() or video_path.stat().st_size == 0:
        dprint(f"GET_RESIZED_FRAME: Video not found or empty: {video_path_str}")
        placeholder = create_color_frame(target_size, (10, 10, 10)) # Dark grey
        cv2.putText(placeholder, "Not Found", (10, target_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return placeholder

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            dprint(f"GET_RESIZED_FRAME: Could not open video: {video_path_str}")
            return create_color_frame(target_size, (20,20,20))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            dprint(f"GET_RESIZED_FRAME: Video has 0 frames: {video_path_str}")
            return create_color_frame(target_size, (30,30,30))
        
        frame_to_get = int(total_frames * frame_ratio)
        frame_to_get = max(0, min(frame_to_get, total_frames - 1)) # Clamp
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_to_get))
        ret, frame = cap.read()
        if not ret or frame is None:
            dprint(f"GET_RESIZED_FRAME: Could not read frame {frame_to_get} from: {video_path_str}")
            return create_color_frame(target_size, (40,40,40))
        
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        dprint(f"GET_RESIZED_FRAME: Exception processing {video_path_str}: {e}")
        return create_color_frame(target_size, (50,50,50)) # Error color
    finally:
        if cap: cap.release()

def draw_multiline_text(image, text_lines, start_pos, font, font_scale, color, thickness, line_spacing):
    x, y = start_pos
    for i, line in enumerate(text_lines):
        line_y = y + (i * (cv2.getTextSize(line, font, font_scale, thickness)[0][1] + line_spacing))
        cv2.putText(image, line, (x, line_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image 

def generate_debug_summary_video(segments_data: list[dict], output_path: str | Path, fps: int, 
                                 num_frames_for_collage: int, 
                                 target_thumb_size: tuple[int, int] = (320, 180)):
    if not DEBUG_MODE: return # Only run if debug mode is on
    if not segments_data:
        dprint("GENERATE_DEBUG_SUMMARY_VIDEO: No segment data provided.")
        return

    dprint(f"Generating animated debug collage with {num_frames_for_collage} frames, at {fps} FPS.")

    thumb_w, thumb_h = target_thumb_size
    padding = 10
    header_h = 50 
    text_line_h_approx = 20
    max_settings_lines = 6
    settings_area_h = (text_line_h_approx * max_settings_lines) + padding 

    num_segments = len(segments_data)
    col_w = thumb_w + (2 * padding)
    canvas_w = num_segments * col_w
    canvas_h = header_h + (thumb_h * 2) + (padding * 3) + settings_area_h + padding
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_small = 0.4
    font_scale_title = 0.6
    text_color = (230, 230, 230)
    title_color = (255, 255, 255)
    line_thickness = 1

    overall_static_template_canvas = np.full((canvas_h, canvas_w, 3), (30, 30, 30), dtype=np.uint8) 
    for idx, seg_data in enumerate(segments_data):
        col_x_start = idx * col_w
        center_x_col = col_x_start + col_w // 2
        title_text = f"Segment {seg_data['segment_index']}"
        (tw, th), _ = cv2.getTextSize(title_text, font, font_scale_title, line_thickness)
        cv2.putText(overall_static_template_canvas, title_text, (center_x_col - tw//2, header_h - padding), font, font_scale_title, title_color, line_thickness, cv2.LINE_AA)
        
        y_offset = header_h
        cv2.putText(overall_static_template_canvas, "Input Guide", (col_x_start + padding, y_offset + text_line_h_approx), font, font_scale_small, text_color, line_thickness)
        y_offset += thumb_h + padding
        cv2.putText(overall_static_template_canvas, "Headless Output", (col_x_start + padding, y_offset + text_line_h_approx), font, font_scale_small, text_color, line_thickness)
        y_offset += thumb_h + padding

        settings_y_start = y_offset
        cv2.putText(overall_static_template_canvas, "Settings:", (col_x_start + padding, settings_y_start + text_line_h_approx), font, font_scale_small, text_color, line_thickness)
        settings_text_lines = []
        payload = seg_data.get("task_payload", {})
        settings_text_lines.append(f"Task ID: {payload.get('task_id', 'N/A')[:10]}...")
        prompt_short = payload.get('prompt', 'N/A')[:35] + ("..." if len(payload.get('prompt', '')) > 35 else "")
        settings_text_lines.append(f"Prompt: {prompt_short}")
        settings_text_lines.append(f"Seed: {payload.get('seed', 'N/A')}, Frames: {payload.get('frames', 'N/A')}")
        settings_text_lines.append(f"Resolution: {payload.get('resolution', 'N/A')}")
        draw_multiline_text(overall_static_template_canvas, settings_text_lines[:max_settings_lines], 
                            (col_x_start + padding, settings_y_start + text_line_h_approx + padding), 
                            font, font_scale_small, text_color, line_thickness, 5)

    error_placeholder_frame = create_color_frame(target_thumb_size, (50, 0, 0)) 
    cv2.putText(error_placeholder_frame, "ERR", (10, target_thumb_size[1]//2), font, 0.8, (255,255,255), 1)
    not_found_placeholder_frame = create_color_frame(target_thumb_size, (0, 50, 0)) 
    cv2.putText(not_found_placeholder_frame, "N/A", (10, target_thumb_size[1]//2), font, 0.8, (255,255,255), 1)
    static_thumbs_cache = {}
    for seg_idx_cache, seg_data_cache in enumerate(segments_data):
        guide_thumb = get_resized_frame(seg_data_cache["guide_video_path"], target_thumb_size, frame_ratio=0.5)
        output_thumb = get_resized_frame(seg_data_cache["raw_headless_output_path"], target_thumb_size, frame_ratio=0.5)
        
        static_thumbs_cache[seg_idx_cache] = {
            'guide': guide_thumb if guide_thumb is not None else not_found_placeholder_frame,
            'output': output_thumb if output_thumb is not None else not_found_placeholder_frame
        }

    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (canvas_w, canvas_h))
        if not writer.isOpened():
            dprint(f"GENERATE_DEBUG_SUMMARY_VIDEO: Failed to open VideoWriter for {output_path}")
            return
        
        dprint(f"GENERATE_DEBUG_SUMMARY_VIDEO: Writing sequentially animated collage to {output_path}")

        for active_seg_idx in range(num_segments):
            dprint(f"Animating segment {active_seg_idx} in collage...")
            caps_for_active_segment = {'guide': None, 'output': None, 'last_frames': {}}
            video_paths_to_load = {
                'guide': segments_data[active_seg_idx]["guide_video_path"],
                'output': segments_data[active_seg_idx]["raw_headless_output_path"]
            }
            for key, path_str in video_paths_to_load.items():
                p = Path(path_str)
                if p.exists() and p.stat().st_size > 0:
                    cap_video = cv2.VideoCapture(str(p))
                    if cap_video.isOpened():
                        caps_for_active_segment[key] = cap_video
                        ret, frame = cap_video.read(); 
                        caps_for_active_segment['last_frames'][key] = cv2.resize(frame, target_thumb_size, cv2.INTER_AREA) if ret and frame is not None else error_placeholder_frame
                        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                    else: caps_for_active_segment['last_frames'][key] = error_placeholder_frame
                else: caps_for_active_segment['last_frames'][key] = not_found_placeholder_frame

            for frame_num in range(num_frames_for_collage):
                current_frame_canvas = overall_static_template_canvas.copy()

                for display_seg_idx in range(num_segments):
                    col_x_start = display_seg_idx * col_w
                    current_y_pos = header_h
                    
                    videos_to_composite = [None, None] # guide, output

                    if display_seg_idx == active_seg_idx:
                        if caps_for_active_segment['guide']:
                            ret, frame = caps_for_active_segment['guide'].read()
                            if ret and frame is not None: videos_to_composite[0] = cv2.resize(frame, target_thumb_size); caps_for_active_segment['last_frames']['guide'] = videos_to_composite[0]
                            else: videos_to_composite[0] = caps_for_active_segment['last_frames'].get('guide', error_placeholder_frame)
                        else: videos_to_composite[0] = caps_for_active_segment['last_frames'].get('guide', not_found_placeholder_frame)
                        if caps_for_active_segment['output']:
                            ret, frame = caps_for_active_segment['output'].read()
                            if ret and frame is not None: videos_to_composite[1] = cv2.resize(frame, target_thumb_size); caps_for_active_segment['last_frames']['output'] = videos_to_composite[1]
                            else: videos_to_composite[1] = caps_for_active_segment['last_frames'].get('output', error_placeholder_frame)
                        else: videos_to_composite[1] = caps_for_active_segment['last_frames'].get('output', not_found_placeholder_frame)
                    else:
                        videos_to_composite[0] = static_thumbs_cache[display_seg_idx]['guide']
                        videos_to_composite[1] = static_thumbs_cache[display_seg_idx]['output']

                    current_frame_canvas[current_y_pos : current_y_pos + thumb_h, col_x_start + padding : col_x_start + padding + thumb_w] = videos_to_composite[0]
                    current_y_pos += thumb_h + padding
                    current_frame_canvas[current_y_pos : current_y_pos + thumb_h, col_x_start + padding : col_x_start + padding + thumb_w] = videos_to_composite[1]
                
                writer.write(current_frame_canvas)
            
            if caps_for_active_segment['guide']: caps_for_active_segment['guide'].release()
            if caps_for_active_segment['output']: caps_for_active_segment['output'].release()
            dprint(f"Finished animating segment {active_seg_idx} in collage.")

        dprint(f"GENERATE_DEBUG_SUMMARY_VIDEO: Finished writing sequentially animated debug collage.")

    except Exception as e:
        dprint(f"GENERATE_DEBUG_SUMMARY_VIDEO: Exception during video writing: {e} - {traceback.format_exc()}")
    finally:
        if writer: writer.release()
        dprint("GENERATE_DEBUG_SUMMARY_VIDEO: Video writer released.")


def generate_different_perspective_debug_video_summary(
    video_stage_data: list[dict], 
    output_path: Path, 
    fps: int, 
    target_resolution: tuple[int, int] # width, height
):
    if not DEBUG_MODE: return # Only run if debug mode is on
    dprint(f"Generating different_perspective DEBUG VIDEO summary at {output_path} ({fps} FPS, {target_resolution[0]}x{target_resolution[1]})")
    
    all_output_frames = [] 
    font_pil = None
    try:
        pil_font = ImageFont.truetype("arial.ttf", size=24) 
    except IOError:
        pil_font = ImageFont.load_default()
        dprint("Arial font not found for debug video summary, using default PIL font.")

    text_color = (255, 255, 255) 
    bg_color = (0,0,0) 
    text_bg_opacity = 128 

    for stage_info in video_stage_data:
        label = stage_info.get('label', 'Unknown Stage')
        file_type = stage_info.get('type', 'image')
        file_path_str = stage_info.get('path')
        display_duration_frames = stage_info.get('display_frames', fps * 2) 

        if not file_path_str:
            print(f"[Debug Video] Missing path for stage '{label}', skipping.")
            continue
        
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"[Debug Video] File not found for stage '{label}': {file_path}, creating placeholder frames.")
            placeholder_frame_np = create_color_frame(target_resolution, (50, 0, 0)) 
            placeholder_pil = Image.fromarray(cv2.cvtColor(placeholder_frame_np, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(placeholder_pil)
            draw.text((20, 20), f"{label}\n(File Not Found)", font=pil_font, fill=text_color)
            placeholder_frame_final_np = cv2.cvtColor(np.array(placeholder_pil), cv2.COLOR_RGB2BGR)
            all_output_frames.extend([placeholder_frame_final_np] * display_duration_frames)
            continue

        current_stage_frames_np = []
        try:
            if file_type == 'image':
                pil_img = Image.open(file_path).convert("RGB")
                pil_img_resized = pil_img.resize(target_resolution, Image.Resampling.LANCZOS)
                np_bgr_frame = cv2.cvtColor(np.array(pil_img_resized), cv2.COLOR_RGB2BGR)
                current_stage_frames_np = [np_bgr_frame] * display_duration_frames
            
            elif file_type == 'video':
                cap_video = cv2.VideoCapture(str(file_path))
                if not cap_video.isOpened():
                    raise IOError(f"Could not open video: {file_path}")
                
                frames_read = 0
                while frames_read < display_duration_frames:
                    ret, frame_np = cap_video.read()
                    if not ret:
                        if current_stage_frames_np: 
                            current_stage_frames_np.extend([current_stage_frames_np[-1]] * (display_duration_frames - frames_read))
                        else: 
                            err_frame = create_color_frame(target_resolution, (0,50,0)) 
                            err_pil = Image.fromarray(cv2.cvtColor(err_frame, cv2.COLOR_BGR2RGB))
                            ImageDraw.Draw(err_pil).text((20,20), f"{label}\n(Video Read Error)", font=pil_font, fill=text_color)
                            current_stage_frames_np.extend([cv2.cvtColor(np.array(err_pil), cv2.COLOR_RGB2BGR)] * (display_duration_frames - frames_read))
                        break 
                    
                    if frame_np.shape[1] != target_resolution[0] or frame_np.shape[0] != target_resolution[1]:
                        frame_np = cv2.resize(frame_np, target_resolution, interpolation=cv2.INTER_AREA)
                    current_stage_frames_np.append(frame_np)
                    frames_read += 1
                cap_video.release()
            else:
                print(f"[Debug Video] Unknown file type '{file_type}' for stage '{label}'. Skipping.")
                continue

            for i in range(len(current_stage_frames_np)):
                frame_pil = Image.fromarray(cv2.cvtColor(current_stage_frames_np[i], cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil, 'RGBA') 
                
                text_x, text_y = 20, 20
                bbox = draw.textbbox((text_x, text_y), label, font=pil_font)
                rect_coords = [(bbox[0]-5, bbox[1]-5), (bbox[2]+5, bbox[3]+5)]
                draw.rectangle(rect_coords, fill=(bg_color[0], bg_color[1], bg_color[2], text_bg_opacity))
                draw.text((text_x, text_y), label, font=pil_font, fill=text_color) 
                
                current_stage_frames_np[i] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            
            all_output_frames.extend(current_stage_frames_np)

        except Exception as e_stage:
            print(f"[Debug Video] Error processing stage '{label}' (path: {file_path}): {e_stage}")
            traceback.print_exc()
            err_frame_np = create_color_frame(target_resolution, (0,0,50)) 
            err_pil = Image.fromarray(cv2.cvtColor(err_frame_np, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(err_pil).text((20,20), f"{label}\n(Stage Processing Error)", font=pil_font, fill=text_color)
            all_output_frames.extend([cv2.cvtColor(np.array(err_pil), cv2.COLOR_RGB2BGR)] * display_duration_frames)
            
    if not all_output_frames:
        dprint("[Debug Video] No frames were generated for the debug video summary.")
        return

    print(f"[Debug Video] Creating final video with {len(all_output_frames)} frames.")
    create_video_from_frames_list(all_output_frames, output_path, fps, target_resolution)
    print(f"Debug video summary for 'different_perspective' saved to: {output_path.resolve()}") 

def download_file(url, dest_folder, filename):
    dest_path = Path(dest_folder) / filename
    if dest_path.exists():
        # Validate existing file before assuming it's good
        if filename.endswith('.safetensors') or 'lora' in filename.lower():
            is_valid, validation_msg = validate_lora_file(dest_path, filename)
            if is_valid:
                print(f"[INFO] File {filename} already exists and is valid in {dest_folder}. {validation_msg}")
                return True
            else:
                print(f"[WARNING] Existing file {filename} failed validation ({validation_msg}). Re-downloading...")
                dest_path.unlink()
        else:
            print(f"[INFO] File {filename} already exists in {dest_folder}.")
            return True
    
    # Use huggingface_hub for HuggingFace URLs for better reliability
    if "huggingface.co" in url:
        try:
            from huggingface_hub import hf_hub_download
            from urllib.parse import urlparse
            
            # Parse HuggingFace URL to extract repo_id and filename
            # Format: https://huggingface.co/USER/REPO/resolve/BRANCH/FILENAME
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            
            if len(path_parts) >= 4 and path_parts[2] == 'resolve':
                repo_id = f"{path_parts[0]}/{path_parts[1]}"
                branch = path_parts[3] if len(path_parts) > 4 else "main"
                hf_filename = '/'.join(path_parts[4:]) if len(path_parts) > 4 else filename
                
                print(f"Downloading {filename} from HuggingFace repo {repo_id} using hf_hub_download...")
                
                # Download using huggingface_hub with automatic checksums and resumption
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_filename,
                    revision=branch,
                    cache_dir=str(dest_folder),
                    resume_download=True,
                    local_files_only=False
                )
                
                # Copy from HF cache to target location if different
                if Path(downloaded_path) != dest_path:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(downloaded_path, dest_path)
                
                # Validate the downloaded file
                if filename.endswith('.safetensors') or 'lora' in filename.lower():
                    is_valid, validation_msg = validate_lora_file(dest_path, filename)
                    if not is_valid:
                        print(f"[ERROR] Downloaded file {filename} failed validation: {validation_msg}")
                        dest_path.unlink(missing_ok=True)
                        return False
                    print(f"Successfully downloaded and validated {filename}. {validation_msg}")
                else:
                    print(f"Successfully downloaded {filename} with integrity verification.")
                return True
                
        except ImportError:
            print(f"[WARNING] huggingface_hub not available, falling back to requests for {url}")
        except Exception as e:
            print(f"[WARNING] HuggingFace download failed for {filename}: {e}, falling back to requests")
    
    # Fallback to requests with basic integrity checks
    try:
        print(f"Downloading {filename} from {url} to {dest_folder}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        # Get expected content length for verification
        expected_size = int(response.headers.get('content-length', 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded_size += len(chunk)
        
        # Verify download integrity
        actual_size = dest_path.stat().st_size
        if expected_size > 0 and actual_size != expected_size:
            print(f"[ERROR] Size mismatch for {filename}: expected {expected_size}, got {actual_size}")
            dest_path.unlink(missing_ok=True)
            return False
        
        # Use comprehensive validation for LoRA files
        if filename.endswith('.safetensors') or 'lora' in filename.lower():
            is_valid, validation_msg = validate_lora_file(dest_path, filename)
            if not is_valid:
                print(f"[ERROR] Downloaded file {filename} failed validation: {validation_msg}")
                dest_path.unlink(missing_ok=True)
                return False
            print(f"Successfully downloaded and validated {filename}. {validation_msg}")
        else:
            # For non-LoRA safetensors files, do basic format check
            if filename.endswith('.safetensors'):
                try:
                    import safetensors.torch as st
                    with st.safe_open(dest_path, framework="pt") as f:
                        pass  # Just verify it can be opened
                    print(f"Successfully downloaded and verified safetensors file {filename}.")
                except ImportError:
                    print(f"[WARNING] safetensors not available for verification of {filename}")
                except Exception as e:
                    print(f"[ERROR] Downloaded safetensors file {filename} appears corrupted: {e}")
                    dest_path.unlink(missing_ok=True)
                    return False
            else:
                print(f"Successfully downloaded {filename}.")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        if dest_path.exists(): # Attempt to clean up partial download
            try: os.remove(dest_path)
            except: pass
        return False

# Added to provide a unique target path generator for files.
def _get_unique_target_path(target_dir: Path, base_name: str, extension: str) -> Path:
    """Generates a unique target Path in the given directory by appending a timestamp and random string."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp_short = datetime.now().strftime("%H%M%S")
    # Use a short UUID/random string to significantly reduce collision probability with just timestamp
    unique_suffix = uuid.uuid4().hex[:6]
    
    # Construct the filename
    # Ensure extension has a leading dot
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    filename = f"{base_name}_{timestamp_short}_{unique_suffix}{extension}"
    return target_dir / filename

def download_image_if_url(image_url_or_path: str, download_target_dir: Path | str | None, task_id_for_logging: str | None = "generic_task", debug_mode: bool = False, descriptive_name: str | None = None) -> str:
    """
    Checks if the given string is an HTTP/HTTPS URL. If so, and if download_target_dir is provided,
    downloads the image to a unique path within download_target_dir.
    Returns the local file path string if downloaded, otherwise returns the original string.
    """
    if not image_url_or_path:
        return image_url_or_path

    parsed_url = urlparse(image_url_or_path)
    if parsed_url.scheme in ['http', 'https'] and download_target_dir:
        # If not in debug mode, return a temp path or the original URL
        if not debug_mode:
            dprint(f"Task {task_id_for_logging}: Debug mode disabled, skipping image download for: {image_url_or_path}")
            return image_url_or_path
            
        target_dir_path = Path(download_target_dir)
        try:
            target_dir_path.mkdir(parents=True, exist_ok=True)
            dprint(f"Task {task_id_for_logging}: Downloading image from URL: {image_url_or_path} to {target_dir_path.resolve()}")
            
            # Use a session for potential keep-alive and connection pooling
            with requests.Session() as s:
                response = s.get(image_url_or_path, stream=True, timeout=300) # 5 min timeout
                response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

            original_filename = Path(parsed_url.path).name
            original_suffix = Path(original_filename).suffix if Path(original_filename).suffix else ".jpg" # Default to .jpg if no suffix
            if not original_suffix.startswith('.'):
                original_suffix = '.' + original_suffix
            
            # Use descriptive naming if provided, otherwise fall back to improved default
            if descriptive_name:
                base_name_for_download = descriptive_name[:50]  # Limit length
            else:
                # Improved default naming
                cleaned_stem = Path(original_filename).stem[:30] if Path(original_filename).stem else "image"
                base_name_for_download = f"input_{cleaned_stem}"
            
            # _get_unique_target_path expects a Path object for target_dir
            local_image_path = _get_unique_target_path(target_dir_path, base_name_for_download, original_suffix)
            
            with open(local_image_path, 'wb') as f:
                # Re-fetch without stream=True if response was already consumed by raise_for_status check,
                # or ensure streaming works correctly if the initial response object can be re-used.
                # For simplicity, re-requesting after status check if necessary, or ensure stream is not prematurely closed.
                # A simple way for non-huge files and to avoid stream issues with one-off downloads:
                if response.raw.closed: # If stream was closed by raise_for_status or other means
                    with requests.Session() as s_final:
                        final_response = s_final.get(image_url_or_path, stream=False, timeout=300) # Not streaming for direct content write
                        final_response.raise_for_status()
                        f.write(final_response.content)
                else: # Stream is still open
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            dprint(f"Task {task_id_for_logging}: Image downloaded successfully to {local_image_path.resolve()}")
            return str(local_image_path.resolve())

        except requests.exceptions.RequestException as e_req:
            dprint(f"Task {task_id_for_logging}: Error downloading image {image_url_or_path}: {e_req}. Returning original path.")
            return image_url_or_path
        except IOError as e_io:
            dprint(f"Task {task_id_for_logging}: IO error saving image from {image_url_or_path}: {e_io}. Returning original path.")
            return image_url_or_path
        except Exception as e_gen:
            dprint(f"Task {task_id_for_logging}: General error processing image URL {image_url_or_path}: {e_gen}. Returning original path.")
            return image_url_or_path
    else:
        # Not an HTTP/HTTPS URL or no download directory specified
        dprint(f"Task {task_id_for_logging}: Not downloading image (not URL or no target dir): {image_url_or_path}")
        return image_url_or_path

def image_to_frame(image_path_str: str | Path, target_resolution_wh: tuple[int, int] | None = None, task_id_for_logging: str | None = "generic_task", image_download_dir: Path | str | None = None) -> np.ndarray | None:
    """
    Load an image, optionally resize, and convert to BGR NumPy array.
    If image_path_str is a URL and image_download_dir is provided, it attempts to download it first.
    """
    resolved_image_path_str = image_path_str # Default to original path

    if isinstance(image_path_str, str): # Only attempt download if it's a string (potentially a URL)
        resolved_image_path_str = download_image_if_url(image_path_str, image_download_dir, task_id_for_logging)
    
    image_path = Path(resolved_image_path_str)

    if not image_path.exists():
        dprint(f"Task {task_id_for_logging}: Image file not found at {image_path} (original input: {image_path_str}).")
        return None
    try:
        img = Image.open(image_path).convert("RGB") # Ensure RGB for consistent processing
        if target_resolution_wh:
            img = img.resize(target_resolution_wh, Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        dprint(f"Error loading image {image_path} (original input {image_path_str}): {e}")
        return None

def _apply_strength_to_image(
    image_path_input: Path | str, # Changed name to avoid confusion
    strength: float,
    output_path: Path,
    target_resolution_wh: tuple[int, int] | None,
    task_id_for_logging: str | None = "generic_task",
    image_download_dir: Path | str | None = None
) -> Path | None:
    """
    Applies a brightness adjustment (strength) to an image, optionally resizes, and saves it.
    If image_path_input is a URL string and image_download_dir is provided, it attempts to download it first.
    """
    resolved_image_path_str = str(image_path_input) 

    if isinstance(image_path_input, str):
         resolved_image_path_str = download_image_if_url(image_path_input, image_download_dir, task_id_for_logging)
    # Check if image_path_input was a Path object representing a URL (less common for this function)
    elif isinstance(image_path_input, Path) and image_path_input.as_posix().startswith(('http://', 'https://')):
         resolved_image_path_str = download_image_if_url(image_path_input.as_posix(), image_download_dir, task_id_for_logging)

    actual_image_path = Path(resolved_image_path_str)

    if not actual_image_path.exists():
        dprint(f"Task {task_id_for_logging}: Source image not found at {actual_image_path} (original input: {image_path_input}) for strength application.")
        return None
    try:
        # Open the potentially downloaded or original local image
        img = Image.open(actual_image_path).convert("RGB") # Ensure RGB
        
        if target_resolution_wh:
            img = img.resize(target_resolution_wh, Image.Resampling.LANCZOS)
        
        # Apply the strength factor using PIL.ImageEnhance for brightness
        enhancer = ImageEnhance.Brightness(img)
        processed_img = enhancer.enhance(strength) # 'strength' is the factor for brightness
        
        # Save the adjusted image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_img.save(output_path) # Save PIL image directly

        dprint(f"Task {task_id_for_logging}: Applied strength {strength} to {actual_image_path.name}, saved to {output_path.name} with resolution {target_resolution_wh if target_resolution_wh else 'original'}")
        return output_path
    except Exception as e:
        dprint(f"Task {task_id_for_logging}: Error in _apply_strength_to_image for {actual_image_path}: {e}")
        traceback.print_exc()
        return None

def _copy_to_folder_with_unique_name(source_path: Path, target_dir: Path, base_name: str, extension: str) -> Path | None:
    """Copies a file to a target directory with a unique name based on timestamp and random string."""
    if not source_path:
        dprint(f"COPY: Source path is None for {base_name}{extension}. Skipping copy.")
        return None
    
    source_path_obj = Path(source_path)
    if not source_path_obj.exists():
        dprint(f"COPY: Source file {source_path_obj} does not exist. Skipping copy.")
        return None

    # Sanitize extension for _get_unique_target_path
    actual_extension = source_path_obj.suffix if source_path_obj.suffix else extension
    if not actual_extension.startswith('.'):
        actual_extension = '.' + actual_extension

    # Determine unique target path using the new helper
    target_file = _get_unique_target_path(target_dir, base_name, actual_extension)
    
    try:
        # target_dir.mkdir(parents=True, exist_ok=True) # _get_unique_target_path handles this
        shutil.copy2(str(source_path_obj), str(target_file))
        dprint(f"COPY: Copied {source_path_obj.name} to {target_file}")
        return target_file # Return the path of the copied file
    except Exception as e_copy:
        dprint(f"COPY: Failed to copy {source_path_obj} to {target_file}: {e_copy}")
        return None
    

def get_image_dimensions_pil(image_path: str | Path) -> tuple[int, int]:
    """Returns the dimensions of an image file as (width, height)."""
    with Image.open(image_path) as img:
        return img.size

# Added to adjust the brightness of an image/frame.
def _adjust_frame_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    """Adjusts the brightness of a given frame.
    The 'factor' is interpreted as a delta from the CLI argument:
    - Positive factor (e.g., 0.1) makes it darker (target_alpha = 1.0 - 0.1 = 0.9).
    - Negative factor (e.g., -0.1) makes it brighter (target_alpha = 1.0 - (-0.1) = 1.1).
    - Zero factor means no change (target_alpha = 1.0).
    """
    # Convert the CLI-style factor to an alpha for cv2.convertScaleAbs
    # CLI factor: positive = darker, negative = brighter
    # cv2 alpha: >1 = brighter, <1 = darker
    cv2_alpha = 1.0 - factor 
    return cv2.convertScaleAbs(frame, alpha=cv2_alpha, beta=0)

def sm_get_unique_target_path(target_dir: Path, name_stem: str, suffix: str) -> Path:
    """Generates a unique target Path in the given directory by appending a number if needed."""
    if not suffix.startswith('.'):
        suffix = f".{suffix}"
    
    final_path = target_dir / f"{name_stem}{suffix}"
    counter = 1
    while final_path.exists():
        final_path = target_dir / f"{name_stem}_{counter}{suffix}"
        counter += 1
    return final_path

def load_pil_images(
    paths_list_or_str: list[str] | str,
    wgp_convert_func: callable,
    image_download_dir: Path | str | None,
    task_id_for_log: str,
    dprint: callable
) -> list[Any] | None:
    """
    Loads one or more images from paths or URLs, downloads them if necessary,
    and applies a conversion function.
    """
    if not paths_list_or_str:
        return None
    
    paths_list = paths_list_or_str if isinstance(paths_list_or_str, list) else [paths_list_or_str]
    images = []

    for p_str in paths_list:
        local_p_str = download_image_if_url(p_str, image_download_dir, task_id_for_log)
        if not local_p_str:
            dprint(f"[Task {task_id_for_log}] Skipping image as download_image_if_url returned nothing for: {p_str}")
            continue

        p = Path(local_p_str.strip())
        if not p.is_file():
            dprint(f"[Task {task_id_for_log}] load_pil_images: Image file not found after potential download: {p} (original: {p_str})")
            continue
        try:
            img = Image.open(p)
            images.append(wgp_convert_func(img))
        except Exception as e:
            print(f"[WARNING] Failed to load image {p}: {e}")
            
    return images if images else None

def _normalize_activated_loras_list(current_activated) -> list:
    """Helper to ensure activated_loras is a proper list."""
    if not isinstance(current_activated, list):
        try:
            return [str(item).strip() for item in str(current_activated).split(',') if item.strip()]
        except Exception:
            return []
    return current_activated

def _apply_special_lora_settings(task_id: str, lora_type: str, lora_basename: str, default_steps: int, 
                                 guidance_scale: float, flow_shift: float, ui_defaults: dict, 
                                 task_params_dict: dict, tea_cache_setting: float = None):
    """
    Shared helper to apply special LoRA settings (CausVid, LightI2X, etc.) to ui_defaults.
    """
    print(f"[Task ID: {task_id}] Applying {lora_type} LoRA settings.")
    
    # [STEPS DEBUG] Add detailed debug for steps logic
    print(f"[STEPS DEBUG] {lora_type}: task_params_dict keys: {list(task_params_dict.keys())}")
    if "steps" in task_params_dict:
        print(f"[STEPS DEBUG] {lora_type}: Found 'steps' = {task_params_dict['steps']}")
    if "num_inference_steps" in task_params_dict:
        print(f"[STEPS DEBUG] {lora_type}: Found 'num_inference_steps' = {task_params_dict['num_inference_steps']}")
    if "video_length" in task_params_dict:
        print(f"[STEPS DEBUG] {lora_type}: Found 'video_length' = {task_params_dict['video_length']}")
    
    # Handle steps logic
    if "steps" in task_params_dict:
        ui_defaults["num_inference_steps"] = task_params_dict["steps"]
        print(f"[Task ID: {task_id}] {lora_type} task using specified steps: {ui_defaults['num_inference_steps']}")
    elif "num_inference_steps" in task_params_dict:
        ui_defaults["num_inference_steps"] = task_params_dict["num_inference_steps"]
        print(f"[Task ID: {task_id}] {lora_type} task using specified num_inference_steps: {ui_defaults['num_inference_steps']}")
    else:
        ui_defaults["num_inference_steps"] = default_steps
        print(f"[Task ID: {task_id}] {lora_type} task defaulting to steps: {ui_defaults['num_inference_steps']}")
    
    # Set guidance and flow shift
    ui_defaults["guidance_scale"] = guidance_scale
    ui_defaults["flow_shift"] = flow_shift
    
    # Set tea cache if specified
    if tea_cache_setting is not None:
        ui_defaults["tea_cache_setting"] = tea_cache_setting
    
    # Handle LoRA activation
    current_activated = _normalize_activated_loras_list(ui_defaults.get("activated_loras", []))
    
    if lora_basename not in current_activated:
        current_activated.append(lora_basename)
    ui_defaults["activated_loras"] = current_activated
    
    # Handle multipliers - simple approach for build_task_state
    current_multipliers_str = ui_defaults.get("loras_multipliers", "")
    multipliers_list = [m.strip() for m in current_multipliers_str.split(" ") if m.strip()] if current_multipliers_str else []
    while len(multipliers_list) < len(current_activated):
        multipliers_list.insert(0, "1.0")
    ui_defaults["loras_multipliers"] = " ".join(multipliers_list)

# --- SM_RESTRUCTURE: Function moved from headless.py ---
def build_task_state(wgp_mod, model_filename, task_params_dict, all_loras_for_model, image_download_dir: Path | str | None = None, apply_reward_lora: bool = False, model_type_override: str = None):
    state = {
        "model_filename": model_filename,
        "validate_success": 1,
        "advanced": True,
        "gen": {"queue": [], "file_list": [], "file_settings_list": [], "prompt_no": 1, "prompts_max": 1},
        "loras": all_loras_for_model,
    }
    # [VACE_FIX] Use model_type_override if provided, otherwise derive from filename
    if model_type_override:
        model_type_key = model_type_override
        print(f"[DEBUG] build_task_state: using model_type_override='{model_type_key}'")
        
        # [VACE_FIX] For VACE models, resolve to their underlying base type for WGP compatibility
        # VACE models should use their base model type (t2v) for configuration, not the composite type
        if model_type_key in ["vace_14B", "vace_1.3B", "vace_multitalk_14B"]:
            # Get the URLs field to find the true base model type
            try:
                vace_urls = wgp_mod.get_model_recursive_prop(model_type_key, "URLs", return_list=False)
                if isinstance(vace_urls, str) and vace_urls in ["t2v", "i2v"]:
                    original_model_type = model_type_key
                    model_type_key = vace_urls  # Use the base model type for WGP
                    print(f"[VACE_FIX] Resolved VACE model '{original_model_type}' → base_type '{model_type_key}' for WGP compatibility")
            except Exception as e:
                print(f"[WARNING] Could not resolve VACE base type for '{model_type_key}': {e}")
    else:
        model_type_key = wgp_mod.get_model_type(model_filename)
        if not model_type_key:
            print(f"[ERROR] Could not determine model type from filename: {model_filename}")
            model_type_key = "t2v"  # Fallback to known good model type
        print(f"[DEBUG] build_task_state: model_filename='{model_filename}' → model_type='{model_type_key}'")
    ui_defaults = wgp_mod.get_default_settings(model_type_key).copy()

    # Override with task_params from JSON, but preserve some crucial ones if CausVid is used
    causvid_active = task_params_dict.get("use_causvid_lora", False)
    lighti2x_active = task_params_dict.get("use_lighti2x_lora", False)

    for key, value in task_params_dict.items():
        if key not in ["output_sub_dir", "model", "task_id", "use_causvid_lora", "use_lighti2x_lora"]:
            if (causvid_active or lighti2x_active) and key in ["steps", "guidance_scale", "flow_shift", "activated_loras", "loras_multipliers"]:
                continue # These will be set by causvid/lighti2x logic if flag is true
            ui_defaults[key] = value
    
    ui_defaults["prompt"] = task_params_dict.get("prompt", "Default prompt")
    ui_defaults["resolution"] = task_params_dict.get("resolution", "832x480")
    # Allow task to specify frames/video_length, steps, guidance_scale, flow_shift unless overridden by CausVid
    if not (causvid_active or lighti2x_active):
        ui_defaults["video_length"] = task_params_dict.get("frames", task_params_dict.get("video_length", 81))
        ui_defaults["num_inference_steps"] = task_params_dict.get("steps", task_params_dict.get("num_inference_steps", 30))
        ui_defaults["guidance_scale"] = task_params_dict.get("guidance_scale", ui_defaults.get("guidance_scale", 5.0))
        ui_defaults["flow_shift"] = task_params_dict.get("flow_shift", ui_defaults.get("flow_shift", 3.0))
    else: # CausVid or LightI2X specific defaults if not touched by their logic yet
        ui_defaults["video_length"] = task_params_dict.get("frames", task_params_dict.get("video_length", 81))
        # steps, guidance_scale, flow_shift will be set below by specialised logic

    ui_defaults["seed"] = task_params_dict.get("seed", -1)
    ui_defaults["lset_name"] = "" 

    current_task_id_for_log = task_params_dict.get('task_id', 'build_task_state_unknown')

    if task_params_dict.get("image_start_paths"):
        loaded = load_pil_images(
            task_params_dict["image_start_paths"],
            wgp_mod.convert_image,
            image_download_dir,
            current_task_id_for_log,
            dprint
        )
        if loaded: ui_defaults["image_start"] = loaded

    if task_params_dict.get("image_end_paths"):
        loaded = load_pil_images(
            task_params_dict["image_end_paths"],
            wgp_mod.convert_image,
            image_download_dir,
            current_task_id_for_log,
            dprint
        )
        if loaded: ui_defaults["image_end"] = loaded

    if task_params_dict.get("image_refs_paths"):
        loaded = load_pil_images(
            task_params_dict["image_refs_paths"],
            wgp_mod.convert_image,
            image_download_dir,
            current_task_id_for_log,
            dprint
        )
        if loaded: ui_defaults["image_refs"] = loaded
    
    for key in ["video_source_path", "video_guide_path", "video_mask_path", "audio_guide_path"]:
        if task_params_dict.get(key):
            ui_defaults[key.replace("_path","")] = task_params_dict[key]

    if task_params_dict.get("prompt_enhancer_mode"):
        ui_defaults["prompt_enhancer"] = task_params_dict["prompt_enhancer_mode"]
        wgp_mod.server_config["enhancer_enabled"] = 1
    elif "prompt_enhancer" not in task_params_dict:
        ui_defaults["prompt_enhancer"] = ""
        wgp_mod.server_config["enhancer_enabled"] = 0

    # --- Custom LoRA Handling (e.g., from lora_name) ---
    custom_lora_name_stem = task_params_dict.get("lora_name")
    task_id_for_dprint = task_params_dict.get('task_id', 'N/A') # For logging

    if custom_lora_name_stem:
        custom_lora_filename = f"{custom_lora_name_stem}.safetensors"
        dprint(f"[Task ID: {task_id_for_dprint}] Custom LoRA specified via lora_name: {custom_lora_filename}")

        # Ensure activated_loras is a list
        activated_loras_val = ui_defaults.get("activated_loras", [])
        if isinstance(activated_loras_val, str):
            # Handles comma-separated string from task_params or previous logic
            current_activated_list = [str(item).strip() for item in activated_loras_val.split(',') if item.strip()]
        elif isinstance(activated_loras_val, list):
            current_activated_list = list(activated_loras_val) # Ensure it's a mutable copy
        else:
            dprint(f"[Task ID: {task_id_for_dprint}] Unexpected type for activated_loras: {type(activated_loras_val)}. Initializing as empty list.")
            current_activated_list = []
        
        if custom_lora_filename not in current_activated_list:
            current_activated_list.append(custom_lora_filename)
            dprint(f"[Task ID: {task_id_for_dprint}] Added '{custom_lora_filename}' to activated_loras list: {current_activated_list}")

            # Handle multipliers: Add a default "1.0" if a LoRA was added and multipliers are potentially mismatched
            loras_multipliers_str = ui_defaults.get("loras_multipliers", "")
            if isinstance(loras_multipliers_str, (list, tuple)):
                loras_multipliers_list = [str(m).strip() for m in loras_multipliers_str if str(m).strip()] # Convert all to string and clean
            elif isinstance(loras_multipliers_str, str):
                loras_multipliers_list = [m.strip() for m in loras_multipliers_str.split(" ") if m.strip()] # Space-separated string
            else:
                dprint(f"[Task ID: {task_id_for_dprint}] Unexpected type for loras_multipliers: {type(loras_multipliers_str)}. Initializing as empty list.")
                loras_multipliers_list = []

            # If number of multipliers is less than activated LoRAs, pad with "1.0"
            while len(loras_multipliers_list) < len(current_activated_list):
                loras_multipliers_list.append("1.0")
                dprint(f"[Task ID: {task_id_for_dprint}] Padded loras_multipliers with '1.0'. Now: {loras_multipliers_list}")
            
            ui_defaults["loras_multipliers"] = " ".join(loras_multipliers_list)
        else:
            dprint(f"[Task ID: {task_id_for_dprint}] Custom LoRA '{custom_lora_filename}' already in activated_loras list.")
            
        ui_defaults["activated_loras"] = current_activated_list # Update ui_defaults
    # --- End Custom LoRA Handling ---

    # Apply special LoRA settings using shared helper
    if causvid_active:
        _apply_special_lora_settings(
            current_task_id_for_log, "CausVid", 
            "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
            default_steps=9, guidance_scale=1.0, flow_shift=1.0,
            ui_defaults=ui_defaults, task_params_dict=task_params_dict
        )

    if lighti2x_active:
        _apply_special_lora_settings(
            current_task_id_for_log, "LightI2X",
            "wan_lcm_r16_fp32_comfy.safetensors", 
            default_steps=4, guidance_scale=1.0, flow_shift=5.0,
            ui_defaults=ui_defaults, task_params_dict=task_params_dict,
            tea_cache_setting=0.0
        )
        # Additional LightI2X-specific settings
        ui_defaults["sample_solver"] = "unipc"
        ui_defaults["denoise_strength"] = 1.0

    if apply_reward_lora:
        print(f"[Task ID: {task_params_dict.get('task_id')}] Applying Reward LoRA settings.")

        reward_lora = {"filename": "Wan2.1-Fun-14B-InP-MPS_reward_lora_wgp.safetensors", "strength": "0.5"}

        # Get current activated LoRAs
        current_activated = ui_defaults.get("activated_loras", [])
        if not isinstance(current_activated, list):
            try:
                current_activated = [str(item).strip() for item in str(current_activated).split(',') if item.strip()]
            except:
                current_activated = []

        # Get current multipliers
        current_multipliers_str = ui_defaults.get("loras_multipliers", "")
        if isinstance(current_multipliers_str, (list, tuple)):
            current_multipliers_list = [str(m).strip() for m in current_multipliers_str if str(m).strip()]
        elif isinstance(current_multipliers_str, str):
            current_multipliers_list = [m.strip() for m in current_multipliers_str.split(" ") if m.strip()]
        else:
            current_multipliers_list = []

        # Pad multipliers to match activated LoRAs before creating map
        while len(current_multipliers_list) < len(current_activated):
            current_multipliers_list.append("1.0")

        # Create a dictionary to map lora to multiplier for easy update (preserves order in Python 3.7+)
        lora_mult_map = dict(zip(current_activated, current_multipliers_list))

        # Add/update reward lora
        lora_mult_map[reward_lora['filename']] = reward_lora['strength']

        ui_defaults["activated_loras"] = list(lora_mult_map.keys())
        ui_defaults["loras_multipliers"] = " ".join(list(lora_mult_map.values()))
        dprint(f"Reward LoRA applied. Activated: {ui_defaults['activated_loras']}, Multipliers: {ui_defaults['loras_multipliers']}")

    # Apply additional LoRAs that may have been passed via task params (e.g. from travel orchestrator)
    processed_additional_loras = task_params_dict.get("processed_additional_loras", {})
    if processed_additional_loras:
        dprint(f"[Task ID: {task_id_for_dprint}] Applying processed additional LoRAs: {processed_additional_loras}")
        
        # Get current activated LoRAs and multipliers again, as they may have been modified by other logic.
        current_activated = ui_defaults.get("activated_loras", [])
        if not isinstance(current_activated, list):
            try:
                current_activated = [str(item).strip() for item in str(current_activated).split(',') if item.strip()]
            except:
                current_activated = []

        current_multipliers_str = ui_defaults.get("loras_multipliers", "")
        if isinstance(current_multipliers_str, (list, tuple)):
            current_multipliers_list = [str(m).strip() for m in current_multipliers_str if str(m).strip()]
        elif isinstance(current_multipliers_str, str):
            current_multipliers_list = [m.strip() for m in current_multipliers_str.split(" ") if m.strip()]
        else:
            current_multipliers_list = []

        # Pad multipliers to match activated LoRAs before creating map
        while len(current_multipliers_list) < len(current_activated):
            current_multipliers_list.append("1.0")

        lora_mult_map = dict(zip(current_activated, current_multipliers_list))
        
        # Add/update additional loras - this will overwrite strength if lora was already present
        lora_mult_map.update(processed_additional_loras)

        ui_defaults["activated_loras"] = list(lora_mult_map.keys())
        ui_defaults["loras_multipliers"] = " ".join(list(lora_mult_map.values()))
        dprint(f"Additional LoRAs applied. Final Activated: {ui_defaults['activated_loras']}, Final Multipliers: {ui_defaults['loras_multipliers']}")

    state[model_type_key] = ui_defaults
    return state, ui_defaults

def prepare_output_path(
    task_id: str, 
    filename: str, 
    main_output_dir_base: Path, 
    *, 
    dprint=lambda *_: None,
    custom_output_dir: str | Path | None = None
) -> tuple[Path, str]:
    """
    Prepares the output path for a task's artifact.

    If `custom_output_dir` is provided, it's used as the base. Otherwise,
    the output is placed in a directory named after the task_id inside
    `main_output_dir_base`.
    """
    # Import DB configuration lazily to avoid circular dependencies.
    try:
        from source import db_operations as db_ops  # type: ignore
    except Exception:  # pragma: no cover
        db_ops = None

    # Decide base directory for the file
    if custom_output_dir:
        output_dir_for_task = Path(custom_output_dir)
        dprint(f"Task {task_id}: Using custom output directory: {output_dir_for_task}")
    else:
        if db_ops and db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
            sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
            output_dir_for_task = sqlite_db_parent / "public" / "files"
            dprint(f"Task {task_id}: Using SQLite public files directory: {output_dir_for_task}")
        else:
            # Flatten: put all artefacts directly in the main output dir, no per-task folder
            output_dir_for_task = main_output_dir_base

            # To avoid name collisions we prefix the filename with the task_id
            if not filename.startswith(task_id):
                filename = f"{task_id}_{filename}"

            dprint(
                f"Task {task_id}: Using flattened output directory: {output_dir_for_task} (filename '{filename}')"
            )

    output_dir_for_task.mkdir(parents=True, exist_ok=True)

    final_save_path = output_dir_for_task / filename

    # Build DB path string
    try:
        if db_ops and db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
            sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
            db_output_location = str(final_save_path.relative_to(sqlite_db_parent / "public"))
        else:
            db_output_location = str(final_save_path.relative_to(Path.cwd()))
    except ValueError:
        db_output_location = str(final_save_path.resolve())

    dprint(f"Task {task_id}: final_save_path='{final_save_path}', db_output_location='{db_output_location}'")

    return final_save_path, db_output_location

def prepare_output_path_with_upload(
    task_id: str, 
    filename: str, 
    main_output_dir_base: Path, 
    *, 
    dprint=lambda *_: None,
    custom_output_dir: str | Path | None = None
) -> tuple[Path, str]:
    """
    Prepares the output path for a task's artifact and handles Supabase upload if configured.
    
    Returns:
        tuple[Path, str]: (local_file_path, db_output_location)
        - local_file_path: Where to save the file locally (for generation)
        - db_output_location: What to store in the database (local path or Supabase URL)
    """
    # Import DB configuration lazily to avoid circular dependencies.
    try:
        from source import db_operations as db_ops  # type: ignore
    except Exception:  # pragma: no cover
        db_ops = None

    # First, get the local path where we'll save the file
    local_save_path, initial_db_location = prepare_output_path(
        task_id, filename, main_output_dir_base, 
        dprint=dprint, custom_output_dir=custom_output_dir
    )
    
    # Return the local path for now - we'll handle Supabase upload after file is created
    return local_save_path, initial_db_location

def upload_and_get_final_output_location(
    local_file_path: Path,
    supabase_object_name: str,  # This parameter is now unused but kept for compatibility
    initial_db_location: str,
    *,
    dprint=lambda *_: None
) -> str:
    """
    Returns the local file path. Upload is now handled by the edge function.
    
    Args:
        local_file_path: Path to the local file
        supabase_object_name: Unused (kept for compatibility)
        initial_db_location: The initial DB location (local path)
        dprint: Debug print function
        
    Returns:
        str: Local file path (upload now handled by edge function)
    """
    # Edge function will handle the upload, so we just return the local path
    dprint(f"File ready for edge function upload: {local_file_path}")
    return str(local_file_path.resolve())

def create_mask_video_from_inactive_indices(
    total_frames: int,
    resolution_wh: tuple[int, int], 
    inactive_frame_indices: set[int] | list[int],
    output_path: Path | str,
    fps: int = 16,
    task_id_for_logging: str = "unknown",
    *, dprint = print
) -> Path | None:
    """
    Create a mask video where:
    - Black frames (0) = inactive/keep original - don't edit these frames
    - White frames (255) = active/generate - model should generate these frames
    
    Args:
        total_frames: Total number of frames in the video
        resolution_wh: (width, height) tuple for video resolution
        inactive_frame_indices: Set or list of frame indices that should be black (inactive)
        output_path: Where to save the mask video
        fps: Frames per second for the output video
        task_id_for_logging: Task ID for debug logging
        dprint: Print function for logging
        
    Returns:
        Path to created mask video, or None if creation failed
    """
    try:
        if total_frames <= 0:
            dprint(f"[WARNING] Task {task_id_for_logging}: Cannot create mask video with {total_frames} frames")
            return None
            
        h, w = resolution_wh[1], resolution_wh[0]  # height, width
        inactive_set = set(inactive_frame_indices) if not isinstance(inactive_frame_indices, set) else inactive_frame_indices
        
        dprint(f"Task {task_id_for_logging}: Creating mask video - total_frames={total_frames}, "
               f"inactive_indices={sorted(list(inactive_set))[:10]}{'...' if len(inactive_set) > 10 else ''}")
        
        # Create mask frames: 0 (black) for inactive, 255 (white) for active
        mask_frames_buf: list[np.ndarray] = [
            np.full((h, w, 3), 0 if idx in inactive_set else 255, dtype=np.uint8)
            for idx in range(total_frames)
        ]
        
        created_mask_video = create_video_from_frames_list(
            mask_frames_buf, 
            Path(output_path), 
            fps, 
            resolution_wh
        )
        
        if created_mask_video and created_mask_video.exists():
            dprint(f"Task {task_id_for_logging}: Mask video created successfully at {created_mask_video}")
            return created_mask_video
        else:
            dprint(f"[WARNING] Task {task_id_for_logging}: Failed to create mask video at {output_path}")
            return None
            
    except Exception as e:
        dprint(f"[ERROR] Task {task_id_for_logging}: Mask video creation failed: {e}")
        return None

def create_simple_first_frame_mask_video(
    total_frames: int,
    resolution_wh: tuple[int, int],
    output_path: Path | str, 
    fps: int = 16,
    task_id_for_logging: str = "unknown",
    *, dprint = print
) -> Path | None:
    """
    Convenience function to create a mask video where only the first frame is inactive (black).
    This is useful for workflows like different_perspective where you want to keep the first frame unchanged
    and generate the rest.
    
    Returns:
        Path to created mask video, or None if creation failed
    """
    return create_mask_video_from_inactive_indices(
        total_frames=total_frames,
        resolution_wh=resolution_wh,
        inactive_frame_indices={0},  # Only first frame is inactive
        output_path=output_path,
        fps=fps,
        task_id_for_logging=task_id_for_logging,
        dprint=dprint
    )

def wait_for_file_stable(path: Path | str, checks: int = 3, interval: float = 1.0, *, dprint=print) -> bool:
    """Return True when the file size stays constant for a few consecutive checks.
    Useful to make sure long-running encoders have finished writing before we
    copy/move the file.
    """
    p = Path(path)
    if not p.exists():
        return False
    last_size = p.stat().st_size
    stable_count = 0
    for _ in range(checks):
        time.sleep(interval)
        new_size = p.stat().st_size
        if new_size == last_size and new_size > 0:
            stable_count += 1
            if stable_count >= checks - 1:
                return True
        else:
            stable_count = 0
            last_size = new_size
    return False

def report_orchestrator_failure(task_params_dict: dict, error_msg: str, dprint: callable = print) -> None:
    """Update the parent orchestrator task to FAILED when a sub-task encounters a fatal error.

    Args:
        task_params_dict: The params payload of the *current* sub-task.
            It is expected to contain a reference to the orchestrator via one
            of the standard keys (e.g. ``orchestrator_task_id_ref``).
        error_msg: Human-readable message describing the failure.
        dprint: Debug print helper (typically passed from the caller).
    """
    # Defer import to avoid potential circular dependencies at module import time
    try:
        from source import db_operations as db_ops  # type: ignore
    except Exception as e:  # pragma: no cover
        dprint(f"[report_orchestrator_failure] Could not import db_operations: {e}")
        return

    orchestrator_id = None
    # Common payload keys that may reference the orchestrator task
    for key in (
        "orchestrator_task_id_ref",
        "orchestrator_task_id",
        "parent_orchestrator_task_id",
        "orchestrator_id",
    ):
        orchestrator_id = task_params_dict.get(key)
        if orchestrator_id:
            break

    if not orchestrator_id:
        dprint(
            f"[report_orchestrator_failure] No orchestrator reference found in payload. Message: {error_msg}"
        )
        return

    # Truncate very long messages to avoid DB column overflow
    truncated_msg = error_msg[:500]

    try:
        db_ops.update_task_status(
            orchestrator_id,
            db_ops.STATUS_FAILED,
            truncated_msg,
        )
        dprint(
            f"[report_orchestrator_failure] Marked orchestrator task {orchestrator_id} as FAILED with message: {truncated_msg}"
        )
    except Exception as e_update:  # pragma: no cover
        dprint(
            f"[report_orchestrator_failure] Failed to update orchestrator status for {orchestrator_id}: {e_update}"
        )

def validate_lora_file(file_path: Path, filename: str) -> tuple[bool, str]:
    """
    Validates a LoRA file for size and format integrity.
    
    Returns:
        (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    file_size = file_path.stat().st_size
    
    # Known LoRA size ranges (in bytes)
    # These are based on common LoRA architectures and rank sizes
    LORA_SIZE_RANGES = {
        # Very small LoRAs (rank 4-8)
        'tiny': (1_000_000, 50_000_000),      # 1MB - 50MB
        # Standard LoRAs (rank 16-32) 
        'standard': (50_000_000, 500_000_000),  # 50MB - 500MB
        # Large LoRAs (rank 64+) or full model fine-tunes
        'large': (500_000_000, 5_000_000_000), # 500MB - 5GB
        # Extremely large (full model weights)
        'xlarge': (5_000_000_000, 50_000_000_000)  # 5GB - 50GB
    }
    
    # Check if file size is within any reasonable range
    in_valid_range = any(
        min_size <= file_size <= max_size 
        for min_size, max_size in LORA_SIZE_RANGES.values()
    )
    
    if not in_valid_range:
        if file_size < 1_000_000:  # Less than 1MB
            return False, f"File too small ({file_size:,} bytes) - likely corrupted or incomplete download"
        elif file_size > 50_000_000_000:  # More than 50GB
            return False, f"File too large ({file_size:,} bytes) - likely not a LoRA file"
    
    # For safetensors files, try to open and inspect
    if filename.endswith('.safetensors'):
        try:
            import safetensors.torch as st
            with st.safe_open(file_path, framework="pt") as f:
                # Get metadata to verify it's actually a LoRA
                metadata = f.metadata()
                keys = list(f.keys())
                
                # LoRAs typically have keys like "lora_down.weight", "lora_up.weight", etc.
                lora_indicators = ['lora_down', 'lora_up', 'lora.down', 'lora.up', 'lora_A', 'lora_B']
                has_lora_keys = any(indicator in key for key in keys for indicator in lora_indicators)
                
                if not has_lora_keys and len(keys) > 100:
                    # Might be a full model checkpoint rather than a LoRA
                    print(f"[WARNING] {filename} appears to be a full model checkpoint ({len(keys)} tensors) rather than a LoRA")
                elif not has_lora_keys:
                    print(f"[WARNING] {filename} doesn't appear to contain LoRA weights (no lora_* keys found)")
                
                # Check for reasonable number of parameters
                if len(keys) == 0:
                    return False, "Safetensors file contains no tensors"
                elif len(keys) > 10000:
                    print(f"[WARNING] {filename} contains many tensors ({len(keys)}) - might be a full model")
                    
        except ImportError:
            print(f"[WARNING] safetensors not available for detailed validation of {filename}")
        except Exception as e:
            return False, f"Safetensors file appears corrupted: {e}"
    
    # Additional checks for common corruption patterns
    if file_size == 0:
        return False, "File is empty"
    
    # For binary files, check they don't start with common error HTML patterns
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(1024)
            if first_bytes.startswith(b'<!DOCTYPE html') or first_bytes.startswith(b'<html'):
                return False, "File appears to be an HTML error page rather than a LoRA file"
    except Exception:
        pass
    
    return True, f"File validated successfully ({file_size:,} bytes)"

def check_loras_in_directory(lora_dir: Path | str, fix_issues: bool = False) -> dict:
    """
    Checks all LoRA files in a directory for integrity issues.
    
    Args:
        lora_dir: Directory containing LoRA files
        fix_issues: If True, removes corrupted files
        
    Returns:
        Dictionary with validation results
    """
    lora_dir = Path(lora_dir)
    if not lora_dir.exists():
        return {"error": f"Directory does not exist: {lora_dir}"}
    
    results = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "issues": [],
        "summary": []
    }
    
    # Look for LoRA-like files
    lora_extensions = ['.safetensors', '.bin', '.pt', '.pth']
    lora_files = []
    
    for ext in lora_extensions:
        lora_files.extend(lora_dir.glob(f"*{ext}"))
        lora_files.extend(lora_dir.glob(f"**/*{ext}"))  # Include subdirectories
    
    # Filter to likely LoRA files
    lora_files = [f for f in lora_files if 'lora' in f.name.lower() or f.suffix == '.safetensors']
    
    results["total_files"] = len(lora_files)
    
    for lora_file in lora_files:
        is_valid, validation_msg = validate_lora_file(lora_file, lora_file.name)
        
        if is_valid:
            results["valid_files"] += 1
            results["summary"].append(f"✓ {lora_file.name}: {validation_msg}")
        else:
            results["invalid_files"] += 1
            issue_msg = f"✗ {lora_file.name}: {validation_msg}"
            results["issues"].append(issue_msg)
            results["summary"].append(issue_msg)
            
            if fix_issues:
                try:
                    lora_file.unlink()
                    results["summary"].append(f"  → Removed corrupted file: {lora_file.name}")
                except Exception as e:
                    results["summary"].append(f"  → Failed to remove {lora_file.name}: {e}")
    
    return results

# --------------------------------------------------------------------------------------------------
