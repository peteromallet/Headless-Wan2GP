"""Travel-between-images task handler."""

import json
import shutil
import traceback
from pathlib import Path
import datetime # ADDED
import subprocess

import cv2 # pip install opencv-python
import numpy as np # ADDED
from PIL import Image # ADDED

# Import from the new common_utils module
from .common_utils import (
    DEBUG_MODE, dprint, generate_unique_task_id, add_task_to_db, poll_task_status,
    extract_video_segment_ffmpeg, stitch_videos_ffmpeg,
    create_pose_interpolated_guide_video, # Used for guide video generation
    generate_debug_summary_video, # For debug mode
    # --- New/assumed helpers for this modification ---
    extract_specific_frame_ffmpeg, # Assumed: (input_video_path, frame_index, output_image_path, input_fps)
    concatenate_videos_ffmpeg,    # Assumed: (list_of_video_paths, output_path, temp_dir_for_list_file)
    get_video_frame_count_and_fps # Assumed: (video_path) -> (frame_count, fps)
)

# --- Helper function for generating unique target paths ---
def _get_unique_target_path(target_dir: Path, base_name: str, extension: str) -> Path:
    """Determines a unique path in target_dir by appending _N if base_name+extension exists."""
    target_dir.mkdir(parents=True, exist_ok=True) # Ensure target directory exists
    # Sanitize extension to ensure it starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension
    
    target_file = target_dir / f"{base_name}{extension}"
    counter = 0
    # Loop to find a unique filename
    while target_file.exists():
        counter += 1
        target_file = target_dir / f"{base_name}_{counter}{extension}"
    return target_file

# --- Helper function for unique file copying (now returns destination path) ---
def _copy_to_folder_with_unique_name(source_path: Path | str | None, target_dir: Path, base_name: str, extension: str) -> Path | None:
    """Copies source_path to target_dir with a unique name, returns the destination Path or None."""
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

# --- Helper Functions (adapted from provided example) ---

def image_to_frame(image_path: str | Path, target_size: tuple[int, int]) -> np.ndarray | None:
    """Loads an image, resizes it, and converts to BGR NumPy array for OpenCV."""
    try:
        img = Image.open(image_path).convert("RGB")
        # Using Image.Resampling.LANCZOS for compatibility with modern Pillow versions
        img = img.resize(target_size, Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading or processing image {image_path}: {e}")
        return None

def create_color_frame(size: tuple[int, int], color_bgr: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Creates a single color BGR frame."""
    height, width = size[1], size[0] # size is (width, height)
    frame = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    return frame

def color_match_video_to_reference(source_video_path: str | Path, reference_video_path: str | Path, 
                                 output_video_path: str | Path, parsed_resolution: tuple[int, int]) -> bool:
    """
    Color matches source_video to reference_video using histogram matching on the last frame of reference
    and first frame of source. Applies the transformation to all frames of source_video.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Get reference frame (last frame of reference video)
        ref_cap = cv2.VideoCapture(str(reference_video_path))
        if not ref_cap.isOpened():
            print(f"Error: Could not open reference video {reference_video_path}")
            return False
        
        ref_frame_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, ref_frame_count - 1)))
        ret, ref_frame = ref_cap.read()
        ref_cap.release()
        
        if not ret or ref_frame is None:
            print(f"Error: Could not read reference frame from {reference_video_path}")
            return False
        
        # Resize reference frame if needed
        if ref_frame.shape[1] != parsed_resolution[0] or ref_frame.shape[0] != parsed_resolution[1]:
            ref_frame = cv2.resize(ref_frame, parsed_resolution, interpolation=cv2.INTER_AREA)
        
        # Get source frame (first frame of source video)
        src_cap = cv2.VideoCapture(str(source_video_path))
        if not src_cap.isOpened():
            print(f"Error: Could not open source video {source_video_path}")
            return False
        
        src_fps = src_cap.get(cv2.CAP_PROP_FPS)
        src_frame_count = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, src_first_frame = src_cap.read()
        if not ret or src_first_frame is None:
            print(f"Error: Could not read first frame from {source_video_path}")
            src_cap.release()
            return False
        
        # Resize source frame if needed
        if src_first_frame.shape[1] != parsed_resolution[0] or src_first_frame.shape[0] != parsed_resolution[1]:
            src_first_frame = cv2.resize(src_first_frame, parsed_resolution, interpolation=cv2.INTER_AREA)
        
        # Calculate histogram matching transformation for each channel
        def match_histogram_channel(source_channel, reference_channel):
            """Match histogram of source channel to reference channel."""
            # Calculate histograms
            src_hist, _ = np.histogram(source_channel.flatten(), 256, [0, 256])
            ref_hist, _ = np.histogram(reference_channel.flatten(), 256, [0, 256])
            
            # Calculate cumulative distribution functions
            src_cdf = src_hist.cumsum()
            ref_cdf = ref_hist.cumsum()
            
            # Normalize CDFs
            src_cdf = src_cdf / src_cdf[-1]
            ref_cdf = ref_cdf / ref_cdf[-1]
            
            # Create lookup table
            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # Find closest match in reference CDF
                closest_idx = np.argmin(np.abs(ref_cdf - src_cdf[i]))
                lookup_table[i] = closest_idx
            
            return lookup_table
        
        # Calculate lookup tables for each BGR channel
        lookup_tables = []
        for channel in range(3):  # BGR channels
            lut = match_histogram_channel(src_first_frame[:, :, channel], ref_frame[:, :, channel])
            lookup_tables.append(lut)
        
        # Create output video writer
        output_path_obj = Path(output_video_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path_obj), fourcc, float(src_fps), parsed_resolution)
        if not out.isOpened():
            print(f"Error: Could not create output video writer for {output_path_obj}")
            src_cap.release()
            return False
        
        # Reset source video to beginning
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process all frames
        frames_processed = 0
        while True:
            ret, frame = src_cap.read()
            if not ret:
                break
            
            # Resize frame if needed
            if frame.shape[1] != parsed_resolution[0] or frame.shape[0] != parsed_resolution[1]:
                frame = cv2.resize(frame, parsed_resolution, interpolation=cv2.INTER_AREA)
            
            # Apply color matching transformation
            matched_frame = frame.copy()
            for channel in range(3):
                matched_frame[:, :, channel] = cv2.LUT(frame[:, :, channel], lookup_tables[channel])
            
            out.write(matched_frame)
            frames_processed += 1
        
        src_cap.release()
        out.release()
        
        dprint(f"Color matching complete: {frames_processed} frames processed from {source_video_path} to {output_video_path}")
        return True
        
    except Exception as e:
        print(f"Error during color matching: {e}")
        traceback.print_exc()
        return False

def create_video_from_frames_list(
    frames_list: list[np.ndarray],
    output_path: str | Path,
    fps: int,
    resolution: tuple[int, int] # width, height
):
    """Creates an MP4 video from a list of NumPy BGR frames."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    try:
        out = cv2.VideoWriter(str(output_path_obj), fourcc, float(fps), resolution)
        if not out.isOpened():
            raise IOError(f"Could not open video writer for {output_path_obj}")

        for frame_np in frames_list:
            if frame_np.shape[1] != resolution[0] or frame_np.shape[0] != resolution[1]:
                frame_np_resized = cv2.resize(frame_np, resolution, interpolation=cv2.INTER_AREA)
                out.write(frame_np_resized)
            else:
                out.write(frame_np)
        # dprint(f"Generated video: {output_path_obj} ({len(frames_list)} frames)") # dprint might not be defined here yet.
        print(f"Generated video via create_video_from_frames_list: {output_path_obj} ({len(frames_list)} frames)")
    finally:
        if out:
            out.release()

# --- Helper: Re-encode a video to H.264 using FFmpeg (ensures consistent codec) ---
def _reencode_to_h264_ffmpeg(
    input_video_path: str | Path,
    output_video_path: str | Path,
    fps: float | None = None,
    resolution: tuple[int, int] | None = None,
    crf: int = 23,
    preset: str = "veryfast"
):
    """Re-encodes the entire input video to H.264 using libx264."""
    inp = Path(input_video_path)
    outp = Path(output_video_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(inp.resolve()),
        "-an",  # no audio
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
    ]
    if fps is not None and fps > 0:
        cmd.extend(["-r", str(fps)])
    if resolution is not None:
        w, h = resolution
        cmd.extend(["-vf", f"scale={w}:{h}"])
    cmd.append(str(outp.resolve()))

    dprint(f"REENCODE_TO_H264_FFMPEG: Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        return outp.exists() and outp.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg re-encode of {inp} -> {outp}:\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
        return False

def run_travel_between_images_task(task_args, common_args, parsed_resolution, main_output_dir, db_file_path):
    print("--- Running Task: Travel Between Images ---")
    dprint(f"Task Args: {task_args}")
    dprint(f"Common Args: {common_args}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Use timestamp as a unique identifier for this run. This will be referenced when
    # naming the run-specific processing folder as well as the persisted final video.
    run_id = timestamp

    run_processing_folder_name: str
    if DEBUG_MODE:
        run_processing_folder_name = f"travel_debug_{run_id}"
    else:
        run_processing_folder_name = f"travel_temp_{run_id}"
    
    run_processing_folder = main_output_dir / run_processing_folder_name
    run_processing_folder.mkdir(parents=True, exist_ok=True)
    dprint(f"All intermediate and final files for this run will be processed in: {run_processing_folder}")


    all_segments_debug_data = [] # For debug collage

    if len(task_args.input_images) < 2:
        print("Error: At least two input images are required for 'travel_between_images' task.")
        # Consider cleanup of run_processing_folder if needed here, though it's empty.
        return 1
    if len(task_args.base_prompts) != len(task_args.input_images) - 1:
        print(f"Error: Number of prompts ({len(task_args.base_prompts)}) must be one less than "
              f"the number of input images ({len(task_args.input_images)}) for 'travel_between_images' task.")
        return 1
    
    if task_args.segment_frames <= 0:
        print("Error: --segment_frames must be positive.")
        return 1
    if task_args.frame_overlap < 0:
        print("Error: --frame_overlap cannot be negative.")
        return 1
    if task_args.frame_overlap > 0 and task_args.segment_frames <= task_args.frame_overlap:
        print(f"Error: --segment_frames ({task_args.segment_frames}) must be greater than --frame_overlap ({task_args.frame_overlap}) when overlap is used.")
        return 1

    generated_segments_for_stitching = []
    path_to_previous_segment_final_untrimmed_output: str | None = None # Stores the UNTRIMMED, (potentially) color-matched output of the PREVIOUS segment
    num_segments_to_generate = len(task_args.input_images) - 1

    for i in range(num_segments_to_generate):
        print(f"--- Processing Segment {i+1} / {num_segments_to_generate} (Travel Task) ---")
        
        task_id = generate_unique_task_id(f"sm_travel_seg{i:02d}_")
        # segment_work_dir is no longer created by this script. 
        # headless.py uses output_sub_dir relative to its own main output.

        current_segment_end_anchor_image_path = Path(task_args.input_images[i+1]).resolve()
        current_prompt = task_args.base_prompts[i]

        if not current_segment_end_anchor_image_path.exists():
            print(f"Error: End anchor image not found: {current_segment_end_anchor_image_path} for segment {i+1}. Skipping segment.")
            continue

        actual_guide_video_path = _get_unique_target_path(run_processing_folder, f"s{i}_guide", ".mp4")
        dprint(f"Guide video for segment {i+1} will be generated at: {actual_guide_video_path}")

        try:
            if i == 0 or task_args.frame_overlap == 0 or path_to_previous_segment_final_untrimmed_output is None:
                # --- First segment or no overlap: Standard guide video generation ---
                print(f"Creating initial guide video: {actual_guide_video_path}")
                current_segment_start_anchor_image_path = Path(task_args.input_images[i]).resolve()
                if not current_segment_start_anchor_image_path.exists():
                    print(f"Error: Start anchor image not found: {current_segment_start_anchor_image_path} for segment {i+1}. Skipping segment.")
                    continue
                
                if task_args.segment_frames <= 0:
                    print(f"Warning: Cannot create guide video with {task_args.segment_frames} frames. Skipping guide for {actual_guide_video_path}.")
                else:
                    frames_for_guide = []
                    start_anchor_frame_np = image_to_frame(current_segment_start_anchor_image_path, parsed_resolution)
                    end_anchor_frame_np = image_to_frame(current_segment_end_anchor_image_path, parsed_resolution)

                    if start_anchor_frame_np is None:
                        raise ValueError(f"Failed to load start anchor image for guide video: {current_segment_start_anchor_image_path}")
                    if end_anchor_frame_np is None:
                        raise ValueError(f"Failed to load end anchor image for guide video: {current_segment_end_anchor_image_path}")

                    gray_frame_bgr = create_color_frame(parsed_resolution, (128, 128, 128))
                    frames_for_guide = [gray_frame_bgr for _ in range(task_args.segment_frames)]
                    
                    if task_args.segment_frames > 0:
                        frames_for_guide[0] = start_anchor_frame_np
                        if task_args.segment_frames > 1: # Ensure there's a last frame to set if more than 1 frame
                             frames_for_guide[task_args.segment_frames - 1] = end_anchor_frame_np
                        elif task_args.segment_frames == 1: # If only one frame, it's already set to start_anchor
                            pass # No need to set end_anchor if it's the same as start for a single frame video

                    create_video_from_frames_list(
                        frames_list=frames_for_guide,
                        output_path=actual_guide_video_path, # Use actual_guide_video_path
                        fps=common_args.fps_helpers,
                        resolution=parsed_resolution
                    )
                # --- End new logic for first segment guide video ---
            else:
                # --- Subsequent segments with overlap: New simplified guide video generation ---
                print(f"Creating guide video for segment {i+1} ({actual_guide_video_path}) using overlap (new logic).")

                if task_args.segment_frames <= 0:
                    print(f"Warning: Cannot create guide video with {task_args.segment_frames} frames. Skipping guide for {actual_guide_video_path}.")
                else:
                    frames_for_guide = []
                    end_anchor_frame_np = image_to_frame(current_segment_end_anchor_image_path, parsed_resolution)
                    if end_anchor_frame_np is None:
                        raise ValueError(f"Failed to load end anchor image for guide video: {current_segment_end_anchor_image_path}")

                    gray_frame_bgr = create_color_frame(parsed_resolution, (128, 128, 128))
                    frames_for_guide = [gray_frame_bgr for _ in range(task_args.segment_frames)]

                    # This is a SUBSEQUENT segment, using overlap from previous headless output
                    cap = cv2.VideoCapture(str(path_to_previous_segment_final_untrimmed_output))
                    if not cap.isOpened():
                        # Clean up the temporary directory if it exists and was created
                        # if guide_construction_temp_dir.exists(): shutil.rmtree(guide_construction_temp_dir)
                        raise IOError(f"Could not open previous video segment: {path_to_previous_segment_final_untrimmed_output}")
                    
                    prev_total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # Ensure frame_overlap is not greater than what's available in prev_total_frames_count
                    actual_overlap_frames = min(task_args.frame_overlap, prev_total_frames_count)
                    
                    start_extraction_idx = max(0, prev_total_frames_count - actual_overlap_frames)
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_extraction_idx))
                    frames_read_for_overlap = 0
                    for k in range(min(actual_overlap_frames, task_args.segment_frames)): # Iterate up to actual_overlap_frames or total_frames for guide
                        ret, frame = cap.read()
                        if not ret:
                            print(f"Warning: Short read from {path_to_previous_segment_final_untrimmed_output} for overlap. Using gray frames for remainder of overlap.")
                            break
                        if frame.shape[1] != parsed_resolution[0] or frame.shape[0] != parsed_resolution[1]:
                            frame = cv2.resize(frame, parsed_resolution, interpolation=cv2.INTER_AREA)
                        frames_for_guide[k] = frame
                        frames_read_for_overlap += 1
                    cap.release()
                    dprint(f"Read {frames_read_for_overlap} frames for overlap from {path_to_previous_segment_final_untrimmed_output}")

                    # Set the last frame to the end_anchor_image for all segments
                    if task_args.segment_frames > 0:
                        # If only one frame and overlap is used, it might have been overwritten by overlap. Ensure last frame is end_anchor.
                        frames_for_guide[task_args.segment_frames - 1] = end_anchor_frame_np

                    create_video_from_frames_list(
                        frames_list=frames_for_guide,
                        output_path=actual_guide_video_path, # Use actual_guide_video_path
                        fps=common_args.fps_helpers,
                        resolution=parsed_resolution
                    )
                    
                    # Remove the old temp dir logic as it's not used by this new approach
                    # guide_construction_temp_dir = segment_work_dir / "guide_construction_temp"
                    # if guide_construction_temp_dir.exists() and not DEBUG_MODE:
                    #    shutil.rmtree(guide_construction_temp_dir)
                # --- End new simplified guide video generation ---

        except Exception as e_helpers:
            print(f"Error creating helper videos for segment {i+1} (Task ID: {task_id}): {e_helpers}. Skipping segment.")
            traceback.print_exc()
            continue
        
        if not actual_guide_video_path.exists() or actual_guide_video_path.stat().st_size == 0:
            print(f"Error: Guide video {actual_guide_video_path} was not created or is empty for segment {i+1}. Skipping segment.")
            continue

        headless_output_sub_dir_name = f"segment_travel_{i:02d}_{task_id}"

        task_payload = {
            "task_id": task_id,
            "prompt": current_prompt,
            "model": common_args.model_name,
            "resolution": common_args.resolution,
            "frames": task_args.segment_frames, # Headless always generates full segment frames
            "seed": common_args.seed + i,
            "video_guide_path": str(actual_guide_video_path.resolve()), # Use resolved actual_guide_video_path          
            "output_sub_dir": headless_output_sub_dir_name, # For local output structure with SQLite
            # output_path is determined by headless.py logic based on main_output_dir and task_id/output_sub_dir
        }

        if common_args.use_causvid_lora:
            task_payload["use_causvid_lora"] = True
            dprint(f"CausVid LoRA enabled for task {task_id}.")
        
        dprint(f"Task payload for headless.py: {json.dumps(task_payload, indent=4)}")
        
        try:
            add_task_to_db(task_payload, db_file_path)
        except Exception as e_db_add:
            print(f"Failed to add task {task_id} to DB: {e_db_add}. Skipping segment.")
            continue

        # Poll for status, headless will save to its own output dir structure
        # e.g., {main_output_dir_from_headless_args}/{output_sub_dir_from_payload}/{task_id}.mp4
        raw_output_video_location_from_db = poll_task_status(task_id, db_file_path, common_args.poll_interval, common_args.poll_timeout)
        
        dprint(f"DB output location for task {task_id}: {raw_output_video_location_from_db}")
        
        if not raw_output_video_location_from_db:
            print(f"Task {task_id} for segment {i+1} failed or timed out. Skipping.")
            continue 
        
        current_headless_raw_output_path = Path(raw_output_video_location_from_db)
        if not current_headless_raw_output_path.exists() or current_headless_raw_output_path.stat().st_size == 0:
            print(f"Error: Headless output video {current_headless_raw_output_path} is missing or empty for task {task_id}. Skipping segment.")
            continue
        
        raw_output_copy_in_processing_folder = _copy_to_folder_with_unique_name(
            source_path=current_headless_raw_output_path,
            target_dir=run_processing_folder,
            base_name=f"s{i}_raw_output",
            extension=current_headless_raw_output_path.suffix or ".mp4"
        )
        if not raw_output_copy_in_processing_folder:
            print(f"Warning: Failed to copy raw headless output {current_headless_raw_output_path} to {run_processing_folder}. The original raw path will be used where needed, but it might not persist if it's in a temporary location managed by another process.")
        
        source_video_for_current_segment_processing = raw_output_copy_in_processing_folder or current_headless_raw_output_path

        # --- Reprocessing for the first segment (i == 0) ---
        initial_raw_output_for_segment_0_debug: Path | None = None # For debug
        reprocessed_output_s0_debug: Path | None = None # For debug
        reprocess_task_id_s0_debug: str | None = None # For debug

        if i == 0:
            dprint(f"Segment {i+1} is the first segment. Preparing for reprocessing its initial output.")
            initial_raw_output_for_segment_0_debug = source_video_for_current_segment_processing # Store for debug

            num_frames_in_initial_output, _ = get_video_frame_count_and_fps(str(source_video_for_current_segment_processing))
            
            if num_frames_in_initial_output is None:
                print(f"Warning: Could not get frame count for initial output {source_video_for_current_segment_processing}. Using task_args.segment_frames ({task_args.segment_frames}) for reprocessing.")
                num_frames_in_initial_output = task_args.segment_frames # Fallback
            
            if num_frames_in_initial_output > 0:
                reprocess_task_id = generate_unique_task_id(f"sm_travel_seg{i:02d}_reproc_")
                reprocess_task_id_s0_debug = reprocess_task_id # Store for debug
                reprocess_headless_output_sub_dir_name = f"segment_travel_{i:02d}_{reprocess_task_id}_reprocess"

                reprocess_task_payload = {
                    "task_id": reprocess_task_id,
                    "prompt": current_prompt, # Reuse original prompt
                    "model": common_args.model_name,
                    "resolution": common_args.resolution,
                    "frames": num_frames_in_initial_output, # Use all frames from the initial output
                    "seed": common_args.seed + i + 1000, # Offset seed
                    "video_guide_path": str(source_video_for_current_segment_processing.resolve()), # Use initial output as guide
                    "output_sub_dir": reprocess_headless_output_sub_dir_name,
                }
                if common_args.use_causvid_lora:
                    reprocess_task_payload["use_causvid_lora"] = True
                
                dprint(f"Reprocessing payload for segment 0: {json.dumps(reprocess_task_payload, indent=4)}")
                
                try:
                    add_task_to_db(reprocess_task_payload, db_file_path)
                    reprocessed_video_location_from_db = poll_task_status(reprocess_task_id, db_file_path, common_args.poll_interval, common_args.poll_timeout)

                    if reprocessed_video_location_from_db:
                        current_reprocessed_headless_output_path = Path(reprocessed_video_location_from_db)
                        if current_reprocessed_headless_output_path.exists() and current_reprocessed_headless_output_path.stat().st_size > 0:
                            reprocessed_output_copy_in_processing_folder = _copy_to_folder_with_unique_name(
                                source_path=current_reprocessed_headless_output_path,
                                target_dir=run_processing_folder,
                                base_name=f"s{i}_reprocessed_output",
                                extension=current_reprocessed_headless_output_path.suffix or ".mp4"
                            )
                            if reprocessed_output_copy_in_processing_folder:
                                dprint(f"First segment reprocessed successfully. Using {reprocessed_output_copy_in_processing_folder.name} as its definitive output.")
                                source_video_for_current_segment_processing = reprocessed_output_copy_in_processing_folder
                                reprocessed_output_s0_debug = source_video_for_current_segment_processing # Store for debug
                            else:
                                print(f"Warning: Failed to copy reprocessed output for segment 0. Using original reprocessed path {current_reprocessed_headless_output_path} (might be temporary).")
                                source_video_for_current_segment_processing = current_reprocessed_headless_output_path
                                reprocessed_output_s0_debug = source_video_for_current_segment_processing # Store for debug
                        else:
                            print(f"Warning: Reprocessed video for segment 0 ({current_reprocessed_headless_output_path}) is missing or empty. Using initial output (path: {initial_raw_output_for_segment_0_debug}).")
                            # source_video_for_current_segment_processing remains initial_raw_output_for_segment_0_debug
                    else:
                        print(f"Warning: Reprocessing task {reprocess_task_id} for segment 0 failed or timed out. Using initial output (path: {initial_raw_output_for_segment_0_debug}).")
                        # source_video_for_current_segment_processing remains initial_raw_output_for_segment_0_debug
                except Exception as e_reprocess:
                    print(f"Error during reprocessing of segment 0: {e_reprocess}. Using initial output (path: {initial_raw_output_for_segment_0_debug}).")
                    traceback.print_exc()
                    # source_video_for_current_segment_processing remains initial_raw_output_for_segment_0_debug
            else:
                print(f"Warning: Initial output for segment 0 ({source_video_for_current_segment_processing}) has no frames or frame count couldn't be determined. Skipping reprocessing.")
                # source_video_for_current_segment_processing remains initial_raw_output_for_segment_0_debug
        # --- End of Reprocessing for the first segment ---

        # Prepare debug data entry (processed_video_path_for_stitch is known, even if file not created yet)
        # This must be done before the actual processing in case of early exit or error during processing.
        segment_debug_info_entry = {}
        if DEBUG_MODE:
            segment_debug_info_entry = {
                "segment_index": i,
                "task_id": task_id, # Original task_id for this segment
                "guide_video_path": str(actual_guide_video_path.resolve()),
                "raw_headless_output_path": str(source_video_for_current_segment_processing.resolve()), # This is the definitive output for *this* segment (reprocessed if i=0 and successful)
                "task_payload": task_payload # Original task payload for this segment
            }
            if i == 0:
                if initial_raw_output_for_segment_0_debug:
                    segment_debug_info_entry["s0_initial_raw_output_path"] = str(initial_raw_output_for_segment_0_debug.resolve())
                if reprocessed_output_s0_debug and reprocessed_output_s0_debug != initial_raw_output_for_segment_0_debug : # Only add if different
                    segment_debug_info_entry["s0_reprocessed_raw_output_path"] = str(reprocessed_output_s0_debug.resolve())
                if reprocess_task_id_s0_debug:
                    segment_debug_info_entry["s0_reprocess_task_id"] = reprocess_task_id_s0_debug
            # Note: "processed_video_path_for_stitch" will be updated/added after trimming logic below.
            # We will also add color matching specific debug info here.
            dprint(f"Collected initial debug info stub for segment {i}: {task_id}")

        # --- Color Matching Block (COMMENTED OUT) ---
        # This section color matches the current segment's output (source_video_for_current_segment_processing)
        # against the final output of the *previous* segment (path_to_previous_segment_final_untrimmed_output).
        # The result (video_product_of_current_segment) is then used for trimming and becomes the reference for the *next* segment.
        video_product_of_current_segment = source_video_for_current_segment_processing # Start with the raw (or copied raw) output
        # current_segment_color_matched_output_path_for_debug: Path | None = None # For debug info

        # if i > 0 and path_to_previous_segment_final_untrimmed_output:
        #     # Apply color matching to match current segment's output to the (color-matched) output of the previous segment
        #     current_segment_color_matched_temp_path = _get_unique_target_path(
        #         run_processing_folder,
        #         f"s{i}_cmatch_to_prev", # Distinct name for this color-matched version
        #         ".mp4"
        #     )
        #     dprint(f"Applying color matching for segment {i+1}: "
        #            f"matching {source_video_for_current_segment_processing.name} "
        #            f"to {Path(path_to_previous_segment_final_untrimmed_output).name}")
        #
        #     color_match_success = color_match_video_to_reference(
        #         source_video_path=source_video_for_current_segment_processing, # Current segment's raw output
        #         reference_video_path=path_to_previous_segment_final_untrimmed_output, # Previous segment's final (color-matched, untrimmed) output
        #         output_video_path=current_segment_color_matched_temp_path,
        #         parsed_resolution=parsed_resolution
        #     )
        #
        #     if color_match_success and current_segment_color_matched_temp_path.exists() and current_segment_color_matched_temp_path.stat().st_size > 0:
        #         video_product_of_current_segment = current_segment_color_matched_temp_path
        #         current_segment_color_matched_output_path_for_debug = current_segment_color_matched_temp_path
        #         dprint(f"Color matching successful for segment {i+1}. Using this color-matched video for further processing and as next reference.")
        #     else:
        #         print(f"Warning: Color matching failed for segment {i+1} against previous segment's output. Using original output for this segment.")
        #         # video_product_of_current_segment remains source_video_for_current_segment_processing
        # else:
        #     dprint(f"Segment {i+1}: No color matching needed (first segment or no previous segment output available to reference).")

        # Update debug info with this segment's color matching outcome
        if DEBUG_MODE and segment_debug_info_entry:
            # segment_debug_info_entry["current_segment_color_matched_output_path"] = str(current_segment_color_matched_output_path_for_debug.resolve()) if current_segment_color_matched_output_path_for_debug else None
            # segment_debug_info_entry["current_segment_color_matching_applied"] = current_segment_color_matched_output_path_for_debug is not None
            segment_debug_info_entry["color_matching_skipped"] = True # Indicate color matching was skipped for this segment.
            segment_debug_info_entry["video_product_before_trimming"] = str(video_product_of_current_segment.resolve())


        # Update path_to_previous_segment_final_untrimmed_output for the *next* iteration's guide video and color matching reference.
        # This is the (potentially color-matched) full output of the *current* segment.
        path_to_previous_segment_final_untrimmed_output = str(video_product_of_current_segment.resolve())

        # --- Process the (potentially color-matched) video for stitching (trimming logic) ---
        actual_processed_segment_for_stitch_path = _get_unique_target_path(
            run_processing_folder,
            f"s{i}_processed_stitch",
            ".mp4"
        )
        dprint(f"Segment {i+1}'s final version for stitch list will be at: {actual_processed_segment_for_stitch_path}")

        # Get frame count from video_product_of_current_segment (which is now color-matched if applicable)
        num_frames_in_segment_output, segment_output_fps = get_video_frame_count_and_fps(str(video_product_of_current_segment))
        if num_frames_in_segment_output is None:
            print(f"Warning: Could not determine frame count/FPS of {video_product_of_current_segment}. Assuming {task_args.segment_frames} frames and {common_args.fps_helpers} FPS.")
            num_frames_in_segment_output = task_args.segment_frames
            segment_output_fps = float(common_args.fps_helpers)
        
        # Logic for preparing segment for stitching (trim end of non-last segments)
        start_extraction_idx_for_stitch = 0
        frames_to_keep_for_stitch = num_frames_in_segment_output

        is_last_overall_segment = (i == num_segments_to_generate - 1)

        if task_args.frame_overlap > 0:
            if not is_last_overall_segment:
                frames_to_keep_for_stitch = max(0, num_frames_in_segment_output - task_args.frame_overlap)
                dprint(f"Segment {i+1} (intermediate with overlap): Trimming end by {task_args.frame_overlap}. Original in {video_product_of_current_segment.name}: {num_frames_in_segment_output}, Keeping: {frames_to_keep_for_stitch} frames.")
            else:
                dprint(f"Segment {i+1} (last overall with overlap): Keeping all {num_frames_in_segment_output} frames from {video_product_of_current_segment.name}.")
        else: # No overlap requested
            dprint(f"Segment {i+1} (no overlap): Keeping all {num_frames_in_segment_output} frames from {video_product_of_current_segment.name}.")

        if frames_to_keep_for_stitch <= 0:
            print(f"Segment {i+1} (Task {task_id}) results in {frames_to_keep_for_stitch} frames after overlap processing. Not adding to final video.")
            # actual_processed_segment_for_stitch_path will be empty/non-existent
        elif frames_to_keep_for_stitch == num_frames_in_segment_output:
            # No trimming needed, but ensure codec consistency by re-encoding to H.264 via FFmpeg
            dprint(f"Segment {i+1} ({task_id}): Re-encoding full output from {video_product_of_current_segment.name} ({frames_to_keep_for_stitch} frames) to H.264 at {actual_processed_segment_for_stitch_path.name} for stitching.")
            reencode_success = _reencode_to_h264_ffmpeg(
                input_video_path=video_product_of_current_segment,
                output_video_path=actual_processed_segment_for_stitch_path,
                fps=segment_output_fps,
                resolution=parsed_resolution
            )
            if not reencode_success:
                print(f"Warning: Failed to re-encode {video_product_of_current_segment} to H.264. Attempting direct copy (may cause codec mismatch).")
                shutil.copy(str(video_product_of_current_segment.resolve()), str(actual_processed_segment_for_stitch_path.resolve()))
        else:
            # Trimming is needed
            dprint(f"Processing {video_product_of_current_segment.name}: Keeping first {frames_to_keep_for_stitch} frames to {actual_processed_segment_for_stitch_path.name}.")
            extract_video_segment_ffmpeg(
                input_video_path=video_product_of_current_segment, # Use the (potentially color-matched) segment
                output_video_path=actual_processed_segment_for_stitch_path,
                start_frame_index=start_extraction_idx_for_stitch,
                num_frames_to_keep=frames_to_keep_for_stitch,
                input_fps=segment_output_fps,
                resolution=parsed_resolution
            )

        # Update debug info with the path of the segment that will actually be stitched
        if DEBUG_MODE and segment_debug_info_entry:
            segment_debug_info_entry["processed_video_path_for_stitch"] = str(actual_processed_segment_for_stitch_path.resolve()) if actual_processed_segment_for_stitch_path.exists() and actual_processed_segment_for_stitch_path.stat().st_size > 0 else None
            all_segments_debug_data.append(segment_debug_info_entry) # Now add the complete entry
            dprint(f"Collected full debug info for segment {i}: {task_id}")
        
        # --- Original logic for adding to stitch list ---
        if actual_processed_segment_for_stitch_path.exists() and actual_processed_segment_for_stitch_path.stat().st_size > 0:
            generated_segments_for_stitching.append(str(actual_processed_segment_for_stitch_path.resolve()))
            dprint(f"Added '{actual_processed_segment_for_stitch_path.name}' to stitch list.")
        else:
            print(f"Warning: Processed segment {actual_processed_segment_for_stitch_path} is empty or missing after processing. Not adding to stitch list.")
            # If debug info was added, its 'processed_video_path_for_stitch' will point to a non-existent/empty file.
            dprint(f"Segment '{actual_processed_segment_for_stitch_path.name}' was not added to stitch list (missing or empty).")

        print(f"Segment {i+1} (Task ID: {task_id}) processing complete.")

    final_video_output_path_in_processing_folder: Path | None = None
    if not generated_segments_for_stitching:
        print("\nNo video segments were successfully generated/processed for stitching.")
    else:
        final_video_output_path_in_processing_folder = _get_unique_target_path(
            run_processing_folder, 
            "final_travel_video", 
            ".mp4"
        )
        print(f"\nStitching {len(generated_segments_for_stitching)} segments into {final_video_output_path_in_processing_folder}...")
        try:
            stitch_videos_ffmpeg(generated_segments_for_stitching, str(final_video_output_path_in_processing_folder))
            # Success message handled in DEBUG_MODE / non-DEBUG_MODE section below
        except Exception as e_stitch:
            print(f"Error during final video stitching: {e_stitch}")
            dprint(f"Exception during stitching: {traceback.format_exc()}")
            print("You may find processed segments in:", run_processing_folder)
            for p_stitch_fail in generated_segments_for_stitching: print(f"- {p_stitch_fail}")
            final_video_output_path_in_processing_folder = None # Mark as failed

    # --- Handle Outputs and Cleanup ---
    if DEBUG_MODE:
        if all_segments_debug_data and final_video_output_path_in_processing_folder and final_video_output_path_in_processing_folder.exists():
            collage_output_path = _get_unique_target_path(
                run_processing_folder,
                "debug_travel_summary_collage",
                ".mp4"
            )
            dprint(f"Generating debug summary collage video at: {collage_output_path}")
            try:
                generate_debug_summary_video(all_segments_debug_data, str(collage_output_path),
                                             fps=common_args.fps_helpers,
                                             num_frames_for_collage=task_args.segment_frames)
                print(f"Debug summary collage video created: {collage_output_path}")
            except Exception as e_collage:
                print(f"Error generating debug summary collage video: {e_collage}")
                dprint(f"Exception during debug collage generation: {traceback.format_exc()}")
        
        if final_video_output_path_in_processing_folder and final_video_output_path_in_processing_folder.exists():
            print(f"\nDEBUG MODE: All run files are located in: {run_processing_folder.resolve()}")
            print(f"Final video (debug): {final_video_output_path_in_processing_folder.resolve()}")
        elif generated_segments_for_stitching: # Stitching was attempted
             print(f"\nDEBUG MODE: All run files are located in: {run_processing_folder.resolve()}")
             print(f"Final video stitching may have failed. Check logs and contents of the folder.")
        else: # No segments were even generated for stitching
            print(f"\nDEBUG MODE: No segments were available for stitching. Run files are in: {run_processing_folder.resolve()}")
        # In DEBUG_MODE, run_processing_folder is kept.
    
    else: # Not DEBUG_MODE
        final_video_persisted_path: Path | None = None
        if final_video_output_path_in_processing_folder and final_video_output_path_in_processing_folder.exists():
            # Persist the final video directly in the base output directory using the
            # run_id as its filename:  output/{run_id}.mp4 (with automatic de-duplication).
            final_destination_for_output_video = _get_unique_target_path(
                main_output_dir,
                run_id,  # e.g. 20240609_142530_123456.mp4
                ".mp4"
            )
            try:
                shutil.copy2(str(final_video_output_path_in_processing_folder), str(final_destination_for_output_video))
                print(f"\nFinal video successfully created: {final_destination_for_output_video.resolve()}")
                final_video_persisted_path = final_destination_for_output_video
            except Exception as e_copy_final:
                print(f"\nError copying final video from {run_processing_folder} to {main_output_dir}: {e_copy_final}")
                print(f"The final video may still be available at: {final_video_output_path_in_processing_folder}")
                # Even if copying fails we will proceed with cleanup (when not in DEBUG_MODE).
        
        elif generated_segments_for_stitching: # Stitching attempted but no final video in processing folder
            print("\nFinal video stitching failed or produced no output. No final video to persist.")
        else: # No segments generated
            print("\nNo video segments were generated or processed. No final video created.")

        # Always clean up the run-specific processing folder when not running in DEBUG_MODE
        print(f"\nCleaning up temporary processing folder: {run_processing_folder}...")
        try:
            shutil.rmtree(run_processing_folder)
            print(f"Removed temporary processing folder: {run_processing_folder}")
        except OSError as e_clean:
            print(f"Error removing temporary processing folder {run_processing_folder}: {e_clean}")


    return 0 