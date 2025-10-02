import math
import subprocess
from pathlib import Path
import traceback
import os
import sys
import json
import time

try:
    import cv2  # pip install opencv-python
    import numpy as np
    from PIL import Image
    import torch
    try:
        import moviepy.editor as mpe  # pip install moviepy
        _MOVIEPY_AVAILABLE = True
    except ImportError:
        _MOVIEPY_AVAILABLE = False
    _COLOR_MATCH_DEPS_AVAILABLE = True

    # Add project root to path to allow absolute imports from source
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Now that root is in path, we can import from Wan2GP and source
    from Wan2GP.postprocessing.rife.inference import temporal_interpolation
    from source.common_utils import (
        dprint, get_video_frame_count_and_fps,
        download_image_if_url, sm_get_unique_target_path, 
        _apply_strength_to_image as sm_apply_strength_to_image,
        create_color_frame as sm_create_color_frame, 
        image_to_frame as sm_image_to_frame, 
        _adjust_frame_brightness as sm_adjust_frame_brightness,
        get_easing_function as sm_get_easing_function,
        wait_for_file_stable as sm_wait_for_file_stable
    )
except ImportError as e_import:
    print(f"Critical import error in video_utils.py: {e_import}")
    traceback.print_exc()
    _COLOR_MATCH_DEPS_AVAILABLE = False

def crossfade_ease(alpha_lin: float) -> float:
    """Cosine ease-in-out function (maps 0..1 to 0..1).
    Used to determine the blending alpha for crossfades.
    """
    return (1 - math.cos(alpha_lin * math.pi)) / 2.0

def _blend_linear(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0-t, b, t, 0)

def _blend_linear_sharp(a: np.ndarray, b: np.ndarray, t: float, amt: float) -> np.ndarray:
    base = _blend_linear(a,b,t)
    if amt<=0: return base
    blur = cv2.GaussianBlur(base,(0,0),3)
    return cv2.addWeighted(base, 1.0+amt*t, blur, -amt*t, 0)

def cross_fade_overlap_frames(
    segment1_frames: list[np.ndarray],
    segment2_frames: list[np.ndarray],
    overlap_count: int,
    mode: str = "linear_sharp",
    sharp_amt: float = 0.3
) -> list[np.ndarray]:
    """
    Cross-fades the overlapping frames between two segments using various modes.
    
    Args:
        segment1_frames: Frames from the first segment (video ending)
        segment2_frames: Frames from the second segment (video starting)
        overlap_count: Number of frames to cross-fade
        mode: Blending mode ("linear", "linear_sharp")
        sharp_amt: Sharpening amount for "linear_sharp" mode (0-1)
    
    Returns:
        List of cross-faded frames for the overlap region
    """
    if overlap_count <= 0:
        return []

    n = min(overlap_count, len(segment1_frames), len(segment2_frames))
    if n <= 0:
        return []

    # Determine target resolution from segment2 (the newer generated video)
    if not segment2_frames:
        return []
    
    target_height, target_width = segment2_frames[0].shape[:2]
    target_resolution = (target_width, target_height)
    
    # Log dimension information for debugging
    seg1_height, seg1_width = segment1_frames[0].shape[:2] if segment1_frames else (0, 0)
    dprint(f"CrossFade: Segment1 resolution: {seg1_width}x{seg1_height}, Segment2 resolution: {target_width}x{target_height}")
    dprint(f"CrossFade: Target resolution set to: {target_width}x{target_height} (from segment2)")
    dprint(f"CrossFade: Processing {n} overlap frames")

    out_frames = []
    for i in range(n):
        t_linear = (i + 1) / float(n)
        alpha = crossfade_ease(t_linear)

        frame_a_np = segment1_frames[-n+i].astype(np.float32)
        frame_b_np = segment2_frames[i].astype(np.float32)
        
        # Log original shapes before any resizing
        original_a_shape = frame_a_np.shape[:2]
        original_b_shape = frame_b_np.shape[:2]
        
        # Ensure both frames have the same dimensions
        if frame_a_np.shape[:2] != (target_height, target_width):
            frame_a_np = cv2.resize(frame_a_np, target_resolution, interpolation=cv2.INTER_AREA).astype(np.float32)
            if i == 0:  # Log only for first frame to avoid spam
                dprint(f"CrossFade: Resized segment1 frame from {original_a_shape} to {frame_a_np.shape[:2]}")
        if frame_b_np.shape[:2] != (target_height, target_width):
            frame_b_np = cv2.resize(frame_b_np, target_resolution, interpolation=cv2.INTER_AREA).astype(np.float32)
            if i == 0:  # Log only for first frame to avoid spam
                dprint(f"CrossFade: Resized segment2 frame from {original_b_shape} to {frame_b_np.shape[:2]}")

        blended_float: np.ndarray
        if mode == "linear_sharp":
            blended_float = _blend_linear_sharp(frame_a_np, frame_b_np, alpha, sharp_amt)
        elif mode == "linear":
            blended_float = _blend_linear(frame_a_np, frame_b_np, alpha)
        else:
            dprint(f"Warning: Unknown crossfade mode '{mode}'. Defaulting to linear.")
            blended_float = _blend_linear(frame_a_np, frame_b_np, alpha)
        
        blended_uint8 = np.clip(blended_float, 0, 255).astype(np.uint8)
        out_frames.append(blended_uint8)
    
    return out_frames

def extract_frames_from_video(video_path: str | Path, start_frame: int = 0, num_frames: int = None, *, dprint_func=print) -> list[np.ndarray]:
    """
    Extracts frames from a video file as numpy arrays with retry logic for recently encoded videos.

    Args:
        video_path: Path to the video file
        start_frame: Starting frame index (0-based)
        num_frames: Number of frames to extract (None = all remaining frames)
        dprint_func: The function to use for printing debug messages

    Returns:
        List of frames as BGR numpy arrays
    """
    frames = []
    max_attempts = 3

    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            if attempt < max_attempts - 1:
                dprint_func(f"Attempt {attempt + 1}: Could not open video {video_path}, retrying in 2 seconds...")
                time.sleep(2.0)
                continue
            else:
                dprint_func(f"Error: Could not open video {video_path} after {max_attempts} attempts")
                return frames

        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check for valid frame count (recently encoded videos might report 0 initially)
        if total_frames_video <= 0:
            cap.release()
            if attempt < max_attempts - 1:
                dprint_func(f"Attempt {attempt + 1}: Video {video_path} reports {total_frames_video} frames, retrying in 2 seconds...")
                time.sleep(2.0)
                continue
            else:
                dprint_func(f"Error: Video {video_path} has invalid frame count {total_frames_video} after {max_attempts} attempts")
                return frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

        frames_to_read = num_frames if num_frames is not None else (total_frames_video - start_frame)
        frames_to_read = min(frames_to_read, total_frames_video - start_frame)

        frames_extracted = 0
        for i in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                dprint_func(f"Warning: Could not read frame {start_frame + i} from {video_path}")
                break
            frames.append(frame)
            frames_extracted += 1

        cap.release()

        # If we successfully extracted frames, return them
        if frames_extracted > 0:
            dprint_func(f"Successfully extracted {frames_extracted} frames from {video_path} on attempt {attempt + 1}")
            return frames

        # If no frames extracted and we have more attempts, retry
        if attempt < max_attempts - 1:
            dprint_func(f"Attempt {attempt + 1}: No frames extracted from {video_path}, retrying in 2 seconds...")
            frames = []  # Reset frames list for next attempt
            time.sleep(2.0)

    dprint_func(f"Error: Failed to extract any frames from {video_path} after {max_attempts} attempts")
    return frames

def create_video_from_frames_list(
    frames_list: list[np.ndarray],
    output_path: str | Path,
    fps: int,
    resolution: tuple[int, int]
) -> Path | None:
    """Creates a video from a list of NumPy BGR frames using FFmpeg subprocess.
    Returns the Path object of the successfully written file, or None if failed.
    """
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
        return None

    try:
        raw_video_data = b''.join(frame.tobytes() for frame in processed_frames)
    except Exception:
        return None

    if not raw_video_data:
        return None

    try:
        proc = subprocess.run(
            ffmpeg_cmd,
            input=raw_video_data,
            capture_output=True,
            timeout=60
        )

        if proc.returncode == 0:
            if output_path_mp4.exists() and output_path_mp4.stat().st_size > 0:
                return output_path_mp4
            return None
        else:
            if output_path_mp4.exists():
                try:
                    output_path_mp4.unlink()
                except Exception:
                    pass
            return None
            
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None
    except Exception:
        return None

def _apply_saturation_to_video_ffmpeg(
    input_video_path: str | Path,
    output_video_path: str | Path,
    saturation_level: float,
    preset: str = "veryfast"
) -> bool:
    """Applies a saturation adjustment to the full video using FFmpeg's eq filter.
    Returns: True if FFmpeg succeeds and the output file exists & is non-empty, else False.
    """
    inp = Path(input_video_path)
    outp = Path(output_video_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(inp.resolve()),
        "-vf", f"eq=saturation={saturation_level}",
        "-c:v", "libx264",
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-an",
        str(outp.resolve())
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        if outp.exists() and outp.stat().st_size > 0:
            return True
        return False
    except subprocess.CalledProcessError:
        return False

def color_match_video_to_reference(source_video_path: str | Path, reference_video_path: str | Path, 
                                 output_video_path: str | Path, parsed_resolution: tuple[int, int]) -> bool:
    """
    Color matches source_video to reference_video using histogram matching on the last frame of reference
    and first frame of source. Applies the transformation to all frames of source_video.
    Returns True if successful, False otherwise.
    """
    try:
        ref_cap = cv2.VideoCapture(str(reference_video_path))
        if not ref_cap.isOpened():
            return False
        
        ref_frame_count_from_cap, _ = get_video_frame_count_and_fps(str(reference_video_path))
        if ref_frame_count_from_cap is None: ref_frame_count_from_cap = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, ref_frame_count_from_cap - 1)))
        ret, ref_frame = ref_cap.read()
        ref_cap.release()
        
        if not ret or ref_frame is None:
            return False
        
        if ref_frame.shape[1] != parsed_resolution[0] or ref_frame.shape[0] != parsed_resolution[1]:
            ref_frame = cv2.resize(ref_frame, parsed_resolution, interpolation=cv2.INTER_AREA)
        
        src_cap = cv2.VideoCapture(str(source_video_path))
        if not src_cap.isOpened():
            return False
        
        src_fps_from_cap = src_cap.get(cv2.CAP_PROP_FPS)

        ret, src_first_frame = src_cap.read()
        if not ret or src_first_frame is None:
            src_cap.release()
            return False
        
        if src_first_frame.shape[1] != parsed_resolution[0] or src_first_frame.shape[0] != parsed_resolution[1]:
            src_first_frame = cv2.resize(src_first_frame, parsed_resolution, interpolation=cv2.INTER_AREA)
        
        def match_histogram_channel(source_channel, reference_channel):
            src_hist, _ = np.histogram(source_channel.flatten(), 256, [0, 256])
            ref_hist, _ = np.histogram(reference_channel.flatten(), 256, [0, 256])
            src_cdf = src_hist.cumsum()
            ref_cdf = ref_hist.cumsum()
            src_cdf = src_cdf / src_cdf[-1]
            ref_cdf = ref_cdf / ref_cdf[-1]
            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                closest_idx = np.argmin(np.abs(ref_cdf - src_cdf[i]))
                lookup_table[i] = closest_idx
            return lookup_table
        
        lookup_tables = []
        for channel in range(3):
            lut = match_histogram_channel(src_first_frame[:, :, channel], ref_frame[:, :, channel])
            lookup_tables.append(lut)
        
        output_path_obj = Path(output_video_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path_obj), fourcc, float(src_fps_from_cap), parsed_resolution)
        if not out.isOpened():
            src_cap.release()
            return False
        
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = src_cap.read()
            if not ret: break
            if frame.shape[1] != parsed_resolution[0] or frame.shape[0] != parsed_resolution[1]:
                frame = cv2.resize(frame, parsed_resolution, interpolation=cv2.INTER_AREA)
            matched_frame = frame.copy()
            for channel in range(3):
                matched_frame[:, :, channel] = cv2.LUT(frame[:, :, channel], lookup_tables[channel])
            out.write(matched_frame)
        
        src_cap.release()
        out.release()
        
        return True
        
    except Exception:
        return False

# ## Video Brightness Adjustment Functions

def get_video_frame_count_and_fps(video_path: str) -> tuple[int, float] | tuple[None, None]:
    """
    Get frame count and FPS from a video file using OpenCV.
    
    Note: OpenCV can sometimes return incorrect frame counts for recently encoded videos.
    Consider using manual frame extraction for critical operations where accuracy is essential.
    """
    
    # Try multiple times in case the video metadata is still being written
    max_attempts = 3
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            if attempt < max_attempts - 1:
                time.sleep(0.5)  # Wait a bit before retrying
                continue
            return None, None
            
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # If we got valid values, return them
        if frames > 0 and fps > 0:
            return frames, fps
            
        # Otherwise, wait and retry
        if attempt < max_attempts - 1:
            time.sleep(0.5)
    
    # If all attempts failed, return what we got (might be 0 or invalid)
    return frames, fps

def adjust_frame_brightness(frame: np.ndarray, brightness_adjust: float) -> np.ndarray:
    if brightness_adjust == 0:
        return frame
    factor = 1 + brightness_adjust
    adjusted = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return adjusted

def apply_brightness_to_video_frames(input_video_path: str, output_video_path: Path, brightness_adjust: float, task_id_for_logging: str) -> Path | None:
    """
    Applies brightness adjustment to a video by processing its frames.
    A brightness_adjust of 0 means no change. Negative values darken, positive values brighten.
    """
    try:
        print(f"Task {task_id_for_logging}: Applying brightness adjustment {brightness_adjust} to {input_video_path}")

        total_frames, fps = get_video_frame_count_and_fps(input_video_path)
        if total_frames is None or fps is None or total_frames == 0:
            print(f"[ERROR] Task {task_id_for_logging}: Could not get frame count or fps for {input_video_path}, or video has 0 frames.")
            return None

        frames = extract_frames_from_video(input_video_path)
        if frames is None:
            print(f"[ERROR] Task {task_id_for_logging}: Could not extract frames from {input_video_path}")
            return None

        adjusted_frames = []
        first_frame = None
        for frame in frames:
            if first_frame is None:
                first_frame = frame
            adjusted_frame = adjust_frame_brightness(frame, brightness_adjust)
            adjusted_frames.append(adjusted_frame)

        if not adjusted_frames or first_frame is None:
            print(f"[ERROR] Task {task_id_for_logging}: No frames to write for brightness-adjusted video.")
            return None

        h, w, _ = first_frame.shape
        resolution = (w, h)

        created_video_path = create_video_from_frames_list(adjusted_frames, output_video_path, fps, resolution)
        if created_video_path and created_video_path.exists():
            print(f"Task {task_id_for_logging}: Successfully created brightness-adjusted video at {created_video_path}")
            return created_video_path
        else:
            print(f"[ERROR] Task {task_id_for_logging}: Failed to create brightness-adjusted video.")
            return None
    except Exception as e:
        print(f"[ERROR] Task {task_id_for_logging}: Exception in apply_brightness_to_video_frames: {e}")
        traceback.print_exc()
        return None

def rife_interpolate_images_to_video(
    image1: Image.Image,
    image2: Image.Image,
    num_frames: int,
    resolution_wh: tuple[int, int],
    output_path: str | Path,
    fps: int = 16,
    dprint_func=print
) -> bool:
    """
    Interpolates between two PIL images using RIFE to generate a video.
    """
    try:
        dprint_func("Imported RIFE modules for interpolation.")

        width_out, height_out = resolution_wh
        dprint_func(f"Parsed resolution: {width_out}x{height_out}")

        def pil_to_tensor_rgb_norm(pil_im: Image.Image):
            pil_resized = pil_im.resize((width_out, height_out), Image.Resampling.LANCZOS)
            np_rgb = np.asarray(pil_resized).astype(np.float32) / 127.5 - 1.0  # [0,255]->[-1,1]
            tensor = torch.from_numpy(np_rgb).permute(2, 0, 1)  # C H W
            return tensor

        t_start = pil_to_tensor_rgb_norm(image1)
        t_end   = pil_to_tensor_rgb_norm(image2)

        sample_in = torch.stack([t_start, t_end], dim=1).unsqueeze(0)  # 1 x 3 x 2 x H x W

        device_for_rife = "cuda" if torch.cuda.is_available() else "cpu"
        sample_in = sample_in.to(device_for_rife)
        dprint_func(f"Input tensor for RIFE prepared on device: {device_for_rife}, shape: {sample_in.shape}")

        exp_val = 3  # x8 (2^3 + 1 = 9 frames output by this RIFE implementation for 2 inputs)
        flownet_ckpt = os.path.join("ckpts", "flownet.pkl")
        dprint_func(f"Checking for RIFE model: {flownet_ckpt}")
        if not os.path.exists(flownet_ckpt):
            dprint_func(f"RIFE Error: flownet.pkl not found at {flownet_ckpt}")
            return False
        dprint_func(f"RIFE model found: {flownet_ckpt}. Exp_val: {exp_val}")
        
        sample_in_for_rife = sample_in[0]

        sample_out_from_rife = temporal_interpolation(flownet_ckpt, sample_in_for_rife, exp_val, device=device_for_rife)
        
        if sample_out_from_rife is None:
            dprint_func("RIFE process returned None.")
            return False

        dprint_func(f"RIFE output tensor shape: {sample_out_from_rife.shape}")

        sample_out_no_batch = sample_out_from_rife.to("cpu")
        total_frames_generated = sample_out_no_batch.shape[1]
        dprint_func(f"RIFE produced {total_frames_generated} frames.")

        if total_frames_generated < num_frames:
            dprint_func(f"Warning: RIFE produced {total_frames_generated} frames, expected {num_frames}. Padding last frame.")
            pad_frames = num_frames - total_frames_generated
        else:
            pad_frames = 0
        
        frames_list_np = []
        for idx in range(min(num_frames, total_frames_generated)):
            frame_tensor = sample_out_no_batch[:, idx]
            frame_np = ((frame_tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            frames_list_np.append(frame_bgr)
        
        if pad_frames > 0 and frames_list_np:
            last_frame_to_pad = frames_list_np[-1].copy()
            frames_list_np.extend([last_frame_to_pad for _ in range(pad_frames)])
        
        if not frames_list_np:
            dprint_func(f"Error: No frames available to write for RIFE video (num_rife_frames: {num_frames}).")
            return False

        output_path_obj = Path(output_path)
        video_written = create_video_from_frames_list(frames_list_np, output_path_obj, fps, resolution_wh)

        if video_written:
            dprint_func(f"RIFE video saved to: {video_written.resolve()}")
            return True
        else:
            dprint_func(f"RIFE output file missing or empty after writing attempt: {output_path_obj}")
            return False

    except Exception as e:
        dprint_func(f"RIFE interpolation failed with exception: {e}")
        traceback.print_exc()
        return False

def prepare_vace_ref_for_segment(
    ref_instruction: dict,
    segment_processing_dir: Path,
    target_resolution_wh: tuple[int, int] | None,
    image_download_dir: Path | str | None = None,
    task_id_for_logging: str | None = "generic_headless_task"
) -> Path | None:
    '''
    Prepares a VACE reference image for a segment based on the given instruction.
    Downloads the image if 'original_path' is a URL and image_download_dir is provided.
    Applies strength adjustment and resizes, saving the result to segment_processing_dir.
    Returns the path to the processed image if successful, or None otherwise.
    '''
    dprint(f"Task {task_id_for_logging} (prepare_vace_ref): VACE Ref instruction: {ref_instruction}, download_dir: {image_download_dir}")

    original_image_path_str = ref_instruction.get("original_path")
    strength_to_apply = ref_instruction.get("strength_to_apply")

    if not original_image_path_str:
        dprint(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: No original_path in VACE ref instruction. Skipping.")
        return None
    
    local_original_image_path_str = download_image_if_url(original_image_path_str, image_download_dir, task_id_for_logging)
    local_original_image_path = Path(local_original_image_path_str)

    if not local_original_image_path.exists():
        dprint(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: VACE ref original image not found (after potential download): {local_original_image_path} (original input: {original_image_path_str})")
        return None

    vace_ref_type = ref_instruction.get("type", "generic")
    segment_idx_for_naming = ref_instruction.get("segment_idx_for_naming", "unknown_idx")
    processed_vace_base_name = f"vace_ref_s{segment_idx_for_naming}_{vace_ref_type}_str{strength_to_apply:.2f}"
    original_suffix = local_original_image_path.suffix if local_original_image_path.suffix else ".png"
    
    output_path_for_processed_vace = sm_get_unique_target_path(segment_processing_dir, processed_vace_base_name, original_suffix)

    effective_target_resolution_wh = None
    if target_resolution_wh:
        effective_target_resolution_wh = ((target_resolution_wh[0] // 16) * 16, (target_resolution_wh[1] // 16) * 16)
        if effective_target_resolution_wh != target_resolution_wh:
            dprint(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: Adjusted VACE ref target resolution from {target_resolution_wh} to {effective_target_resolution_wh}")

    final_processed_path = sm_apply_strength_to_image(
        image_path_input=local_original_image_path, 
        strength=strength_to_apply,
        output_path=output_path_for_processed_vace,
        target_resolution_wh=effective_target_resolution_wh,
        task_id_for_logging=task_id_for_logging,
        image_download_dir=None
    )

    if final_processed_path and final_processed_path.exists():
        dprint(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: Prepared VACE ref: {final_processed_path}")
        return final_processed_path
    else:
        dprint(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: Failed to apply strength/save VACE ref from {local_original_image_path}. Skipping.")
        traceback.print_exc()
        return None

def create_guide_video_for_travel_segment(
    segment_idx_for_logging: int,
    end_anchor_image_index: int,
    is_first_segment_from_scratch: bool,
    total_frames_for_segment: int,
    parsed_res_wh: tuple[int, int],
    fps_helpers: int,
    input_images_resolved_for_guide: list[str],
    path_to_previous_segment_video_output_for_guide: str | None,
    output_target_dir: Path,
    guide_video_base_name: str,
    segment_image_download_dir: Path | None,
    task_id_for_logging: str,
    full_orchestrator_payload: dict,
    segment_params: dict,
    single_image_journey: bool = False,
    predefined_output_path: Path | None = None,
    structure_video_path: str | None = None,
    structure_video_treatment: str = "adjust",
    structure_video_motion_strength: float = 1.0,
    structure_motion_video_url: str | None = None,
    structure_motion_frame_offset: int = 0,
    *,
    dprint=print
) -> Path | None:
    """Creates the guide video for a travel segment with all fading and adjustments."""
    try:
        # Initialize guidance tracker for structure video feature
        from source.structure_video_guidance import GuidanceTracker, apply_structure_motion_with_tracking
        guidance_tracker = GuidanceTracker(total_frames_for_segment)
        # Use predefined path if provided (for UUID-based naming), otherwise generate unique path
        if predefined_output_path:
            actual_guide_video_path = predefined_output_path
        else:
            actual_guide_video_path = sm_get_unique_target_path(output_target_dir, guide_video_base_name, ".mp4")
        
        # Extract debug mode from orchestrator payload or segment params
        debug_mode = segment_params.get("debug_mode_enabled", full_orchestrator_payload.get("debug_mode_enabled", False))
        
        gray_frame_bgr = sm_create_color_frame(parsed_res_wh, (128, 128, 128))

        fade_in_p = json.loads(full_orchestrator_payload["fade_in_params_json_str"])
        fade_out_p = json.loads(full_orchestrator_payload["fade_out_params_json_str"])
        strength_adj = segment_params.get("subsequent_starting_strength_adjustment", 0.0)
        desat_factor = segment_params.get("desaturate_subsequent_starting_frames", 0.0)
        bright_adj = segment_params.get("adjust_brightness_subsequent_starting_frames", 0.0)
        frame_overlap_from_previous = segment_params.get("frame_overlap_from_previous", 0)

        fi_low, fi_high, fi_curve, fi_factor = float(fade_in_p.get("low_point",0)), float(fade_in_p.get("high_point",1)), str(fade_in_p.get("curve_type","ease_in_out")), float(fade_in_p.get("duration_factor",0))
        fo_low, fo_high, fo_curve, fo_factor = float(fade_out_p.get("low_point",0)), float(fade_out_p.get("high_point",1)), str(fade_out_p.get("curve_type","ease_in_out")), float(fade_out_p.get("duration_factor",0))

        if total_frames_for_segment <= 0:
            dprint(f"Task {task_id_for_logging}: Guide video has 0 frames. Skipping creation.")
            return None

        dprint(f"Task {task_id_for_logging}: Interpolating guide video with {total_frames_for_segment} frames...")
        frames_for_guide_list = [sm_create_color_frame(parsed_res_wh, (128,128,128)).copy() for _ in range(total_frames_for_segment)]

        # Check for consolidated keyframe positions (frame consolidation optimization)
        consolidated_keyframe_positions = segment_params.get("consolidated_keyframe_positions")
        if consolidated_keyframe_positions and not single_image_journey:
            dprint(f"Task {task_id_for_logging}: CONSOLIDATED SEGMENT - placing keyframes at positions {consolidated_keyframe_positions}")

            # For consolidated segments, we need to determine which input images to use
            # If this is the first segment, use images sequentially starting from 0
            # If this is a subsequent segment, only place the final target image (end anchor)
            if is_first_segment_from_scratch:
                # First segment: place images starting from index 0
                for img_idx, frame_pos in enumerate(consolidated_keyframe_positions):
                    if img_idx < len(input_images_resolved_for_guide) and frame_pos <= total_frames_for_segment - 1:
                        img_path = input_images_resolved_for_guide[img_idx]
                        keyframe_np = sm_image_to_frame(img_path, parsed_res_wh, task_id_for_logging=task_id_for_logging, image_download_dir=segment_image_download_dir, debug_mode=debug_mode)
                        if keyframe_np is not None:
                            frames_for_guide_list[frame_pos] = keyframe_np.copy()
                            # Mark keyframe as guided
                            guidance_tracker.mark_single_frame(frame_pos)
                            dprint(f"Task {task_id_for_logging}: Placed image {img_idx} at frame {frame_pos}")
            else:
                # Subsequent consolidated segment: handle overlap + place end anchor
                # First, extract overlap frames from previous video if needed
                if frame_overlap_from_previous > 0 and path_to_previous_segment_video_output_for_guide:
                    dprint(f"Task {task_id_for_logging}: CONSOLIDATED SEGMENT - extracting {frame_overlap_from_previous} overlap frames")

                    # Extract the overlap frames from the previous video
                    try:
                        all_prev_frames = extract_frames_from_video(path_to_previous_segment_video_output_for_guide, dprint_func=dprint)
                        if all_prev_frames and len(all_prev_frames) >= frame_overlap_from_previous:
                            overlap_frames = all_prev_frames[-frame_overlap_from_previous:]
                            # Place overlap frames at the beginning of this video
                            for overlap_idx, overlap_frame in enumerate(overlap_frames):
                                if overlap_idx < total_frames_for_segment:
                                    frames_for_guide_list[overlap_idx] = overlap_frame.copy()
                                    # Mark as guided for structure video tracking
                                    guidance_tracker.mark_single_frame(overlap_idx)
                                    dprint(f"Task {task_id_for_logging}: Placed overlap frame {overlap_idx} from previous video")
                    except Exception as e:
                        dprint(f"Task {task_id_for_logging}: WARNING - Could not extract overlap frames: {e}")

                # Then place ALL keyframe images at their positions (including intermediate ones)
                # For subsequent consolidated segments, we need to place:
                # 1. Intermediate keyframes at their positions
                # 2. The end anchor at the final position

                # Calculate which input images correspond to each keyframe position
                # The consolidated end anchor tells us which image should be the final one
                if len(consolidated_keyframe_positions) > 0:
                    # For subsequent segments, we need to place all keyframes that belong to this segment
                    # The end anchor index tells us which image should be at the final position
                    final_frame_pos = consolidated_keyframe_positions[-1]

                    # Calculate which input images should go at each keyframe position
                    # This depends on which segment we are and how keyframes were distributed
                    start_image_idx = end_anchor_image_index - len(consolidated_keyframe_positions) + 1

                    for i, frame_pos in enumerate(consolidated_keyframe_positions):
                        image_idx = start_image_idx + i
                        if image_idx >= 0 and image_idx < len(input_images_resolved_for_guide) and frame_pos <= total_frames_for_segment - 1:
                            img_path = input_images_resolved_for_guide[image_idx]
                            keyframe_np = sm_image_to_frame(img_path, parsed_res_wh, task_id_for_logging=task_id_for_logging, image_download_dir=segment_image_download_dir, debug_mode=debug_mode)
                            if keyframe_np is not None:
                                frames_for_guide_list[frame_pos] = keyframe_np.copy()
                                # Mark keyframe as guided
                                guidance_tracker.mark_single_frame(frame_pos)
                                dprint(f"Task {task_id_for_logging}: Placed image {image_idx} at consolidated keyframe position {frame_pos}")

                    dprint(f"Task {task_id_for_logging}: Placed {len(consolidated_keyframe_positions)} keyframes for consolidated segment (end anchor: image {end_anchor_image_index})")

            # Apply structure motion to unguidanced frames before creating video
            if structure_video_path:
                dprint(f"[GUIDANCE_TRACK] Pre-structure guidance summary:")
                dprint(guidance_tracker.debug_summary())
                
                frames_for_guide_list = apply_structure_motion_with_tracking(
                    frames_for_guide_list=frames_for_guide_list,
                    guidance_tracker=guidance_tracker,
                    structure_video_path=structure_video_path,
                    structure_video_treatment=structure_video_treatment,
                    parsed_res_wh=parsed_res_wh,
                    fps_helpers=fps_helpers,
                    motion_strength=structure_video_motion_strength,
                    structure_motion_video_url=structure_motion_video_url,
                    segment_processing_dir=output_target_dir,
                    structure_motion_frame_offset=structure_motion_frame_offset,
                    dprint=dprint
                )
                
                dprint(f"[GUIDANCE_TRACK] Post-structure guidance summary:")
                dprint(guidance_tracker.debug_summary())

            return create_video_from_frames_list(frames_for_guide_list, predefined_output_path or output_target_dir / guide_video_base_name, fps_helpers, parsed_res_wh)

        end_anchor_frame_np = None

        if not single_image_journey:
            end_anchor_img_path_str: str
            if end_anchor_image_index < len(input_images_resolved_for_guide):
                end_anchor_img_path_str = input_images_resolved_for_guide[end_anchor_image_index]
            else:
                 raise ValueError(f"Seg {segment_idx_for_logging}: End anchor index {end_anchor_image_index} out of bounds for input images list ({len(input_images_resolved_for_guide)} images available).")
            
            end_anchor_frame_np = sm_image_to_frame(end_anchor_img_path_str, parsed_res_wh, task_id_for_logging=task_id_for_logging, image_download_dir=segment_image_download_dir, debug_mode=debug_mode)
            if end_anchor_frame_np is None: raise ValueError(f"Failed to load end anchor image: {end_anchor_img_path_str}")
        else:
            # For single image journeys, we don't need an end anchor - only set the first frame
            dprint(f"Task {task_id_for_logging}: Single image journey - skipping end anchor setup, will only set first frame")
        
        num_end_anchor_duplicates = 1
        start_anchor_frame_np = None

        if is_first_segment_from_scratch:
            start_anchor_img_path_str = input_images_resolved_for_guide[0]
            start_anchor_frame_np = sm_image_to_frame(start_anchor_img_path_str, parsed_res_wh, task_id_for_logging=task_id_for_logging, image_download_dir=segment_image_download_dir, debug_mode=debug_mode)
            if start_anchor_frame_np is None: raise ValueError(f"Failed to load start anchor: {start_anchor_img_path_str}")
            if frames_for_guide_list:
                frames_for_guide_list[0] = start_anchor_frame_np.copy()
                # Mark first frame as guided
                guidance_tracker.mark_single_frame(0)

            if single_image_journey:
                dprint(f"Task {task_id_for_logging}: Guide video for single image journey. Only first frame is set, all other frames remain gray/masked.")
            else:
                # This is the original logic for fading between start and end.
                pot_max_idx_start_fade = total_frames_for_segment - num_end_anchor_duplicates - 1
                avail_frames_start_fade = max(0, pot_max_idx_start_fade)
                num_start_fade_steps = int(avail_frames_start_fade * fo_factor)
                if num_start_fade_steps > 0:
                    easing_fn_out = sm_get_easing_function(fo_curve)
                    for k_fo in range(num_start_fade_steps):
                        idx_in_guide = 1 + k_fo
                        if idx_in_guide >= total_frames_for_segment: break
                        alpha_lin = 1.0 - ((k_fo + 1) / float(num_start_fade_steps))
                        e_alpha = fo_low + (fo_high - fo_low) * easing_fn_out(alpha_lin)
                        frames_for_guide_list[idx_in_guide] = cv2.addWeighted(frames_for_guide_list[idx_in_guide].astype(np.float32), 1.0 - e_alpha, start_anchor_frame_np.astype(np.float32), e_alpha, 0).astype(np.uint8)
                        # Mark fade frames as guided
                        guidance_tracker.mark_single_frame(idx_in_guide)
                
                min_idx_end_fade = 1
                max_idx_end_fade = total_frames_for_segment - num_end_anchor_duplicates - 1
                avail_frames_end_fade = max(0, max_idx_end_fade - min_idx_end_fade + 1)
                num_end_fade_steps = int(avail_frames_end_fade * fi_factor)
                if num_end_fade_steps > 0:
                    actual_end_fade_start_idx = max(min_idx_end_fade, max_idx_end_fade - num_end_fade_steps + 1)
                    easing_fn_in = sm_get_easing_function(fi_curve)
                    for k_fi in range(num_end_fade_steps):
                        idx_in_guide = actual_end_fade_start_idx + k_fi
                        if idx_in_guide >= total_frames_for_segment: break
                        alpha_lin = (k_fi + 1) / float(num_end_fade_steps)
                        e_alpha = fi_low + (fi_high - fi_low) * easing_fn_in(alpha_lin)
                        base_f = frames_for_guide_list[idx_in_guide]
                        frames_for_guide_list[idx_in_guide] = cv2.addWeighted(base_f.astype(np.float32), 1.0 - e_alpha, end_anchor_frame_np.astype(np.float32), e_alpha, 0).astype(np.uint8)
                        # Mark fade frames as guided
                        guidance_tracker.mark_single_frame(idx_in_guide)
                elif fi_factor > 0 and avail_frames_end_fade > 0:
                    for k_fill in range(min_idx_end_fade, max_idx_end_fade + 1):
                        if k_fill < total_frames_for_segment:
                            frames_for_guide_list[k_fill] = end_anchor_frame_np.copy()
                            # Mark filled frames as guided
                            guidance_tracker.mark_single_frame(k_fill)

        elif path_to_previous_segment_video_output_for_guide: # Continued or Subsequent
            dprint(f"GuideBuilder (Seg {segment_idx_for_logging}): Subsequent segment logic started.")
            dprint(f"GuideBuilder: Prev video path: {path_to_previous_segment_video_output_for_guide}")
            dprint(f"GuideBuilder: Overlap from prev setting: {frame_overlap_from_previous}")
            dprint(f"GuideBuilder: Prev video exists: {Path(path_to_previous_segment_video_output_for_guide).exists()}")
            
            if not Path(path_to_previous_segment_video_output_for_guide).exists():
                raise ValueError(f"Previous video path does not exist: {path_to_previous_segment_video_output_for_guide}")

            # Wait for file to be stable before reading (important for recently encoded videos)
            dprint(f"GuideBuilder: Waiting for previous video file to stabilize...")
            file_stable = sm_wait_for_file_stable(path_to_previous_segment_video_output_for_guide, checks=3, interval=1.0, dprint=dprint)
            if not file_stable:
                dprint(f"GuideBuilder: WARNING - File stability check failed, proceeding anyway")

            # Get the expected frame count for the previous segment from orchestrator data
            expected_prev_segment_frames = None
            if segment_idx_for_logging > 0 and full_orchestrator_payload:
                segment_frames_expanded = full_orchestrator_payload.get("segment_frames_expanded", [])
                if segment_idx_for_logging - 1 < len(segment_frames_expanded):
                    expected_prev_segment_frames = segment_frames_expanded[segment_idx_for_logging - 1]
                    dprint(f"GuideBuilder: Previous segment expected to have {expected_prev_segment_frames} frames based on orchestrator data")

            # If we have the expected frame count, use it directly
            if expected_prev_segment_frames and expected_prev_segment_frames > 0:
                dprint(f"GuideBuilder: Using known frame count {expected_prev_segment_frames} from orchestrator data")
                prev_vid_total_frames = expected_prev_segment_frames
                
                # Calculate overlap frames to extract
                actual_overlap_to_use = min(frame_overlap_from_previous, prev_vid_total_frames)
                start_extraction_idx = max(0, prev_vid_total_frames - actual_overlap_to_use)
                dprint(f"GuideBuilder: Extracting {actual_overlap_to_use} frames starting from index {start_extraction_idx}")
                
                # Extract the frames directly
                overlap_frames_raw = extract_frames_from_video(path_to_previous_segment_video_output_for_guide, start_extraction_idx, actual_overlap_to_use, dprint_func=dprint)
                
                # Verify we got the expected number of frames
                if len(overlap_frames_raw) != actual_overlap_to_use:
                    dprint(f"GuideBuilder: WARNING - Expected {actual_overlap_to_use} frames but got {len(overlap_frames_raw)}. Falling back to manual extraction.")
                    # Fall back to extracting all frames
                    all_prev_frames = extract_frames_from_video(path_to_previous_segment_video_output_for_guide, dprint_func=dprint)
                    prev_vid_total_frames = len(all_prev_frames)
                    actual_overlap_to_use = min(frame_overlap_from_previous, prev_vid_total_frames)
                    overlap_frames_raw = all_prev_frames[-actual_overlap_to_use:] if actual_overlap_to_use > 0 else []
            else:
                # Fallback: No orchestrator data, use OpenCV or manual extraction
                dprint(f"GuideBuilder: No orchestrator frame count data available, falling back to frame detection")
                prev_vid_total_frames, prev_vid_fps = get_video_frame_count_and_fps(path_to_previous_segment_video_output_for_guide)
                dprint(f"GuideBuilder: Frame count from cv2: {prev_vid_total_frames}, fps: {prev_vid_fps}")

                if not prev_vid_total_frames:  # Handles None or 0
                    dprint(f"GuideBuilder: Fallback triggered due to zero/None frame count. Manually reading frames.")
                    # Fallback: read all frames to determine length
                    all_prev_frames = extract_frames_from_video(path_to_previous_segment_video_output_for_guide, dprint_func=dprint)
                    prev_vid_total_frames = len(all_prev_frames)
                    dprint(f"GuideBuilder: Manual frame count from fallback: {prev_vid_total_frames}")
                    if prev_vid_total_frames == 0:
                        raise ValueError("Previous segment video appears to have zero frames – cannot build guide overlap.")
                    # Decide how many overlap frames we can reuse
                    actual_overlap_to_use = min(frame_overlap_from_previous, prev_vid_total_frames)
                    overlap_frames_raw = all_prev_frames[-actual_overlap_to_use:]
                    dprint(f"GuideBuilder: Using fallback - extracting last {actual_overlap_to_use} frames from {prev_vid_total_frames} total frames")
                else:
                    dprint(f"GuideBuilder: Using cv2 frame count.")
                    actual_overlap_to_use = min(frame_overlap_from_previous, prev_vid_total_frames)
                    start_extraction_idx = max(0, prev_vid_total_frames - actual_overlap_to_use)
                    dprint(f"GuideBuilder: Extracting {actual_overlap_to_use} frames starting from index {start_extraction_idx}")
                    overlap_frames_raw = extract_frames_from_video(path_to_previous_segment_video_output_for_guide, start_extraction_idx, actual_overlap_to_use, dprint_func=dprint)

            # Log the final overlap calculation
            dprint(f"GuideBuilder: Calculated actual_overlap_to_use: {actual_overlap_to_use if 'actual_overlap_to_use' in locals() else 'Not calculated'}")
            dprint(f"GuideBuilder: Extracted raw overlap frames count: {len(overlap_frames_raw) if 'overlap_frames_raw' in locals() else 'Not extracted'}")

            # Check video resolution to understand if it matches our target
            if overlap_frames_raw and len(overlap_frames_raw) > 0:
                first_frame_shape = overlap_frames_raw[0].shape
                prev_height, prev_width = first_frame_shape[0], first_frame_shape[1]
                dprint(f"GuideBuilder: Previous video resolution from extracted frames: {prev_width}x{prev_height} (target: {parsed_res_wh[0]}x{parsed_res_wh[1]})")
                if prev_width != parsed_res_wh[0] or prev_height != parsed_res_wh[1]:
                    dprint(f"GuideBuilder: Resolution mismatch detected! Previous video will be resized during guide creation.")

            frames_read_for_overlap = 0
            for k, frame_fp in enumerate(overlap_frames_raw):
                if k >= total_frames_for_segment: break
                original_shape = frame_fp.shape
                if frame_fp.shape[1]!=parsed_res_wh[0] or frame_fp.shape[0]!=parsed_res_wh[1]: 
                    frame_fp = cv2.resize(frame_fp, parsed_res_wh, interpolation=cv2.INTER_AREA)
                    dprint(f"GuideBuilder: Resized frame {k} from {original_shape} to {frame_fp.shape}")
                frames_for_guide_list[k] = frame_fp.copy()
                # Mark overlap frames as guided
                guidance_tracker.mark_single_frame(k)
                frames_read_for_overlap += 1
            
            dprint(f"GuideBuilder: Frames copied into guide list: {frames_read_for_overlap}")
            
            # Log details about what frames were actually placed in the guide
            if frames_read_for_overlap > 0:
                dprint(f"GuideBuilder: Guide frames 0-{frames_read_for_overlap-1} now contain frames from previous video")
                dprint(f"GuideBuilder: Guide frames {frames_read_for_overlap}-{total_frames_for_segment-1} are still gray frames (will be modified by fade logic)")
            
            if frames_read_for_overlap > 0:
                if fo_factor > 0.0:
                    num_init_fade_steps = min(int(frames_read_for_overlap * fo_factor), frames_read_for_overlap)
                    easing_fn_fo_ol = sm_get_easing_function(fo_curve)
                    for k_fo_ol in range(num_init_fade_steps):
                        alpha_l = 1.0 - ((k_fo_ol + 1) / float(num_init_fade_steps))
                        eff_s = fo_low + (fo_high - fo_low) * easing_fn_fo_ol(alpha_l)
                        eff_s = np.clip(eff_s + strength_adj, 0, 1)
                        base_f=frames_for_guide_list[k_fo_ol]
                        frames_for_guide_list[k_fo_ol] = cv2.addWeighted(gray_frame_bgr.astype(np.float32),1-eff_s,base_f.astype(np.float32),eff_s,0).astype(np.uint8)
                        if desat_factor > 0:
                            g=cv2.cvtColor(frames_for_guide_list[k_fo_ol],cv2.COLOR_BGR2GRAY)
                            gb=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
                            frames_for_guide_list[k_fo_ol]=cv2.addWeighted(frames_for_guide_list[k_fo_ol],1-desat_factor,gb,desat_factor,0)
                        if bright_adj!=0:
                            frames_for_guide_list[k_fo_ol]=sm_adjust_frame_brightness(frames_for_guide_list[k_fo_ol],bright_adj)
                else:
                    eff_s=np.clip(fo_high+strength_adj,0,1)
                    if abs(eff_s-1.0)>1e-5 or desat_factor>0 or bright_adj!=0:
                        for k_s_ol in range(frames_read_for_overlap):
                            base_f=frames_for_guide_list[k_s_ol];frames_for_guide_list[k_s_ol]=cv2.addWeighted(gray_frame_bgr.astype(np.float32),1-eff_s,base_f.astype(np.float32),eff_s,0).astype(np.uint8)
                            if desat_factor>0: g=cv2.cvtColor(frames_for_guide_list[k_s_ol],cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR);frames_for_guide_list[k_s_ol]=cv2.addWeighted(frames_for_guide_list[k_s_ol],1-desat_factor,gb,desat_factor,0)
                            if bright_adj!=0: frames_for_guide_list[k_s_ol]=sm_adjust_frame_brightness(frames_for_guide_list[k_s_ol],bright_adj)
            
            if not single_image_journey:
                min_idx_efs = frames_read_for_overlap; max_idx_efs = total_frames_for_segment - num_end_anchor_duplicates - 1
                avail_f_efs = max(0, max_idx_efs - min_idx_efs + 1); num_efs_steps = int(avail_f_efs * fi_factor)
                if num_efs_steps > 0:
                    actual_efs_start_idx = max(min_idx_efs, max_idx_efs - num_efs_steps + 1)
                    easing_fn_in_s = sm_get_easing_function(fi_curve)
                    for k_fi_s in range(num_efs_steps):
                        idx = actual_efs_start_idx+k_fi_s
                        if idx >= total_frames_for_segment: break
                        if idx < min_idx_efs: continue
                        alpha_l=(k_fi_s+1)/float(num_efs_steps);e_alpha=fi_low+(fi_high-fi_low)*easing_fn_in_s(alpha_l);e_alpha=np.clip(e_alpha,0,1)
                        base_f=frames_for_guide_list[idx];frames_for_guide_list[idx]=cv2.addWeighted(base_f.astype(np.float32),1-e_alpha,end_anchor_frame_np.astype(np.float32),e_alpha,0).astype(np.uint8)
                        # Mark fade frames as guided
                        guidance_tracker.mark_single_frame(idx)
                elif fi_factor > 0 and avail_f_efs > 0:
                    for k_fill in range(min_idx_efs, max_idx_efs + 1):
                        if k_fill < total_frames_for_segment:
                            frames_for_guide_list[k_fill] = end_anchor_frame_np.copy()
                            # Mark filled frames as guided
                            guidance_tracker.mark_single_frame(k_fill)
        
        if not single_image_journey and total_frames_for_segment > 0 and end_anchor_frame_np is not None:
            for k_dup in range(min(num_end_anchor_duplicates, total_frames_for_segment)):
                idx_s = total_frames_for_segment - 1 - k_dup
                if idx_s >= 0:
                    frames_for_guide_list[idx_s] = end_anchor_frame_np.copy()
                    # Mark end anchor duplicates as guided
                    guidance_tracker.mark_single_frame(idx_s)
                else: break

        if is_first_segment_from_scratch and total_frames_for_segment > 0 and start_anchor_frame_np is not None:
            frames_for_guide_list[0] = start_anchor_frame_np.copy()
            # Already marked above, but ensure it's marked
            guidance_tracker.mark_single_frame(0)

        # Apply structure motion to unguidanced frames before creating video
        if structure_video_path:
            dprint(f"[GUIDANCE_TRACK] Pre-structure guidance summary:")
            dprint(guidance_tracker.debug_summary())
            
            frames_for_guide_list = apply_structure_motion_with_tracking(
                frames_for_guide_list=frames_for_guide_list,
                guidance_tracker=guidance_tracker,
                structure_video_path=structure_video_path,
                structure_video_treatment=structure_video_treatment,
                parsed_res_wh=parsed_res_wh,
                fps_helpers=fps_helpers,
                motion_strength=structure_video_motion_strength,
                structure_motion_video_url=structure_motion_video_url,
                segment_processing_dir=output_target_dir,
                structure_motion_frame_offset=structure_motion_frame_offset,
                dprint=dprint
            )
            
            dprint(f"[GUIDANCE_TRACK] Post-structure guidance summary:")
            dprint(guidance_tracker.debug_summary())

        if frames_for_guide_list:
            guide_video_file_path = create_video_from_frames_list(frames_for_guide_list, actual_guide_video_path, fps_helpers, parsed_res_wh)
            if guide_video_file_path and guide_video_file_path.exists():
                return guide_video_file_path
        
        return None

    except Exception as e:
        dprint(f"ERROR creating guide video for segment {segment_idx_for_logging}: {e}")
        traceback.print_exc()
        return None

def _cm_enhance_saturation(image_bgr, saturation_factor=0.5):
    """
    Adjust saturation of an image by the given factor.
    saturation_factor: 1.0 = no change, 0.5 = 50% reduction, 1.3 = 30% increase, etc.
    """
    if not _COLOR_MATCH_DEPS_AVAILABLE: return image_bgr
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_float = hsv.astype(np.float32)
    h, s, v = cv2.split(hsv_float)
    s_adjusted = s * saturation_factor
    s_adjusted = np.clip(s_adjusted, 0, 255)
    hsv_adjusted = cv2.merge([h, s_adjusted, v])
    hsv_adjusted_uint8 = hsv_adjusted.astype(np.uint8)
    adjusted_bgr = cv2.cvtColor(hsv_adjusted_uint8, cv2.COLOR_HSV2BGR)
    return adjusted_bgr

def _cm_transfer_mean_std_lab(source_bgr, target_bgr):
    if not _COLOR_MATCH_DEPS_AVAILABLE: return source_bgr
    MIN_ALLOWED_STD_RATIO_FOR_LUMINANCE = 0.1
    MIN_ALLOWED_STD_RATIO_FOR_COLOR = 0.4
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)

    source_lab_float = source_lab.astype(np.float32)
    target_lab_float = target_lab.astype(np.float32)

    s_l, s_a, s_b = cv2.split(source_lab_float)
    t_l, t_a, t_b = cv2.split(target_lab_float)

    channels_out = []
    for i, (s_chan, t_chan) in enumerate(zip([s_l, s_a, s_b], [t_l, t_a, t_b])):
        s_mean_val, s_std_val = cv2.meanStdDev(s_chan)
        t_mean_val, t_std_val = cv2.meanStdDev(t_chan)
        s_mean, s_std = s_mean_val[0][0], s_std_val[0][0]
        t_mean, t_std = t_mean_val[0][0], t_std_val[0][0]

        std_ratio = t_std / s_std if s_std > 1e-5 else 1.0
        
        min_ratio = MIN_ALLOWED_STD_RATIO_FOR_LUMINANCE if i == 0 else MIN_ALLOWED_STD_RATIO_FOR_COLOR
        effective_std_ratio = max(std_ratio, min_ratio)

        if s_std > 1e-5:
            transformed_chan = (s_chan - s_mean) * effective_std_ratio + t_mean
        else: 
            transformed_chan = np.full_like(s_chan, t_mean)
        
        channels_out.append(transformed_chan)

    result_lab_float = cv2.merge(channels_out)
    result_lab_clipped = np.clip(result_lab_float, 0, 255)
    result_lab_uint8 = result_lab_clipped.astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab_uint8, cv2.COLOR_LAB2BGR)
    return result_bgr

def apply_color_matching_to_video(video_path: str, start_ref_path: str, end_ref_path: str, output_path: str, dprint):
    if not all([_COLOR_MATCH_DEPS_AVAILABLE, Path(video_path).exists(), Path(start_ref_path).exists(), Path(end_ref_path).exists()]):
        dprint(f"Color Matching: Skipping due to missing deps or files. Deps:{_COLOR_MATCH_DEPS_AVAILABLE}, Video:{Path(video_path).exists()}, Start:{Path(start_ref_path).exists()}, End:{Path(end_ref_path).exists()}")
        return None

    frames = extract_frames_from_video(video_path)
    frame_count, fps = get_video_frame_count_and_fps(video_path)
    if not frames or not frame_count or not fps:
        dprint("Color Matching: Frame extraction or metadata retrieval failed.")
        return None

    # Get resolution from the first frame
    h, w, _ = frames[0].shape
    resolution = (w, h)
    
    start_ref_bgr = cv2.imread(start_ref_path)
    end_ref_bgr = cv2.imread(end_ref_path)
    start_ref_resized = cv2.resize(start_ref_bgr, resolution)
    end_ref_resized = cv2.resize(end_ref_bgr, resolution)

    total_frames = len(frames)
    accumulated_frames = []

    for i, frame_bgr in enumerate(frames):
        frame_bgr_desaturated = _cm_enhance_saturation(frame_bgr, saturation_factor=0.5)
        
        corrected_start_bgr = _cm_transfer_mean_std_lab(frame_bgr_desaturated, start_ref_resized)
        corrected_end_bgr = _cm_transfer_mean_std_lab(frame_bgr_desaturated, end_ref_resized)
        
        t = i / (total_frames - 1) if total_frames > 1 else 1.0
        w_original = (0.5 * t) if t < 0.5 else (0.5 - 0.5 * t)
        w_correct = 1.0 - w_original
        w_start = (1.0 - t) * w_correct
        w_end = t * w_correct

        blend_float = (w_start * corrected_start_bgr.astype(np.float32) + 
                       w_end * corrected_end_bgr.astype(np.float32) + 
                       w_original * frame_bgr.astype(np.float32))
        
        blended_frame_bgr = np.clip(blend_float, 0, 255).astype(np.uint8)
        accumulated_frames.append(blended_frame_bgr)
    
    if accumulated_frames:
        created_video_path = create_video_from_frames_list(accumulated_frames, output_path, fps, resolution)
        dprint(f"Color Matching: Successfully created color matched video at {created_video_path}")
        return created_video_path
    
    dprint("Color Matching: Failed to produce any frames.")
    return None

def extract_last_frame_as_image(video_path: str | Path, output_dir: Path, task_id_for_log: str) -> str | None:
    """
    Extracts the last frame of a video and saves it as a PNG image.
    """
    if not _COLOR_MATCH_DEPS_AVAILABLE: 
        dprint(f"Task {task_id_for_log} extract_last_frame_as_image: Skipping due to missing CV2/Numpy dependencies.")
        return None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            dprint(f"[ERROR Task {task_id_for_log}] extract_last_frame_as_image: Could not open video {video_path}")
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            dprint(f"Task {task_id_for_log} extract_last_frame_as_image: Video has 0 frames {video_path}")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()

        if ret:
            output_path = output_dir / f"last_frame_ref_{Path(video_path).stem}.png"
            cv2.imwrite(str(output_path), frame)
            return str(output_path.resolve())
        dprint(f"Task {task_id_for_log} extract_last_frame_as_image: Failed to read last frame from {video_path}")
        return None
    except Exception as e:
        dprint(f"[ERROR Task {task_id_for_log}] extract_last_frame_as_image: Exception extracting frame from {video_path}: {e}")
        traceback.print_exc()
        return None

# --- New utility: Overlay input images above video (Start & End side-by-side) ---

def overlay_start_end_images_above_video(
    start_image_path: str | Path,
    end_image_path: str | Path,
    input_video_path: str | Path,
    output_video_path: str | Path,
    *,
    dprint=print,
) -> bool:
    """Creates a composite video that shows *start_image* (left) and *end_image* (right)
    on a row above the *input_video*.

    Layout:
        START | END  (static images, full video duration)
        ----------------------
               VIDEO           (original video frames)

    The resulting video keeps the original width of *input_video*.  Each image is
    scaled to exactly half that width and the same height as the video to ensure
    perfect alignment.  The final output therefore has a height of
    ``video_height * 2`` and a width equal to ``video_width``.

    Args:
        start_image_path: Path to the starting image.
        end_image_path:   Path to the ending image.
        input_video_path: Source video that was generated.
        output_video_path: Desired path for the composite video.
        dprint: Debug print function.

    Returns:
        True if the composite video was created successfully, else False.
    """
    try:
        start_image_path = Path(start_image_path)
        end_image_path = Path(end_image_path)
        input_video_path = Path(input_video_path)
        output_video_path = Path(output_video_path)

        if not (start_image_path.exists() and end_image_path.exists() and input_video_path.exists()):
            dprint(
                f"overlay_start_end_images_above_video: One or more input paths are missing.\n"
                f"  start_image_path = {start_image_path}\n"
                f"  end_image_path   = {end_image_path}\n"
                f"  input_video_path = {input_video_path}"
            )
            return False

        # ---------------------------------------------------------
        #   Preferred implementation: MoviePy (simpler, robust)
        # ---------------------------------------------------------
        if _MOVIEPY_AVAILABLE:
            try:
                video_clip = mpe.VideoFileClip(str(input_video_path))

                half_width_px = int(video_clip.w / 2)

                img1_clip = mpe.ImageClip(str(start_image_path)).resize(width=half_width_px).set_duration(video_clip.duration)
                img2_clip = mpe.ImageClip(str(end_image_path)).resize(width=half_width_px).set_duration(video_clip.duration)

                # top row (images side-by-side)
                top_row = mpe.clips_array([[img1_clip, img2_clip]])

                # Build composite video
                final_h = top_row.h + video_clip.h
                composite = mpe.CompositeVideoClip([
                    top_row.set_position((0, 0)),
                    video_clip.set_position((0, top_row.h))
                ], size=(video_clip.w, final_h))

                # Write video
                composite.write_videofile(
                    str(output_video_path.with_suffix('.mp4')),
                    codec="libx264",
                    audio=False,
                    fps=video_clip.fps or fps,
                    preset="veryfast",
                )

                video_clip.close(); img1_clip.close(); img2_clip.close(); composite.close()

                if output_video_path.exists() and output_video_path.stat().st_size > 0:
                    return True
                else:
                    dprint("overlay_start_end_images_above_video: MoviePy output missing after write.")
            except Exception as e_mov:
                dprint(f"overlay_start_end_images_above_video: MoviePy path failed – {e_mov}. Falling back to ffmpeg.")

        # ---------------------------------------------------------
        #   Fallback: FFmpeg filter_complex (no MoviePy or MoviePy failed)
        # ---------------------------------------------------------
        if not _MOVIEPY_AVAILABLE:
            try:
                # ---------------------------------------------------------
                #   Determine the resolution of the **input video**
                # ---------------------------------------------------------
                cap = cv2.VideoCapture(str(input_video_path))
                if not cap.isOpened():
                    dprint(f"overlay_start_end_images_above_video: Could not open video {input_video_path}")
                    return False
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if video_width == 0 or video_height == 0:
                    dprint(
                        f"overlay_start_end_images_above_video: Failed to read resolution from {input_video_path}"
                    )
                    return False

                half_width = video_width // 2  # Integer division for even-pixel alignment

                # Ensure output directory exists
                output_video_path = output_video_path.with_suffix('.mp4')
                output_video_path.parent.mkdir(parents=True, exist_ok=True)

                # ---------------------------------------------------------
                #   Build & run ffmpeg command
                # ---------------------------------------------------------
                # 1) Scale both images to half width / full height
                # 2) hstack -> side-by-side static banner (labelled [top])
                # 3) vstack banner with original video (labelled [vid])
                # NOTE: We explicitly scale the *video* as well into label [vid]
                #       to guarantee width/height match.  This is mostly defensive –
                #       if the video is already the desired size the scale is a NOP.

                filter_complex = (
                    f"[1:v]scale={half_width}:{video_height}[left];"  # scale start img
                    f"[2:v]scale={half_width}:{video_height}[right];"  # scale end img
                    f"[left][right]hstack=inputs=2[top];"              # combine images
                    f"[0:v]scale={video_width}:{video_height}[vid];"   # ensure video size
                    f"[top][vid]vstack=inputs=2[output]"              # stack banner + video
                )

                # Determine FPS from the input video for consistent output
                fps = 0.0
                try:
                    cap2 = cv2.VideoCapture(str(input_video_path))
                    if cap2.isOpened():
                        fps = cap2.get(cv2.CAP_PROP_FPS)
                    cap2.release()
                except Exception:
                    fps = 0.0
                if fps is None or fps <= 0.1:
                    fps = 16  # sensible default

                ffmpeg_cmd = [
                    "ffmpeg", "-y",  # overwrite output
                    "-loglevel", "error",
                    "-i", str(input_video_path),
                    "-loop", "1", "-i", str(start_image_path),
                    "-loop", "1", "-i", str(end_image_path),
                    "-filter_complex", filter_complex,
                    "-map", "[output]",
                    "-r", str(int(round(fps))),  # set output fps
                    "-shortest",  # stop when primary video stream ends
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(output_video_path.resolve()),
                ]

                dprint(f"overlay_start_end_images_above_video: Running ffmpeg to create composite video.\nCommand: {' '.join(ffmpeg_cmd)}")

                proc = subprocess.run(ffmpeg_cmd, capture_output=True)
                if proc.returncode != 0:
                    dprint(
                        f"overlay_start_end_images_above_video: ffmpeg failed (returncode={proc.returncode}).\n"
                        f"stderr: {proc.stderr.decode(errors='ignore')[:500]}"
                    )
                    # Clean up partially written file if any
                    if output_video_path.exists():
                        try:
                            output_video_path.unlink()
                        except Exception:
                            pass
                    return False

                if not output_video_path.exists() or output_video_path.stat().st_size == 0:
                    dprint(
                        f"overlay_start_end_images_above_video: Output video not created or empty at {output_video_path}"
                    )
                    return False

                return True

            except Exception as e_ffmpeg:
                dprint(f"overlay_start_end_images_above_video: ffmpeg failed – {e_ffmpeg}. Falling back to MoviePy.")
                return False

    except Exception as e_ov:
        dprint(f"overlay_start_end_images_above_video: Exception – {e_ov}")
        traceback.print_exc()
        try:
            if output_video_path and output_video_path.exists():
                output_video_path.unlink()
        except Exception:
            pass
        return False