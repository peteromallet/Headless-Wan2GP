import math
import subprocess
from pathlib import Path
import traceback

import cv2 # pip install opencv-python
import numpy as np

# Import dprint from common_utils (assuming it's discoverable)
from .common_utils import dprint, get_video_frame_count_and_fps # Added get_video_frame_count_and_fps

# --- Easing function for cross-fading ---
def ease(alpha_lin: float) -> float:
    """cosine ease-in-out  (0..1 -> 0..1)"""
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

    out_frames = []
    for i in range(n):
        t_linear = (i + 1) / float(n)
        alpha = ease(t_linear)

        frame_a_np = segment1_frames[-n+i].astype(np.float32)
        frame_b_np = segment2_frames[i].astype(np.float32)

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

def extract_frames_from_video(video_path: str | Path, start_frame: int = 0, num_frames: int = None) -> list[np.ndarray]:
    """
    Extracts frames from a video file as numpy arrays.
    
    Args:
        video_path: Path to the video file
        start_frame: Starting frame index (0-based)
        num_frames: Number of frames to extract (None = all remaining frames)
    
    Returns:
        List of frames as BGR numpy arrays
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        dprint(f"Error: Could not open video {video_path}")
        return frames
    
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    
    frames_to_read = num_frames if num_frames is not None else (total_frames_video - start_frame)
    frames_to_read = min(frames_to_read, total_frames_video - start_frame)
    
    for i in range(frames_to_read):
        ret, frame = cap.read()
        if not ret:
            dprint(f"Warning: Could not read frame {start_frame + i} from {video_path}")
            break
        frames.append(frame)
    
    cap.release()
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

    dprint(f"Attempting to write video to: {output_path_mp4} using FFmpeg: {' '.join(ffmpeg_cmd)}")

    processed_frames = []
    for frame_idx, frame_np in enumerate(frames_list):
        if frame_np is None:
            dprint(f"Warning: Frame {frame_idx} is None. Skipping.")
            continue
        if not isinstance(frame_np, np.ndarray):
            dprint(f"Warning: Frame {frame_idx} is not a numpy array ({type(frame_np)}). Skipping.")
            continue
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)
        if frame_np.shape[0] != resolution[1] or frame_np.shape[1] != resolution[0] or frame_np.shape[2] != 3:
            dprint(f"Warning: Frame {frame_idx} has incorrect shape {frame_np.shape}, expected ({resolution[1]}, {resolution[0]}, 3). Resizing.")
            try:
                frame_np = cv2.resize(frame_np, resolution, interpolation=cv2.INTER_AREA)
            except Exception as e_resize:
                dprint(f"Error resizing frame {frame_idx}: {e_resize}. Skipping.")
                continue
        processed_frames.append(frame_np)

    if not processed_frames:
        dprint("Error: No valid frames to write to video.")
        return None

    try:
        raw_video_data = b''.join(frame.tobytes() for frame in processed_frames)
    except Exception as e_data:
        dprint(f"Error creating raw video data: {e_data}")
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
                dprint(f"Generated video (FFmpeg/libx264): {output_path_mp4} ({len(processed_frames)} frames)")
                return output_path_mp4
            else:
                dprint(f"FFmpeg process completed (rc=0) but output file {output_path_mp4} is missing or empty.")
                dprint(f"FFmpeg stdout: {proc.stdout.decode(errors='ignore')}")
                dprint(f"FFmpeg stderr: {proc.stderr.decode(errors='ignore')}")
                return None
        else:
            dprint(f"FFmpeg failed for {output_path_mp4}. Return code: {proc.returncode}")
            dprint(f"FFmpeg stdout: {proc.stdout.decode(errors='ignore')}")
            dprint(f"FFmpeg stderr: {proc.stderr.decode(errors='ignore')}")
            if output_path_mp4.exists():
                try: output_path_mp4.unlink()
                except Exception as e_unlink: dprint(f"Could not remove partially written/failed MP4 file {output_path_mp4}: {e_unlink}")
            return None
            
    except subprocess.TimeoutExpired:
        dprint(f"FFmpeg process timed out after 60 seconds for {output_path_mp4}")
        return None
    except FileNotFoundError:
        dprint("Error: ffmpeg command not found. Please ensure FFmpeg is installed and in your PATH.")
        return None
    except Exception as e_proc:
        dprint(f"Error running FFmpeg subprocess: {e_proc}")
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

    dprint(f"SATURATION_ADJUST: Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        if outp.exists() and outp.stat().st_size > 0:
            return True
        else:
            dprint(f"SATURATION_ADJUST: Output file {outp} missing or empty after ffmpeg run.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg saturation adjust of {inp} -> {outp}:\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
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
            print(f"Error: Could not open reference video {reference_video_path}")
            return False
        
        ref_frame_count_from_cap, _ = get_video_frame_count_and_fps(str(reference_video_path)) # Use helper
        if ref_frame_count_from_cap is None: ref_frame_count_from_cap = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Fallback

        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, ref_frame_count_from_cap - 1)))
        ret, ref_frame = ref_cap.read()
        ref_cap.release()
        
        if not ret or ref_frame is None:
            print(f"Error: Could not read reference frame from {reference_video_path}")
            return False
        
        if ref_frame.shape[1] != parsed_resolution[0] or ref_frame.shape[0] != parsed_resolution[1]:
            ref_frame = cv2.resize(ref_frame, parsed_resolution, interpolation=cv2.INTER_AREA)
        
        src_cap = cv2.VideoCapture(str(source_video_path))
        if not src_cap.isOpened():
            print(f"Error: Could not open source video {source_video_path}")
            return False
        
        src_fps_from_cap = src_cap.get(cv2.CAP_PROP_FPS)
        # src_frame_count_from_cap = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Not strictly needed here

        ret, src_first_frame = src_cap.read()
        if not ret or src_first_frame is None:
            print(f"Error: Could not read first frame from {source_video_path}")
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
            print(f"Error: Could not create output video writer for {output_path_obj}")
            src_cap.release()
            return False
        
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames_processed = 0
        while True:
            ret, frame = src_cap.read()
            if not ret: break
            if frame.shape[1] != parsed_resolution[0] or frame.shape[0] != parsed_resolution[1]:
                frame = cv2.resize(frame, parsed_resolution, interpolation=cv2.INTER_AREA)
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