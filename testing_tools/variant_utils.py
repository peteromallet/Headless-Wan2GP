#!/usr/bin/env python3
"""
Utility functions for processing variants.json files in base_tester.py.
Handles image cropping, guide video generation, and mask video creation.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    from PIL import Image
    _DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available for full variant processing: {e}")
    _DEPS_AVAILABLE = False

# Standard aspect ratios and their resolutions
STANDARD_RESOLUTIONS = {
    (9, 21): (438, 1024),   # 9:21 ultra-tall
    (9, 16): (508, 902),    # 9:16 portrait
    (2, 3): (512, 768),     # 2:3 portrait
    (3, 4): (576, 768),     # 3:4 portrait
    (1, 1): (670, 670),     # 1:1 square
    (3, 2): (768, 512),     # 3:2 landscape
    (4, 3): (768, 576),     # 4:3 landscape
    (16, 9): (902, 508),    # 16:9 landscape
    (21, 9): (1024, 438),   # 21:9 ultra-wide
}

def load_variants_json(variants_file: Path) -> Dict[str, Any]:
    """Load and validate variants.json file."""
    try:
        with open(variants_file, 'r') as f:
            data = json.load(f)

        if 'variants' not in data:
            raise ValueError("variants.json must contain a 'variants' key")

        return data
    except Exception as e:
        print(f"Error loading variants.json: {e}")
        return {}

def get_closest_aspect_ratio(image_path: str) -> Tuple[int, int]:
    """
    Determine the closest standard aspect ratio for an image.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) for the closest standard resolution
    """
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        img_ratio = img_width / img_height

        # Find the closest standard aspect ratio
        closest_ratio = None
        min_diff = float('inf')

        for (aspect_w, aspect_h), (res_w, res_h) in STANDARD_RESOLUTIONS.items():
            standard_ratio = aspect_w / aspect_h
            diff = abs(img_ratio - standard_ratio)

            if diff < min_diff:
                min_diff = diff
                closest_ratio = (res_w, res_h)

        return closest_ratio or (512, 512)

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return (512, 512)  # Default fallback

def crop_image_to_resolution(image_path: str, target_resolution: Tuple[int, int], output_path: str) -> bool:
    """
    Crop an image to the target resolution, maintaining aspect ratio by center cropping.

    Args:
        image_path: Path to input image
        target_resolution: Target (width, height)
        output_path: Path for output image

    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            target_width, target_height = target_resolution
            target_ratio = target_width / target_height

            img_width, img_height = img.size
            img_ratio = img_width / img_height

            if img_ratio > target_ratio:
                # Image is wider, crop width
                new_width = int(img_height * target_ratio)
                new_height = img_height
                left = (img_width - new_width) // 2
                top = 0
                right = left + new_width
                bottom = img_height
            else:
                # Image is taller, crop height
                new_width = img_width
                new_height = int(img_width / target_ratio)
                left = 0
                top = (img_height - new_height) // 2
                right = img_width
                bottom = top + new_height

            cropped = img.crop((left, top, right, bottom))
            resized = cropped.resize(target_resolution, Image.Resampling.LANCZOS)

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            resized.save(output_path)
            return True

    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")
        return False

def create_guide_video(
    start_image_path: str,
    end_image_path: str,
    resolution: Tuple[int, int],
    length: int,
    output_path: str,
    fps: int = 16
) -> bool:
    """
    Create a guide video with start and end frames, grey frames in between.

    Args:
        start_image_path: Path to starting image
        end_image_path: Path to ending image
        resolution: Video resolution (width, height)
        length: Video length in frames
        output_path: Output video path
        fps: Frames per second

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load and resize images
        start_img = cv2.imread(start_image_path)
        end_img = cv2.imread(end_image_path)

        if start_img is None or end_img is None:
            print(f"Error: Could not load images {start_image_path} or {end_image_path}")
            return False

        width, height = resolution
        start_frame = cv2.resize(start_img, (width, height))
        end_frame = cv2.resize(end_img, (width, height))

        # Create grey frame
        grey_frame = np.full((height, width, 3), 128, dtype=np.uint8)  # Mid-grey

        # Create frame list
        frames = []
        for i in range(length):
            if i == 0:
                frames.append(start_frame.copy())
            elif i == length - 1:
                frames.append(end_frame.copy())
            else:
                frames.append(grey_frame.copy())

        # Write video using FFmpeg
        return write_frames_to_video(frames, output_path, fps, resolution)

    except Exception as e:
        print(f"Error creating guide video: {e}")
        return False

def create_mask_video(
    resolution: Tuple[int, int],
    length: int,
    output_path: str,
    fps: int = 16
) -> bool:
    """
    Create a mask video with black frames for active (first and last) frames,
    white frames for middle frames.

    Args:
        resolution: Video resolution (width, height)
        length: Video length in frames
        output_path: Output video path
        fps: Frames per second

    Returns:
        True if successful, False otherwise
    """
    try:
        width, height = resolution

        # Create black and white frames
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black
        white_frame = np.full((height, width, 3), 255, dtype=np.uint8)  # White

        # Create frame list
        frames = []
        for i in range(length):
            if i == 0 or i == length - 1:
                frames.append(black_frame.copy())  # Active frames are black
            else:
                frames.append(white_frame.copy())  # Middle frames are white

        # Write video using FFmpeg
        return write_frames_to_video(frames, output_path, fps, resolution)

    except Exception as e:
        print(f"Error creating mask video: {e}")
        return False

def write_frames_to_video(frames: List[np.ndarray], output_path: str, fps: int, resolution: Tuple[int, int]) -> bool:
    """
    Write a list of frames to a video file using FFmpeg subprocess.

    Args:
        frames: List of BGR frames as numpy arrays
        output_path: Output video path
        fps: Frames per second
        resolution: Video resolution (width, height)

    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        width, height = resolution

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "23",
            str(output_path_obj.resolve())
        ]

        # Prepare frame data
        raw_video_data = b''.join(frame.tobytes() for frame in frames)

        if not raw_video_data:
            print("Error: No frame data to write")
            return False

        # Run FFmpeg
        proc = subprocess.run(
            ffmpeg_cmd,
            input=raw_video_data,
            capture_output=True,
            timeout=60
        )

        if proc.returncode == 0:
            if output_path_obj.exists() and output_path_obj.stat().st_size > 0:
                print(f"Successfully created video: {output_path_obj}")
                return True
            else:
                print(f"Error: Video file not created or empty: {output_path_obj}")
                return False
        else:
            print(f"FFmpeg error (return code {proc.returncode}): {proc.stderr.decode()}")
            return False

    except subprocess.TimeoutExpired:
        print("Error: FFmpeg timeout")
        return False
    except FileNotFoundError:
        print("Error: FFmpeg not found")
        return False
    except Exception as e:
        print(f"Error writing video: {e}")
        return False

def process_variant(
    variant_data: Dict[str, Any],
    variant_num: int,
    experiment_folder: Path,
    temp_dir: Path
) -> Dict[str, str]:
    """
    Process a single variant from variants.json.

    Args:
        variant_data: Dictionary containing variant specification
        variant_num: Variant number for file naming
        experiment_folder: Experiment folder path
        temp_dir: Temporary directory for processing

    Returns:
        Dictionary with paths to generated files
    """
    try:
        # Extract variant parameters
        prompt = variant_data.get('prompt', '')
        length = variant_data.get('length', 81)
        start_image = variant_data.get('starting_image', '')
        end_image = variant_data.get('ending_image', '')

        if not start_image or not end_image:
            print(f"Error: Variant {variant_num} missing start or end image")
            return {}

        # Resolve image paths (could be relative to experiment folder)
        start_image_path = str(experiment_folder / start_image) if not Path(start_image).is_absolute() else start_image
        end_image_path = str(experiment_folder / end_image) if not Path(end_image).is_absolute() else end_image

        if not Path(start_image_path).exists() or not Path(end_image_path).exists():
            print(f"Error: Variant {variant_num} image files not found: {start_image_path}, {end_image_path}")
            return {}

        # Get closest aspect ratio
        target_resolution = get_closest_aspect_ratio(start_image_path)
        print(f"Variant {variant_num}: Using resolution {target_resolution[0]}x{target_resolution[1]}")

        # Create processed image paths
        start_processed = temp_dir / f"{variant_num}_start_cropped.png"
        end_processed = temp_dir / f"{variant_num}_end_cropped.png"

        # Crop images to target resolution
        if not crop_image_to_resolution(start_image_path, target_resolution, str(start_processed)):
            print(f"Error: Failed to crop start image for variant {variant_num}")
            return {}

        if not crop_image_to_resolution(end_image_path, target_resolution, str(end_processed)):
            print(f"Error: Failed to crop end image for variant {variant_num}")
            return {}

        # Create guide and mask videos
        guide_video_path = temp_dir / f"{variant_num}_video.mp4"
        mask_video_path = temp_dir / f"{variant_num}_mask.mp4"

        if not create_guide_video(str(start_processed), str(end_processed), target_resolution, length, str(guide_video_path)):
            print(f"Error: Failed to create guide video for variant {variant_num}")
            return {}

        if not create_mask_video(target_resolution, length, str(mask_video_path)):
            print(f"Error: Failed to create mask video for variant {variant_num}")
            return {}

        # Create prompt file
        prompt_file = temp_dir / f"{variant_num}_prompt.json"
        with open(prompt_file, 'w') as f:
            json.dump({"prompt": prompt}, f)

        return {
            'video': str(guide_video_path),
            'mask': str(mask_video_path),
            'prompt_file': str(prompt_file),
            'resolution': target_resolution,
            'length': length,
            'prompt': prompt
        }

    except Exception as e:
        print(f"Error processing variant {variant_num}: {e}")
        return {}

def create_example_variants_json(output_path: str) -> bool:
    """
    Create an example variants.json file for documentation purposes.

    Args:
        output_path: Where to save the example file

    Returns:
        True if successful
    """
    example_data = {
        "variants": [
            {
                "prompt": "A beautiful sunset over mountains",
                "length": 81,
                "starting_image": "images/sunset_start.jpg",
                "ending_image": "images/sunset_end.jpg"
            },
            {
                "prompt": "Ocean waves crashing on beach",
                "length": 49,
                "starting_image": "images/ocean_start.jpg",
                "ending_image": "images/ocean_end.jpg"
            },
            {
                "prompt": "Forest path in autumn",
                "length": 65,
                "starting_image": "images/forest_start.jpg",
                "ending_image": "images/forest_end.jpg"
            }
        ]
    }

    try:
        with open(output_path, 'w') as f:
            json.dump(example_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error creating example variants.json: {e}")
        return False