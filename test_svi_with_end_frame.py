#!/usr/bin/env python3
"""
Test script for SVI encoding with end frame support.

Uses the new wan_2_2_i2v_lightning_svi_3_3 model config to generate
a video from the last frames of a source video to a target end image.

Usage:
    python test_svi_with_end_frame.py [--start samples/video.mp4] [--end samples/1.png]
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
WAN2GP_ROOT = PROJECT_ROOT / "Wan2GP"

def extract_last_frames(video_path: str, num_frames: int = 3, output_dir: str = None) -> str:
    """
    Extract the last N frames from a video file and save as a short video clip.
    
    Returns path to the extracted clip.
    """
    import subprocess
    import json
    
    # Get video info
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    # Find video stream
    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    
    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")
    
    # Get frame count and fps
    nb_frames = int(video_stream.get("nb_frames", 0))
    fps_parts = video_stream.get("r_frame_rate", "24/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    duration = float(info.get("format", {}).get("duration", 0))
    
    if nb_frames == 0:
        # Estimate from duration
        nb_frames = int(duration * fps)
    
    print(f"Video info: {nb_frames} frames, {fps:.2f} fps, {duration:.2f}s")
    
    # Calculate start time for last N frames
    start_frame = max(0, nb_frames - num_frames)
    start_time = start_frame / fps
    
    print(f"Extracting frames {start_frame}-{nb_frames} (last {num_frames} frames)")
    
    # Create output path
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="svi_test_")
    output_path = os.path.join(output_dir, "start_clip.mp4")
    
    # Extract frames as video clip
    extract_cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-i", video_path,
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(extract_cmd, capture_output=True)
    
    print(f"Extracted clip saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test SVI encoding with end frame")
    parser.add_argument("--start", default="samples/video.mp4", help="Source video (will use last 3 frames)")
    parser.add_argument("--end", default="samples/1.png", help="Target end image")
    parser.add_argument("--anchor", default=None, help="Anchor/reference image for SVI (optional, defaults to first frame of start)")
    parser.add_argument("--prompt", default="A smooth cinematic transition between scenes", help="Generation prompt")
    parser.add_argument("--frames", type=int, default=81, help="Output video length in frames")
    parser.add_argument("--num-start-frames", type=int, default=3, help="Number of frames to extract from end of start video")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be done")
    args = parser.parse_args()
    
    # Resolve paths
    start_video = PROJECT_ROOT / args.start
    end_image = PROJECT_ROOT / args.end
    anchor_image = PROJECT_ROOT / args.anchor if args.anchor else None
    output_dir = PROJECT_ROOT / args.output_dir
    
    # Validate inputs
    if not start_video.exists():
        print(f"❌ Start video not found: {start_video}")
        sys.exit(1)
    if not end_image.exists():
        print(f"❌ End image not found: {end_image}")
        sys.exit(1)
    if anchor_image and not anchor_image.exists():
        print(f"❌ Anchor image not found: {anchor_image}")
        sys.exit(1)
    
    print("=" * 60)
    print("SVI + End Frame Test")
    print("=" * 60)
    print(f"Start video: {start_video}")
    print(f"End image: {end_image}")
    print(f"Anchor image: {anchor_image or '(will use start frame)'}")
    print(f"Prompt: {args.prompt}")
    print(f"Output frames: {args.frames}")
    print("=" * 60)
    
    # Extract last frames from start video
    print("\n📹 Extracting last frames from start video...")
    start_clip = extract_last_frames(str(start_video), num_frames=args.num_start_frames)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - would execute:")
        print(f"  Model: wan_2_2_i2v_lightning_svi_3_3")
        print(f"  Start clip: {start_clip}")
        print(f"  End image: {end_image}")
        print(f"  Anchor: {anchor_image or 'first frame'}")
        print(f"  Frames: {args.frames}")
        return
    
    # Change to Wan2GP directory (required for wgp.py imports)
    os.chdir(WAN2GP_ROOT)
    sys.path.insert(0, str(WAN2GP_ROOT))
    
    # Import orchestrator
    from headless_wgp import WanOrchestrator
    
    # Initialize orchestrator
    print("\n🔧 Initializing WanOrchestrator...")
    orchestrator = WanOrchestrator(
        wan_root=str(WAN2GP_ROOT),
        main_output_dir=str(output_dir)
    )
    
    # Load the SVI model
    model_key = "wan_2_2_i2v_lightning_svi_3_3"
    print(f"\n📦 Loading model: {model_key}")
    orchestrator.load_model(model_key)
    
    # Load images
    from PIL import Image
    
    # For I2V, we need to provide:
    # - image_start: The starting frame(s) - from our extracted clip
    # - image_end: The target end frame
    # - input_ref_images: The anchor image for SVI (optional)
    
    # Load the start clip as PIL images
    import cv2
    cap = cv2.VideoCapture(start_clip)
    start_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_frames.append(Image.fromarray(frame_rgb))
    cap.release()
    
    print(f"Loaded {len(start_frames)} start frames")
    
    # Use last frame as image_start (or all frames if your model supports multi-frame input)
    image_start = start_frames[-1] if start_frames else None
    
    # Load end image
    image_end = Image.open(str(end_image)).convert("RGB")
    print(f"Loaded end image: {image_end.size}")
    
    # Load or use anchor image
    if anchor_image:
        anchor = Image.open(str(anchor_image)).convert("RGB")
        print(f"Using custom anchor image: {anchor.size}")
    else:
        # For SVI, anchor should match image_start for consistent generation
        # (WGP defaults to pre_video_frame = last frame of source, which is image_start)
        anchor = image_start
        print(f"Using image_start as anchor (recommended for SVI): {anchor.size if anchor else 'None'}")
    
    # Generate
    print("\n🎬 Starting generation...")
    print(f"[DEBUG] image_start: {type(image_start)}, size: {image_start.size if image_start else 'None'}")
    print(f"[DEBUG] image_end: {type(image_end)}, size: {image_end.size if image_end else 'None'}")
    print(f"[DEBUG] anchor: {type(anchor)}, size: {anchor.size if anchor else 'None'}")
    output_path = orchestrator.generate(
        prompt=args.prompt,
        image_start=image_start,
        image_end=image_end,
        image_refs=[anchor],  # SVI anchor image (WGP expects image_refs, not input_ref_images)
        image_prompt_type="T",  # Use image as prompt
        video_prompt_type="I",  # CRITICAL: "I" enables image_refs to be passed through to SVI
        video_length=args.frames,
        resolution="768x576",
    )
    
    print("\n" + "=" * 60)
    print("✅ Generation complete!")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

