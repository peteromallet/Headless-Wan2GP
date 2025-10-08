"""
Visualization utilities for creating debug/preview collages of video generation tasks.

This module provides tools to create informative visualizations that show:
- Input images with current segment highlighted
- Structure/guidance videos side-by-side
- Timeline progress indicators
- Segment boundaries and transitions
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import cv2


def _apply_video_treatment(clip, target_duration: float, target_fps: int, treatment: str, video_name: str = "video"):
    """
    Apply treatment to video clip to match target duration using FRAME SAMPLING.

    This matches the exact logic from structure_video_guidance.py:load_structure_video_frames()
    by sampling specific frame indices rather than changing playback speed.

    Args:
        clip: MoviePy VideoFileClip
        target_duration: Target duration in seconds
        target_fps: Target FPS for output
        treatment: "adjust" (linear frame sampling) or "clip" (FPS-based sampling)
        video_name: Name for logging

    Returns:
        Treated VideoFileClip with resampled frames
    """
    import cv2
    from moviepy.editor import ImageSequenceClip

    current_fps = clip.fps
    current_frame_count = int(clip.duration * clip.fps)
    target_frame_count = int(target_duration * target_fps)

    if treatment == "adjust":
        # ADJUST MODE: Linear interpolation sampling (matches generation logic line 201)
        # Formula: frame_indices = [int(i * (video_frame_count - 1) / (frames_to_load - 1)) for i in range(frames_to_load)]

        if current_frame_count >= target_frame_count:
            # Compress: Sample evenly across video
            frame_indices = [int(i * (current_frame_count - 1) / (target_frame_count - 1))
                           for i in range(target_frame_count)]
            print(f"  {video_name}: adjust mode - sampling {target_frame_count} from {current_frame_count} frames (compress)")
        else:
            # Stretch: Repeat frames to reach target count
            frame_indices = [int(i * (current_frame_count - 1) / (target_frame_count - 1))
                           for i in range(target_frame_count)]
            duplicates = target_frame_count - len(set(frame_indices))
            print(f"  {video_name}: adjust mode - sampling {target_frame_count} from {current_frame_count} frames (stretch, {duplicates} duplicates)")

    elif treatment == "clip":
        # CLIP MODE: FPS-based temporal sampling (matches generation logic line 212-218)
        # Use _resample_frame_indices logic

        def resample_frame_indices(video_fps, video_frames_count, max_target_frames, target_fps):
            """Matches _resample_frame_indices from structure_video_guidance.py"""
            import math
            video_frame_duration = 1 / video_fps
            target_frame_duration = 1 / target_fps

            target_time = 0
            frame_no = math.ceil(target_time / video_frame_duration)
            cur_time = frame_no * video_frame_duration
            frame_ids = []

            while True:
                if len(frame_ids) >= max_target_frames:
                    break
                diff = round((target_time - cur_time) / video_frame_duration, 5)
                add_frames_count = math.ceil(diff)
                frame_no += add_frames_count
                if frame_no >= video_frames_count:
                    break
                frame_ids.append(frame_no)
                cur_time += add_frames_count * video_frame_duration
                target_time += target_frame_duration

            return frame_ids[:max_target_frames]

        frame_indices = resample_frame_indices(current_fps, current_frame_count, target_frame_count, target_fps)

        # If video too short, loop (matches generation logic line 221-225)
        if len(frame_indices) < target_frame_count:
            print(f"  {video_name}: clip mode - video too short, looping to fill {target_frame_count} frames")
            while len(frame_indices) < target_frame_count:
                remaining = target_frame_count - len(frame_indices)
                frame_indices.extend(frame_indices[:remaining])

        print(f"  {video_name}: clip mode - sampled {len(frame_indices)} frames from {current_frame_count}")

    else:
        raise ValueError(f"Invalid treatment: {treatment}. Must be 'adjust' or 'clip'")

    # Extract frames at the specified indices
    print(f"  {video_name}: extracting {len(frame_indices)} frames...")
    frames = []
    for idx in frame_indices:
        time_at_frame = idx / clip.fps
        frame = clip.get_frame(time_at_frame)
        frames.append(frame)

    # Create new clip from resampled frames
    resampled_clip = ImageSequenceClip(frames, fps=target_fps)

    print(f"  {video_name}: resampled to {len(frames)} frames @ {target_fps}fps ({resampled_clip.duration:.2f}s)")

    return resampled_clip


def create_travel_visualization(
    output_video_path: str,
    structure_video_path: str,
    guidance_video_path: Optional[str],
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]] = None,
    viz_output_path: Optional[str] = None,
    layout: str = "side_by_side",
    fps: int = 16,
    show_guidance: bool = True,
    structure_video_treatment: str = "adjust",
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
) -> str:
    """
    Create a visualization collage showing the generation process.

    Args:
        output_video_path: Path to final output video
        structure_video_path: Path to structure/flow video
        guidance_video_path: Path to guidance video (optional)
        input_image_paths: List of input image paths
        segment_frames: List of frame counts per segment (raw, before overlap subtraction)
        segment_prompts: Optional list of prompts per segment
        viz_output_path: Where to save visualization (default: adds _viz suffix)
        layout: Layout type - "side_by_side", "triple", "grid", or "vertical"
        fps: FPS for output video
        show_guidance: Whether to include guidance video
        structure_video_treatment: "adjust" (stretch/compress) or "clip" (temporal sample)
        frame_overlaps: Optional list of overlap counts between segments
        structure_video_type: Optional structure video type label for overlay (e.g., "depth", "canny")
        structure_video_strength: Optional structure video strength value for overlay

    Returns:
        Path to created visualization video
    """
    try:
        from moviepy.editor import (
            VideoFileClip, ImageClip, CompositeVideoClip,
            TextClip, concatenate_videoclips, clips_array
        )
        from moviepy.video.fx import resize
    except ImportError:
        raise ImportError(
            "MoviePy is required for visualization. Install with: pip install moviepy"
        )

    # Determine output path
    if viz_output_path is None:
        output_path = Path(output_video_path)
        viz_output_path = str(output_path.parent / f"{output_path.stem}_viz.mp4")

    # Load videos
    print(f"📹 Loading videos for visualization...")
    output_clip = VideoFileClip(output_video_path)
    structure_clip = VideoFileClip(structure_video_path)

    if guidance_video_path and show_guidance:
        guidance_clip = VideoFileClip(guidance_video_path)
    else:
        guidance_clip = None

    # Apply treatment to structure video
    print(f"📐 Structure video treatment: {structure_video_treatment}")
    structure_clip = _apply_video_treatment(
        structure_clip,
        target_duration=output_clip.duration,
        target_fps=fps,
        treatment=structure_video_treatment,
        video_name="structure"
    )

    # Apply treatment to guidance video if present
    if guidance_clip:
        guidance_clip = _apply_video_treatment(
            guidance_clip,
            target_duration=output_clip.duration,
            target_fps=fps,
            treatment=structure_video_treatment,
            video_name="guidance"
        )

    # Create layout based on type
    if layout == "side_by_side":
        result = _create_side_by_side_layout(
            output_clip=output_clip,
            structure_clip=structure_clip,
            guidance_clip=guidance_clip,
            input_image_paths=input_image_paths,
            segment_frames=segment_frames,
            segment_prompts=segment_prompts,
            fps=fps,
            frame_overlaps=frame_overlaps,
            structure_video_type=structure_video_type,
            structure_video_strength=structure_video_strength
        )
    elif layout == "triple":
        result = _create_triple_layout(
            output_clip=output_clip,
            structure_clip=structure_clip,
            guidance_clip=guidance_clip,
            input_image_paths=input_image_paths,
            segment_frames=segment_frames,
            segment_prompts=segment_prompts,
            fps=fps,
            frame_overlaps=frame_overlaps,
            structure_video_type=structure_video_type,
            structure_video_strength=structure_video_strength
        )
    elif layout == "grid":
        result = _create_grid_layout(
            output_clip=output_clip,
            structure_clip=structure_clip,
            guidance_clip=guidance_clip,
            input_image_paths=input_image_paths,
            segment_frames=segment_frames,
            segment_prompts=segment_prompts,
            fps=fps,
            frame_overlaps=frame_overlaps,
            structure_video_type=structure_video_type,
            structure_video_strength=structure_video_strength
        )
    elif layout == "vertical":
        result = _create_vertical_layout(
            output_clip=output_clip,
            structure_clip=structure_clip,
            guidance_clip=guidance_clip,
            input_image_paths=input_image_paths,
            segment_frames=segment_frames,
            segment_prompts=segment_prompts,
            fps=fps,
            frame_overlaps=frame_overlaps,
            structure_video_type=structure_video_type,
            structure_video_strength=structure_video_strength
        )
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Write output
    print(f"💾 Writing visualization to: {viz_output_path}")
    result.write_videofile(
        viz_output_path,
        fps=fps,
        codec='libx264',
        audio=False,
        preset='medium',
        ffmpeg_params=['-pix_fmt', 'yuv420p'],  # Force compatible pixel format
        logger=None  # Suppress moviepy progress bar
    )

    # Cleanup
    output_clip.close()
    structure_clip.close()
    if guidance_clip:
        guidance_clip.close()
    result.close()

    print(f"✅ Visualization saved: {viz_output_path}")
    return viz_output_path


def _create_side_by_side_layout(
    output_clip,
    structure_clip,
    guidance_clip,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
):
    """
    Create side-by-side layout with timeline on top.

    Layout:
    ┌─────────────────────────────────────┐
    │  [Img1] [Img2] [Img3] [Img4]        │
    │  ════════█═══════════════════        │
    ├──────────────────┬──────────────────┤
    │  Structure Video │  Output Video    │
    └──────────────────┴──────────────────┘
    """
    from moviepy.editor import clips_array, CompositeVideoClip

    # Get dimensions
    target_height = 400
    target_width = structure_clip.w * target_height // structure_clip.h

    # Resize clips
    structure_resized = structure_clip.resize(height=target_height)
    output_resized = output_clip.resize(height=target_height)

    # Add structure video overlay if metadata provided
    if structure_video_type and structure_video_strength is not None:
        from moviepy.editor import TextClip, CompositeVideoClip

        # Create overlay text
        overlay_text = f"{structure_video_type} | {structure_video_strength:.1f}"

        try:
            text_clip = TextClip(
                overlay_text,
                fontsize=16,
                color='white',
                font='DejaVu-Sans-Bold',
                bg_color='black',
                method='caption',
                size=(None, None)
            ).set_duration(structure_resized.duration).set_position(('left', 'top'))

            structure_resized = CompositeVideoClip([structure_resized, text_clip])
        except Exception as e:
            print(f"Warning: Could not add structure overlay: {e}")

    # Create side-by-side video
    video_array = clips_array([[structure_resized, output_resized]])

    # Create timeline overlay
    timeline_clip = _create_timeline_clip(
        duration=output_clip.duration,
        width=video_array.w,
        height=150,
        input_image_paths=input_image_paths,
        segment_frames=segment_frames,
        segment_prompts=segment_prompts,
        fps=fps,
        frame_overlaps=frame_overlaps
    )

    # Position timeline at top
    timeline_clip = timeline_clip.set_position(("center", 0))

    # Composite
    final = CompositeVideoClip(
        [video_array, timeline_clip],
        size=(video_array.w, video_array.h + 150)
    )

    return final


def _create_triple_layout(
    output_clip,
    structure_clip,
    guidance_clip,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
):
    """
    Create triple view layout.

    Layout:
    ┌──────────┬──────────┬──────────┐
    │ Structure│ Guidance │  Output  │
    ├──────────┴──────────┴──────────┤
    │ [Img1] [Img2] [Img3] [Img4]    │
    └────────────────────────────────┘
    """
    from moviepy.editor import clips_array, CompositeVideoClip

    # Get dimensions
    target_height = 400
    target_width = structure_clip.w * target_height // structure_clip.h

    # Resize clips
    structure_resized = structure_clip.resize(height=target_height)
    output_resized = output_clip.resize(height=target_height)

    # Add structure video overlay if metadata provided
    if structure_video_type and structure_video_strength is not None:
        from moviepy.editor import TextClip, CompositeVideoClip

        # Create overlay text
        overlay_text = f"{structure_video_type} | {structure_video_strength:.1f}"

        try:
            text_clip = TextClip(
                overlay_text,
                fontsize=16,
                color='white',
                font='DejaVu-Sans-Bold',
                bg_color='black',
                method='caption',
                size=(None, None)
            ).set_duration(structure_resized.duration).set_position(('left', 'top'))

            structure_resized = CompositeVideoClip([structure_resized, text_clip])
        except Exception as e:
            print(f"Warning: Could not add structure overlay: {e}")

    if guidance_clip:
        guidance_resized = guidance_clip.resize(height=target_height)
        video_array = clips_array([[structure_resized, guidance_resized, output_resized]])
    else:
        video_array = clips_array([[structure_resized, output_resized]])

    # Create timeline overlay
    timeline_clip = _create_timeline_clip(
        duration=output_clip.duration,
        width=video_array.w,
        height=100,
        input_image_paths=input_image_paths,
        segment_frames=segment_frames,
        segment_prompts=segment_prompts,
        fps=fps,
        frame_overlaps=frame_overlaps
    )

    # Position timeline at bottom
    timeline_clip = timeline_clip.set_position(("center", video_array.h))

    # Composite
    final = CompositeVideoClip(
        [video_array, timeline_clip],
        size=(video_array.w, video_array.h + 100)
    )

    return final


def _create_grid_layout(
    output_clip,
    structure_clip,
    guidance_clip,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
):
    """
    Create 2x2 grid layout.

    Layout:
    ┌──────────────┬──────────────┐
    │  Structure   │   Guidance   │
    ├──────────────┼──────────────┤
    │   Output     │   Timeline   │
    └──────────────┴──────────────┘
    """
    from moviepy.editor import clips_array

    # Get dimensions
    target_height = 300
    target_width = structure_clip.w * target_height // structure_clip.h

    # Resize clips
    structure_resized = structure_clip.resize(height=target_height)
    output_resized = output_clip.resize(height=target_height)

    # Add structure video overlay if metadata provided
    if structure_video_type and structure_video_strength is not None:
        from moviepy.editor import TextClip, CompositeVideoClip

        # Create overlay text
        overlay_text = f"{structure_video_type} | {structure_video_strength:.1f}"

        try:
            text_clip = TextClip(
                overlay_text,
                fontsize=16,
                color='white',
                font='DejaVu-Sans-Bold',
                bg_color='black',
                method='caption',
                size=(None, None)
            ).set_duration(structure_resized.duration).set_position(('left', 'top'))

            structure_resized = CompositeVideoClip([structure_resized, text_clip])
        except Exception as e:
            print(f"Warning: Could not add structure overlay: {e}")

    if guidance_clip:
        guidance_resized = guidance_clip.resize(height=target_height)
    else:
        # Create blank clip
        from moviepy.editor import ColorClip
        guidance_resized = ColorClip(
            size=(target_width, target_height),
            color=(0, 0, 0),
            duration=output_clip.duration
        )

    # Create timeline as clip
    timeline_clip = _create_timeline_clip(
        duration=output_clip.duration,
        width=target_width,
        height=target_height,
        input_image_paths=input_image_paths,
        segment_frames=segment_frames,
        segment_prompts=segment_prompts,
        fps=fps,
        frame_overlaps=frame_overlaps
    )

    # Arrange in 2x2 grid
    video_array = clips_array([
        [structure_resized, guidance_resized],
        [output_resized, timeline_clip]
    ])

    return video_array


def _create_vertical_layout(
    output_clip,
    structure_clip,
    guidance_clip,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
):
    """
    Create vertical layout with images on left, videos stacked on right.

    Layout:
    ┌────────┬───────────────┐
    │ Img 1  │               │
    │ Img 2  │  Structure    │
    │ Img 3  │     Video     │
    │ Img 4  │               │
    │ Img 5  │───────────────│
    │  ...   │               │
    │        │  Output       │
    │        │    Video      │
    │        │               │
    └────────┴───────────────┘
    """
    from moviepy.editor import clips_array, CompositeVideoClip

    # Target dimensions for videos
    video_target_height = 300

    # Resize structure and output videos
    structure_resized = structure_clip.resize(height=video_target_height)
    output_resized = output_clip.resize(height=video_target_height)

    # Ensure resized clips have even dimensions for H.264 compatibility
    def ensure_even_dimensions(clip):
        w, h = clip.w, clip.h
        if w % 2 != 0:
            w += 1
        if h % 2 != 0:
            h += 1
        if w != clip.w or h != clip.h:
            return clip.resize(width=w, height=h)
        return clip

    structure_resized = ensure_even_dimensions(structure_resized)
    output_resized = ensure_even_dimensions(output_resized)

    print(f"  structure_resized: {structure_resized.w}x{structure_resized.h}")
    print(f"  output_resized: {output_resized.w}x{output_resized.h}")

    # Add structure video overlay if metadata provided
    if structure_video_type and structure_video_strength is not None:
        from moviepy.editor import TextClip, CompositeVideoClip

        # Create overlay text
        overlay_text = f"{structure_video_type} | {structure_video_strength:.1f}"

        try:
            text_clip = TextClip(
                overlay_text,
                fontsize=16,
                color='white',
                font='DejaVu-Sans-Bold',
                bg_color='black',
                method='caption',
                size=(None, None)
            ).set_duration(structure_resized.duration).set_position(('left', 'top'))

            structure_resized = CompositeVideoClip([structure_resized, text_clip])
        except Exception as e:
            print(f"Warning: Could not add structure overlay: {e}")

    # Stack videos vertically on the right
    video_stack = clips_array([[structure_resized], [output_resized]])

    # Create timeline clip for the left side (vertical arrangement of images)
    # Make it match the height of the video stack
    timeline_height = video_stack.h
    timeline_width = 200  # Fixed width for the image column

    # Ensure the TOTAL width (timeline + video_stack) is even for H.264 compatibility
    video_stack_width = video_stack.w
    total_width = timeline_width + video_stack_width

    # If total is odd, adjust timeline width to make it even
    if total_width % 2 != 0:
        timeline_width += 1  # Change timeline from 200 to 201

    # Also ensure height is even
    if video_stack.h % 2 != 0:
        timeline_height = video_stack.h + 1
        video_stack = video_stack.resize(height=timeline_height)

    timeline_clip = _create_timeline_clip(
        duration=output_clip.duration,
        width=timeline_width,
        height=timeline_height,
        input_image_paths=input_image_paths,
        segment_frames=segment_frames,
        segment_prompts=segment_prompts,
        fps=fps,
        frame_overlaps=frame_overlaps,
        vertical=True  # Stack images vertically for vertical layout
    )

    # Combine timeline (left) with video_stack (right)
    final = clips_array([[timeline_clip, video_stack]])

    # Ensure final dimensions are even for H.264 compatibility
    final_width = final.w
    final_height = final.h
    if final_width % 2 != 0 or final_height % 2 != 0:
        if final_width % 2 != 0:
            final_width += 1
        if final_height % 2 != 0:
            final_height += 1
        final = final.resize(width=final_width, height=final_height)

    return final


def _create_timeline_clip(
    duration: float,
    width: int,
    height: int,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    vertical: bool = False
):
    """
    Create an animated timeline clip showing:
    - Thumbnail strip of input images
    - Progress bar with current segment highlighted
    - Current frame number and segment info

    Args:
        frame_overlaps: List of overlap frame counts between segments.
                       Overlaps are subtracted when calculating final positions.
    """
    from moviepy.editor import VideoClip

    # Calculate segment boundaries ACCOUNTING FOR OVERLAPS first
    # This determines where each image should be positioned
    overlaps = frame_overlaps if frame_overlaps else [0] * (len(segment_frames) - 1)

    segment_boundaries = []
    cumulative_start = 0

    for i, seg_frames in enumerate(segment_frames):
        segment_end = cumulative_start + seg_frames - 1
        segment_boundaries.append((cumulative_start, segment_end))

        # Next segment starts at current_end + 1 - overlap
        if i < len(overlaps):
            cumulative_start = segment_end + 1 - overlaps[i]
        else:
            cumulative_start = segment_end + 1

    # Calculate actual total frames (accounting for overlaps)
    total_frames = sum(segment_frames) - sum(overlaps)

    # Calculate image positions:
    # - First image at frame 0 (start of video)
    # - Remaining images at their segment end frames
    # - If we have N+1 images for N segments, the extra image is at the very end
    image_frame_positions = [0]  # First image always at frame 0

    for start, end in segment_boundaries:
        image_frame_positions.append(end)

    # If we have more images than segments+1, distribute them
    # For now, we expect len(images) == len(segments) + 1
    # So we should have: [0, seg0_end, seg1_end, seg2_end] = [0, 70, 139, 183]

    # But if len(images) == len(segments), use segment ends: [seg0_end, seg1_end, seg2_end]
    if len(input_image_paths) == len(segment_frames):
        image_frame_positions = [end for start, end in segment_boundaries]
    else:
        # Standard case: N+1 images for N segments
        image_frame_positions = [0] + [end for start, end in segment_boundaries]

    # Calculate width available for images/bar (with margins)
    margin = 20
    available_width = width - (2 * margin)

    # Load and prepare input images with consistent sizing
    num_images = len(input_image_paths)

    if vertical:
        # For vertical layout, make images much larger
        # Use most of the width, leaving room for progress bar and margins
        max_thumb_width = int(width * 0.8)  # 80% of width for images
        available_height = height - (2 * margin)
        # Divide height by number of images
        max_thumb_height = int(available_height / (num_images + 0.5))  # Leave some spacing
    else:
        # Original horizontal sizing
        spacing = 10
        total_spacing = spacing * (num_images - 1)
        max_thumb_width = int((available_width - total_spacing) / num_images)
        max_thumb_width = int(max_thumb_width * 0.9)
        max_thumb_height = int(height * 0.6)  # Use 60% of height for images

    thumbnails = []
    for img_path in input_image_paths:
        img = Image.open(img_path).convert('RGB')
        # Resize to fixed width, maintaining aspect ratio
        aspect = img.height / img.width
        target_h = int(max_thumb_width * aspect)
        if target_h > max_thumb_height:
            target_h = max_thumb_height
            thumb_w = int(max_thumb_height / aspect)
        else:
            thumb_w = max_thumb_width
        img = img.resize((thumb_w, target_h), Image.Resampling.LANCZOS)
        thumbnails.append(np.array(img))

    def make_frame(t):
        """Generate frame at time t."""
        # Create canvas
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray

        # Convert to PIL for drawing
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)

        # Calculate current frame and determine active image based on halfway point
        current_frame = int(t * fps)

        # For N+1 images with N segments, determine which image should be highlighted
        # Switchover happens halfway between current and next image positions
        active_image_index = 0
        if len(input_image_paths) == len(segment_frames) + 1:
            # N+1 pattern: calculate halfway points between images
            for i in range(len(image_frame_positions) - 1):
                current_img_frame = image_frame_positions[i]
                next_img_frame = image_frame_positions[i + 1]
                halfway_frame = (current_img_frame + next_img_frame) / 2

                if current_frame >= halfway_frame:
                    active_image_index = i + 1
                else:
                    break
            # Clamp to last image
            active_image_index = min(active_image_index, len(input_image_paths) - 1)
        else:
            # N pattern: use segment boundaries
            current_segment = 0
            for i, (start, end) in enumerate(segment_boundaries):
                if start <= current_frame <= end:
                    current_segment = i
                    break
            if current_frame > segment_boundaries[-1][1]:
                current_segment = len(segment_boundaries) - 1
            active_image_index = current_segment

        # Calculate image positions
        if vertical:
            # Stack images vertically with active image at the top
            available_height = height - (2 * margin) - 50  # Reserve space for text at bottom
            num_images = len(image_frame_positions)

            # Always put active image at the top
            image_y_positions = []
            top_y = margin + 30  # Top position for active image

            # Active image goes at top
            image_y_positions.append(top_y)

            # Remaining images distributed below
            if num_images > 1:
                remaining_height = available_height - 100  # Space below the top image
                spacing_y = remaining_height / max(num_images - 1, 1)
                for i in range(1, num_images):
                    y_pos = top_y + 100 + int((i - 1) * spacing_y)
                    image_y_positions.append(y_pos)

            # Reorder y positions based on active image index
            # Move active image position to first slot
            ordered_positions = image_y_positions.copy()
            if active_image_index < len(ordered_positions):
                # Swap: put active image's position at index 0
                temp_positions = [0] * num_images
                temp_positions[active_image_index] = ordered_positions[0]  # Active gets top spot

                # Distribute others
                other_idx = 1
                for i in range(num_images):
                    if i != active_image_index:
                        temp_positions[i] = ordered_positions[other_idx]
                        other_idx += 1
                image_y_positions = temp_positions

            # For vertical layout, x is centered
            image_x_positions = [width // 2] * num_images
        else:
            # Original horizontal layout
            bar_start_x = margin
            bar_end_x = margin + available_width
            bar_width = available_width

            image_x_positions = []
            for i, frame_pos in enumerate(image_frame_positions):
                # Calculate x position based on frame position in timeline
                progress_ratio = frame_pos / (total_frames - 1) if total_frames > 1 else 0
                x_pos = bar_start_x + int(progress_ratio * bar_width)
                image_x_positions.append(x_pos)

            # For horizontal layout, y is at top
            image_y_positions = [10] * len(image_x_positions)

        # Draw thumbnails at their calculated positions
        for i, (thumb, x_center, y_center) in enumerate(zip(thumbnails, image_x_positions, image_y_positions)):
            # Determine if this image should be highlighted based on active_image_index
            should_highlight = (i == active_image_index)

            # Feature 1: Scale active image to 1.15x
            if should_highlight:
                # Scale thumbnail to 1.15x
                thumb_img = Image.fromarray(thumb)
                scaled_w = int(thumb_img.width * 1.15)
                scaled_h = int(thumb_img.height * 1.15)
                thumb_img = thumb_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

                # Recalculate position to keep centered horizontally and push up vertically
                x_start = x_center - scaled_w // 2
                # Push the active image up by 20 pixels
                y_start = y_center - scaled_h // 2 - 20
                # Ensure thumbnail doesn't overflow
                x_start = max(margin, min(x_start, width - margin - scaled_w))
                y_start = max(margin, min(y_start, height - margin - scaled_h))

                # Draw highlight border around scaled image
                border_color = (255, 100, 0)
                draw.rectangle(
                    [x_start - 3, y_start - 3,
                     x_start + scaled_w + 3, y_start + scaled_h + 3],
                    outline=border_color,
                    width=4
                )

                # Paste scaled thumbnail
                img.paste(thumb_img, (x_start, y_start))
            else:
                # Normal size thumbnail
                x_start = x_center - thumb.shape[1] // 2
                y_start = y_center - thumb.shape[0] // 2
                # Ensure thumbnail doesn't overflow
                x_start = max(margin, min(x_start, width - margin - thumb.shape[1]))
                y_start = max(margin, min(y_start, height - margin - thumb.shape[0]))

                # Paste thumbnail
                thumb_img = Image.fromarray(thumb)
                img.paste(thumb_img, (x_start, y_start))

        # Draw progress bar
        if vertical:
            # Vertical progress bar on the right side
            bar_width = 12
            bar_x = width - margin - bar_width
            bar_start_y = margin
            bar_end_y = height - margin
            bar_height = bar_end_y - bar_start_y

            # Background bar
            draw.rectangle(
                [bar_x, bar_start_y, bar_x + bar_width, bar_end_y],
                fill=(200, 200, 200),
                outline=(150, 150, 150)
            )

            # Progress fill - from top to current position
            progress = min(t / duration, 1.0)
            progress_y = bar_start_y + int(bar_height * progress)

            # Ensure at least 1 pixel of progress
            if progress_y <= bar_start_y and progress > 0:
                progress_y = bar_start_y + 1

            draw.rectangle(
                [bar_x, bar_start_y, bar_x + bar_width, progress_y],
                fill=(0, 150, 255),
                outline=None
            )
        else:
            # Horizontal progress bar
            max_thumb_h = max(thumb.shape[0] for thumb in thumbnails)

            bar_y = 10 + max_thumb_h + 10
            bar_height = 12

            # Background bar
            draw.rectangle(
                [bar_start_x, bar_y, bar_end_x, bar_y + bar_height],
                fill=(200, 200, 200),
                outline=(150, 150, 150)
            )

            # Progress fill - aligned from bar_start_x to current position
            progress = min(t / duration, 1.0)
            progress_x = bar_start_x + int(bar_width * progress)

            # Ensure at least 1 pixel of progress
            if progress_x <= bar_start_x and progress > 0:
                progress_x = bar_start_x + 1

            draw.rectangle(
                [bar_start_x, bar_y, progress_x, bar_y + bar_height],
                fill=(0, 150, 255),
                outline=None
            )

        # Draw image markers on the progress bar (only for horizontal layout)
        if not vertical:
            for i, x_pos in enumerate(image_x_positions):
                # Determine if this marker should be highlighted (same as active_image_index)
                marker_active = (i == active_image_index)

                # Draw vertical line marker
                marker_color = (255, 100, 0) if marker_active else (100, 100, 100)
                draw.line(
                    [(x_pos, bar_y - 5), (x_pos, bar_y + bar_height + 5)],
                    fill=marker_color,
                    width=3 if marker_active else 2
                )

                # Draw small circle at marker position
                circle_r = 4
                draw.ellipse(
                    [x_pos - circle_r, bar_y + bar_height//2 - circle_r,
                     x_pos + circle_r, bar_y + bar_height//2 + circle_r],
                    fill=marker_color,
                    outline=(255, 255, 255)
                )

        # Calculate current segment for text display (based on segment boundaries)
        current_segment = 0
        for i, (start, end) in enumerate(segment_boundaries):
            if start <= current_frame <= end:
                current_segment = i
                break
        if current_frame > segment_boundaries[-1][1]:
            current_segment = len(segment_boundaries) - 1

        # Draw text info
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        text = f"Frame {current_frame}/{total_frames-1} | Segment {current_segment + 1}/{len(segment_boundaries)}"
        if segment_prompts and current_segment < len(segment_prompts):
            prompt_preview = segment_prompts[current_segment][:50]
            text += f" | {prompt_preview}..."

        # Position text below progress bar (horizontal) or at bottom (vertical)
        if vertical:
            text_y = height - margin - 30
        else:
            text_y = bar_y + 20
        draw.text((margin, text_y), text, fill=(50, 50, 50), font=font)

        return np.array(img)

    return VideoClip(make_frame, duration=duration)


def create_simple_comparison(
    video1_path: str,
    video2_path: str,
    output_path: str,
    labels: Optional[Tuple[str, str]] = None,
    orientation: str = "horizontal"
) -> str:
    """
    Create a simple side-by-side or stacked comparison of two videos.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Where to save comparison
        labels: Optional tuple of (label1, label2)
        orientation: "horizontal" or "vertical"

    Returns:
        Path to created comparison video
    """
    try:
        from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
    except ImportError:
        raise ImportError("MoviePy is required. Install with: pip install moviepy")

    print(f"📹 Creating comparison video...")

    # Load videos
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)

    # Add labels if provided
    if labels:
        try:
            from moviepy.editor import TextClip
            label1 = TextClip(labels[0], fontsize=24, color='white', bg_color='black')
            label1 = label1.set_duration(clip1.duration).set_position(("center", "top"))
            clip1 = CompositeVideoClip([clip1, label1])

            label2 = TextClip(labels[1], fontsize=24, color='white', bg_color='black')
            label2 = label2.set_duration(clip2.duration).set_position(("center", "top"))
            clip2 = CompositeVideoClip([clip2, label2])
        except:
            print("⚠️ Could not add labels (text rendering failed)")

    # Arrange clips
    if orientation == "horizontal":
        final = clips_array([[clip1, clip2]])
    else:  # vertical
        final = clips_array([[clip1], [clip2]])

    # Write output
    print(f"💾 Writing comparison to: {output_path}")
    final.write_videofile(
        output_path,
        codec='libx264',
        audio=False,
        preset='medium',
        logger=None
    )

    # Cleanup
    clip1.close()
    clip2.close()
    final.close()

    print(f"✅ Comparison saved: {output_path}")
    return output_path


# Lightweight alternative using OpenCV (no MoviePy dependency)
def create_opencv_side_by_side(
    video1_path: str,
    video2_path: str,
    output_path: str,
    fps: Optional[int] = None
) -> str:
    """
    Create side-by-side comparison using only OpenCV (faster, no MoviePy needed).

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Where to save comparison
        fps: Output FPS (defaults to video1 FPS)

    Returns:
        Path to created comparison video
    """
    print(f"📹 Creating OpenCV side-by-side comparison...")

    # Open videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get properties
    if fps is None:
        fps = int(cap1.get(cv2.CAP_PROP_FPS))

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Match heights
    target_height = min(height1, height2)
    target_width1 = int(width1 * target_height / height1)
    target_width2 = int(width2 * target_height / height2)

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (target_width1 + target_width2, target_height)
    )

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Resize frames
        frame1 = cv2.resize(frame1, (target_width1, target_height))
        frame2 = cv2.resize(frame2, (target_width2, target_height))

        # Concatenate horizontally
        combined = np.hstack([frame1, frame2])

        out.write(combined)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    # Cleanup
    cap1.release()
    cap2.release()
    out.release()

    print(f"✅ Comparison saved: {output_path} ({frame_count} frames)")
    return output_path
