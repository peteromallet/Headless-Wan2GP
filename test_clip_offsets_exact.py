"""
Test different frame offsets using the EXACT same approach as join_clips.py
"""

import sys
from pathlib import Path

# Import the same functions join_clips uses
from source.video_utils import (
    extract_frames_from_video,
    stitch_videos_with_crossfade,
    create_video_from_frames_list
)
from source.common_utils import get_video_frame_count_and_fps

def test_offset(bridge_path, clip2_path, offset, output_dir, blend_frames=3):
    """Test a specific offset using the exact same logic as join_clips.py"""
    
    print(f"\n{'='*80}")
    print(f"Testing offset: {offset:+d} frames")
    print(f"{'='*80}")
    
    # Get video info (same as join_clips.py)
    bridge_frame_count, fps = get_video_frame_count_and_fps(str(bridge_path))
    clip2_frame_count, _ = get_video_frame_count_and_fps(str(clip2_path))
    
    print(f"Bridge: {bridge_frame_count} frames @ {fps} fps")
    print(f"Clip2: {clip2_frame_count} frames @ {fps} fps")
    
    # Calculate clip2 start frame based on offset
    # Base calculation (what the code currently does): 0
    # With offset: 0 + offset
    base_skip = 0
    frames_to_skip_clip2 = base_skip + offset
    
    if frames_to_skip_clip2 < 0:
        print(f"WARNING: Offset {offset} results in negative skip. Clamping to 0.")
        frames_to_skip_clip2 = 0
    
    if frames_to_skip_clip2 >= clip2_frame_count:
        print(f"ERROR: Offset {offset} exceeds video length. Skipping.")
        return None
    
    print(f"Clip2 will skip first {frames_to_skip_clip2} frames (offset: {offset:+d})")
    
    # Extract all frames from bridge
    print("Extracting frames from bridge...")
    bridge_frames = extract_frames_from_video(str(bridge_path), dprint_func=lambda *args: None)
    print(f"  Extracted {len(bridge_frames)} frames")
    
    # Extract frames from clip2, starting at offset
    print(f"Extracting frames from clip2 starting at frame {frames_to_skip_clip2}...")
    clip2_frames = extract_frames_from_video(
        str(clip2_path), 
        start_frame=frames_to_skip_clip2,
        dprint_func=lambda *args: None
    )
    print(f"  Extracted {len(clip2_frames)} frames")
    
    # Create temporary video files (same approach as join_clips.py)
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=output_dir) as bridge_tmp:
        bridge_tmp_path = Path(bridge_tmp.name)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=output_dir) as clip2_tmp:
        clip2_tmp_path = Path(clip2_tmp.name)
    
    # Write frames to temporary videos
    print("Creating temporary videos...")
    from source.video_utils import create_video_from_frames_list
    
    height, width = bridge_frames[0].shape[:2]
    resolution_wh = (width, height)
    
    create_video_from_frames_list(bridge_frames, bridge_tmp_path, fps, resolution_wh)
    create_video_from_frames_list(clip2_frames, clip2_tmp_path, fps, resolution_wh)
    
    # Use stitch_videos_with_crossfade (EXACT same as join_clips.py)
    output_path = output_dir / f"test_offset_{offset:+03d}.mp4"
    
    print(f"Stitching with {blend_frames}-frame crossfade...")
    
    try:
        stitch_videos_with_crossfade(
            video_paths=[bridge_tmp_path, clip2_tmp_path],
            blend_frame_counts=[blend_frames],
            output_path=output_path,
            fps=fps,
            crossfade_mode="linear_sharp",
            crossfade_sharp_amt=0.3,
            dprint=lambda *args: None
        )
        
        # Add text overlay showing offset
        import subprocess
        output_with_text = output_dir / f"test_offset_{offset:+03d}_labeled.mp4"
        subprocess.run([
            'ffmpeg', '-y', '-i', str(output_path),
            '-vf', f"drawtext=text='Offset\\: {offset:+d}':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.8:x=(w-text_w)/2:y=50",
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            str(output_with_text)
        ], check=True, capture_output=True)
        
        # Remove unlabeled version
        output_path.unlink()
        output_with_text.rename(output_path)
        
        print(f"✓ Created: {output_path}")
        
        # Cleanup temp files
        try:
            bridge_tmp_path.unlink()
            clip2_tmp_path.unlink()
        except:
            pass
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_clip_offsets_exact.py <bridge.mp4> <clip2.mp4> [blend_frames]")
        sys.exit(1)
    
    print(f"DEBUG: sys.argv = {sys.argv}")
    print(f"DEBUG: len(sys.argv) = {len(sys.argv)}")
    if len(sys.argv) > 3:
        print(f"DEBUG: sys.argv[3] = '{sys.argv[3]}' (len={len(sys.argv[3])})")

    bridge_path = Path(sys.argv[1])
    clip2_path = Path(sys.argv[2])
    blend_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    if not bridge_path.exists():
        print(f"Error: Bridge not found: {bridge_path}")
        sys.exit(1)
    
    if not clip2_path.exists():
        print(f"Error: Clip2 not found: {clip2_path}")
        sys.exit(1)
    
    output_dir = Path("test_offsets_exact")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("CLIP OFFSET TESTING (using exact join_clips.py approach)")
    print("="*80)
    print(f"Bridge: {bridge_path}")
    print(f"Clip2: {clip2_path}")
    print(f"Blend frames: {blend_frames}")
    print(f"Output directory: {output_dir}")
    
    results = []
    for offset in range(-4, 5):
        result = test_offset(bridge_path, clip2_path, offset, output_dir, blend_frames)
        if result:
            results.append((offset, result))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Generated {len(results)} test videos:")
    for offset, path in results:
        print(f"  Offset {offset:+2d}: {path}")
    print(f"\nAll videos saved to: {output_dir.absolute()}")
    print("\nReview each video to find the offset with the smoothest transition.")

if __name__ == "__main__":
    main()
