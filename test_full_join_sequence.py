"""
Test different frame offsets for the FULL join sequence: starting → bridge → ending
This matches the exact flow of join_clips.py
"""

import sys
from pathlib import Path

from source.video_utils import (
    extract_frames_from_video,
    stitch_videos_with_crossfade,
    create_video_from_frames_list
)
from source.common_utils import get_video_frame_count_and_fps

def test_full_sequence(starting_path, bridge_path, ending_path, offset, output_dir, 
                       context_frame_count=24, gap_frame_count=16, blend_frames=3):
    """Test the full 3-video sequence with different clip2 offsets."""
    
    print(f"\n{'='*80}")
    print(f"Testing FULL SEQUENCE with offset: {offset:+d} frames")
    print(f"{'='*80}")
    
    # Get video info
    start_frame_count, fps = get_video_frame_count_and_fps(str(starting_path))
    bridge_frame_count, _ = get_video_frame_count_and_fps(str(bridge_path))
    end_frame_count, _ = get_video_frame_count_and_fps(str(ending_path))
    
    print(f"Starting video: {start_frame_count} frames @ {fps} fps")
    print(f"Bridge video: {bridge_frame_count} frames")
    print(f"Ending video: {end_frame_count} frames")
    print(f"Parameters: context={context_frame_count}, gap={gap_frame_count}, blend={blend_frames}")
    
    # REPLACE MODE calculations (matching the logs)
    frames_to_replace_from_before = gap_frame_count // 2
    frames_to_replace_from_after = gap_frame_count - frames_to_replace_from_before
    num_preserved_after = context_frame_count - frames_to_replace_from_after
    
    # Trim starting video (same as join_clips.py)
    frames_to_remove_clip1 = context_frame_count - blend_frames
    frames_to_keep_clip1 = start_frame_count - frames_to_remove_clip1
    
    print(f"\nClip1 (starting) trimming:")
    print(f"  Keep first {frames_to_keep_clip1} frames (remove last {frames_to_remove_clip1})")
    
    # Trim ending video with offset
    # Base calculation from logs: frames_to_skip_clip2 = 22 (with +1 fix)
    # Without fix: 21
    base_frames_to_skip = frames_to_replace_from_after + (num_preserved_after - blend_frames)
    frames_to_skip_clip2 = base_frames_to_skip + offset
    
    if frames_to_skip_clip2 < 0:
        print(f"WARNING: Offset {offset} results in negative skip. Clamping to 0.")
        frames_to_skip_clip2 = 0
    
    if frames_to_skip_clip2 >= end_frame_count:
        print(f"ERROR: Offset {offset} exceeds video length. Skipping.")
        return None
    
    print(f"\nClip2 (ending) trimming:")
    print(f"  Base skip: {base_frames_to_skip} frames")
    print(f"  With offset {offset:+d}: skip {frames_to_skip_clip2} frames")
    print(f"  Preserved section in bridge: clip2[{frames_to_replace_from_after}:{context_frame_count}]")
    
    # Extract frames
    print(f"\nExtracting frames...")
    starting_frames = extract_frames_from_video(str(starting_path), num_frames=frames_to_keep_clip1, dprint_func=lambda *args: None)
    bridge_frames = extract_frames_from_video(str(bridge_path), dprint_func=lambda *args: None)
    ending_frames = extract_frames_from_video(str(ending_path), start_frame=frames_to_skip_clip2, dprint_func=lambda *args: None)
    
    print(f"  Starting: {len(starting_frames)} frames")
    print(f"  Bridge: {len(bridge_frames)} frames")
    print(f"  Ending: {len(ending_frames)} frames")
    
    # Create temp videos
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=output_dir) as f1:
        start_tmp = Path(f1.name)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=output_dir) as f2:
        bridge_tmp = Path(f2.name)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=output_dir) as f3:
        end_tmp = Path(f3.name)
    
    height, width = starting_frames[0].shape[:2]
    resolution_wh = (width, height)
    
    create_video_from_frames_list(starting_frames, start_tmp, fps, resolution_wh)
    create_video_from_frames_list(bridge_frames, bridge_tmp, fps, resolution_wh)
    create_video_from_frames_list(ending_frames, end_tmp, fps, resolution_wh)
    
    # Stitch all 3 videos (matching join_clips.py)
    output_path = output_dir / f"full_sequence_offset_{offset:+03d}.mp4"
    
    print(f"\nStitching 3 videos with {blend_frames}-frame crossfades...")
    
    try:
        stitch_videos_with_crossfade(
            video_paths=[start_tmp, bridge_tmp, end_tmp],
            blend_frame_counts=[blend_frames, blend_frames],  # Two boundaries
            output_path=output_path,
            fps=fps,
            crossfade_mode="linear_sharp",
            crossfade_sharp_amt=0.3,
            dprint=lambda *args: None
        )
        
        # Add text overlay
        import subprocess
        output_with_text = output_dir / f"full_sequence_offset_{offset:+03d}_labeled.mp4"
        subprocess.run([
            'ffmpeg', '-y', '-i', str(output_path),
            '-vf', f"drawtext=text='Full Sequence - Offset\\: {offset:+d}':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.8:x=(w-text_w)/2:y=50",
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            str(output_with_text)
        ], check=True, capture_output=True)
        
        output_path.unlink()
        output_with_text.rename(output_path)
        
        print(f"✓ Created: {output_path}")
        
        # Cleanup
        try:
            start_tmp.unlink()
            bridge_tmp.unlink()
            end_tmp.unlink()
        except:
            pass
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) < 4:
        print("Usage: python test_full_join_sequence.py <starting.mp4> <bridge.mp4> <ending.mp4>")
        sys.exit(1)
    
    starting_path = Path(sys.argv[1])
    bridge_path = Path(sys.argv[2])
    ending_path = Path(sys.argv[3])
    
    for p, name in [(starting_path, "Starting"), (bridge_path, "Bridge"), (ending_path, "Ending")]:
        if not p.exists():
            print(f"Error: {name} video not found: {p}")
            sys.exit(1)
    
    output_dir = Path("test_full_sequence")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("FULL JOIN SEQUENCE TESTING (matching join_clips.py)")
    print("="*80)
    print(f"Starting: {starting_path}")
    print(f"Bridge: {bridge_path}")
    print(f"Ending: {ending_path}")
    print(f"Output directory: {output_dir}")
    
    results = []
    for offset in range(-4, 5):
        result = test_full_sequence(starting_path, bridge_path, ending_path, offset, output_dir)
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
