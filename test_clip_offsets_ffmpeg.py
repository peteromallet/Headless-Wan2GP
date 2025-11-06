"""
Test different frame offsets when attaching two clips using ffmpeg.
"""

import sys
import subprocess
from pathlib import Path
import tempfile

def get_video_info(video_path):
    """Get frame count and fps using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames,r_frame_rate',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, None
    
    parts = result.stdout.strip().split(',')
    fps_frac = parts[0].split('/')
    fps = float(fps_frac[0]) / float(fps_frac[1]) if len(fps_frac) == 2 else float(fps_frac[0])
    frame_count = int(parts[1])
    
    return frame_count, fps

def test_offset_ffmpeg(clip1_path, clip2_path, offset, output_dir, blend_frames=3):
    """Test a specific offset using ffmpeg's xfade filter."""
    print(f"\n{'='*80}")
    print(f"Testing offset: {offset:+d} frames")
    print(f"{'='*80}")
    
    clip1_frame_count, fps = get_video_info(clip1_path)
    clip2_frame_count, _ = get_video_info(clip2_path)
    
    print(f"Clip1 (bridge): {clip1_frame_count} frames @ {fps} fps")
    print(f"Clip2: {clip2_frame_count} frames @ {fps} fps")
    
    # Calculate clip2 start frame
    clip2_start_frame = 0 + offset
    
    if clip2_start_frame < 0:
        print(f"WARNING: Offset {offset} results in negative start frame. Clamping to 0.")
        clip2_start_frame = 0
    
    if clip2_start_frame >= clip2_frame_count:
        print(f"ERROR: Offset {offset} exceeds video length. Skipping.")
        return None
    
    print(f"Clip2 will start at frame {clip2_start_frame}")
    
    # Trim clip1: remove last blend_frames
    clip1_duration_frames = clip1_frame_count - blend_frames
    clip1_duration_sec = clip1_duration_frames / fps
    
    # Trim clip2: skip first (clip2_start_frame + blend_frames) frames
    clip2_skip_frames = clip2_start_frame + blend_frames
    clip2_skip_sec = clip2_skip_frames / fps
    
    print(f"Clip1: keeping first {clip1_duration_frames} frames ({clip1_duration_sec:.3f}s)")
    print(f"Clip2: skipping first {clip2_skip_frames} frames ({clip2_skip_sec:.3f}s)")
    
    # Create trimmed clips
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp1:
        clip1_trimmed = tmp1.name
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp2:
        clip2_trimmed = tmp2.name
    
    try:
        # Trim clip1
        subprocess.run([
            'ffmpeg', '-y', '-i', str(clip1_path),
            '-t', str(clip1_duration_sec),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            clip1_trimmed
        ], check=True, capture_output=True)
        
        # Trim clip2
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(clip2_skip_sec), '-i', str(clip2_path),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            clip2_trimmed
        ], check=True, capture_output=True)
        
        # Now concatenate with crossfade using xfade filter
        output_path = output_dir / f"test_offset_{offset:+03d}.mp4"
        
        # Calculate xfade parameters
        blend_duration = blend_frames / fps
        offset_time = clip1_duration_sec - blend_duration
        
        # Add text overlay showing offset
        filter_complex = f"[0:v][1:v]xfade=transition=fade:duration={blend_duration}:offset={offset_time}[v];[v]drawtext=text='Offset\\: {offset:+d}':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.8:x=(w-text_w)/2:y=50[outv]"
        
        subprocess.run([
            'ffmpeg', '-y',
            '-i', clip1_trimmed,
            '-i', clip2_trimmed,
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-r', str(fps),
            str(output_path)
        ], check=True, capture_output=True)
        
        print(f"✓ Created: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(e.stderr.decode() if e.stderr else "")
        return None
    finally:
        # Cleanup temp files
        try:
            Path(clip1_trimmed).unlink()
            Path(clip2_trimmed).unlink()
        except:
            pass

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_clip_offsets_ffmpeg.py <bridge.mp4> <clip2.mp4> [blend_frames]")
        sys.exit(1)
    
    clip1_path = Path(sys.argv[1])
    clip2_path = Path(sys.argv[2])
    blend_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    if not clip1_path.exists():
        print(f"Error: Bridge video not found: {clip1_path}")
        sys.exit(1)
    
    if not clip2_path.exists():
        print(f"Error: Clip2 not found: {clip2_path}")
        sys.exit(1)
    
    output_dir = Path("test_offsets")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("CLIP OFFSET TESTING (using ffmpeg)")
    print("="*80)
    print(f"Bridge: {clip1_path}")
    print(f"Clip2: {clip2_path}")
    print(f"Blend frames: {blend_frames}")
    print(f"Output directory: {output_dir}")
    
    results = []
    for offset in range(-4, 5):
        result = test_offset_ffmpeg(clip1_path, clip2_path, offset, output_dir, blend_frames)
        if result:
            results.append((offset, result))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Generated {len(results)} test videos:")
    for offset, path in results:
        print(f"  Offset {offset:+2d}: {path}")
    print(f"\nAll videos saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
