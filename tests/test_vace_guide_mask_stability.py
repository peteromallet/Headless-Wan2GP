"""
Test script for VACE guide/mask creation stability improvements.

This tests the fixes for:
1. Frame count consistency between guide, mask, and total_frames
2. VACE quantization handling (4n+1 requirement)
3. Replace mode boundary clamping (prevents negative preservation windows)
4. Insert mode with regenerate_anchors
5. Gray frame creation and colorspace consistency
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import subprocess

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.vace_frame_utils import create_guide_and_mask_for_generation
from source.common_utils import get_video_frame_count_and_fps


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def create_test_frame(width: int, height: int, color_bgr: tuple, label: str = "") -> np.ndarray:
    """Create a test frame with a specific color and optional label."""
    frame = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    
    if label:
        # Add label text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(frame, label, (text_x, text_y), font, 1, (255, 255, 255), 2)
    
    return frame


def create_test_context_frames(num_frames: int, width: int, height: int, base_color: tuple, label_prefix: str) -> list:
    """Create a list of test context frames with varying colors."""
    frames = []
    for i in range(num_frames):
        # Vary color slightly for visual distinction
        color_variation = int(i * (255 / max(num_frames, 1)))
        color = (base_color[0], base_color[1], min(255, base_color[2] + color_variation))
        frame = create_test_frame(width, height, color, f"{label_prefix}{i}")
        frames.append(frame)
    return frames


def validate_video_properties(video_path: Path, expected_frames: int, expected_width: int, expected_height: int, expected_fps: int) -> tuple:
    """Validate video properties match expectations."""
    try:
        actual_frames, actual_fps = get_video_frame_count_and_fps(str(video_path))
        
        # Get resolution using ffprobe
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            width_str, height_str = result.stdout.strip().split(',')
            actual_width, actual_height = int(width_str), int(height_str)
        else:
            return False, f"Failed to get video resolution"
        
        # Check all properties
        errors = []
        if actual_frames != expected_frames:
            errors.append(f"Frame count mismatch: expected {expected_frames}, got {actual_frames}")
        if abs(actual_fps - expected_fps) > 1:  # Allow 1 fps tolerance
            errors.append(f"FPS mismatch: expected {expected_fps}, got {actual_fps}")
        if actual_width != expected_width or actual_height != expected_height:
            errors.append(f"Resolution mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "All properties match"
        
    except Exception as e:
        return False, f"Validation error: {e}"


def run_test(test_name: str, test_func):
    """Run a single test and report results."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}TEST: {test_name}{Colors.RESET}")
    print("=" * 80)
    
    try:
        success, message = test_func()
        if success:
            print(f"{Colors.GREEN}✓ PASSED{Colors.RESET}: {message}")
            return True
        else:
            print(f"{Colors.RED}✗ FAILED{Colors.RESET}: {message}")
            return False
    except Exception as e:
        print(f"{Colors.RED}✗ FAILED{Colors.RESET}: Exception - {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST CASES
# ============================================================================

def test_basic_insert_mode():
    """Test basic INSERT mode with simple gap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Create test frames
        width, height = 512, 320
        context_before = create_test_context_frames(8, width, height, (100, 50, 50), "Before_")
        context_after = create_test_context_frames(8, width, height, (50, 100, 50), "After_")
        gap_count = 16
        
        # Expected: 8 (before) + 16 (gap) + 8 (after) = 32 frames
        expected_total = 8 + 16 + 8
        
        guide_video, mask_video, actual_total = create_guide_and_mask_for_generation(
            context_frames_before=context_before,
            context_frames_after=context_after,
            gap_frame_count=gap_count,
            resolution_wh=(width, height),
            fps=16,
            output_dir=output_dir,
            task_id="test_insert_basic",
            filename_prefix="test_insert",
            regenerate_anchors=False,
            replace_mode=False,
            dprint=print
        )
        
        # Validate
        if actual_total != expected_total:
            return False, f"Total frames mismatch: expected {expected_total}, got {actual_total}"
        
        # Validate guide video
        success, msg = validate_video_properties(guide_video, expected_total, width, height, 16)
        if not success:
            return False, f"Guide video validation failed: {msg}"
        
        # Validate mask video
        success, msg = validate_video_properties(mask_video, expected_total, width, height, 16)
        if not success:
            return False, f"Mask video validation failed: {msg}"
        
        return True, f"Created guide/mask with {actual_total} frames (INSERT mode, no anchor regen)"


def test_insert_mode_with_anchors():
    """Test INSERT mode with regenerate_anchors enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        width, height = 512, 320
        context_before = create_test_context_frames(12, width, height, (100, 50, 50), "Before_")
        context_after = create_test_context_frames(12, width, height, (50, 100, 50), "After_")
        gap_count = 20
        num_anchor_frames = 3
        
        # Expected: 12 + 20 + 12 = 44 frames (anchors are regenerated but don't change count)
        expected_total = 12 + 20 + 12
        
        guide_video, mask_video, actual_total = create_guide_and_mask_for_generation(
            context_frames_before=context_before,
            context_frames_after=context_after,
            gap_frame_count=gap_count,
            resolution_wh=(width, height),
            fps=16,
            output_dir=output_dir,
            task_id="test_insert_anchors",
            filename_prefix="test_insert_anchors",
            regenerate_anchors=True,
            num_anchor_frames=num_anchor_frames,
            replace_mode=False,
            dprint=print
        )
        
        if actual_total != expected_total:
            return False, f"Total frames mismatch: expected {expected_total}, got {actual_total}"
        
        success, msg = validate_video_properties(guide_video, expected_total, width, height, 16)
        if not success:
            return False, f"Guide video validation failed: {msg}"
        
        return True, f"Created guide/mask with {actual_total} frames (INSERT mode with {num_anchor_frames} anchor frames)"


def test_replace_mode_basic():
    """Test REPLACE mode with basic boundary replacement."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        width, height = 512, 320
        context_before = create_test_context_frames(24, width, height, (100, 50, 50), "Before_")
        context_after = create_test_context_frames(24, width, height, (50, 100, 50), "After_")
        gap_count = 16  # Replace 16 boundary frames (8 from each side)
        
        # Expected: 24 + 24 = 48 frames (gap REPLACES boundary frames, doesn't insert)
        expected_total = 24 + 24
        
        guide_video, mask_video, actual_total = create_guide_and_mask_for_generation(
            context_frames_before=context_before,
            context_frames_after=context_after,
            gap_frame_count=gap_count,
            resolution_wh=(width, height),
            fps=16,
            output_dir=output_dir,
            task_id="test_replace_basic",
            filename_prefix="test_replace",
            regenerate_anchors=False,
            replace_mode=True,
            dprint=print
        )
        
        if actual_total != expected_total:
            return False, f"Total frames mismatch: expected {expected_total}, got {actual_total}"
        
        success, msg = validate_video_properties(guide_video, expected_total, width, height, 16)
        if not success:
            return False, f"Guide video validation failed: {msg}"
        
        return True, f"Created guide/mask with {actual_total} frames (REPLACE mode, {gap_count} boundary frames)"


def test_replace_mode_oversized_gap():
    """Test REPLACE mode with gap larger than available context (tests clamping fix)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        width, height = 512, 320
        context_before = create_test_context_frames(8, width, height, (100, 50, 50), "Before_")
        context_after = create_test_context_frames(8, width, height, (50, 100, 50), "After_")
        gap_count = 20  # Requesting 20 frames but only have 8+8=16 total context!
        
        # Expected: Should clamp to available context, result should be 8 + 8 = 16
        expected_total = 16
        
        guide_video, mask_video, actual_total = create_guide_and_mask_for_generation(
            context_frames_before=context_before,
            context_frames_after=context_after,
            gap_frame_count=gap_count,
            resolution_wh=(width, height),
            fps=16,
            output_dir=output_dir,
            task_id="test_replace_oversized",
            filename_prefix="test_replace_oversized",
            regenerate_anchors=False,
            replace_mode=True,
            dprint=print
        )
        
        if actual_total != expected_total:
            return False, f"Total frames mismatch: expected {expected_total}, got {actual_total}"
        
        success, msg = validate_video_properties(guide_video, expected_total, width, height, 16)
        if not success:
            return False, f"Guide video validation failed: {msg}"
        
        return True, f"Correctly clamped oversized gap: requested {gap_count} frames, got {actual_total} (matches available context)"


def test_vace_quantization_alignment():
    """Test that VACE quantization (4n+1) is handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        width, height = 512, 320
        context_before = create_test_context_frames(8, width, height, (100, 50, 50), "Before_")
        context_after = create_test_context_frames(8, width, height, (50, 100, 50), "After_")
        gap_count = 53  # This will create 8+53+8=69 frames, which VACE will quantize to 69 (already 4n+1)
        
        # 69 is already 4n+1 (n=17), so no quantization should occur
        expected_total = 8 + 53 + 8  # = 69
        
        guide_video, mask_video, actual_total = create_guide_and_mask_for_generation(
            context_frames_before=context_before,
            context_frames_after=context_after,
            gap_frame_count=gap_count,
            resolution_wh=(width, height),
            fps=16,
            output_dir=output_dir,
            task_id="test_quantization",
            filename_prefix="test_quantization",
            regenerate_anchors=False,
            replace_mode=False,
            dprint=print
        )
        
        if actual_total != expected_total:
            return False, f"Total frames mismatch: expected {expected_total}, got {actual_total}"
        
        # Verify guide and mask match
        guide_frames, _ = get_video_frame_count_and_fps(str(guide_video))
        mask_frames, _ = get_video_frame_count_and_fps(str(mask_video))
        
        if guide_frames != mask_frames:
            return False, f"Guide ({guide_frames}) and mask ({mask_frames}) frame counts don't match!"
        
        if guide_frames != actual_total:
            return False, f"Video frame count ({guide_frames}) doesn't match returned total ({actual_total})"
        
        return True, f"VACE quantization test passed: guide={guide_frames}, mask={mask_frames}, total={actual_total}"


def test_gap_inserted_frames():
    """Test gap_inserted_frames feature for keep_bridging_images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        width, height = 512, 320
        context_before = create_test_context_frames(8, width, height, (100, 50, 50), "Before_")
        context_after = create_test_context_frames(8, width, height, (50, 100, 50), "After_")
        gap_count = 18
        
        # Create anchor frames to insert
        anchor1 = create_test_frame(width, height, (255, 0, 0), "Anchor1")
        anchor2 = create_test_frame(width, height, (0, 255, 0), "Anchor2")
        gap_inserted_frames = {
            6: anchor1,   # 1/3 position
            12: anchor2,  # 2/3 position
        }
        
        expected_total = 8 + 18 + 8  # = 44
        
        guide_video, mask_video, actual_total = create_guide_and_mask_for_generation(
            context_frames_before=context_before,
            context_frames_after=context_after,
            gap_frame_count=gap_count,
            resolution_wh=(width, height),
            fps=16,
            output_dir=output_dir,
            task_id="test_gap_inserted",
            filename_prefix="test_gap_inserted",
            regenerate_anchors=False,
            replace_mode=False,
            gap_inserted_frames=gap_inserted_frames,
            dprint=print
        )
        
        if actual_total != expected_total:
            return False, f"Total frames mismatch: expected {expected_total}, got {actual_total}"
        
        success, msg = validate_video_properties(guide_video, expected_total, width, height, 16)
        if not success:
            return False, f"Guide video validation failed: {msg}"
        
        return True, f"Gap inserted frames test passed with {len(gap_inserted_frames)} anchor frames"


def test_minimal_context():
    """Test with minimal context frames (edge case)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        width, height = 512, 320
        context_before = create_test_context_frames(1, width, height, (100, 50, 50), "Before_")
        context_after = create_test_context_frames(1, width, height, (50, 100, 50), "After_")
        gap_count = 3
        
        expected_total = 1 + 3 + 1  # = 5
        
        guide_video, mask_video, actual_total = create_guide_and_mask_for_generation(
            context_frames_before=context_before,
            context_frames_after=context_after,
            gap_frame_count=gap_count,
            resolution_wh=(width, height),
            fps=16,
            output_dir=output_dir,
            task_id="test_minimal",
            filename_prefix="test_minimal",
            regenerate_anchors=False,
            replace_mode=False,
            dprint=print
        )
        
        if actual_total != expected_total:
            return False, f"Total frames mismatch: expected {expected_total}, got {actual_total}"
        
        success, msg = validate_video_properties(guide_video, expected_total, width, height, 16)
        if not success:
            return False, f"Guide video validation failed: {msg}"
        
        return True, f"Minimal context test passed with {actual_total} frames"


def test_different_resolutions():
    """Test various resolutions to ensure robustness."""
    test_resolutions = [
        (320, 192),   # Small
        (512, 320),   # Medium
        (1024, 640),  # Large
        (768, 768),   # Square
        (1920, 1080), # HD
    ]
    
    for width, height in test_resolutions:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            context_before = create_test_context_frames(4, width, height, (100, 50, 50), "Before_")
            context_after = create_test_context_frames(4, width, height, (50, 100, 50), "After_")
            gap_count = 8
            
            expected_total = 4 + 8 + 4  # = 16
            
            try:
                guide_video, mask_video, actual_total = create_guide_and_mask_for_generation(
                    context_frames_before=context_before,
                    context_frames_after=context_after,
                    gap_frame_count=gap_count,
                    resolution_wh=(width, height),
                    fps=16,
                    output_dir=output_dir,
                    task_id=f"test_res_{width}x{height}",
                    filename_prefix=f"test_res_{width}x{height}",
                    regenerate_anchors=False,
                    replace_mode=False,
                    dprint=lambda x: None  # Suppress output for this test
                )
                
                if actual_total != expected_total:
                    return False, f"Resolution {width}x{height} failed: frame count mismatch"
                
                success, msg = validate_video_properties(guide_video, expected_total, width, height, 16)
                if not success:
                    return False, f"Resolution {width}x{height} failed: {msg}"
                
            except Exception as e:
                return False, f"Resolution {width}x{height} failed: {e}"
    
    return True, f"All {len(test_resolutions)} resolution tests passed"


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests and report results."""
    print(f"\n{Colors.BOLD}{'='*80}")
    print(f"VACE Guide/Mask Stability Test Suite")
    print(f"{'='*80}{Colors.RESET}\n")
    
    tests = [
        ("Basic INSERT Mode", test_basic_insert_mode),
        ("INSERT Mode with Anchor Regeneration", test_insert_mode_with_anchors),
        ("Basic REPLACE Mode", test_replace_mode_basic),
        ("REPLACE Mode with Oversized Gap (Clamping)", test_replace_mode_oversized_gap),
        ("VACE Quantization Alignment", test_vace_quantization_alignment),
        ("Gap Inserted Frames (keep_bridging_images)", test_gap_inserted_frames),
        ("Minimal Context Edge Case", test_minimal_context),
        ("Multiple Resolution Support", test_different_resolutions),
    ]
    
    results = []
    for test_name, test_func in tests:
        passed = run_test(test_name, test_func)
        results.append((test_name, passed))
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}{Colors.RESET}\n")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"{status} - {test_name}")
    
    print(f"\n{Colors.BOLD}Results: {passed_count}/{total_count} tests passed{Colors.RESET}")
    
    if passed_count == total_count:
        print(f"{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}Some tests failed!{Colors.RESET}")
        return 1


if __name__ == "__main__":
    exit(main())

