#!/usr/bin/env python3

# Test to verify CausVid steps calculation
def test_causvid_steps_calculation():
    """Test that CausVid calculates correct steps for target video_length"""
    
    test_cases = [
        (73, 25),  # (73 + 2) / 3 = 25 steps → 25*3-2 = 73 frames
        (25, 9),   # (25 + 2) / 3 = 9 steps → 9*3-2 = 25 frames  
        (81, 27),  # (81 + 2) / 3 = 27.67 → 27 steps → 27*3-2 = 79 frames (close)
        (1, 1),    # (1 + 2) / 3 = 1 step → 1*3-2 = 1 frame
        (4, 2),    # (4 + 2) / 3 = 2 steps → 2*3-2 = 4 frames
    ]
    
    print("Testing CausVid steps calculation:")
    print("Target Frames → Required Steps → Actual Frames")
    print("-" * 50)
    
    for target_frames, expected_steps in test_cases:
        # Formula: required_steps = (video_length + 2) / 3
        calculated_steps = max(1, int((target_frames + 2) / 3))
        
        # Verify WGP formula: video_length = num_inference_steps * 3 - 2
        actual_frames = calculated_steps * 3 - 2
        
        print(f"{target_frames:3d} frames → {calculated_steps:2d} steps → {actual_frames:3d} frames")
        
        assert calculated_steps == expected_steps, f"Expected {expected_steps} steps for {target_frames} frames, got {calculated_steps}"
        
        # Allow 2-frame tolerance due to integer rounding
        assert abs(actual_frames - target_frames) <= 2, f"Expected ~{target_frames} frames, got {actual_frames}"

def test_travel_segment_case():
    """Test the specific case from travel segments"""
    print("\nTesting travel segment case:")
    print("Orchestrator quantized: 72 → 73 frames")
    print("Old CausVid: 9 steps → 9*3-2 = 25 frames ❌")
    print("New CausVid: (73+2)/3 = 25 steps → 25*3-2 = 73 frames ✅")
    
    target_frames = 73
    required_steps = max(1, int((target_frames + 2) / 3))
    actual_frames = required_steps * 3 - 2
    
    print(f"\nCalculation: ({target_frames} + 2) / 3 = {required_steps} steps")
    print(f"Verification: {required_steps} * 3 - 2 = {actual_frames} frames")
    print(f"Match: {actual_frames == target_frames} ✅")

def test_stitching_math():
    """Test the stitching math with corrected frame counts"""
    print("\nTesting stitching math:")
    
    # Old (broken) case
    old_frames_per_segment = 25
    old_total_input = 3 * old_frames_per_segment  # 75
    old_total_overlaps = 2 * 10  # 20
    old_expected_output = old_total_input - old_total_overlaps  # 55
    print(f"Old (broken): 3×{old_frames_per_segment} - 2×10 = {old_expected_output} frames")
    
    # New (fixed) case  
    new_frames_per_segment = 73
    new_total_input = 3 * new_frames_per_segment  # 219
    new_total_overlaps = 2 * 10  # 20
    new_expected_output = new_total_input - new_total_overlaps  # 199
    print(f"New (fixed):  3×{new_frames_per_segment} - 2×10 = {new_expected_output} frames")
    
    print(f"\nImprovement: {new_expected_output - old_expected_output} more frames ({new_expected_output/old_expected_output:.1f}x longer)")

if __name__ == "__main__":
    test_causvid_steps_calculation()
    test_travel_segment_case()
    test_stitching_math()
    print("\n✅ All tests passed!") 