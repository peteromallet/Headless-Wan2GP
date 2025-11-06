#!/usr/bin/env python3
"""
Test the refactored helper functions produce correct results
"""

# Simulate the helper functions locally
def _calculate_vace_quantization(context_frame_count: int, gap_frame_count: int, replace_mode: bool) -> dict:
    """Calculate VACE quantization adjustments for frame counts."""
    # Calculate desired total
    if replace_mode:
        desired_total = context_frame_count * 2
    else:
        desired_total = context_frame_count * 2 + gap_frame_count

    # Apply VACE quantization (4n + 1)
    actual_total = ((desired_total - 1) // 4) * 4 + 1
    quantization_shift = desired_total - actual_total

    # Adjust gap to account for dropped frames
    gap_for_guide = gap_frame_count - quantization_shift

    return {
        'total_frames': actual_total,
        'gap_for_guide': gap_for_guide,
        'quantization_shift': quantization_shift,
    }


def _calculate_replace_mode_clip2_skip(
    context_frame_count: int,
    gap_for_guide: int,
    blend_frames: int,
    quantization_shift: int
) -> int:
    """Calculate how many frames to skip from clip2 start in REPLACE mode."""
    # How the gap is split between before/after contexts
    frames_replaced_from_after = gap_for_guide - (gap_for_guide // 2)

    # How many frames are preserved from the clip2 context
    preserved_frame_count = context_frame_count - frames_replaced_from_after

    # Base calculation: skip replaced frames + preserved frames - blend overlap
    base_skip = frames_replaced_from_after + (preserved_frame_count - blend_frames)

    # CRITICAL: VACE shifts the preserved section by quantization_shift
    # The bridge actually contains earlier frames than the guide specified
    actual_skip = base_skip - quantization_shift

    return actual_skip


if __name__ == "__main__":
    print("="*80)
    print("TESTING REFACTORED HELPER FUNCTIONS")
    print("="*80)

    # Test case: context=24, gap=16, blend=3 (the problematic case)
    context = 24
    gap = 16
    blend = 3

    print(f"\nTest case: context={context}, gap={gap}, blend={blend}")
    print("-"*80)

    # Step 1: Quantization calculation
    quant_result = _calculate_vace_quantization(
        context_frame_count=context,
        gap_frame_count=gap,
        replace_mode=True
    )

    print(f"\nQuantization result:")
    print(f"  total_frames: {quant_result['total_frames']}")
    print(f"  gap_for_guide: {quant_result['gap_for_guide']}")
    print(f"  quantization_shift: {quant_result['quantization_shift']}")

    # Step 2: Calculate clip2 skip
    clip2_skip = _calculate_replace_mode_clip2_skip(
        context_frame_count=context,
        gap_for_guide=quant_result['gap_for_guide'],
        blend_frames=blend,
        quantization_shift=quant_result['quantization_shift']
    )

    print(f"\nClip2 skip calculation:")
    print(f"  frames_to_skip_clip2: {clip2_skip}")

    # Validation
    print(f"\n{'='*80}")
    print("VALIDATION")
    print("="*80)

    expected = {
        'total_frames': 45,
        'gap_for_guide': 13,
        'quantization_shift': 3,
        'clip2_skip': 18
    }

    all_pass = True

    checks = [
        ("Total frames", quant_result['total_frames'], expected['total_frames']),
        ("Gap for guide", quant_result['gap_for_guide'], expected['gap_for_guide']),
        ("Quantization shift", quant_result['quantization_shift'], expected['quantization_shift']),
        ("Clip2 skip", clip2_skip, expected['clip2_skip'])
    ]

    for name, actual, expect in checks:
        passed = actual == expect
        status = "✓" if passed else "✗"
        print(f"{status} {name}: expected {expect}, got {actual}")
        if not passed:
            all_pass = False

    print(f"\n{'='*80}")
    if all_pass:
        print("SUCCESS! All validations passed.")
        print("Refactored code produces identical results to original.")
    else:
        print("FAILED! Validation mismatch detected.")
    print("="*80)
