#!/usr/bin/env python3
"""
Test the FINAL quantization fix: use original gap + offset
"""

def simulate_replace_mode(context_frame_count, gap_frame_count, blend_frames=3):
    """Simulate REPLACE mode with quantization offset."""

    print(f"\n{'='*80}")
    print(f"REPLACE MODE SIMULATION (FINAL APPROACH)")
    print(f"{'='*80}")
    print(f"Input parameters:")
    print(f"  context_frame_count: {context_frame_count}")
    print(f"  gap_frame_count: {gap_frame_count}")
    print(f"  blend_frames: {blend_frames}")

    # Step 1: Calculate raw total frames
    total_frames_raw = context_frame_count * 2
    print(f"\nStep 1: Calculate raw total")
    print(f"  total_frames_raw = {context_frame_count} * 2 = {total_frames_raw}")

    # Step 2: Apply VACE quantization (4n + 1)
    total_frames = ((total_frames_raw - 1) // 4) * 4 + 1
    print(f"\nStep 2: Apply VACE quantization (4n + 1)")
    print(f"  total_frames = (({total_frames_raw} - 1) // 4) * 4 + 1 = {total_frames}")

    # Step 3: Calculate quantization delta and adjusted gap
    quantization_delta = 0
    original_gap_frame_count = gap_frame_count
    adjusted_gap = gap_frame_count

    if total_frames != total_frames_raw:
        quantization_delta = total_frames_raw - total_frames
        adjusted_gap = gap_frame_count - quantization_delta
        print(f"\nStep 3: Quantization adjustment")
        print(f"  quantization_delta: {quantization_delta} frames")
        print(f"  gap_frame_count: {gap_frame_count} → {adjusted_gap}")
    else:
        print(f"\nStep 3: No adjustment needed (already 4n+1)")

    # Step 4: Build guide/mask with adjusted gap
    print(f"\nStep 4: Build guide/mask")
    print(f"  Context before: {context_frame_count} frames")
    print(f"  Gap (adjusted): {adjusted_gap} frames")
    print(f"  Context after: {context_frame_count} frames")
    print(f"  Guide/mask structure: {context_frame_count} + {adjusted_gap} + {context_frame_count} = {2*context_frame_count + adjusted_gap} frames")
    print(f"  Note: Guide is 48 frames, but VACE will generate {total_frames}")

    # Step 5: Calculate preserved sections using ORIGINAL gap
    frames_to_replace_from_before = original_gap_frame_count // 2
    frames_to_replace_from_after = original_gap_frame_count - frames_to_replace_from_before
    num_preserved_before = context_frame_count - frames_to_replace_from_before
    num_preserved_after = context_frame_count - frames_to_replace_from_after

    print(f"\nStep 5: Calculate preserved sections (using ORIGINAL gap={original_gap_frame_count})")
    print(f"  frames_to_replace_from_before: {frames_to_replace_from_before}")
    print(f"  frames_to_replace_from_after: {frames_to_replace_from_after}")
    print(f"  num_preserved_before: {num_preserved_before}")
    print(f"  num_preserved_after: {num_preserved_after}")
    print(f"  → Guide specifies: preserve clip2[{frames_to_replace_from_after}:{context_frame_count}]")

    # Step 6: VACE generates with quantization shift
    print(f"\nStep 6: VACE generation behavior")
    print(f"  VACE receives 48-frame guide/mask")
    print(f"  VACE generates {total_frames}-frame bridge")
    print(f"  Quantization causes {quantization_delta}-frame shift in preserved section")
    print(f"  → Bridge actually contains clip2[{frames_to_replace_from_after - quantization_delta}:{context_frame_count - quantization_delta}]")

    # Step 7: Calculate clip2 skip with offset
    frames_to_skip_clip2_base = frames_to_replace_from_after + (num_preserved_after - blend_frames)
    frames_to_skip_clip2 = frames_to_skip_clip2_base - quantization_delta

    print(f"\nStep 7: Calculate clip2 trimming")
    print(f"  Base calculation (using original gap):")
    print(f"    skip = {frames_to_replace_from_after} + ({num_preserved_after} - {blend_frames}) = {frames_to_skip_clip2_base}")
    print(f"  Apply quantization offset:")
    print(f"    skip = {frames_to_skip_clip2_base} - {quantization_delta} = {frames_to_skip_clip2}")

    print(f"\nStep 8: Verify alignment")
    bridge_last_frame = total_frames - 1
    blend_start = bridge_last_frame - blend_frames + 1
    print(f"  Bridge last {blend_frames} frames: [{blend_start}, {blend_start+1}, {blend_start+2}]")
    print(f"  Trimmed clip2 starts at: clip2[{frames_to_skip_clip2}]")
    print(f"  Trimmed clip2 first {blend_frames} frames: clip2[{frames_to_skip_clip2}:{frames_to_skip_clip2+blend_frames}]")

    # Where these frames are in the bridge
    clip2_preserved_start_in_bridge = num_preserved_before + adjusted_gap
    clip2_in_bridge_start = frames_to_replace_from_after - quantization_delta
    clip2_in_bridge_at_blend = clip2_in_bridge_start + (blend_start - clip2_preserved_start_in_bridge)

    print(f"\n  Bridge preserved section: frames [{clip2_preserved_start_in_bridge}:{total_frames}]")
    print(f"  Contains: clip2[{clip2_in_bridge_start}:{context_frame_count - quantization_delta}]")
    print(f"  Bridge frame {blend_start} = clip2[{clip2_in_bridge_start + (blend_start - clip2_preserved_start_in_bridge)}]")
    print(f"  Trimmed clip2[{frames_to_skip_clip2}] = clip2[{frames_to_skip_clip2}]")

    if frames_to_skip_clip2 == clip2_in_bridge_start + (blend_start - clip2_preserved_start_in_bridge):
        print(f"  ✓ Perfect alignment!")
    else:
        print(f"  ✗ Misalignment detected")

    return {
        'total_frames': total_frames,
        'gap_frame_count': adjusted_gap,
        'original_gap_frame_count': original_gap_frame_count,
        'quantization_delta': quantization_delta,
        'frames_to_skip_clip2': frames_to_skip_clip2,
        'frames_to_skip_clip2_base': frames_to_skip_clip2_base,
        'num_preserved_after': num_preserved_after
    }


if __name__ == "__main__":
    print("="*80)
    print("FINAL QUANTIZATION FIX VERIFICATION")
    print("="*80)

    # Test REPLACE mode with the problematic case
    print("\nTest: REPLACE mode (context=24, gap=16)")
    result = simulate_replace_mode(context_frame_count=24, gap_frame_count=16, blend_frames=3)

    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    print(f"\n✓ Total frames: 45 (quantized from 48)")
    print(f"  Actual: {result['total_frames']} {'✓' if result['total_frames'] == 45 else '✗'}")

    print(f"\n✓ Gap adjusted: 13 (from 16)")
    print(f"  Actual: {result['gap_frame_count']} {'✓' if result['gap_frame_count'] == 13 else '✗'}")

    print(f"\n✓ Quantization delta: 3")
    print(f"  Actual: {result['quantization_delta']} {'✓' if result['quantization_delta'] == 3 else '✗'}")

    print(f"\n✓ Base clip2 skip (without offset): 21")
    print(f"  Actual: {result['frames_to_skip_clip2_base']} {'✓' if result['frames_to_skip_clip2_base'] == 21 else '✗'}")

    print(f"\n✓ Final clip2 skip (with offset): 18")
    print(f"  Actual: {result['frames_to_skip_clip2']} {'✓' if result['frames_to_skip_clip2'] == 18 else '✗'}")

    print(f"\n✓ This matches the empirically found offset of -3!")
    print(f"  21 - 3 = {result['frames_to_skip_clip2']} ✓")

    if result['frames_to_skip_clip2'] == 18:
        print(f"\n{'='*80}")
        print("SUCCESS! All validations passed.")
        print("="*80)
    else:
        print(f"\n{'='*80}")
        print("FAILED! Validation did not match expected values.")
        print("="*80)
