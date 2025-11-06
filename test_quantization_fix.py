#!/usr/bin/env python3
"""
Test the quantization fix by simulating the frame calculations
"""

def simulate_replace_mode(context_frame_count, gap_frame_count, blend_frames=3):
    """Simulate the REPLACE mode calculations with quantization fix"""

    print(f"\n{'='*80}")
    print(f"REPLACE MODE SIMULATION")
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

    # Step 3: Track quantization delta
    quantization_delta = 0
    original_gap_frame_count = gap_frame_count

    if total_frames != total_frames_raw:
        quantization_delta = total_frames_raw - total_frames
        gap_frame_count_adjusted = gap_frame_count - quantization_delta
        print(f"\nStep 3: Frame count changed - track delta")
        print(f"  quantization_delta: {quantization_delta} frames dropped")
        print(f"  gap_frame_count (for generation): {gap_frame_count} → {gap_frame_count_adjusted}")
        print(f"  original_gap_frame_count (for preserved calculations): {original_gap_frame_count}")
    else:
        print(f"\nStep 3: No adjustment needed (already 4n+1)")

    # Step 4: Calculate preserved sections using ORIGINAL gap (preserved sections stay same size!)
    frames_to_replace_from_before = original_gap_frame_count // 2
    frames_to_replace_from_after = original_gap_frame_count - frames_to_replace_from_before
    num_preserved_before = context_frame_count - frames_to_replace_from_before
    num_preserved_after = context_frame_count - frames_to_replace_from_after

    print(f"\nStep 4: Calculate preserved sections")
    print(f"  frames_to_replace_from_before: {frames_to_replace_from_before}")
    print(f"  frames_to_replace_from_after: {frames_to_replace_from_after}")
    print(f"  num_preserved_before: {num_preserved_before}")
    print(f"  num_preserved_after: {num_preserved_after}")

    # Step 5: Verify structure
    clip1_preserved_end = num_preserved_before
    gap_start = context_frame_count - frames_to_replace_from_before
    gap_end = gap_start + gap_frame_count
    clip2_preserved_start = frames_to_replace_from_after

    print(f"\nStep 5: Verify transition structure")
    print(f"  Clip1 preserved: frames 0-{clip1_preserved_end-1} ({clip1_preserved_end} frames)")
    print(f"  Gap section: frames {gap_start}-{gap_end-1} ({gap_frame_count} frames)")
    print(f"  Clip2 preserved: frames {clip2_preserved_start}-{context_frame_count-1} ({num_preserved_after} frames)")
    print(f"  Total: {clip1_preserved_end + gap_frame_count + num_preserved_after} frames")

    # Step 6: Calculate clip2 trimming with quantization offset
    frames_to_skip_clip2_base = frames_to_replace_from_after + (num_preserved_after - blend_frames)
    frames_to_skip_clip2 = frames_to_skip_clip2_base - quantization_delta

    print(f"\nStep 6: Calculate clip2 trimming")
    print(f"  base_skip = {frames_to_replace_from_after} + ({num_preserved_after} - {blend_frames}) = {frames_to_skip_clip2_base}")
    print(f"  quantization_offset = -{quantization_delta}")
    print(f"  frames_to_skip_clip2 = {frames_to_skip_clip2_base} - {quantization_delta} = {frames_to_skip_clip2}")

    print(f"\nAlignment verification:")
    print(f"  Bridge frame {clip2_preserved_start} should match clip2 frame 0")
    print(f"  Clip2 skips {frames_to_skip_clip2} frames, starts from frame {frames_to_skip_clip2}")
    print(f"  Bridge preserved: frames [{clip2_preserved_start}:{context_frame_count}] ({num_preserved_after} frames)")
    print(f"  Blend region: bridge frames [{context_frame_count-blend_frames}:{context_frame_count}] with clip2 frames [0:{blend_frames}]")

    return {
        'total_frames': total_frames,
        'gap_frame_count': gap_frame_count,
        'original_gap_frame_count': original_gap_frame_count,
        'quantization_delta': quantization_delta,
        'frames_to_skip_clip2': frames_to_skip_clip2,
        'clip2_preserved_start': clip2_preserved_start,
        'num_preserved_after': num_preserved_after
    }

def simulate_insert_mode(context_frame_count, gap_frame_count, blend_frames=3):
    """Simulate the INSERT mode calculations with quantization fix"""

    print(f"\n{'='*80}")
    print(f"INSERT MODE SIMULATION")
    print(f"{'='*80}")
    print(f"Input parameters:")
    print(f"  context_frame_count: {context_frame_count}")
    print(f"  gap_frame_count: {gap_frame_count}")
    print(f"  blend_frames: {blend_frames}")

    # Step 1: Calculate raw total frames
    total_frames_raw = context_frame_count * 2 + gap_frame_count
    print(f"\nStep 1: Calculate raw total")
    print(f"  total_frames_raw = {context_frame_count} * 2 + {gap_frame_count} = {total_frames_raw}")

    # Step 2: Apply VACE quantization (4n + 1)
    total_frames = ((total_frames_raw - 1) // 4) * 4 + 1
    print(f"\nStep 2: Apply VACE quantization (4n + 1)")
    print(f"  total_frames = (({total_frames_raw} - 1) // 4) * 4 + 1 = {total_frames}")

    # Step 3: Adjust gap if needed
    quantization_delta = 0
    if total_frames != total_frames_raw:
        quantization_delta = total_frames_raw - total_frames
        gap_frame_count_adjusted = gap_frame_count - quantization_delta
        print(f"\nStep 3: Frame count changed - adjust gap")
        print(f"  quantization_delta: {quantization_delta} frames dropped")
        print(f"  gap_frame_count: {gap_frame_count} → {gap_frame_count_adjusted}")
        gap_frame_count = gap_frame_count_adjusted
    else:
        print(f"\nStep 3: No adjustment needed (already 4n+1)")

    # Step 4: Verify structure
    clip1_end = context_frame_count
    gap_start = context_frame_count
    gap_end = gap_start + gap_frame_count
    clip2_start = context_frame_count + gap_frame_count

    print(f"\nStep 4: Verify transition structure")
    print(f"  Clip1 frames: 0-{clip1_end-1} ({context_frame_count} frames)")
    print(f"  Gap frames: {gap_start}-{gap_end-1} ({gap_frame_count} frames)")
    print(f"  Clip2 frames: {clip2_start}-{total_frames-1} ({context_frame_count} frames)")
    print(f"  Total: {total_frames} frames")

    # Step 5: Calculate clip2 trimming (INSERT mode uses simple formula)
    frames_to_skip_clip2 = context_frame_count - blend_frames

    print(f"\nStep 5: Calculate clip2 trimming")
    print(f"  frames_to_skip_clip2 = {context_frame_count} - {blend_frames} = {frames_to_skip_clip2}")
    print(f"  (INSERT mode: independent of gap size)")

    print(f"\nAlignment verification:")
    print(f"  Bridge frame {clip2_start} should match clip2 frame {frames_to_skip_clip2}")
    print(f"  Clip2 skips first {frames_to_skip_clip2} frames")
    print(f"  Blend region: bridge frames [{clip2_start-blend_frames}:{clip2_start}] with clip2 frames [0:{blend_frames}] (after skip)")

    return {
        'total_frames': total_frames,
        'gap_frame_count': gap_frame_count,
        'quantization_delta': quantization_delta,
        'frames_to_skip_clip2': frames_to_skip_clip2
    }

if __name__ == "__main__":
    print("="*80)
    print("QUANTIZATION FIX VERIFICATION")
    print("="*80)

    # Test REPLACE mode with the problematic case
    print("\nTest 1: REPLACE mode (context=24, gap=16)")
    result = simulate_replace_mode(context_frame_count=24, gap_frame_count=16, blend_frames=3)

    print("\n" + "="*80)
    print("EXPECTED RESULTS:")
    print("="*80)
    print(f"✓ Total frames should be 45 (not 48)")
    print(f"  Actual: {result['total_frames']} {'✓' if result['total_frames'] == 45 else '✗'}")
    print(f"\n✓ Preserved sections use original gap (16)")
    print(f"  Actual: {result['original_gap_frame_count']} {'✓' if result['original_gap_frame_count'] == 16 else '✗'}")
    print(f"  (This keeps preserved section sizes constant: {result['num_preserved_after']} frames each)")
    print(f"\n✓ Quantization delta is 3 frames")
    print(f"  Actual: {result['quantization_delta']} {'✓' if result['quantization_delta'] == 3 else '✗'}")
    print(f"\n✓ Clip2 skip should be 18 (not 21)")
    print(f"  Actual: {result['frames_to_skip_clip2']} {'✓' if result['frames_to_skip_clip2'] == 18 else '✗'}")
    print(f"\n✓ This matches the -3 offset that was manually found to work!")
    print(f"  Original calculation: 21")
    print(f"  With fix: {result['frames_to_skip_clip2']}")
    print(f"  Difference: {21 - result['frames_to_skip_clip2']} (matches -3 offset!)")

    # Test INSERT mode (no quantization needed)
    print("\n" + "="*80)
    print("\nTest 2: INSERT mode (context=8, gap=53) - already 4n+1")
    result2 = simulate_insert_mode(context_frame_count=8, gap_frame_count=53, blend_frames=3)

    # Test INSERT mode WITH quantization
    print("\n" + "="*80)
    print("\nTest 3: INSERT mode (context=8, gap=50) - requires quantization")
    result3 = simulate_insert_mode(context_frame_count=8, gap_frame_count=50, blend_frames=3)

    print("\n" + "="*80)
    print("EXPECTED RESULTS:")
    print("="*80)
    print(f"✓ Total frames should be 69 (already 4n+1)")
    print(f"  Actual: {result2['total_frames']} {'✓' if result2['total_frames'] == 69 else '✗'}")
    print(f"\n✓ Gap should remain 53 (no adjustment)")
    print(f"  Actual: {result2['gap_frame_count']} {'✓' if result2['gap_frame_count'] == 53 else '✗'}")

    print("\n" + "="*80)
    print("EXPECTED RESULTS (Test 3):")
    print("="*80)
    print(f"✓ Total frames should be 65 (quantized from 66)")
    print(f"  Actual: {result3['total_frames']} {'✓' if result3['total_frames'] == 65 else '✗'}")
    print(f"\n✓ Gap should be adjusted to 49 (from 50)")
    print(f"  Actual: {result3['gap_frame_count']} {'✓' if result3['gap_frame_count'] == 49 else '✗'}")
    print(f"\n✓ Quantization delta should be 1")
    print(f"  Actual: {result3['quantization_delta']} {'✓' if result3['quantization_delta'] == 1 else '✗'}")
    print(f"\n✓ In INSERT mode, clip2 skip is independent of gap (context - blend_frames)")
    print(f"  Expected: {8 - 3} = 5")
    print(f"  Actual: {result3['frames_to_skip_clip2']} {'✓' if result3['frames_to_skip_clip2'] == 5 else '✗'}")
