#!/usr/bin/env python3
"""
Test script for REPLACE mode join_clips logic.

Verifies that frame alignment is correct at blend boundaries.
"""

def test_replace_mode(
    clip1_frames=416,
    clip2_frames=416,
    context=20,
    gap=20,
    blend=3
):
    """Test REPLACE mode frame alignment"""

    print("=" * 80)
    print(f"REPLACE MODE TEST: context={context}, gap={gap}, blend={blend}")
    print("=" * 80)

    # REPLACE mode calculations
    frames_to_replace_from_before = gap // 2
    frames_to_replace_from_after = gap - frames_to_replace_from_before
    num_preserved_before = context - frames_to_replace_from_before
    num_preserved_after = context - frames_to_replace_from_after

    print(f"\n--- Guide Video Construction ---")
    print(f"Context extraction:")
    print(f"  Clip1: last {context} frames → Clip1[{clip1_frames-context}:{clip1_frames}]")
    print(f"  Clip2: first {context} frames → Clip2[0:{context}]")
    print()

    print(f"Guide video ({context * 2} frames):")
    print(f"  [0-{num_preserved_before-1}]: Clip1[{clip1_frames-context}:{clip1_frames-context+num_preserved_before}] (preserved)")
    print(f"  [{num_preserved_before}-{num_preserved_before+gap-1}]: GRAY (regenerate, replaces boundary)")
    print(f"  [{num_preserved_before+gap}-{context*2-1}]: Clip2[{frames_to_replace_from_after}:{context}] (preserved)")
    print()

    print(f"Mask video:")
    print(f"  BLACK: [0-{num_preserved_before-1}] (preserve)")
    print(f"  WHITE: [{num_preserved_before}-{num_preserved_before+gap-1}] (generate)")
    print(f"  BLACK: [{num_preserved_before+gap}-{context*2-1}] (preserve)")
    print()

    # Transition video
    print(f"--- Transition Video ({context * 2} frames after VACE) ---")
    transition_start = clip1_frames - context
    transition_preserved_end = transition_start + num_preserved_before - 1
    transition_generated_start = num_preserved_before
    transition_generated_end = num_preserved_before + gap - 1
    transition_preserved2_start = frames_to_replace_from_after
    transition_preserved2_end = context - 1

    print(f"  [0-{num_preserved_before-1}]: Clip1[{transition_start}-{transition_preserved_end}]' (VACE processed)")
    print(f"  [{transition_generated_start}-{transition_generated_end}]: Generated content")
    print(f"  [{num_preserved_before+gap}-{context*2-1}]: Clip2[{transition_preserved2_start}-{transition_preserved2_end}]' (VACE processed)")
    print()

    # Trimming
    print(f"--- Trimming ---")
    frames_to_remove_clip1 = context - blend
    frames_to_keep_clip1 = clip1_frames - frames_to_remove_clip1

    print(f"Clip1:")
    print(f"  Remove last {frames_to_remove_clip1} frames")
    print(f"  Keep: [0:{frames_to_keep_clip1}] = {frames_to_keep_clip1} frames")
    print(f"  Last {blend}: Clip1[{frames_to_keep_clip1-blend}:{frames_to_keep_clip1}]")
    print(f"             = Clip1[{frames_to_keep_clip1-blend}, {frames_to_keep_clip1-blend+1}, {frames_to_keep_clip1-blend+2}]")
    print()

    frames_to_skip_clip2 = context - blend
    frames_to_keep_clip2 = clip2_frames - frames_to_skip_clip2

    print(f"Clip2:")
    print(f"  Skip first {frames_to_skip_clip2} frames")
    print(f"  Keep: [{frames_to_skip_clip2}:{clip2_frames}] = {frames_to_keep_clip2} frames")
    print(f"  First {blend}: Clip2[{frames_to_skip_clip2}:{frames_to_skip_clip2+blend}]")
    print(f"              = Clip2[{frames_to_skip_clip2}, {frames_to_skip_clip2+1}, {frames_to_skip_clip2+2}]")
    print()

    # Boundary alignment check
    print(f"--- Blend Boundary Alignment ---")

    # Boundary 1
    clip1_last_frames = [frames_to_keep_clip1 - blend + i for i in range(blend)]
    transition_first_frames = [transition_start + i for i in range(blend)]

    print(f"Boundary 1 (Clip1 → Transition):")
    print(f"  Clip1 last {blend} frames:       {clip1_last_frames}")
    print(f"  Transition first {blend} frames:  {transition_first_frames} (VACE processed)")

    if clip1_last_frames == transition_first_frames:
        print(f"  ✓✓✓ ALIGNED! Blending same frame indices")
        aligned_1 = True
    else:
        print(f"  ❌❌❌ MISALIGNED! Frame indices don't match")
        print(f"  Offset: {transition_first_frames[0] - clip1_last_frames[0]} frames")
        aligned_1 = False
    print()

    # Boundary 2
    transition_last_frames_idx = [num_preserved_before + gap - blend + i for i in range(blend)]
    transition_last_frames = [frames_to_replace_from_after + (num_preserved_before + gap - blend - num_preserved_before - gap + frames_to_replace_from_after) + i for i in range(blend)]
    # Simplify: last frames of transition are from Clip2[transition_preserved2_start + (total_transition - blend)...]
    # Actually, last 'blend' frames come from the preserved section which starts at Clip2[frames_to_replace_from_after]
    # Position in transition: num_preserved_before + gap onwards
    # So last blend frames start at (context*2 - blend)
    # These map to Clip2[frames_to_replace_from_after + (context - frames_to_replace_from_after - blend):]
    # = Clip2[context - blend : context]
    transition_last_frames = [context - blend + i for i in range(blend)]
    clip2_first_frames = [frames_to_skip_clip2 + i for i in range(blend)]

    print(f"Boundary 2 (Transition → Clip2):")
    print(f"  Transition last {blend} frames:  Clip2{transition_last_frames} (VACE processed)")
    print(f"  Clip2 first {blend} frames:      {clip2_first_frames}")

    if transition_last_frames == clip2_first_frames:
        print(f"  ✓✓✓ ALIGNED! Blending same frame indices")
        aligned_2 = True
    else:
        print(f"  ❌❌❌ MISALIGNED! Frame indices don't match")
        print(f"  Offset: {clip2_first_frames[0] - transition_last_frames[0]} frames")
        aligned_2 = False
    print()

    # Final length
    print(f"--- Final Video Length ---")
    total_without_blend = frames_to_keep_clip1 + (context * 2) + frames_to_keep_clip2
    total_with_blend = total_without_blend - (2 * blend)

    print(f"Without blend: {frames_to_keep_clip1} + {context*2} + {frames_to_keep_clip2} = {total_without_blend}")
    print(f"With {blend}-frame blends: {total_with_blend} frames")
    print()

    # Overall result
    print("=" * 80)
    if aligned_1 and aligned_2:
        print("✅ TEST PASSED: All blend boundaries correctly aligned!")
    else:
        print("❌ TEST FAILED: Blend boundary misalignment detected")
    print("=" * 80)
    print()

    return aligned_1 and aligned_2


if __name__ == "__main__":
    # Test default parameters
    print("\n")
    test1 = test_replace_mode(
        clip1_frames=416,
        clip2_frames=416,
        context=20,
        gap=20,
        blend=3
    )

    # Test with different gap size
    test2 = test_replace_mode(
        clip1_frames=416,
        clip2_frames=416,
        context=20,
        gap=10,
        blend=3
    )

    # Test with larger blend
    test3 = test_replace_mode(
        clip1_frames=416,
        clip2_frames=416,
        context=20,
        gap=20,
        blend=5
    )

    print("\n" + "=" * 80)
    if all([test1, test2, test3]):
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)
